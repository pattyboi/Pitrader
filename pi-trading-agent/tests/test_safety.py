from pathlib import Path
from types import SimpleNamespace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import logging
import sqlite3
import threading

import pytest

from adaptive_news_model import AdaptiveNewsModel
from congress_context import CongressTradeAnalyzer
from news_context import NewsContext, WorldEventAnalyzer
from symbol_reference import SymbolReference
from wsb_context import WallStreetBetsAnalyzer, WallStreetBetsSnapshot
from strategy import AssetRotationStrategy
from trade_memory import TradeMemory
from main import _DropOptionalLumiwealthWarning, format_market_open_time


def test_market_open_time_is_logged_in_eastern_time() -> None:
    assert format_market_open_time(datetime(2026, 7, 14, 13, 30, tzinfo=timezone.utc)) == "9:30 AM ET"
    assert format_market_open_time(datetime(2026, 1, 14, 14, 30, tzinfo=timezone.utc)) == "9:30 AM ET"


def test_congress_context_reports_disclosure_aggregates_without_a_trade_signal() -> None:
    analyzer = CongressTradeAnalyzer(
        fetcher=lambda _url, _timeout: [
            {"ticker": "SPY", "trade_count": 8, "filer_count": 3, "purchases": 6, "sales": 1}
        ]
    )

    context = analyzer.analyze(["SPY", "QQQ"])

    assert context.available
    assert context.matched_symbols == 1
    assert "SPY: 8 disclosed trades" in context.highlights[0]
    assert "not a trading signal" in context.explanation


def test_opportunistic_probability_uses_settled_a_to_b_outcomes(tmp_path: Path) -> None:
    memory = TradeMemory(tmp_path / "memory.duckdb", 1, 10)
    memory.backfill_history(
        [
            ("2026-01-02", 100.0, 100.0, 5.0, True),
            ("2026-01-05", 101.0, 102.0, 5.0, True),  # B beat A
            ("2026-01-06", 102.0, 102.0, 5.0, True),  # B did not beat A
        ]
    )

    probability = memory.opportunity_probability()

    assert probability.observations == 2
    assert probability.wins == 1
    assert probability.probability == 0.5


def test_wsb_context_parses_public_tracker_rows() -> None:
    page = '''<tr><td><a href="https://altindex.com/ticker/nvda">NVIDIA</a></td>
    <td>140<br /><span>12%</span></td>
    <td><span class="badge--sentiment-bullish">Bullish</span></td></tr>'''

    context = WallStreetBetsAnalyzer(fetcher=lambda _url, _timeout: page).analyze(["NVDA"])

    assert context.available
    assert context.mentions[0].symbol == "NVDA"
    assert context.mentions[0].mentions == 140
    assert context.mentions[0].sentiment == "bullish"


def test_wsb_snapshot_reuses_one_fetch_for_24_hours(tmp_path: Path) -> None:
    page = '''<tr><td><a href="https://altindex.com/ticker/nvda">NVIDIA</a></td>
    <td>140<br /></td><td><span class="badge--sentiment-bullish">Bullish</span></td></tr>'''
    calls = []
    snapshot = WallStreetBetsSnapshot(
        tmp_path / "wsb.json",
        WallStreetBetsAnalyzer(fetcher=lambda _url, _timeout: calls.append(True) or page),
    )

    assert snapshot.refresh_if_due()
    assert not snapshot.refresh_if_due()
    assert snapshot.context(["NVDA"]).available
    assert len(calls) == 1


def test_only_optional_lumiwealth_api_key_warning_is_silenced() -> None:
    noise_filter = _DropOptionalLumiwealthWarning()

    assert not noise_filter.filter(logging.makeLogRecord({"msg": "LUMIWEALTH_API_KEY not set. Not sending an update to the cloud"}))
    assert noise_filter.filter(logging.makeLogRecord({"msg": "Alpaca API authentication failed"}))


def test_news_model_discards_stale_pending_return(tmp_path: Path) -> None:
    model = AdaptiveNewsModel(tmp_path / "news.json", 1, 10)
    model.update("2026-01-02", 100.0, -2)

    result = model.update("2026-01-10", 120.0, 1)

    assert result.observations == 0


def test_news_model_keeps_weekend_next_session_return(tmp_path: Path) -> None:
    model = AdaptiveNewsModel(tmp_path / "news.json", 1, 10)
    model.update("2026-01-02", 100.0, -2)

    result = model.update("2026-01-05", 102.0, 1)

    assert result.observations == 1


def test_portfolio_ignores_unmanaged_account_positions() -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy.get_positions = lambda: [
        SimpleNamespace(asset=SimpleNamespace(symbol="AAPL", asset_type="stock"), quantity="2", avg_fill_price="150.0"),
        SimpleNamespace(asset=SimpleNamespace(symbol="SPY", asset_type="stock"), quantity="3", avg_fill_price="400.0"),
    ]

    held, entry_prices = strategy._portfolio_held_positions({"SPY"})
    assert held == {"SPY": 3}
    assert entry_prices == {"SPY": 400.0}


def test_live_spread_percent_reads_bid_ask_from_the_quote() -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy.get_quote = lambda symbol: SimpleNamespace(bid=99.0, ask=101.0)

    # (101 - 99) / mid(100) * 100 = 2.0%
    assert strategy._live_spread_percent("THIN") == 2.0


def test_live_spread_percent_is_capped_against_a_bad_print() -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy.get_quote = lambda symbol: SimpleNamespace(bid=1.0, ask=50.0)

    assert strategy._live_spread_percent("BAD") == AssetRotationStrategy._PORTFOLIO_LIVE_SPREAD_CAP_PERCENT


def test_live_spread_percent_fails_open_on_a_missing_or_invalid_quote() -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)

    strategy.get_quote = lambda symbol: SimpleNamespace(bid=None, ask=101.0)
    assert strategy._live_spread_percent("NOBID") is None

    strategy.get_quote = lambda symbol: SimpleNamespace(bid=101.0, ask=99.0)  # crossed/invalid
    assert strategy._live_spread_percent("CROSSED") is None

    def _raise(symbol: str) -> None:
        raise RuntimeError("data source unavailable")

    strategy.get_quote = _raise
    assert strategy._live_spread_percent("DOWN") is None


def test_realizable_sale_price_prefers_the_live_bid_over_last_trade() -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy.get_quote = lambda symbol: SimpleNamespace(bid=99.0, ask=101.0)
    strategy.get_last_price = lambda symbol: 100.5  # would overstate an exit vs. the real bid

    assert strategy._realizable_sale_price("THIN") == 99.0


def test_realizable_sale_price_falls_back_to_last_trade_without_a_quote() -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy.get_quote = lambda symbol: None
    strategy.get_last_price = lambda symbol: 100.5

    assert strategy._realizable_sale_price("NOQUOTE") == 100.5


def test_realizable_sale_price_is_none_when_nothing_is_available() -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy.get_quote = lambda symbol: None
    strategy.get_last_price = lambda symbol: None

    assert strategy._realizable_sale_price("DARK") is None


def test_walk_forward_validation_never_uses_a_trade_to_select_itself() -> None:
    # The first five results establish a 2% gross edge. The sixth return is
    # then an out-of-sample trade selected from those five prior results.
    outcomes = AssetRotationStrategy._walk_forward_net_returns(
        [2.0, 2.0, 2.0, 2.0, 2.0, -1.0],
        round_trip_cost_percent=0.2,
        minimum_observations=5,
        entry_threshold_percent=1.0,
    )

    assert outcomes == [-1.2]


def test_walk_forward_validation_accounts_for_costs_before_selection() -> None:
    outcomes = AssetRotationStrategy._walk_forward_net_returns(
        [1.1, 1.1, 1.1, 1.1, 1.1, 3.0],
        round_trip_cost_percent=0.2,
        minimum_observations=5,
        entry_threshold_percent=1.0,
    )

    assert outcomes == []


def test_holding_horizon_is_due_on_the_next_trading_day_interval() -> None:
    assert AssetRotationStrategy._holding_is_due("2026-01-02", date(2026, 1, 5), 1)
    assert not AssetRotationStrategy._holding_is_due("2026-01-05", date(2026, 1, 5), 1)


def test_decision_memory_uses_duckdb_and_imports_legacy_sqlite(tmp_path: Path) -> None:
    legacy_path = tmp_path / ".trade_memory.sqlite3"
    with sqlite3.connect(legacy_path) as conn:
        conn.execute(
            """
            CREATE TABLE observations (
                evaluation_date TEXT PRIMARY KEY, price_a REAL, price_b REAL,
                dip_percent REAL, news_score INTEGER, signal_present INTEGER,
                decision TEXT, decision_reason TEXT, relative_return_percent REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE executions (
                id INTEGER PRIMARY KEY, evaluation_date TEXT, symbol TEXT,
                side TEXT, price REAL, quantity REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO observations VALUES
            ('2026-01-02', 100, 100, -2, NULL, 1, 'hold', 'legacy', 1.5)
            """
        )
        conn.commit()

    memory = TradeMemory(tmp_path / ".trade_memory.duckdb", 1, 10)
    result = memory.update_and_forecast("2026-01-05", 101.0, 102.0, -2.0, None, True)

    assert (tmp_path / ".trade_memory.duckdb").is_file()
    assert result.observations == 1
    memory.record_execution("2026-01-05", "SPY", "buy", 102.0, 1.0)


def test_posture_adjusted_edge_keeps_the_expected_profit_floor_unchanged() -> None:
    # A signal exactly at the minimum-profit floor must still read as exactly
    # that floor before any posture adjustment is layered on: the floor
    # itself is never touched by the reasoning pattern, only what gets added
    # or subtracted around it.
    signal = {"expected_profit": 1.0, "return_stdev": 0.0, "win_probability": 0.5}

    assert AssetRotationStrategy._posture_adjusted_edge(signal, "conservative", None, None) == 1.0
    assert AssetRotationStrategy._posture_adjusted_edge(signal, "risky", None, None) == 1.0


def test_conservative_posture_penalizes_variance_harder_than_risky() -> None:
    volatile = {"expected_profit": 3.0, "return_stdev": 4.0, "win_probability": 0.5}

    conservative = AssetRotationStrategy._posture_adjusted_edge(volatile, "conservative", None, None)
    risky = AssetRotationStrategy._posture_adjusted_edge(volatile, "risky", None, None)

    assert conservative < risky < 3.0


def test_conservative_posture_discounts_bad_news_harder_than_risky() -> None:
    signal = {"expected_profit": 2.0, "return_stdev": 0.0, "win_probability": 0.5}

    conservative = AssetRotationStrategy._posture_adjusted_edge(signal, "conservative", -8, None)
    risky = AssetRotationStrategy._posture_adjusted_edge(signal, "risky", -8, None)

    assert conservative < risky < 2.0


def test_risky_posture_leans_into_wsb_bullish_momentum_conservative_ignores_it() -> None:
    signal = {"expected_profit": 2.0, "return_stdev": 0.0, "win_probability": 0.5}

    assert AssetRotationStrategy._posture_adjusted_edge(signal, "risky", None, "bullish") > 2.0
    assert AssetRotationStrategy._posture_adjusted_edge(signal, "conservative", None, "bullish") == 2.0


def test_posture_adjusted_edge_never_exceeds_the_configured_clamp() -> None:
    extreme = {"expected_profit": 5.0, "return_stdev": 1000.0, "win_probability": 1.0}

    conservative = AssetRotationStrategy._posture_adjusted_edge(extreme, "conservative", 10, "bearish")

    assert conservative >= 5.0 - AssetRotationStrategy._POSTURE_MAX_ADJUSTMENT_PERCENT


def _exit_test_strategy(bid_by_symbol: dict[str, float | None]) -> AssetRotationStrategy:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy.parameters = {
        "portfolio_take_profit_percent": 1.0,
        "portfolio_stop_loss_percent": 0.5,
        "portfolio_holding_horizon_max_days": 15,
    }
    strategy._realizable_sale_price = lambda symbol: bid_by_symbol.get(symbol)
    return strategy


@pytest.mark.parametrize(
    ("bid", "expected_reason"),
    [
        (101.0, "take-profit"),  # +1.00% >= 1.0% target
        (99.5, "stop-loss"),  # -0.50% <= -0.5% stop
        (100.4, None),  # between bounds: left to run
        (99.6, None),  # a loss smaller than the stop: left to run
    ],
)
def test_portfolio_exit_reasons_apply_the_configured_bounds(
    bid: float, expected_reason: str | None
) -> None:
    strategy = _exit_test_strategy({"SPY": bid})
    today = date(2026, 7, 15)

    reasons = strategy._portfolio_exit_reasons(
        {"SPY": Decimal("1")}, {"SPY": 100.0}, {"SPY": today.isoformat()}, today
    )

    if expected_reason is None:
        assert reasons == {}
    else:
        assert expected_reason in reasons["SPY"]


def test_portfolio_exit_reasons_backstop_catches_a_stagnant_or_unpriceable_holding() -> None:
    strategy = _exit_test_strategy({"FLAT": 100.2, "DARK": None})
    today = date(2026, 7, 15)
    long_ago = (today - timedelta(days=20)).isoformat()

    reasons = strategy._portfolio_exit_reasons(
        {"FLAT": Decimal("1"), "DARK": Decimal("1")},
        {"FLAT": 100.0},  # DARK has no cost basis at all
        {"FLAT": long_ago, "DARK": long_ago},
        today,
    )

    assert "backstop" in reasons["FLAT"]  # between bounds for 20 days: horizon fires
    assert "backstop" in reasons["DARK"]  # unpriceable: only the horizon can exit it


def test_portfolio_exit_reasons_label_a_gain_as_take_profit_even_when_overdue() -> None:
    strategy = _exit_test_strategy({"SPY": 102.0})
    today = date(2026, 7, 15)

    reasons = strategy._portfolio_exit_reasons(
        {"SPY": Decimal("1")}, {"SPY": 100.0}, {"SPY": (today - timedelta(days=30)).isoformat()}, today
    )

    assert "take-profit" in reasons["SPY"]


def test_portfolio_exit_reasons_never_sell_a_fresh_holding_without_cost_basis() -> None:
    strategy = _exit_test_strategy({"NEW": 100.0})
    today = date(2026, 7, 15)

    reasons = strategy._portfolio_exit_reasons(
        {"NEW": Decimal("1")}, {}, {"NEW": today.isoformat()}, today
    )

    assert reasons == {}


def test_optimal_position_count_never_exceeds_the_configured_ceiling() -> None:
    identical_candidates = [(2.0, 1.0)] * 5

    assert AssetRotationStrategy._optimal_position_count(1000.0, 5.0, identical_candidates, 1) == 1


def test_optimal_position_count_is_capped_by_the_minimum_order_floor() -> None:
    # $12 of capital and a $5 minimum order can fund at most 2 positions,
    # regardless of how generous the configured ceiling or candidate pool is.
    identical_candidates = [(2.0, 1.0)] * 5

    assert AssetRotationStrategy._optimal_position_count(12.0, 5.0, identical_candidates, 5) <= 2


def test_optimal_position_count_diversifies_across_equally_good_candidates() -> None:
    # Three independent candidates with identical edge/risk: the Sharpe-like
    # score strictly improves with n under the zero-correlation assumption,
    # so spreading across all three should beat concentrating in one.
    identical_candidates = [(2.0, 1.0), (2.0, 1.0), (2.0, 1.0)]

    assert AssetRotationStrategy._optimal_position_count(300.0, 5.0, identical_candidates, 3) == 3


def test_optimal_position_count_excludes_a_much_weaker_third_candidate() -> None:
    # The first two candidates are strong and identical; the third has a
    # far worse edge-to-risk ratio and should drag the basket score down
    # enough that including it is not worth it.
    candidates = [(3.0, 1.0), (3.0, 1.0), (0.1, 5.0)]

    assert AssetRotationStrategy._optimal_position_count(300.0, 5.0, candidates, 3) == 2


def test_optimal_position_count_fails_open_to_one_on_bad_inputs() -> None:
    assert AssetRotationStrategy._optimal_position_count(100.0, 5.0, [], 3) == 1
    assert AssetRotationStrategy._optimal_position_count(0.0, 5.0, [(2.0, 1.0)], 3) == 1
    assert AssetRotationStrategy._optimal_position_count(100.0, 5.0, [(2.0, 1.0)], 0) == 1


def test_score_articles_attributes_score_to_only_the_tagged_symbol() -> None:
    articles = [
        {"headline": "Company reports layoffs", "summary": "", "symbols": ["TSLA"]},
        {"headline": "Ceasefire reached in the region", "summary": "", "symbols": ["XYZ"]},
        {"headline": "A quiet update with no news content", "summary": "", "symbols": ["ACME"]},
    ]

    scoring = WorldEventAnalyzer.score_articles(articles, lookback_hours=24, refine=False)

    assert scoring.total_score == 0  # -1 (layoffs) + 1 (ceasefire) + 0
    assert scoring.per_symbol_scores == {"TSLA": -1, "XYZ": 1, "ACME": 0}


def test_score_articles_with_refine_off_matches_score_text_exactly() -> None:
    articles = [
        {"headline": "Recession fears grow amid tariff threats", "summary": "layoffs expected"},
        {"headline": "Stimulus and rate cut announced", "summary": "trade agreement reached"},
    ]

    scoring = WorldEventAnalyzer.score_articles(articles, lookback_hours=24, refine=False)
    manual_total = sum(
        WorldEventAnalyzer.score_text(f"{a['headline']} {a.get('summary', '')}")[0] for a in articles
    )

    assert scoring.total_score == manual_total


def test_score_articles_refine_decays_older_articles_toward_the_floor() -> None:
    now = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    fresh = [{"headline": "Recession warning", "summary": "", "created_at": now}]
    stale = [{"headline": "Recession warning", "summary": "", "created_at": now - timedelta(hours=24)}]

    fresh_score = WorldEventAnalyzer.score_articles(fresh, lookback_hours=24, refine=True, now=now).total_score
    stale_score = WorldEventAnalyzer.score_articles(stale, lookback_hours=24, refine=True, now=now).total_score

    assert fresh_score == -1
    assert stale_score == 0  # -1 * floor(0.4) rounds to 0, weaker than the fresh copy


def test_score_articles_refine_dampens_duplicate_phrase_occurrences() -> None:
    now = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    articles = [
        {"headline": "Layoffs announced at Company A", "summary": "", "created_at": now},
        {"headline": "Layoffs announced at Company B", "summary": "", "created_at": now},
        {"headline": "Layoffs announced at Company C", "summary": "", "created_at": now},
    ]

    refined = WorldEventAnalyzer.score_articles(articles, lookback_hours=24, refine=True, now=now)
    unrefined = WorldEventAnalyzer.score_articles(articles, lookback_hours=24, refine=False, now=now)

    assert unrefined.total_score == -3  # three full -1 hits
    assert refined.total_score == -2  # -1 + -0.6 + -0.3 rounded


def test_symbol_reference_verifies_only_when_both_sources_agree(tmp_path: Path) -> None:
    alpaca_names = {"AAPL": "Apple Inc. Common Stock", "MYST": "Mystery Co"}
    sec_names = {"AAPL": "Apple Inc.", "ACME": "Acme Corp"}
    reference = SymbolReference(
        tmp_path / "symbols.duckdb",
        refresh_days=7,
        alpaca_fetcher=lambda url, headers: (
            {"symbol": url.rsplit("/", 1)[-1], "name": alpaca_names[url.rsplit("/", 1)[-1]]}
            if url.rsplit("/", 1)[-1] in alpaca_names
            else None
        ),
        sec_fetcher=lambda url, timeout: [{"ticker": k, "title": v} for k, v in sec_names.items()],
    )

    assert reference.refresh(["AAPL", "MYST", "ACME", "GARBAGE"], "key", "secret") is True
    assert reference.verified_symbols() == {"AAPL", "MYST", "ACME"}  # GARBAGE dropped, others kept


def test_symbol_reference_refresh_is_gated_by_the_interval(tmp_path: Path) -> None:
    calls = {"count": 0}

    def counting_fetcher(url: str, headers: dict) -> dict:
        calls["count"] += 1
        return {"symbol": "AAPL", "name": "Apple Inc."}

    reference = SymbolReference(
        tmp_path / "symbols.duckdb",
        refresh_days=7,
        alpaca_fetcher=counting_fetcher,
        sec_fetcher=lambda url, timeout: [{"ticker": "AAPL", "title": "Apple Inc."}],
    )

    assert reference.refresh(["AAPL"], "key", "secret") is True
    assert reference.refresh(["AAPL"], "key", "secret") is False
    assert calls["count"] == 1


def test_symbol_reference_refreshes_new_discovery_symbols_within_interval(tmp_path: Path) -> None:
    calls: list[str] = []
    names = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corp."}

    reference = SymbolReference(
        tmp_path / "symbols.duckdb",
        refresh_days=7,
        alpaca_fetcher=lambda url, headers: (
            calls.append(url.rsplit("/", 1)[-1])
            or {"name": names[url.rsplit("/", 1)[-1]]}
        ),
        sec_fetcher=lambda url, timeout: [
            {"ticker": symbol, "title": name} for symbol, name in names.items()
        ],
    )

    assert reference.refresh(["AAPL"], "key", "secret") is True
    assert reference.refresh(["AAPL", "MSFT"], "key", "secret") is True

    assert calls == ["AAPL", "MSFT"]
    assert reference.verified_symbols() == {"AAPL", "MSFT"}


def test_checked_submission_treats_lumibot_error_status_as_rejection() -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    messages: list[str] = []
    order = SimpleNamespace(status="unprocessed", error_message="broker said no")
    strategy.submit_order = lambda submitted: (
        setattr(submitted, "status", "error") or submitted
    )
    strategy.log_message = lambda message, **kwargs: messages.append(message)

    assert not strategy._submit_order_checked(order, "SPY buy")
    assert "Broker rejected SPY buy" in messages[-1]


def test_portfolio_rotation_is_staged_before_sell_and_rolled_back_on_rejection() -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy.parameters = {}
    strategy.vars = SimpleNamespace(portfolio_pending_rotation={})
    strategy.log_message = lambda *args, **kwargs: None
    strategy.create_order = lambda *args, **kwargs: SimpleNamespace(
        status="unprocessed", error_message=""
    )

    def reject_after_observing_staged_state(order):
        assert strategy.vars.portfolio_pending_rotation["SPY"]["to"] == "QQQ"
        order.status = "error"
        return order

    strategy.submit_order = reject_after_observing_staged_state

    accepted = strategy._submit_portfolio_rotation_sell(
        "SPY", "QQQ", Decimal("2"), 1000.0, "replacement"
    )

    assert not accepted
    assert strategy.vars.portfolio_pending_rotation == {}


def test_symbol_reference_refresh_runs_outside_trading_thread() -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy.parameters = {"symbol_reference_enabled": True}
    strategy._symbol_reference_refresh_lock = threading.Lock()
    strategy._symbol_reference_pending_symbols = set()
    strategy._symbol_reference_refresh_running = False
    strategy.log_message = lambda *args, **kwargs: None
    started = threading.Event()
    release = threading.Event()

    class BlockingReference:
        def refresh(self, symbols, api_key, secret_key):
            started.set()
            assert release.wait(timeout=2)
            return True

    strategy._symbol_reference = lambda: BlockingReference()
    strategy._refresh_symbol_reference(["SPY"])

    assert started.wait(timeout=1)
    assert strategy._symbol_reference_refresh_running
    release.set()


def test_symbol_reference_refresh_queues_symbols_seen_during_active_refresh() -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy.parameters = {"symbol_reference_enabled": True}
    strategy._symbol_reference_refresh_lock = threading.Lock()
    strategy._symbol_reference_pending_symbols = set()
    strategy._symbol_reference_refresh_running = False
    strategy.log_message = lambda *args, **kwargs: None
    first_started = threading.Event()
    release_first = threading.Event()
    finished = threading.Event()
    batches: list[list[str]] = []

    class RecordingReference:
        def refresh(self, symbols, api_key, secret_key):
            batches.append(symbols)
            if len(batches) == 1:
                first_started.set()
                assert release_first.wait(timeout=2)
            else:
                finished.set()
            return True

    strategy._symbol_reference = lambda: RecordingReference()
    strategy._refresh_symbol_reference(["SPY"])
    assert first_started.wait(timeout=1)

    strategy._refresh_symbol_reference(["QQQ"])
    release_first.set()

    assert finished.wait(timeout=1)
    assert batches == [["SPY"], ["QQQ"]]


def test_portfolio_history_requests_use_bounded_concurrency() -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy._PORTFOLIO_HISTORY_WORKERS = 2
    rendezvous = threading.Barrier(2)

    def signal(symbol: str):
        rendezvous.wait(timeout=1)
        return {"symbol": symbol}

    strategy._portfolio_signal = signal

    assert strategy._portfolio_signals(["SPY", "QQQ"]) == [
        {"symbol": "SPY"},
        {"symbol": "QQQ"},
    ]


def test_symbol_reference_scan_text_finds_untagged_company_mentions(tmp_path: Path) -> None:
    reference = SymbolReference(
        tmp_path / "symbols.duckdb",
        refresh_days=7,
        alpaca_fetcher=lambda url, headers: {"symbol": "AAPL", "name": "Apple Inc. Common Stock"},
        sec_fetcher=lambda url, timeout: [{"ticker": "AAPL", "title": "Apple Inc."}],
    )
    reference.refresh(["AAPL"], "key", "secret")

    found = reference.scan_text_for_symbols("Apple reported record quarterly earnings", {"AAPL"})

    assert found == {"AAPL"}
    assert reference.scan_text_for_symbols("Nothing relevant here", {"AAPL"}) == set()


def test_symbol_news_scores_filters_unverified_tags_and_falls_back_when_empty() -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy.parameters = {"symbol_reference_enabled": True}
    strategy.log_message = lambda *args, **kwargs: None

    class FakeReference:
        def verified_symbols(self) -> set[str]:
            return {"TSLA"}  # SPURIOUS was never recognized by either source

        def scan_text_for_symbols(self, text: str, candidates) -> set[str]:
            return set()

    strategy._symbol_reference = lambda: FakeReference()
    news_context = NewsContext(
        available=True,
        score=-4,
        per_symbol_scores={"TSLA": -2, "SPURIOUS": -5},
        per_article=[],
    )

    scores = strategy._symbol_news_scores(news_context, {"TSLA", "SPURIOUS", "UNCOVERED"})

    assert scores == {"TSLA": -2}  # SPURIOUS dropped; UNCOVERED absent (caller falls back to market-wide)


def test_symbol_news_scores_extends_coverage_via_text_scan() -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy.parameters = {"symbol_reference_enabled": True}
    strategy.log_message = lambda *args, **kwargs: None

    class FakeReference:
        def verified_symbols(self) -> set[str]:
            return set()  # nothing cached yet: fail open, no filtering

        def scan_text_for_symbols(self, text: str, candidates) -> set[str]:
            return {"AAPL"} if "Apple" in text else set()

    strategy._symbol_reference = lambda: FakeReference()
    news_context = NewsContext(
        available=True,
        score=1,
        per_symbol_scores={},
        per_article=[{"headline": "Apple beats earnings", "summary": "", "symbols": [], "score": 1}],
    )

    scores = strategy._symbol_news_scores(news_context, {"AAPL"})

    assert scores == {"AAPL": 1}
