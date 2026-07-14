from pathlib import Path
from types import SimpleNamespace
from datetime import date, datetime, timedelta, timezone
import logging
import sqlite3

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
        SimpleNamespace(asset=SimpleNamespace(symbol="AAPL", asset_type="stock"), quantity="2"),
        SimpleNamespace(asset=SimpleNamespace(symbol="SPY", asset_type="stock"), quantity="3"),
    ]

    assert strategy._portfolio_held_positions({"SPY"}) == {"SPY": 3}


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
