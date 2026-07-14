from pathlib import Path
from types import SimpleNamespace
from datetime import date, datetime, timezone
import logging
import sqlite3

from adaptive_news_model import AdaptiveNewsModel
from congress_context import CongressTradeAnalyzer
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
