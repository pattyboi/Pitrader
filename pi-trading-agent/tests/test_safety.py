from pathlib import Path
from types import SimpleNamespace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
import logging
import os
import sqlite3
import stat
import subprocess
import threading

import duckdb
import pandas as pd
import pytest

import article_filter
import autonomous_universe
import llm_news
from adaptive_news_model import AdaptiveNewsModel
from autonomous_universe import AutonomousUniverse
from llm_news import LLMNewsAnalyzer
from market_sessions import is_next_trading_session
from news_context import NewsContext, WorldEventAnalyzer
from portfolio_memory import PortfolioMemory
from symbol_reference import SymbolReference
from strategy import AssetRotationStrategy
from trade_memory import TradeMemory
from main import _DropOptionalLumiwealthWarning, format_market_open_time


def test_market_open_time_is_logged_in_eastern_time() -> None:
    assert format_market_open_time(datetime(2026, 7, 14, 13, 30, tzinfo=timezone.utc)) == "9:30 AM ET"
    assert format_market_open_time(datetime(2026, 1, 14, 14, 30, tzinfo=timezone.utc)) == "9:30 AM ET"


def test_next_trading_session_uses_the_exchange_holiday_calendar() -> None:
    assert is_next_trading_session("2026-07-02", "2026-07-06")
    assert not is_next_trading_session("2026-07-13", "2026-07-16")


def test_cpu_watchdog_skips_a_negative_percentage_after_a_service_restart(tmp_path: Path) -> None:
    """A restart resets the cgroup's cpu.stat counter; the script must not log a negative %."""
    project_dir = tmp_path / "project"
    scripts_dir = project_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    script_source = Path(__file__).resolve().parent.parent / "scripts" / "cpu_watchdog.sh"
    script_path = scripts_dir / "cpu_watchdog.sh"
    script_path.write_text(script_source.read_text(encoding="utf-8"), encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

    cgroup_root = tmp_path / "cgroup"
    cgroup_dir = cgroup_root / "system.slice" / "trading-agent.service"
    cgroup_dir.mkdir(parents=True)
    cpu_stat_file = cgroup_dir / "cpu.stat"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_systemctl = fake_bin / "systemctl"
    fake_systemctl.write_text(
        "#!/usr/bin/env bash\necho '/system.slice/trading-agent.service'\n", encoding="utf-8"
    )
    fake_systemctl.chmod(fake_systemctl.stat().st_mode | stat.S_IEXEC)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["CGROUP_ROOT"] = str(cgroup_root)

    def run_sample(usage_usec: int) -> None:
        cpu_stat_file.write_text(f"usage_usec {usage_usec}\n", encoding="utf-8")
        subprocess.run([str(script_path)], check=True, env=env, cwd=str(project_dir))

    run_sample(1_000_000)  # first sample: no prior state, nothing logged
    run_sample(2_000_000)  # normal delta: logs one (large but non-negative) sample
    run_sample(500_000)  # counter reset by a restart: must be skipped, not logged negative

    log_lines = (project_dir / ".cpu_watchdog.log").read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1
    assert not log_lines[0].split(",")[1].startswith("-")


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


def test_portfolio_memory_pools_observations_across_symbols(tmp_path: Path) -> None:
    # Unlike TradeMemory (one A/B observation/day), every symbol's settled
    # history feeds the same pooled model -- this is what lets "multiple
    # symbols a day" warm up far faster than a single pair ever could.
    memory = PortfolioMemory(tmp_path / "portfolio_memory.duckdb", minimum_observations=2, maximum_observations=50)
    memory.backfill_history("SPY", [("2026-01-02", 5.0, 1.0), ("2026-01-05", 6.0, 2.0)])
    memory.backfill_history("AAPL", [("2026-01-03", 5.5, 1.5)])

    forecast = memory.update_and_forecast("2026-01-10", "MSFT", price=100.0, dip_percent=5.5, news_score=0)

    assert forecast.observations == 3
    assert forecast.ready is True


def test_portfolio_memory_warms_up_before_forecasting(tmp_path: Path) -> None:
    memory = PortfolioMemory(tmp_path / "portfolio_memory.duckdb", minimum_observations=5, maximum_observations=50)
    memory.backfill_history("SPY", [("2026-01-02", 5.0, 1.0)])

    forecast = memory.update_and_forecast("2026-01-10", "MSFT", price=100.0, dip_percent=5.5, news_score=0)

    assert forecast.ready is False
    assert forecast.predicted_edge_percent is None


def test_portfolio_memory_settlement_is_scoped_to_the_same_symbol(tmp_path: Path) -> None:
    db_path = tmp_path / "portfolio_memory.duckdb"
    memory = PortfolioMemory(db_path, minimum_observations=1, maximum_observations=50)
    memory.update_and_forecast("2026-01-02", "SPY", price=100.0, dip_percent=5.0, news_score=0)
    memory.update_and_forecast("2026-01-02", "AAPL", price=50.0, dip_percent=6.0, news_score=0)

    # Settling SPY on a later day must never resolve AAPL's still-open row --
    # a next-session return can only ever be measured from that same symbol's
    # own later price.
    memory.update_and_forecast("2026-01-05", "SPY", price=101.0, dip_percent=4.0, news_score=0)

    with duckdb.connect(str(db_path)) as conn:
        aapl_return = conn.execute(
            "SELECT next_session_return_percent FROM observations WHERE symbol = 'AAPL'"
        ).fetchone()[0]
    assert aapl_return is None


def test_portfolio_memory_regression_ignores_non_signal_observations(tmp_path: Path) -> None:
    # Broadening daily coverage to every evaluated symbol (not just today's
    # dip-signal ones) must never dilute the pooled fit with ordinary
    # non-dip market days -- signal_present is what keeps them out.
    db_path = tmp_path / "portfolio_memory.duckdb"
    memory = PortfolioMemory(db_path, minimum_observations=2, maximum_observations=50)
    memory.backfill_history("SPY", [("2026-01-02", 5.0, 1.0), ("2026-01-05", 6.0, 2.0)])
    with duckdb.connect(str(db_path)) as conn:
        memory._create_schema(conn)
        conn.execute(
            """
            INSERT INTO observations
                (evaluation_date, symbol, price, dip_percent, news_score,
                 next_session_return_percent, signal_present)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("2026-01-06", "AAPL", 100.0, 1.0, 0, 9.0, 0),
        )
        conn.commit()

    forecast = memory.update_and_forecast("2026-01-10", "MSFT", price=100.0, dip_percent=5.5, news_score=0)

    assert forecast.observations == 2


def test_portfolio_memory_records_daily_facts_for_a_non_qualifying_symbol(tmp_path: Path) -> None:
    db_path = tmp_path / "portfolio_memory.duckdb"
    memory = PortfolioMemory(db_path, minimum_observations=1, maximum_observations=50)

    memory.update_and_forecast(
        "2026-01-02",
        "AAPL",
        price=150.0,
        dip_percent=1.0,
        news_score=2,
        signal_present=False,
        live_spread_percent=0.3,
        recent_avg_volume=1_000_000.0,
        historical_expected_profit=0.8,
        historical_win_probability=0.6,
        historical_return_stdev=1.2,
    )

    with duckdb.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT signal_present, live_spread_percent, recent_avg_volume, "
            "historical_expected_profit, historical_win_probability, historical_return_stdev "
            "FROM observations WHERE symbol = 'AAPL'"
        ).fetchone()

    assert row == pytest.approx((0, 0.3, 1_000_000.0, 0.8, 0.6, 1.2))


def test_autonomous_universe_next_batch_rotates_via_duckdb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    universe = AutonomousUniverse(tmp_path / "universe.duckdb", refresh_days=7, batch_size=2)
    payload = [{"symbol": s, "tradable": True, "fractionable": True} for s in ["AAPL", "MSFT", "NVDA", "TSLA"]]
    monkeypatch.setattr(
        autonomous_universe.requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload),
    )

    first = universe.next_batch("key", "secret")
    second = universe.next_batch("key", "secret")

    assert first == ["AAPL", "MSFT"]
    assert second == ["NVDA", "TSLA"]


def test_autonomous_universe_excludes_unpriceable_symbols_from_future_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    universe = AutonomousUniverse(tmp_path / "universe.duckdb", refresh_days=7, batch_size=2)
    payload = [{"symbol": s, "tradable": True, "fractionable": True} for s in ["AAPL", "MSFT", "NVDA", "TSLA"]]
    monkeypatch.setattr(
        autonomous_universe.requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload),
    )
    universe.remember(["MSFT"])

    universe.exclude_unpriceable(["MSFT"])

    first = universe.next_batch("key", "secret")
    second = universe.next_batch("key", "secret")

    # MSFT is dropped both from the remembered set and every future rotated
    # batch, so it never resurfaces once it's confirmed to have no price data.
    assert "MSFT" not in first
    assert "MSFT" not in second
    assert first == ["AAPL"]
    assert second == ["NVDA", "TSLA"]


def test_autonomous_universe_remember_refreshes_recency_of_a_re_mentioned_symbol(tmp_path: Path) -> None:
    universe = AutonomousUniverse(tmp_path / "universe.duckdb", refresh_days=7, batch_size=5)
    universe.remember(["AAA"], limit=2)
    universe.remember(["BBB"], limit=2)
    universe.remember(["AAA"], limit=2)  # re-mention refreshes AAA's recency
    universe.remember(["CCC"], limit=2)  # trims BBB, the one never re-mentioned

    with duckdb.connect(str(tmp_path / "universe.duckdb")) as conn:
        learned = [
            row[0]
            for row in conn.execute(
                "SELECT symbol FROM learned_symbols ORDER BY last_seen_rank"
            ).fetchall()
        ]
    assert learned == ["AAA", "CCC"]


def test_autonomous_candidates_are_not_managed_until_a_buy_is_confirmed(
    tmp_path: Path,
) -> None:
    universe = AutonomousUniverse(tmp_path / "universe.duckdb", refresh_days=7, batch_size=5)
    universe.remember(["AAPL"])

    assert universe.managed_symbols() == []

    universe.remember_owned(["AAPL"])

    assert universe.managed_symbols() == ["AAPL"]

    universe.forget_owned(["AAPL"])

    assert universe.managed_symbols() == []


def test_autonomous_universe_migrates_legacy_json_once(tmp_path: Path) -> None:
    legacy = tmp_path / "universe.json"
    legacy.write_text(
        json.dumps({"symbols": ["AAPL", "MSFT"], "cursor": 1, "refreshed": "2026-07-01", "learned": ["AAPL"]})
    )
    database_path = tmp_path / "universe.duckdb"

    universe = AutonomousUniverse(database_path, refresh_days=7, batch_size=5, legacy_json_path=legacy)
    assert universe.managed_symbols() == []
    with duckdb.connect(str(database_path)) as conn:
        cursor = conn.execute("SELECT value FROM universe_state WHERE name = 'cursor'").fetchone()[0]
        learned = conn.execute("SELECT symbol FROM learned_symbols").fetchone()[0]
    assert cursor == "1"
    assert learned == "AAPL"

    # Migration only ever runs once: changing the legacy file afterward must
    # not re-import or overwrite already-migrated state.
    legacy.write_text(json.dumps({"symbols": [], "cursor": 0, "refreshed": "2026-07-02", "learned": ["ZZZZ"]}))
    universe_again = AutonomousUniverse(database_path, refresh_days=7, batch_size=5, legacy_json_path=legacy)
    assert universe_again.managed_symbols() == []


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


def test_on_abrupt_closing_dumps_every_thread_stack_to_the_diagnostic_file(tmp_path: Path) -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    diagnostic_path = tmp_path / "shutdown.log"
    strategy.parameters = {"shutdown_diagnostic_file": str(diagnostic_path)}

    strategy.on_abrupt_closing()

    assert "Current thread" in diagnostic_path.read_text(encoding="utf-8")


def test_on_abrupt_closing_is_a_no_op_without_a_configured_path() -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy.parameters = {}

    strategy.on_abrupt_closing()  # must not raise


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

    assert AssetRotationStrategy._posture_adjusted_edge(signal, "conservative", None) == 1.0
    assert AssetRotationStrategy._posture_adjusted_edge(signal, "risky", None) == 1.0


def test_conservative_posture_penalizes_variance_harder_than_risky() -> None:
    volatile = {"expected_profit": 3.0, "return_stdev": 4.0, "win_probability": 0.5}

    conservative = AssetRotationStrategy._posture_adjusted_edge(volatile, "conservative", None)
    risky = AssetRotationStrategy._posture_adjusted_edge(volatile, "risky", None)

    assert conservative < risky < 3.0


def test_conservative_posture_discounts_bad_news_harder_than_risky() -> None:
    signal = {"expected_profit": 2.0, "return_stdev": 0.0, "win_probability": 0.5}

    conservative = AssetRotationStrategy._posture_adjusted_edge(signal, "conservative", -8)
    risky = AssetRotationStrategy._posture_adjusted_edge(signal, "risky", -8)

    assert conservative < risky < 2.0


def test_learned_edge_only_shifts_ranking_when_ready() -> None:
    ready = {
        "expected_profit": 1.0, "return_stdev": 0.0, "win_probability": 0.5,
        "learned_edge_ready": True, "learned_edge": 4.0,
    }
    not_ready = {**ready, "learned_edge_ready": False}

    adjusted = AssetRotationStrategy._posture_adjusted_edge(ready, "risky", None)
    unadjusted = AssetRotationStrategy._posture_adjusted_edge(not_ready, "risky", None)

    assert unadjusted == 1.0
    assert adjusted > unadjusted


def test_risky_posture_leans_into_the_learned_edge_more_than_conservative() -> None:
    signal = {
        "expected_profit": 1.0, "return_stdev": 0.0, "win_probability": 0.5,
        "learned_edge_ready": True, "learned_edge": 4.0,
    }

    conservative = AssetRotationStrategy._posture_adjusted_edge(signal, "conservative", None)
    risky = AssetRotationStrategy._posture_adjusted_edge(signal, "risky", None)

    assert 1.0 < conservative < risky


def test_posture_adjusted_edge_never_exceeds_the_configured_clamp() -> None:
    extreme = {"expected_profit": 5.0, "return_stdev": 1000.0, "win_probability": 1.0}

    conservative = AssetRotationStrategy._posture_adjusted_edge(extreme, "conservative", 10)

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


def _signal_test_strategy(last_price: float, volume: list[float]) -> AssetRotationStrategy:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy.parameters = {
        "portfolio_analysis_days": 252,
        "recent_high_lookback_days": 3,
        "dip_threshold_percent": 5.0,
        "portfolio_round_trip_cost_percent": 0.20,
        "portfolio_oos_min_observations": 10,
        "portfolio_min_expected_profit_percent": 1.0,
        "portfolio_discovery_min_price_dollars": 5.0,
        "portfolio_discovery_min_avg_volume": 100000,
    }
    bars = SimpleNamespace(
        df=pd.DataFrame(
            {
                "high": [100.0, 100.0, 100.0, 100.0, 100.0],
                "close": [100.0, 100.0, 100.0, 90.0, 95.0],
                "volume": volume,
            }
        )
    )
    strategy.get_historical_prices = lambda *_args, **_kwargs: bars
    strategy.get_last_price = lambda _symbol: last_price
    strategy._get_bid_ask = lambda _symbol: None
    return strategy


def test_portfolio_signal_rejects_a_symbol_below_the_price_floor() -> None:
    strategy = _signal_test_strategy(last_price=3.0, volume=[200000] * 5)

    assert strategy._portfolio_signal("PENNY") is None


def test_portfolio_signal_rejects_a_symbol_below_the_volume_floor() -> None:
    strategy = _signal_test_strategy(last_price=50.0, volume=[1000] * 5)

    assert strategy._portfolio_signal("THIN") is None


def test_portfolio_signal_accepts_a_symbol_clearing_both_floors() -> None:
    strategy = _signal_test_strategy(last_price=90.0, volume=[200000] * 5)

    assert strategy._portfolio_signal("OK") is not None


def test_portfolio_signal_exposes_round_trip_cost_for_memory_blending() -> None:
    # PortfolioMemory blending nets its learned edge against this same
    # per-symbol cost basis, so it must be visible on the signal, not just
    # baked into expected_profit.
    strategy = _signal_test_strategy(last_price=90.0, volume=[200000] * 5)

    result = strategy._portfolio_signal("OK")

    assert result["round_trip_cost"] == pytest.approx(0.20)


def test_portfolio_signal_qualifies_true_when_dip_and_history_both_present() -> None:
    strategy = _signal_test_strategy(last_price=90.0, volume=[200000] * 5)

    result = strategy._portfolio_signal("OK")

    assert result["qualifies"] is True


def test_portfolio_signal_still_recorded_when_todays_dip_is_below_threshold() -> None:
    # Below-threshold symbols used to be discarded entirely (None); they now
    # still carry daily learning context (historical backtest ran), just
    # with qualifies=False so they can never leak into trading eligibility.
    strategy = _signal_test_strategy(last_price=98.0, volume=[200000] * 5)

    result = strategy._portfolio_signal("OK")

    assert result is not None
    assert result["qualifies"] is False
    assert result["expected_profit"] is not None


def test_portfolio_signal_still_context_only_without_any_historical_dip() -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy.parameters = {
        "portfolio_analysis_days": 252,
        "recent_high_lookback_days": 3,
        "dip_threshold_percent": 5.0,
        "portfolio_round_trip_cost_percent": 0.20,
        "portfolio_oos_min_observations": 10,
        "portfolio_min_expected_profit_percent": 1.0,
        "portfolio_discovery_min_price_dollars": 5.0,
        "portfolio_discovery_min_avg_volume": 100000,
    }
    bars = SimpleNamespace(
        df=pd.DataFrame({"high": [100.0] * 5, "close": [100.0] * 5, "volume": [200000] * 5})
    )
    strategy.get_historical_prices = lambda *_args, **_kwargs: bars
    strategy.get_last_price = lambda _symbol: 90.0  # today's dip alone clears the threshold
    strategy._get_bid_ask = lambda _symbol: None

    result = strategy._portfolio_signal("NEW")

    assert result is not None
    assert result["qualifies"] is False  # no historical comparable dip to estimate an edge from
    assert result["expected_profit"] is None
    assert result["observations"] == 0


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

        def aliases_for_symbols(self, candidates) -> dict:
            return {}

        def scan_text_for_aliases(self, text: str, aliases) -> set[str]:
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

        def aliases_for_symbols(self, candidates) -> dict:
            return {"AAPL": ("apple",)}

        def scan_text_for_aliases(self, text: str, aliases) -> set[str]:
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


def test_estimate_tokens_uses_the_real_tokenizer_when_available() -> None:
    assert LLMNewsAnalyzer._estimate_tokens("") == 0
    # A dense, space-free run (e.g. a table cell or a long ticker/number
    # blob) is exactly the case a words/chars heuristic undercounts but a
    # real BPE encoder still segments correctly.
    assert LLMNewsAnalyzer._estimate_tokens("x" * 400) > 40


def test_estimate_tokens_falls_back_to_words_times_token_ratio_without_tiktoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_news, "_TOKEN_ENCODER", None)
    assert LLMNewsAnalyzer._estimate_tokens("") == 0
    assert LLMNewsAnalyzer._estimate_tokens(" ".join(["word"] * 10)) == 13


def test_estimate_tokens_never_raises_on_special_token_lookalikes() -> None:
    # tiktoken's default encode() raises on text resembling a special
    # token; arbitrary headline/article text must never trip that up.
    assert LLMNewsAnalyzer._estimate_tokens("breaking: <|endoftext|> leaked") > 0


def test_prioritize_articles_keeps_the_highest_signal_article_within_budget() -> None:
    articles = [
        {"headline": "low", "summary": "", "score": 1},
        {"headline": "high signal", "summary": "", "score": 9},  # 2 words, fills the budget exactly
        {"headline": "medium", "summary": "", "score": 4},
    ]

    kept = LLMNewsAnalyzer._prioritize_articles(articles, budget_tokens=2)

    assert kept == [articles[1]]  # only the highest |score| article fits


def test_prioritize_articles_restores_original_order_for_the_kept_subset() -> None:
    articles = [
        {"headline": "a" * 8, "summary": "", "score": 1},
        {"headline": "b" * 8, "summary": "", "score": 9},
    ]

    kept = LLMNewsAnalyzer._prioritize_articles(articles, budget_tokens=4)

    assert kept == [articles[0], articles[1]]  # both fit; original order kept, not rank order


def test_prioritize_articles_returns_empty_for_non_positive_budget_or_no_articles() -> None:
    articles = [{"headline": "A", "summary": "", "score": 5}]

    assert LLMNewsAnalyzer._prioritize_articles(articles, budget_tokens=0) == []
    assert LLMNewsAnalyzer._prioritize_articles([], budget_tokens=1000) == []


def test_prioritize_articles_treats_a_missing_score_as_zero() -> None:
    # explain_exit/check_red_flag callers pass articles without a "score" key.
    articles = [{"headline": "First"}, {"headline": "Second"}]

    kept = LLMNewsAnalyzer._prioritize_articles(articles, budget_tokens=100)

    assert kept == articles


def test_assess_includes_all_articles_when_well_within_budget() -> None:
    analyzer = LLMNewsAnalyzer(model="test-model")
    analyzer._chat = lambda *args, **kwargs: json.dumps(
        {"score": 2, "risk_level": "constructive", "reasoning": "fine"}
    )

    assessment = analyzer.assess([{"headline": "Only headline", "summary": "short"}])

    assert assessment.explanation == "test-model assessed 1 articles; score +2 (constructive)."


def test_assess_bounds_article_count_to_the_prompt_token_budget() -> None:
    analyzer = LLMNewsAnalyzer(model="test-model")
    analyzer._chat = lambda *args, **kwargs: json.dumps(
        {"score": 0, "risk_level": "normal", "reasoning": "fine"}
    )
    articles = [
        {"headline": f"Headline number {i}", "summary": "word " * 100, "score": i}
        for i in range(20)
    ]

    assessment = analyzer.assess(articles)

    assert assessment.available
    assert f"assessed {len(articles)} articles" not in assessment.explanation


def test_article_filter_estimate_tokens_uses_the_real_tokenizer_when_available() -> None:
    assert article_filter._estimate_tokens("") == 0
    assert article_filter._estimate_tokens("x" * 400) > 40


def test_article_filter_estimate_tokens_falls_back_to_words_times_token_ratio_without_tiktoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(article_filter, "_TOKEN_ENCODER", None)
    assert article_filter._estimate_tokens("") == 0
    assert article_filter._estimate_tokens(" ".join(["word"] * 10)) == 13


def test_article_filter_estimate_tokens_never_raises_on_special_token_lookalikes() -> None:
    assert article_filter._estimate_tokens("breaking: <|endoftext|> leaked") > 0


def test_extract_financial_context_returns_none_for_a_low_density_article(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(article_filter, "DB_PATH", tmp_path / "verdicts.duckdb")
    monkeypatch.setattr(article_filter.trafilatura, "fetch_url", lambda url: "<html></html>")
    monkeypatch.setattr(
        article_filter.trafilatura,
        "extract",
        lambda *args, **kwargs: "Too short to bother the model with today.",
    )

    def _fail_post(*args, **kwargs):
        raise AssertionError("a low-density article must never reach the model")

    monkeypatch.setattr(article_filter.requests, "post", _fail_post)

    assert article_filter.extract_financial_context("https://example.com/a", ["AAPL"]) is None


def test_extract_financial_context_returns_cached_value_without_fetching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "verdicts.duckdb"
    monkeypatch.setattr(article_filter, "DB_PATH", db_path)
    cached = {"sentiment": "bullish", "confidence": 0.9, "affected_tickers": ["AAPL"],
              "key_risks": [], "catalyst_type": "earnings"}
    article_filter._save_verdict(
        date.today().isoformat(),
        "https://example.com/a",
        article_filter._watchlist_digest(["AAPL"]),
        cached,
    )

    def _fail_fetch(url):
        raise AssertionError("a cache hit must never fetch the article")

    monkeypatch.setattr(article_filter.trafilatura, "fetch_url", _fail_fetch)

    result = article_filter.extract_financial_context("https://example.com/a", ["AAPL"])

    assert result == cached


def test_extract_financial_context_parses_a_valid_model_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "verdicts.duckdb"
    monkeypatch.setattr(article_filter, "DB_PATH", db_path)
    article_text = (
        "Apple AAPL reports strong quarterly earnings and raises guidance for outlook. "
        "Analysts issued an upgrade after the surge in revenue beat estimates broadly. "
        "The rally followed a dividend increase and a new buyback program announced today. "
        "Some risk remains from a pending investigation into supplier tariff exposure now. "
        "The company also discussed merger talks and a possible acquisition target overseas. "
        "Executives said the recession fears and inflation outlook remain a modest headwind. "
        "A weather report and a recipe roundup filled out the rest of the newsletter."
    )
    monkeypatch.setattr(article_filter.trafilatura, "fetch_url", lambda url: "<html></html>")
    monkeypatch.setattr(article_filter.trafilatura, "extract", lambda *args, **kwargs: article_text)
    expected = {
        "sentiment": "bullish",
        "confidence": 0.8,
        "affected_tickers": ["AAPL"],
        "key_risks": ["supplier tariff exposure"],
        "catalyst_type": "earnings",
    }

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"response": json.dumps(expected)}

    monkeypatch.setattr(article_filter.requests, "post", lambda *args, **kwargs: FakeResponse())

    result = article_filter.extract_financial_context("https://example.com/a", ["AAPL"])

    assert result == expected
    stored = article_filter._load_verdict(
        date.today().isoformat(),
        "https://example.com/a",
        article_filter._watchlist_digest(["AAPL"]),
    )
    assert stored == expected


def test_discovery_article_context_is_advisory_and_scoped_to_the_symbols_own_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy.parameters = {"llm_news_enabled": True}
    logged: list[str] = []
    strategy.log_message = lambda message, **kwargs: logged.append(message)

    news_context = NewsContext(
        available=True,
        per_article=[
            {
                "headline": "Widget Co under investigation",
                "summary": "",
                "symbols": ["ZZZZ"],
                "score": -3,
                "url": "https://example.com/widget-co",
            },
        ],
    )
    captured = {}

    def fake_extract(url: str, watchlist: list[str]) -> dict:
        captured["url"] = url
        captured["watchlist"] = watchlist
        return {
            "sentiment": "bearish",
            "confidence": 0.7,
            "affected_tickers": ["ZZZZ"],
            "key_risks": ["regulatory investigation"],
            "catalyst_type": "corporate",
        }

    monkeypatch.setattr(article_filter, "extract_financial_context", fake_extract)

    report: dict = {}
    strategy._check_discovery_article_context(["ZZZZ"], news_context, {"ZZZZ": -3}, report)

    assert captured == {"url": "https://example.com/widget-co", "watchlist": ["ZZZZ"]}
    assert report["discovery_article_context"] == "ZZZZ: bearish (corporate): regulatory investigation"
    assert any("Discovery article context: ZZZZ" in message for message in logged)


def test_discovery_article_context_skips_symbols_without_negative_coverage_or_a_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy.parameters = {"llm_news_enabled": True}
    strategy.log_message = lambda *args, **kwargs: None

    def _fail_extract(url: str, watchlist: list[str]) -> dict:
        raise AssertionError("must not spend a call when there's nothing to screen")

    monkeypatch.setattr(article_filter, "extract_financial_context", _fail_extract)

    news_context = NewsContext(
        available=True,
        per_article=[
            {"headline": "Neutral coverage", "summary": "", "symbols": ["AAAA"], "score": 0, "url": "https://x/a"},
            {"headline": "No URL on file", "summary": "", "symbols": ["BBBB"], "score": -2, "url": ""},
        ],
    )
    report: dict = {}

    strategy._check_discovery_article_context(
        ["AAAA", "BBBB"], news_context, {"AAAA": 0, "BBBB": -2}, report
    )

    assert "discovery_article_context" not in report


def test_portfolio_builds_reserve_cash_across_same_pass_orders() -> None:
    strategy = AssetRotationStrategy.__new__(AssetRotationStrategy)
    strategy.get_cash = lambda: 100.0
    budgets: list[float] = []
    strategy._buy_portfolio_symbol = lambda symbol, price, budget: (
        budgets.append(budget) or "submitted"
    )
    desired = [
        {"symbol": "AAPL", "price": 10.0},
        {"symbol": "MSFT", "price": 20.0},
    ]
    claimed: set[str] = set()
    actions: list[str] = []

    submitted = strategy._submit_portfolio_builds(
        desired, {}, claimed, effective_max_positions=2, actions=actions
    )

    assert submitted == 2
    assert budgets == pytest.approx([50.0, 50.0])
    assert sum(budgets) <= 100.0


def test_portfolio_memory_does_not_label_a_multisession_gap_as_next_session(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "portfolio_memory.duckdb"
    memory = PortfolioMemory(db_path, minimum_observations=1, maximum_observations=50)
    memory.update_and_forecast("2026-07-13", "AAPL", 100.0, 5.0, 0)  # Monday
    memory.update_and_forecast("2026-07-16", "AAPL", 110.0, 5.0, 0)  # Thursday

    with duckdb.connect(str(db_path)) as conn:
        settled = conn.execute(
            "SELECT next_session_return_percent FROM observations "
            "WHERE evaluation_date = '2026-07-13' AND symbol = 'AAPL'"
        ).fetchone()[0]
    assert settled is None


def test_trade_memory_does_not_label_a_multisession_gap_as_next_session(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "trade_memory.duckdb"
    memory = TradeMemory(db_path, 1, 50)
    memory.update_and_forecast("2026-07-13", 100.0, 100.0, 5.0, 0, True)
    memory.update_and_forecast("2026-07-16", 100.0, 110.0, 5.0, 0, True)

    with duckdb.connect(str(db_path)) as conn:
        settled = conn.execute(
            "SELECT relative_return_percent FROM observations "
            "WHERE evaluation_date = '2026-07-13'"
        ).fetchone()[0]
    assert settled is None


def test_adaptive_news_model_does_not_learn_from_a_multisession_gap(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "news.json"
    model = AdaptiveNewsModel(state_path, minimum_observations=1, maximum_observations=50)
    model.update("2026-07-13", 100.0, -1)
    model.update("2026-07-16", 110.0, -1)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["observations"] == []


def test_article_context_cache_is_scoped_to_the_watchlist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(article_filter, "DB_PATH", tmp_path / "verdicts.duckdb")
    monkeypatch.setattr(article_filter.trafilatura, "fetch_url", lambda url: "raw")
    monkeypatch.setattr(
        article_filter.trafilatura,
        "extract",
        lambda *args, **kwargs: " ".join(
            ["Company earnings revenue profit guidance market shares growth quarter results outlook."]
            * 20
        ),
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        article_filter,
        "_query_model",
        lambda text, watchlist: calls.append(list(watchlist)) or {"watchlist": list(watchlist)},
    )

    first = article_filter.extract_financial_context("https://example.com/a", ["AAPL"])
    second = article_filter.extract_financial_context("https://example.com/a", ["MSFT"])

    assert first != second
    assert calls == [["AAPL"], ["MSFT"]]


def test_symbol_reference_does_not_defer_retry_when_all_sources_fail(
    tmp_path: Path,
) -> None:
    def fail(*args, **kwargs):
        raise OSError("offline")

    reference = SymbolReference(
        tmp_path / "symbols.duckdb",
        refresh_days=7,
        alpaca_fetcher=fail,
        sec_fetcher=fail,
    )

    assert reference.refresh(["AAPL"], "key", "secret") is False
    assert reference.refresh(["AAPL"], "key", "secret") is False
    with duckdb.connect(str(tmp_path / "symbols.duckdb")) as conn:
        assert conn.execute(
            "SELECT value FROM refresh_state WHERE name = 'last_refreshed'"
        ).fetchone() is None


def test_prioritize_articles_skips_an_oversize_article_and_keeps_shorter_ones() -> None:
    articles = [
        {"headline": "high", "summary": "x" * 400, "score": 10},
        {"headline": "short", "summary": "", "score": 9},
    ]

    kept = LLMNewsAnalyzer._prioritize_articles(articles, budget_tokens=2)

    assert kept == [articles[1]]
