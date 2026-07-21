"""Daily dip-buying and asset-rotation strategy for Lumibot."""

import faulthandler
import json
import math
import os
import smtplib
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date as date_type, timedelta
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable

from lumibot.strategies import Strategy
import numpy as np

import article_filter
import decision_math
import email_render
import signal_snapshot
import trade_counter
from adaptive_news_model import AdaptiveNewsModel, LearningResult
from autonomous_universe import AutonomousUniverse
from llm_news import LLMNewsAnalyzer, LLMNewsAssessment, RedFlagCheck
from news_context import NewsContext, WorldEventAnalyzer
from portfolio_memory import PortfolioMemory, PortfolioMemoryInput
from runtime_state import DuckDBStateStore
from trade_memory import OpportunityProbability, RotationForecast, TradeMemory
from symbol_reference import SymbolReference


class AssetRotationStrategy(Strategy):
    """Run the default dip-signal portfolio with a learned A/B opportunity."""

    parameters = {
        "asset_a": "SPY",
        "asset_b": "QQQ",
        "dip_threshold_percent": 5.0,
        "recent_high_lookback_days": 20,
        "email_report_enabled": False,
        "news_context_enabled": True,
        "news_learning_enabled": True,
        "llm_news_enabled": False,
        "decision_memory_enabled": True,
        "decision_memory_block_enabled": False,
        "portfolio_oos_min_observations": 10,
        "portfolio_oos_min_net_profit_percent": 0.0,
        "portfolio_round_trip_cost_percent": 0.20,
        "portfolio_take_profit_percent": 1.0,
        "portfolio_stop_loss_percent": 0.5,
        "portfolio_holding_horizon_max_days": 15,
        "portfolio_risk_posture": "conservative",
    }

    # Fraction of cash withheld from the Asset B buy so the market order is not
    # rejected (or filled into a deficit) if the price moves before execution.
    CASH_BUFFER_FRACTION = 0.01

    # These constants and the three methods aliased below live in
    # decision_math.py -- pure, asset-class-agnostic math shared with
    # CryptoRotationStrategy. Kept as class attributes here so existing call
    # sites (self._posture_adjusted_edge(...), etc.) and tests are unaffected.
    _POSTURE_VARIANCE_PENALTY = decision_math.POSTURE_VARIANCE_PENALTY
    _POSTURE_CONSISTENCY_WEIGHT = decision_math.POSTURE_CONSISTENCY_WEIGHT
    _POSTURE_NEWS_DISCOUNT_PER_POINT = decision_math.POSTURE_NEWS_DISCOUNT_PER_POINT
    _POSTURE_LLM_SCORE_WEIGHT = decision_math.POSTURE_LLM_SCORE_WEIGHT
    _POSTURE_LEARNED_EDGE_WEIGHT = decision_math.POSTURE_LEARNED_EDGE_WEIGHT
    _POSTURE_MAX_ADJUSTMENT_PERCENT = decision_math.POSTURE_MAX_ADJUSTMENT_PERCENT

    # Order statuses that mean an order can no longer fill. Anything else is
    # treated as still working so the agent never submits a duplicate.
    _TERMINAL_ORDER_STATUSES = {
        "fill",
        "filled",
        "cancel",
        "canceled",
        "cancelled",
        "cash_settled",
        "error",
        "expired",
        "rejected",
    }
    _FAILED_ORDER_STATUSES = {"cancel", "canceled", "cancelled", "error", "expired", "rejected"}
    _PORTFOLIO_HISTORY_WORKERS = 4
    # LLM checks that can affect a purchase run synchronously before orders.
    # Alpaca's free quote feed is IEX-only, not full NBBO -- a bad or thin
    # print should never make a symbol's live spread reading look enormous
    # and swamp the flat cost estimate it is only meant to floor.
    _PORTFOLIO_LIVE_SPREAD_CAP_PERCENT = 5.0

    def initialize(self) -> None:
        """Poll frequently; _due_portfolio_iteration_window gates actual evaluations
        to at most twice a trading day (see on_trading_iteration)."""
        self.sleeptime = "10M"
        self._rotation_lock = threading.Lock()
        self._portfolio_state_lock = threading.RLock()
        self._symbol_reference_refresh_lock = threading.Lock()
        self._symbol_reference_pending_symbols: set[str] = set()
        self._symbol_reference_refresh_running = False
        self._exit_narrative_lock = threading.Lock()
        self._pending_exit_narratives: list[tuple[str, str, NewsContext]] = []
        self._exit_narrative_worker_running = False
        self._unpriceable_symbols_lock = threading.Lock()
        self._unpriceable_symbols_this_iteration: set[str] = set()
        # Historical bars are fetched during the first evaluation, after the
        # broker has supplied current market data.
        self.vars.decision_memory_backfill_attempted = False
        self.vars.portfolio_memory_backfilled_symbols = set()
        self.vars.portfolio_pending_rotation = self._load_portfolio_rotation()
        self.vars.portfolio_holding_dates = self._load_portfolio_holding_dates()
        self.vars.portfolio_iteration_state = self._load_portfolio_iteration_state()
        if self.vars.portfolio_pending_rotation:
            pending = self.vars.portfolio_pending_rotation
            summary = ", ".join(
                f"{source} to {entry['to']} ({entry['kind']})"
                for source, entry in sorted(pending.items())
            )
            self.log_message(
                f"Restored {len(pending)} in-progress portfolio rotation(s): "
                f"{summary}; reconciling next cycle.",
                color="yellow",
            )
        # Warm the slow, optional symbol metadata before the market opens. Any
        # later discovery symbols are queued by _run_portfolio_iteration, but
        # enrichment must never delay price evaluation or order submission.
        self._refresh_symbol_reference(
            [str(symbol).strip().upper() for symbol in self.parameters.get("portfolio_symbols", [])]
        )

    def on_abrupt_closing(self) -> None:
        """Snapshot every thread's stack the instant a stop signal arrives.

        Lumibot's StrategyExecutor.stop() sets stop_event and calls this hook
        synchronously in the main thread before any join or systemd
        TimeoutStopSec countdown. main.py's MarketOpenLoggingAlpaca already
        makes the market-open/market-close waits interruptible on that same
        stop_event, so a stop should exit within a second either way -- but
        some mid-day stops still needed systemd's SIGKILL, and nothing so far
        has captured what every thread was actually doing at that moment.
        Overwritten (not appended) each time, so it stays a single snapshot
        rather than a growing log.
        """
        raw_path = self.parameters.get("shutdown_diagnostic_file")
        if not raw_path:
            return
        try:
            with open(str(raw_path), "w", encoding="utf-8") as handle:
                faulthandler.dump_traceback(file=handle, all_threads=True)
        except OSError:
            pass

    def _portfolio_rotation_state_path(self) -> Path | None:
        raw = self.parameters.get("portfolio_rotation_state_file")
        return Path(str(raw)) if raw else None

    def _runtime_state(self) -> DuckDBStateStore | None:
        raw = self.parameters.get("runtime_state_database_file")
        if not raw:
            return None
        path = Path(str(raw))
        cached = getattr(self, "_runtime_state_store", None)
        if cached is None or cached.database_path != path:
            cached = DuckDBStateStore(path)
            self._runtime_state_store = cached
        return cached

    def _load_runtime_value(
        self, key: str, legacy_path: Path | None, *, plain_text: bool = False
    ) -> tuple[bool, Any]:
        store = self._runtime_state()
        if store is not None:
            found, value = store.get(key)
            if found:
                return True, value
        if legacy_path is None or not legacy_path.exists():
            return False, None
        text = legacy_path.read_text(encoding="utf-8")
        value = text.strip() if plain_text else json.loads(text)
        if store is not None:
            store.set(key, value)
        return True, value

    def _save_runtime_value(
        self,
        key: str,
        value: Any,
        legacy_path: Path | None,
        *,
        delete_empty: bool = False,
        plain_text: bool = False,
    ) -> None:
        store = self._runtime_state()
        if store is not None:
            # Persist an explicit empty value as a tombstone. Deleting the key
            # would make the next restart re-import a stale legacy file.
            store.set(key, value)
            return
        if legacy_path is None:
            return
        if delete_empty and not value:
            legacy_path.unlink(missing_ok=True)
            return
        temporary_path = legacy_path.with_suffix(legacy_path.suffix + ".tmp")
        serialized = str(value) if plain_text else json.dumps(value, sort_keys=True)
        temporary_path.write_text(serialized + "\n", encoding="utf-8")
        temporary_path.replace(legacy_path)

    def _portfolio_state_guard(self) -> threading.RLock:
        """Return the shared lock protecting callback/iteration state swaps."""
        lock = getattr(self, "_portfolio_state_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._portfolio_state_lock = lock
        return lock

    def _unpriceable_symbols_guard(self) -> threading.Lock:
        """Return the lock protecting the current iteration's no-bars symbol set."""
        lock = getattr(self, "_unpriceable_symbols_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._unpriceable_symbols_lock = lock
        return lock

    def _load_portfolio_rotation(self) -> dict[str, dict[str, Any]]:
        """Restore every staged portfolio rotation after a restart.

        Keyed by source ("from") symbol so a sell-fill callback can look its
        entry up in O(1). Transparently migrates the old single-record shape
        ({"from", "to", "budget"}) written by earlier versions into the new
        keyed shape, defaulting its kind to "replacement" since that format
        couldn't distinguish an Opportunistic Opportunity swap.
        """
        try:
            found, state = self._load_runtime_value(
                "portfolio_rotation", self._portfolio_rotation_state_path()
            )
            if not found:
                return {}
        except Exception:
            return {}
        if isinstance(state, dict) and all(
            isinstance(state.get(key), str) and state[key] for key in ("from", "to")
        ):
            entry = self._parse_rotation_entry(state)
            return {state["from"].upper(): entry} if entry else {}
        if not isinstance(state, dict):
            return {}
        restored: dict[str, dict[str, Any]] = {}
        for source, raw_entry in state.items():
            if not isinstance(source, str) or not source.strip() or not isinstance(raw_entry, dict):
                continue
            entry = self._parse_rotation_entry(raw_entry)
            if entry:
                restored[source.upper()] = entry
        return restored

    @staticmethod
    def _parse_rotation_entry(raw_entry: dict[str, Any]) -> dict[str, Any] | None:
        """Validate a single {to, budget, kind} rotation record, or None."""
        target = raw_entry.get("to")
        if not isinstance(target, str) or not target.strip():
            return None
        try:
            budget = float(raw_entry.get("budget", 0))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(budget) or budget <= 0:
            return None
        kind = raw_entry.get("kind")
        if kind not in ("replacement", "opportunistic"):
            kind = "replacement"
        return {"to": target.upper(), "budget": budget, "kind": kind}

    def _set_portfolio_rotation(self, state: dict[str, dict[str, Any]]) -> bool:
        """Persist the whole rotation collection in one transaction.

        Callers always pass a freshly rebuilt dict rather than mutating the
        live one in place, matching the same discipline already used for
        portfolio_holding_dates, so a concurrent read from the broker
        callback thread never observes a partially updated collection.
        """
        with self._portfolio_state_guard():
            previous = self.vars.portfolio_pending_rotation
            self.vars.portfolio_pending_rotation = state
            try:
                self._save_runtime_value(
                    "portfolio_rotation",
                    state,
                    self._portfolio_rotation_state_path(),
                    delete_empty=True,
                )
            except Exception as exc:
                self.vars.portfolio_pending_rotation = previous
                self.log_message(f"Could not persist portfolio rotation state: {exc}", color="red")
                return False
            return True

    def _add_portfolio_rotation(self, source: str, target: str, budget: float, kind: str) -> bool:
        """Stage a new rotation, refusing if either symbol is already in flight.

        Fills are matched by symbol, not order id, so a symbol referenced as
        both a "from" in one entry and a "to" in another would be ambiguous.
        Returns False (and logs, without changing state) rather than risk
        creating that ambiguity.
        """
        with self._portfolio_state_guard():
            pending = self.vars.portfolio_pending_rotation
            claimed = set(pending.keys()) | {entry["to"] for entry in pending.values()}
            if source in claimed or target in claimed:
                self.log_message(
                    f"Refused to stage {source} to {target}: one of these symbols "
                    "already has an in-flight portfolio rotation.",
                    color="red",
                )
                return False
            updated = dict(pending)
            updated[source] = {"to": target, "budget": budget, "kind": kind}
            return self._set_portfolio_rotation(updated)

    def _remove_portfolio_rotation(self, source: str) -> None:
        with self._portfolio_state_guard():
            pending = self.vars.portfolio_pending_rotation
            if source not in pending:
                return
            updated = dict(pending)
            del updated[source]
            self._set_portfolio_rotation(updated)

    def _portfolio_holding_state_path(self) -> Path | None:
        raw = self.parameters.get("portfolio_holding_state_file")
        return Path(str(raw)) if raw else None

    def _load_portfolio_holding_dates(self) -> dict[str, str]:
        """Restore broker-confirmed portfolio entry dates after a restart."""
        try:
            found, state = self._load_runtime_value(
                "portfolio_holding_dates", self._portfolio_holding_state_path()
            )
            if not found:
                return {}
            if not isinstance(state, dict):
                return {}
            return {
                str(symbol).upper(): value
                for symbol, value in state.items()
                if isinstance(value, str)
                and str(symbol).strip()
                and self._valid_iso_date(value)
            }
        except Exception:
            return {}

    @staticmethod
    def _valid_iso_date(value: str) -> bool:
        try:
            date_type.fromisoformat(value)
            return True
        except ValueError:
            return False

    def _set_portfolio_holding_dates(self, dates: dict[str, str]) -> None:
        with self._portfolio_state_guard():
            previous = self.vars.portfolio_holding_dates
            self.vars.portfolio_holding_dates = dates
            try:
                self._save_runtime_value(
                    "portfolio_holding_dates", dates, self._portfolio_holding_state_path()
                )
            except Exception as exc:
                self.vars.portfolio_holding_dates = previous
                self.log_message(f"Could not persist portfolio holding dates: {exc}", color="red")

    def _record_portfolio_entry(self, symbol: str) -> None:
        with self._portfolio_state_guard():
            dates = dict(self.vars.portfolio_holding_dates)
            dates[str(symbol).upper()] = self.get_datetime().date().isoformat()
            self._set_portfolio_holding_dates(dates)

    def _remove_portfolio_entry(self, symbol: str) -> None:
        with self._portfolio_state_guard():
            dates = dict(self.vars.portfolio_holding_dates)
            dates.pop(str(symbol).upper(), None)
            self._set_portfolio_holding_dates(dates)

    def _portfolio_iteration_state_path(self) -> Path | None:
        raw = self.parameters.get("portfolio_iteration_state_file")
        return Path(str(raw)) if raw else None

    @staticmethod
    def _default_portfolio_iteration_state() -> dict[str, Any]:
        return {"date": "", "windows_completed": [], "opportunistic_swap_done": False}

    def _load_portfolio_iteration_state(self) -> dict[str, Any]:
        """Restore today's iteration progress after a restart.

        Needed now that a trading day can run more than one evaluation
        (see _due_portfolio_iteration_window): without this surviving a
        restart, a crash between the day's two windows would forget the
        first one ran and re-fire it immediately on the next poll.
        """
        try:
            found, state = self._load_runtime_value(
                "portfolio_iteration", self._portfolio_iteration_state_path()
            )
            if not found:
                return self._default_portfolio_iteration_state()
            if not isinstance(state, dict):
                return self._default_portfolio_iteration_state()
            date_value = str(state.get("date", ""))
            windows = state.get("windows_completed", [])
            windows = [str(w) for w in windows] if isinstance(windows, list) else []
            return {
                "date": date_value if self._valid_iso_date(date_value) else "",
                "windows_completed": windows,
                "opportunistic_swap_done": bool(state.get("opportunistic_swap_done", False)),
            }
        except Exception:
            return self._default_portfolio_iteration_state()

    def _save_portfolio_iteration_state(self) -> None:
        try:
            self._save_runtime_value(
                "portfolio_iteration",
                self.vars.portfolio_iteration_state,
                self._portfolio_iteration_state_path(),
            )
        except Exception as exc:
            self.log_message(f"Could not persist portfolio iteration state: {exc}", color="red")

    def _nightly_preeval_state_path(self) -> Path | None:
        raw = self.parameters.get("nightly_preeval_state_file")
        return Path(str(raw)) if raw else None

    def _save_nightly_preeval_state(self, summary: str, symbol_count: int) -> None:
        """Persist the overnight pass's findings so the live iteration's
        email can show what was learned last night (see
        _load_nightly_preeval_learnings). Keyed by date the same way
        _save_portfolio_iteration_state is, so stale state from a missed
        night is never mistaken for tonight's."""
        try:
            state = {
                "date": self.get_datetime().date().isoformat(),
                "symbol_count": symbol_count,
                "summary": summary,
            }
            self._save_runtime_value(
                "nightly_preeval", state, self._nightly_preeval_state_path()
            )
        except Exception as exc:
            self.log_message(f"Could not persist nightly pre-evaluation state: {exc}", color="red")

    def _load_nightly_preeval_learnings(self) -> dict[str, Any]:
        """Today's nightly pre-evaluation summary, or {} if it hasn't run
        yet today (feature disabled, timer hasn't fired, or a stale/corrupt
        record) -- read once per iteration so the email can show what the
        overnight pass found. Fails open like every other state read here.
        """
        try:
            found, state = self._load_runtime_value(
                "nightly_preeval", self._nightly_preeval_state_path()
            )
            if not found:
                return {}
            if not isinstance(state, dict):
                return {}
            if state.get("date") != self.get_datetime().date().isoformat():
                return {}
            return {
                "summary": str(state.get("summary", "")),
                "symbol_count": int(state.get("symbol_count", 0)),
            }
        except Exception:
            return {}

    @staticmethod
    def _due_portfolio_iteration_window(
        now: Any,
        market_open: Any,
        second_offset_minutes: int,
        windows_completed: list[str],
    ) -> str | None:
        """Return the label of the iteration window that is due, or None.

        "open" is due once `now` reaches market open; "midday" is due once
        `now` reaches market_open + second_offset_minutes. Each label fires
        at most once per trading day -- `windows_completed` is reset by the
        caller whenever the calendar date rolls over -- regardless of how
        often this is polled (sleeptime="10M", see initialize).
        """
        completed = set(windows_completed)
        windows = (
            ("open", market_open),
            ("midday", market_open + timedelta(minutes=second_offset_minutes)),
        )
        for label, due_at in windows:
            if label not in completed and now >= due_at:
                return label
        return None

    @staticmethod
    def _holding_is_due(entry_date: str, today: date_type, maximum_days: int) -> bool:
        """Return whether a confirmed entry has reached its configured horizon."""
        try:
            return today - date_type.fromisoformat(entry_date) >= timedelta(days=maximum_days)
        except ValueError:
            return False

    def _cached_orders(self) -> list[Any]:
        """Fetch open orders once per iteration; _has_active_order is called
        from several loop sites per iteration and they all describe the same
        broker-side order book, so refetching per call is a pure waste of a
        network round-trip. Cleared by _invalidate_orders_cache()."""
        cached = getattr(self, "_orders_cache", None)
        if cached is None:
            cached = self.get_orders() or []
            self._orders_cache = cached
        return cached

    def _invalidate_orders_cache(self) -> None:
        self._orders_cache = None

    def _has_active_order(self, symbol: str, side: str) -> bool:
        """Best-effort check for a working order; unknown states count as active."""
        try:
            orders = self._cached_orders()
        except Exception as exc:
            self.log_message(
                f"Could not read orders ({type(exc).__name__}: {exc}); "
                "assuming one may still be working.",
                color="yellow",
            )
            return True
        for order in orders:
            order_symbol = getattr(getattr(order, "asset", None), "symbol", None)
            order_side = str(getattr(order, "side", "")).lower()
            if order_symbol != symbol or order_side != side.lower():
                continue
            status = str(getattr(order, "status", "")).lower()
            if status not in self._TERMINAL_ORDER_STATUSES:
                return True
        return False

    @staticmethod
    def _order_status(order: Any) -> str:
        """Normalize Lumibot enum/string statuses for submission checks."""
        status = getattr(order, "status", "")
        value = getattr(status, "value", status)
        return str(value).strip().lower()

    def _submit_order_checked(self, order: Any, description: str) -> bool:
        """Submit and reject Lumibot's non-raising synchronous error result.

        Alpaca's Lumibot broker catches API exceptions, sets ``order.status``
        to ``error``, and returns the order. Callers therefore cannot use the
        absence of an exception as proof that the broker accepted it.
        """
        submitted = self.submit_order(order)
        if submitted is None:
            self.log_message(
                f"Broker did not accept {description}: submission returned no order.",
                color="red",
            )
            return False
        status = self._order_status(submitted)
        if status in self._FAILED_ORDER_STATUSES:
            error = str(getattr(submitted, "error_message", "") or "").strip()
            suffix = f": {error}" if error else ""
            self.log_message(
                f"Broker rejected {description} (status={status}){suffix}.",
                color="red",
            )
            return False
        trade_counter.record_trade(
            str(self.parameters.get("portfolio_trade_count_file", "")),
            self.get_datetime().date().isoformat(),
        )
        self._invalidate_orders_cache()
        return True

    def _submit_portfolio_rotation_sell(
        self,
        source: str,
        target: str,
        quantity: Decimal,
        budget: float,
        kind: str,
    ) -> bool:
        """Persist rotation intent before exposing its sell to the broker."""
        if not self._add_portfolio_rotation(source, target, budget, kind):
            return False
        order = self.create_order(
            source,
            quantity=quantity,
            side="sell",
            order_type="market",
            time_in_force="day",
        )
        try:
            accepted = self._submit_order_checked(order, f"{source} sell for {kind} rotation")
        except Exception:
            # The persisted intent was created only for this submission. A
            # later callback cannot have filled an order that raised before it
            # was accepted, so removing it is safe.
            self._remove_portfolio_rotation(source)
            raise
        if not accepted:
            self._remove_portfolio_rotation(source)
            return False
        return True

    def _send_daily_email(self, report: dict[str, Any]) -> None:
        """Send at most one successful summary email per calendar day."""
        if not bool(self.parameters.get("email_report_enabled", False)):
            return

        try:
            report_date = self.get_datetime().date().isoformat()
            raw_state_file = self.parameters.get("email_state_file")
            state_file = Path(str(raw_state_file)) if raw_state_file else None
            found, last_report_date = self._load_runtime_value(
                "last_email_report", state_file, plain_text=True
            )
            if found and str(last_report_date) == report_date:
                return

            message = EmailMessage()
            message["Subject"] = (
                f"Trading Agent Daily Report - {report_date} - {report['status']}"
            )
            message["From"] = str(self.parameters["email_from_address"])
            message["To"] = str(self.parameters["email_to_address"])
            nightly_learned_line = (
                f"Learned at night ({report['nightly_learned_symbol_count']} symbols checked): "
                f"{report['nightly_learned_summary']}"
                if "nightly_learned_summary" in report
                else "Learned at night: not run last night"
            )
            lines = [
                "Raspberry Pi Trading Agent Daily Summary",
                "",
                f"Date: {report_date}",
                f"Evaluation time: {self.get_datetime().isoformat()}",
                "Mode: portfolio",
                *(
                    [f"Summary: {report['daily_narrative']}", ""]
                    if report.get("daily_narrative")
                    else []
                ),
                f"Risk posture: {report.get('portfolio_risk_posture', 'unavailable')}",
                f"Holdings: {report.get('portfolio_holdings', 'unavailable')}",
                f"Signal candidates: {report.get('portfolio_candidates', 'unavailable')}",
                f"Effective max positions today: "
                f"{report.get('portfolio_effective_max_positions', 'unavailable')} "
                f"(configured ceiling {self.parameters.get('portfolio_max_positions', 'unavailable')})",
                f"Discovered symbols: {report.get('discovered_symbols', 'none')}",
                f"Discovery status: {report.get('discovery_status', 'ok')}",
                f"Discovery red flags: {report.get('discovery_red_flags', 'none')}",
                f"Discovery article context: {report.get('discovery_article_context', 'none')}",
                f"Dip threshold: {report['threshold']:.2f}%",
                f"News risk level: {report.get('news_risk_level', 'unavailable')}",
                f"News score: {report.get('news_score', 'unavailable')}",
                f"News articles checked: {report.get('news_article_count', 'unavailable')}",
                f"News explanation: {report.get('news_explanation', 'unavailable')}",
                f"LLM risk level: {report.get('llm_risk_level', 'unavailable')}",
                f"LLM score: {report.get('llm_score', 'unavailable')}",
                f"LLM reasoning: {report.get('llm_reasoning', 'unavailable')}",
                nightly_learned_line,
                f"Learning observations: {report.get('learning_observations', 'unavailable')}",
                f"Learned return forecast: {report.get('learned_forecast', 'not ready')}",
                f"Learning explanation: {report.get('learning_explanation', 'unavailable')}",
                f"LLM-score learning observations: {report.get('llm_learning_observations', 'unavailable')}",
                f"LLM-score learned return forecast: {report.get('llm_learned_forecast', 'not ready')}",
                f"LLM-score learning explanation: {report.get('llm_learning_explanation', 'unavailable')}",
                f"Opportunistic Opportunity: {report.get('opportunistic_opportunity_status', 'unavailable')}",
                f"Opportunistic Opportunity probability: {report.get('opportunistic_opportunity_probability', 'unavailable')}",
                f"Opportunistic Opportunity evidence: {report.get('opportunistic_opportunity_explanation', 'unavailable')}",
                "Portfolio actions this iteration:",
                *[f"- {action}" for action in report.get("portfolio_actions", [])],
                "Notable scored headlines:",
                *[f"- {headline}" for headline in report.get("news_headlines", [])],
                f"Result: {report['status']}",
                "",
                "Review all orders and positions in the Alpaca dashboard.",
                "This automated message is not financial advice.",
            ]
            message.set_content("\n".join(lines))
            message.add_alternative(
                self._render_email_html(report, report_date),
                subtype="html",
            )

            host = str(self.parameters["email_smtp_host"])
            port = int(self.parameters["email_smtp_port"])
            # The password comes from the environment so it never travels
            # through Lumibot's parameters dict, which may be logged.
            password = os.environ.get("EMAIL_SMTP_PASSWORD") or str(
                self.parameters.get("email_smtp_password", "")
            )
            with smtplib.SMTP(host, port, timeout=15) as smtp:
                smtp.ehlo()
                if bool(self.parameters["email_use_tls"]):
                    # Verify the server certificate; the stdlib default does not.
                    smtp.starttls(context=ssl.create_default_context())
                    smtp.ehlo()
                smtp.login(str(self.parameters["email_smtp_username"]), password)
                smtp.send_message(message)

            self._save_runtime_value(
                "last_email_report", report_date, state_file, plain_text=True
            )
            self.log_message(f"Daily email report sent for {report_date}.", color="green")
        except Exception as exc:
            self.log_message(
                f"Daily email report failed safely: {type(exc).__name__}: {exc}",
                color="red",
            )

    # These render helpers live in email_render.py -- generic HTML-table
    # builders with no report-shape or equity-specific coupling, shared with
    # CryptoRotationStrategy's own email report. Aliased under their old
    # names so existing call sites in _render_email_html below are unaffected.
    _email_status_theme = staticmethod(email_render.email_status_theme)
    _email_value = staticmethod(email_render.email_value)
    _email_kv_section = staticmethod(email_render.email_kv_section)
    _email_bullet_section = staticmethod(email_render.email_bullet_section)

    def _render_email_html(self, report: dict[str, Any], report_date: str) -> str:
        """Build a styled HTML alternative body mirroring the plain-text report."""
        status = str(report["status"])
        mode_label = "Portfolio mode"

        snapshot_rows = [
            ("Risk posture", report.get("portfolio_risk_posture", "unavailable")),
            ("Holdings", report.get("portfolio_holdings", "unavailable")),
            ("Signal candidates", report.get("portfolio_candidates", "unavailable")),
            (
                "Effective max positions today",
                f"{report.get('portfolio_effective_max_positions', 'unavailable')} "
                f"(configured ceiling {self.parameters.get('portfolio_max_positions', 'unavailable')})",
            ),
            ("Discovered symbols", report.get("discovered_symbols", "none")),
            ("Discovery status", report.get("discovery_status", "ok")),
            ("Discovery red flags", report.get("discovery_red_flags", "none")),
            ("Discovery article context", report.get("discovery_article_context", "none")),
            ("Dip threshold", f"{report['threshold']:.2f}%"),
        ]

        signal_rows = [
            ("News risk level", report.get("news_risk_level", "unavailable")),
            ("News score", report.get("news_score", "unavailable")),
            ("News articles checked", report.get("news_article_count", "unavailable")),
            ("News explanation", report.get("news_explanation", "unavailable")),
            ("LLM risk level", report.get("llm_risk_level", "unavailable")),
            ("LLM score", report.get("llm_score", "unavailable")),
            ("LLM reasoning", report.get("llm_reasoning", "unavailable")),
        ]

        forecast_rows = [
            ("Learning observations", report.get("learning_observations", "unavailable")),
            ("Learned return forecast", report.get("learned_forecast", "not ready")),
            ("Learning explanation", report.get("learning_explanation", "unavailable")),
            (
                "LLM-score learning observations",
                report.get("llm_learning_observations", "unavailable"),
            ),
            (
                "LLM-score learned return forecast",
                report.get("llm_learned_forecast", "not ready"),
            ),
            (
                "LLM-score learning explanation",
                report.get("llm_learning_explanation", "unavailable"),
            ),
            (
                "Opportunistic Opportunity",
                report.get("opportunistic_opportunity_status", "unavailable"),
            ),
            (
                "Opportunistic Opportunity probability",
                report.get("opportunistic_opportunity_probability", "unavailable"),
            ),
            (
                "Opportunistic Opportunity evidence",
                report.get("opportunistic_opportunity_explanation", "unavailable"),
            ),
        ]

        nightly_symbol_count = report.get("nightly_learned_symbol_count")
        nightly_title = (
            f"Learned at night ({nightly_symbol_count} symbols checked)"
            if nightly_symbol_count is not None
            else "Learned at night (not run last night)"
        )
        nightly_items = [
            item
            for item in str(report.get("nightly_learned_summary") or "").split("; ")
            if item
        ]

        sections = "".join(
            [
                self._email_kv_section("Snapshot", snapshot_rows),
                self._email_bullet_section("Portfolio actions", report.get("portfolio_actions", [])),
                self._email_kv_section("News & Risk Signals", signal_rows),
                self._email_bullet_section(nightly_title, nightly_items),
                self._email_kv_section("Learning & Forecasts", forecast_rows),
                self._email_bullet_section(
                    "Notable scored headlines", report.get("news_headlines", [])
                ),
            ]
        )

        return email_render.render_email_shell(
            report_date=report_date,
            mode_label=mode_label,
            status=status,
            narrative=str(report.get("daily_narrative") or ""),
            sections_html=sections,
        )

    def _get_news_context(self) -> NewsContext:
        """Return recent headline context; callers enforce configured outage policy."""
        if not bool(self.parameters.get("news_context_enabled", True)):
            return NewsContext(
                available=False,
                risk_level="disabled",
                explanation="News context is disabled in config.json.",
            )
        try:
            analyzer = WorldEventAnalyzer(
                lookback_hours=int(self.parameters["news_lookback_hours"]),
                max_articles=int(self.parameters["news_max_articles"]),
                block_score=int(self.parameters["news_high_risk_score"]),
                refine_scoring=bool(self.parameters.get("news_score_refinement_enabled", False)),
                rss_enabled=bool(self.parameters.get("news_rss_enabled", False)),
                rss_feed_urls=list(self.parameters.get("news_rss_feed_urls", [])),
            )
            context = analyzer.analyze()
            self.log_message(
                f"World-event context: risk={context.risk_level}, "
                f"score={context.score}, articles={context.article_count}. "
                f"{context.explanation}",
                color="yellow" if context.score < 0 else "blue",
            )
            for headline in context.headlines:
                self.log_message(f"News evidence: {headline}", color="blue")
            return context
        except Exception as exc:
            self.log_message(
                f"News context unavailable; configured safety policy will decide: "
                f"{type(exc).__name__}: {exc}",
                color="red",
            )
            return NewsContext(
                available=False,
                risk_level="unavailable",
                explanation=f"News retrieval failed: {type(exc).__name__}: {exc}",
            )

    def _get_llm_news_assessment(
        self,
        news_context: NewsContext,
        symbols: list[str],
        held_symbols: set[str],
        symbol_news_scores: dict[str, int],
    ) -> LLMNewsAssessment:
        """Ask the local Ollama model to assess today's headlines, failing open on problems.

        `symbols`/`held_symbols`/`symbol_news_scores` give the model today's
        actual evaluation universe and per-symbol coverage (the same
        cross-checked scores ranking uses -- see `_symbol_news_scores`) so it
        can reason about risk to the symbols this agent might actually trade
        today, not just generic market headlines. The score it returns is
        still aggregate market risk (the only thing `_market_veto_reason`
        consumes); this only makes that judgment better-informed.
        """
        if not bool(self.parameters.get("llm_news_enabled", False)):
            return LLMNewsAssessment(
                available=False,
                risk_level="disabled",
                explanation="LLM news assessment is disabled in config.json.",
            )
        if not news_context.available or not news_context.articles:
            return LLMNewsAssessment(
                available=False,
                risk_level="unavailable",
                explanation=(
                    "No news articles were available for the LLM assessment."
                ),
            )
        try:
            analyzer = LLMNewsAnalyzer(
                model=str(self.parameters["llm_news_model"]),
                base_url=str(self.parameters.get("llm_news_base_url", "")),
                block_score=int(self.parameters["llm_news_block_score"]),
            )
            assessment = analyzer.assess(
                news_context.per_article,
                symbols=symbols,
                held_symbols=held_symbols,
                symbol_scores=symbol_news_scores,
            )
            self.log_message(
                f"LLM news assessment: risk={assessment.risk_level}, "
                f"score={assessment.score:+d}. {assessment.reasoning}",
                color="yellow" if assessment.score < 0 else "blue",
            )
            return assessment
        except Exception as exc:
            self.log_message(
                f"LLM news assessment unavailable; price strategy will "
                f"continue: {type(exc).__name__}: {exc}",
                color="red",
            )
            return LLMNewsAssessment(
                available=False,
                risk_level="unavailable",
                explanation=(
                    f"LLM assessment failed: {type(exc).__name__}: {exc}"
                ),
            )

    def _llm_assessment_for_iteration(
        self,
        news_context: NewsContext,
        symbols: list[str],
        held_symbols: set[str],
        symbol_news_scores: dict[str, int],
    ) -> LLMNewsAssessment:
        """Return the current iteration's assessment before any order decision."""
        return self._get_llm_news_assessment(
            news_context, symbols, held_symbols, symbol_news_scores
        )

    def _update_news_learning(
        self,
        price_b: float,
        state_file_key: str,
        news_score: int | None,
        log_prefix: str = "",
    ) -> LearningResult:
        """Shared body for _update_adaptive_learning/_update_llm_adaptive_learning:
        persist one day's (price, news_score) observation to the named state
        file and return its explainable forecast."""
        if not bool(self.parameters.get("news_learning_enabled", True)):
            return LearningResult(
                observations=0,
                ready=False,
                predicted_return_percent=None,
                slope=None,
                correlation=None,
                explanation="Adaptive news learning is disabled in config.json.",
            )
        try:
            model = AdaptiveNewsModel(
                state_path=Path(str(self.parameters[state_file_key])),
                minimum_observations=int(
                    self.parameters["news_learning_min_observations"]
                ),
                maximum_observations=int(
                    self.parameters["news_learning_max_observations"]
                ),
            )
            result = model.update(
                evaluation_date=self.get_datetime().date().isoformat(),
                current_price=price_b,
                news_score=news_score,
            )
            self.log_message(f"{log_prefix}{result.explanation}", color="blue")
            return result
        except Exception as exc:
            self.log_message(
                f"{log_prefix}Adaptive news learning failed safely: {type(exc).__name__}: {exc}",
                color="red",
            )
            return LearningResult(
                0,
                False,
                None,
                None,
                None,
                f"Learning update failed: {type(exc).__name__}: {exc}",
            )

    def _update_adaptive_learning(
        self,
        price_b: float,
        news_context: NewsContext,
    ) -> LearningResult:
        """Update the persistent model and return its explainable forecast."""
        return self._update_news_learning(
            price_b,
            "news_learning_state_file",
            news_context.score if news_context.available else None,
        )

    def _update_llm_adaptive_learning(
        self,
        price_b: float,
        llm_assessment: LLMNewsAssessment,
    ) -> LearningResult:
        """Same regression as _update_adaptive_learning, trained on the LLM
        score instead of the deterministic keyword score, in its own state
        file. The two scores can diverge on the same day's news (a keyword
        scorer only matches a fixed phrase list; the LLM reads the full
        headline set) -- training both against realized next-session returns
        is how that disagreement gets resolved with evidence instead of by
        assumption. Purely observational for now: this forecast is reported
        alongside the keyword one but does not feed _market_veto_reason."""
        return self._update_news_learning(
            price_b,
            "news_learning_llm_state_file",
            llm_assessment.score if llm_assessment.available else None,
            log_prefix="LLM-score ",
        )

    def _update_decision_memory(
        self,
        price_a: float,
        price_b: float,
        dip_percent: float,
        news_context: NewsContext,
    ) -> RotationForecast:
        """Learn whether comparable past rotations favored B over A."""
        if not bool(self.parameters.get("decision_memory_enabled", True)):
            return RotationForecast(
                0, False, None, None, "Decision memory is disabled in config.json."
            )

        try:
            memory = self._decision_memory(
                int(self.parameters["decision_memory_min_observations"]),
                int(self.parameters["decision_memory_max_observations"]),
            )
            result = memory.update_and_forecast(
                evaluation_date=self.get_datetime().date().isoformat(),
                price_a=price_a,
                price_b=price_b,
                dip_percent=dip_percent,
                news_score=news_context.score if news_context.available else None,
                signal_present=dip_percent >= float(self.parameters["dip_threshold_percent"]),
            )
            self.log_message(result.explanation, color="blue")
            return result
        except Exception as exc:
            self.log_message(
                f"Decision memory failed safely: {type(exc).__name__}: {exc}",
                color="red",
            )
            return RotationForecast(
                0,
                False,
                None,
                None,
                f"Decision memory failed: {type(exc).__name__}: {exc}",
            )

    def _opportunistic_opportunity(
        self,
        asset_a: str,
        asset_b: str,
        price_a: float | None,
        price_b: float | None,
        news_context: NewsContext,
    ) -> dict[str, float | int | str | None]:
        """Evaluate the A/B rotation as a portfolio-only, data-backed option."""
        unavailable: dict[str, float | int | str | None] = {
            "status": "unavailable", "probability": None
        }
        if price_a is None or price_b is None or min(float(price_a), float(price_b)) <= 0:
            return unavailable
        bars = self.get_historical_prices(
            asset_b, int(self.parameters["recent_high_lookback_days"]), "day"
        )
        if bars is None or bars.df is None or bars.df.empty or "high" not in bars.df:
            return unavailable
        highs = [float(value) for value in bars.df["high"].dropna() if math.isfinite(float(value)) and float(value) > 0]
        if not highs:
            return unavailable
        recent_high = max(highs)
        dip = ((recent_high - float(price_b)) / recent_high) * 100.0
        self._backfill_decision_memory(asset_a, asset_b)
        forecast = self._update_decision_memory(float(price_a), float(price_b), dip, news_context)
        try:
            probability = self._decision_memory(1, 1).opportunity_probability()
        except Exception as exc:
            self.log_message(
                f"Opportunity probability lookup failed safely: {type(exc).__name__}: {exc}",
                color="red",
            )
            probability = OpportunityProbability(observations=0, wins=0, probability=None)
        return {
            "status": "ready" if forecast.ready else "warming up",
            "dip": dip,
            "predicted_edge": forecast.predicted_edge_percent,
            "observations": probability.observations,
            "wins": probability.wins,
            "probability": probability.probability,
            "forecast_explanation": forecast.explanation,
        }

    def _backfill_decision_memory(self, asset_a: str, asset_b: str) -> None:
        """Seed decision memory from settled daily bars once per process start."""
        if self.vars.decision_memory_backfill_attempted:
            return
        days = int(self.parameters.get("decision_memory_backfill_days", 0))
        if not bool(self.parameters.get("decision_memory_enabled", True)) or days < 2:
            self.vars.decision_memory_backfill_attempted = True
            return
        try:
            bars_a = self.get_historical_prices(asset_a, days, "day")
            bars_b = self.get_historical_prices(asset_b, days, "day")
            if (
                bars_a is None
                or bars_b is None
                or bars_a.df is None
                or bars_b.df is None
                or bars_a.df.empty
                or bars_b.df.empty
                or not {"close"}.issubset(bars_a.df.columns)
                or not {"close", "high"}.issubset(bars_b.df.columns)
            ):
                self.log_message("Decision-memory historical backfill unavailable; continuing normally.", color="yellow")
                return

            a_closes = {
                str(index.date() if hasattr(index, "date") else index): float(value)
                for index, value in bars_a.df["close"].dropna().items()
                if math.isfinite(float(value)) and float(value) > 0
            }
            b_rows = [
                (str(index.date() if hasattr(index, "date") else index), float(row["close"]), float(row["high"]))
                for index, row in bars_b.df[["close", "high"]].dropna().iterrows()
                if math.isfinite(float(row["close"]))
                and math.isfinite(float(row["high"]))
                and float(row["close"]) > 0
                and float(row["high"]) > 0
            ]
            lookback = int(self.parameters["recent_high_lookback_days"])
            threshold = float(self.parameters["dip_threshold_percent"])
            dips = decision_math.historical_dips(
                np.asarray([row[2] for row in b_rows], dtype=np.float64),
                np.asarray([row[1] for row in b_rows], dtype=np.float64),
                lookback,
            )
            history = []
            for (date, close_b, _), dip in zip(b_rows[lookback:], dips):
                close_a = a_closes.get(date)
                if close_a is None:
                    continue
                history.append(
                    (date, close_a, close_b, float(dip), float(dip) >= threshold)
                )
            inserted = self._decision_memory(
                1, int(self.parameters["decision_memory_max_observations"])
            ).backfill_history(history)
            self.log_message(
                f"Decision-memory historical backfill added {inserted} settled daily observations.",
                color="blue",
            )
            self.vars.decision_memory_backfill_attempted = True
        except Exception as exc:
            self.log_message(
                f"Decision-memory historical backfill failed safely: {type(exc).__name__}: {exc}",
                color="yellow",
            )

    def _decision_memory(self, minimum_observations: int, maximum_observations: int) -> TradeMemory:
        """Instance-cached like _portfolio_memory(). Call sites intentionally
        use different (min, max) windows (e.g. the opportunity-probability
        lookup wants only the most recent observation), so each distinct
        pairing keeps its own cached instance rather than sharing a single
        slot -- this still gets every call site off a fresh DuckDB connection
        and schema-creation pass per invocation."""
        key = (
            str(self.parameters["decision_memory_database_file"]),
            int(minimum_observations),
            int(maximum_observations),
        )
        cache = getattr(self, "_decision_memory_instances", None)
        if cache is None:
            cache = {}
            self._decision_memory_instances = cache
        cached = cache.get(key)
        if cached is None:
            cached = TradeMemory(Path(key[0]), key[1], key[2])
            cache[key] = cached
        return cached

    def _portfolio_memory(self) -> PortfolioMemory:
        key = (
            str(self.parameters["portfolio_memory_database_file"]),
            int(self.parameters["portfolio_memory_min_observations"]),
            int(self.parameters["portfolio_memory_max_observations"]),
        )
        cached = getattr(self, "_portfolio_memory_instance", None)
        if cached is None or getattr(self, "_portfolio_memory_key", None) != key:
            cached = PortfolioMemory(
                database_path=Path(key[0]),
                minimum_observations=key[1],
                maximum_observations=key[2],
            )
            self._portfolio_memory_instance = cached
            self._portfolio_memory_key = key
        return cached

    def _update_portfolio_memories(
        self,
        signals: list[dict[str, Any]],
        symbol_news_scores: dict[str, int],
        market_wide_news_score: int | None,
    ) -> dict[str, RotationForecast]:
        """Record and forecast every evaluated symbol with one pooled fit."""
        if not signals:
            return {}
        if not bool(self.parameters.get("portfolio_memory_enabled", True)):
            disabled = RotationForecast(
                0, False, None, None, "Portfolio memory is disabled in config.json."
            )
            return {str(signal["symbol"]): disabled for signal in signals}
        inputs = [
            PortfolioMemoryInput(
                symbol=str(signal["symbol"]),
                price=float(signal["price"]),
                dip_percent=float(signal["dip"]),
                news_score=symbol_news_scores.get(
                    str(signal["symbol"]), market_wide_news_score
                ),
                signal_present=bool(signal.get("qualifies")),
                live_spread_percent=signal.get("live_spread_percent"),
                recent_avg_volume=signal.get("recent_avg_volume"),
                historical_expected_profit=signal.get("expected_profit"),
                historical_win_probability=signal.get("win_probability"),
                historical_return_stdev=signal.get("return_stdev"),
            )
            for signal in signals
        ]
        try:
            return self._portfolio_memory().update_many_and_forecast(
                self.get_datetime().date().isoformat(), inputs
            )
        except Exception as exc:
            self.log_message(
                f"Portfolio memory batch update failed safely: {type(exc).__name__}: {exc}",
                color="yellow",
            )
            failed = RotationForecast(
                0, False, None, None, f"Portfolio memory failed: {type(exc).__name__}: {exc}"
            )
            return {item.symbol: failed for item in inputs}

    def _update_portfolio_memory(
        self,
        symbol: str,
        price: float,
        dip_percent: float,
        news_score: int | None,
        signal_present: bool,
        live_spread_percent: float | None = None,
        recent_avg_volume: float | None = None,
        historical_expected_profit: float | None = None,
        historical_win_probability: float | None = None,
        historical_return_stdev: float | None = None,
    ) -> RotationForecast:
        """Record today's context for one symbol and forecast its next-session return.

        Called once per *evaluated* symbol per day -- not just one clearing
        today's dip threshold -- pooling every symbol's history into one
        model; see PortfolioMemory for why a pooled fit (rather than one
        model per symbol) is what lets many symbols a day actually accelerate
        warm-up. `signal_present` keeps the forecast itself trained only on
        decision-specific dip days, exactly like trade_memory.py's own
        signal_present column, even though every symbol's daily context is
        now durably recorded regardless.
        """
        if not bool(self.parameters.get("portfolio_memory_enabled", True)):
            return RotationForecast(
                0, False, None, None, "Portfolio memory is disabled in config.json."
            )
        try:
            result = self._portfolio_memory().update_and_forecast(
                evaluation_date=self.get_datetime().date().isoformat(),
                symbol=symbol,
                price=price,
                dip_percent=dip_percent,
                news_score=news_score,
                signal_present=signal_present,
                live_spread_percent=live_spread_percent,
                recent_avg_volume=recent_avg_volume,
                historical_expected_profit=historical_expected_profit,
                historical_win_probability=historical_win_probability,
                historical_return_stdev=historical_return_stdev,
            )
            return result
        except Exception as exc:
            self.log_message(
                f"Portfolio memory update failed safely for {symbol}: {type(exc).__name__}: {exc}",
                color="yellow",
            )
            return RotationForecast(
                0, False, None, None, f"Portfolio memory failed: {type(exc).__name__}: {exc}"
            )

    def _backfill_portfolio_memory(
        self,
        symbol: str,
        history: list[tuple[str, float, float]] | None = None,
    ) -> None:
        """Seed one symbol's pooled memory from settled daily bars, once ever.

        Mirrors _backfill_decision_memory's price-only, once-per-symbol
        approach, but tracked in a set (not a single bool) since autonomous
        discovery can introduce new symbols throughout the process lifetime.
        """
        backfilled = self.vars.portfolio_memory_backfilled_symbols
        if symbol in backfilled:
            return
        backfilled.add(symbol)
        if not bool(self.parameters.get("portfolio_memory_enabled", True)):
            return
        try:
            if history is None:
                bars = self.get_historical_prices(
                    symbol, int(self.parameters["portfolio_analysis_days"]), "day"
                )
                if bars is None or bars.df is None or bars.df.empty or not {"high", "close"}.issubset(bars.df.columns):
                    return
                frame = bars.df[["high", "close"]].dropna()
                values = frame.to_numpy(dtype=np.float64, copy=False)
                valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
                values = values[valid]
                dates = frame.index[valid]
                lookback = int(self.parameters["recent_high_lookback_days"])
                threshold = float(self.parameters["dip_threshold_percent"])
                dips, next_returns = decision_math.historical_dip_returns(
                    values[:, 0], values[:, 1], lookback
                )
                selected = dips >= threshold
                history = [
                    (
                        str(date.date() if hasattr(date, "date") else date),
                        float(dip),
                        float(next_return),
                    )
                    for date, dip, next_return in zip(
                        dates[lookback:-1][selected], dips[selected], next_returns[selected]
                    )
                ]
            inserted = self._portfolio_memory().backfill_history(symbol, history)
            if inserted:
                self.log_message(
                    f"Portfolio-memory historical backfill added {inserted} settled observations for {symbol}.",
                    color="blue",
                )
        except Exception as exc:
            self.log_message(
                f"Portfolio-memory historical backfill failed safely for {symbol}: {type(exc).__name__}: {exc}",
                color="yellow",
            )

    def _record_memory_decision(self, report: dict[str, Any]) -> None:
        """Persist the final decision label after an observation was recorded."""
        if not report.get("decision_memory_recorded"):
            return
        try:
            self._decision_memory(1, 1).record_decision(
                self.get_datetime().date().isoformat(),
                str(report.get("status", "unknown")),
                str(report.get("decision_reason", report.get("status", ""))),
            )
        except Exception as exc:
            self.log_message(
                f"Could not label decision-memory entry: {type(exc).__name__}: {exc}",
                color="red",
            )

    @staticmethod
    def _quantity(position: Any) -> Decimal:
        """Return a safe, non-negative quantity for a Lumibot position."""
        if position is None:
            return Decimal("0")
        try:
            return max(Decimal(str(position.quantity)), Decimal("0"))
        except (AttributeError, InvalidOperation, TypeError, ValueError):
            return Decimal("0")

    def _managed_portfolio_symbols(self) -> set[str]:
        """Return symbols this strategy is permitted to count or sell.

        A shared Alpaca account may contain manual investments.  Portfolio
        mode must never adopt or liquidate them merely because they are stocks.
        Static symbols are explicitly opted in; discovered symbols become
        managed only after they have been persisted in the learned universe.
        """
        symbols = {
            str(symbol).strip().upper()
            for symbol in self.parameters["portfolio_symbols"]
            if str(symbol).strip()
        }
        if bool(self.parameters.get("portfolio_autonomous_discovery", False)):
            try:
                symbols.update(self._autonomous_universe().managed_symbols())
            except Exception as exc:
                self.log_message(
                    f"Could not read managed discovery symbols: {type(exc).__name__}: {exc}",
                    color="yellow",
                )
        symbols.update(
            str(symbol).upper()
            for symbol in getattr(self.vars, "portfolio_holding_dates", {})
        )
        for source, entry in self.vars.portfolio_pending_rotation.items():
            symbols.update((str(source).upper(), str(entry["to"]).upper()))
        return symbols

    def _portfolio_held_positions(
        self, managed_symbols: set[str]
    ) -> tuple[dict[str, Decimal], dict[str, float]] | None:
        """Return (quantities, avg entry prices) for managed long stock positions.

        None on broker-read failure. The entry price comes straight from the
        broker's own cost basis (Alpaca's avg_entry_price, surfaced by Lumibot
        as Position.avg_fill_price) rather than anything tracked locally, so
        it reflects the true fill price even across restarts or partial fills.
        """
        try:
            positions = self.get_positions() or []
        except Exception as exc:
            self.log_message(
                f"Could not read account positions ({type(exc).__name__}: {exc}); "
                "skipping this portfolio evaluation.",
                color="red",
            )
            return None
        held: dict[str, Decimal] = {}
        entry_prices: dict[str, float] = {}
        for position in positions:
            asset = getattr(position, "asset", None)
            symbol = getattr(asset, "symbol", None)
            asset_type = str(getattr(asset, "asset_type", "stock") or "stock").lower()
            normalized_symbol = str(symbol).upper() if symbol else ""
            if (
                not normalized_symbol
                or normalized_symbol not in managed_symbols
                or asset_type not in ("stock", "us_equity")
            ):
                continue
            quantity = self._quantity(position)
            if quantity > 0:
                held[normalized_symbol] = quantity
                avg_fill_price = getattr(position, "avg_fill_price", None)
                if avg_fill_price is not None:
                    try:
                        price_value = float(avg_fill_price)
                        if math.isfinite(price_value) and price_value > 0:
                            entry_prices[normalized_symbol] = price_value
                    except (TypeError, ValueError):
                        pass
        return held, entry_prices

    def _market_veto_reason(
        self,
        news_context: NewsContext,
        llm_assessment: LLMNewsAssessment,
        learning_result: LearningResult | None,
    ) -> str | None:
        """Return the first market-level veto that blocks opening a trade.

        Used by the portfolio path so it honors the same configured guards as
        the A/B path. Completing an in-flight rotation is never vetoed.
        """
        news_blocking_enabled = bool(
            self.parameters.get("news_context_enabled", True)
        ) and bool(self.parameters["news_block_on_high_risk"])
        if (
            news_blocking_enabled
            and bool(self.parameters.get("news_fail_closed_on_unavailable", True))
            and not news_context.available
        ):
            return "Trade blocked: world-event risk context is unavailable"
        if (
            news_context.available
            and news_blocking_enabled
            and news_context.score <= int(self.parameters["news_high_risk_score"])
        ):
            return f"Trade blocked: high world-event risk score {news_context.score}"
        llm_blocking_enabled = bool(self.parameters.get("llm_news_enabled", False))
        if (
            llm_blocking_enabled
            and bool(self.parameters.get("llm_news_fail_closed_on_unavailable", True))
            and not llm_assessment.available
        ):
            return "Trade blocked: LLM news assessment is unavailable"
        if (
            llm_assessment.available
            and llm_blocking_enabled
            and llm_assessment.score <= int(self.parameters["llm_news_block_score"])
        ):
            return f"Trade blocked: LLM news assessment score {llm_assessment.score:+d}"
        if (
            learning_result is not None
            and bool(self.parameters["news_learning_block_enabled"])
            and learning_result.ready
            and learning_result.predicted_return_percent is not None
            and learning_result.correlation is not None
            and abs(learning_result.correlation)
            >= float(self.parameters["news_learning_min_correlation"])
            and learning_result.predicted_return_percent
            <= float(self.parameters["news_predicted_return_block_percent"])
        ):
            return (
                "Trade blocked: adaptive model forecast "
                f"{learning_result.predicted_return_percent:+.2f}%"
            )
        return None

    _ACCOUNT_VALUE_CACHE_SECONDS = 30.0

    def _account_total_value_dollars(self) -> float:
        """The shared Alpaca account's total value (cash + net equity).

        Falls back to cash alone if a fresh broker equity read fails, and
        treats a non-finite (NaN/inf) or negative reading as zero available
        value -- all three failure modes can only push downstream
        calculations more conservative, never let them overspend, matching
        this codebase's fail-open convention for a bad broker read. Cached
        briefly (_ACCOUNT_VALUE_CACHE_SECONDS) since Lumibot's
        get_portfolio_value()/get_cash() each force their own fresh broker
        round-trip on every call with no caching between them -- without
        this, every buy attempt in a multi-candidate pass would pay for its
        own redundant network round-trip for a total that barely moves
        within one iteration.
        """
        cached = getattr(self, "_account_value_cache", None)
        now = time.monotonic()
        if cached is not None and now - cached[0] < self._ACCOUNT_VALUE_CACHE_SECONDS:
            return cached[1]
        total_value = self.get_portfolio_value()
        if (
            total_value is None
            or not math.isfinite(float(total_value))
            or float(total_value) <= 0
        ):
            total_value = self.get_cash()
        total_value = float(total_value or 0.0)
        if not math.isfinite(total_value) or total_value < 0:
            total_value = 0.0
        self._account_value_cache = (now, total_value)
        return total_value

    def _crypto_reserve_dollars(self) -> float:
        """Cash held back for the separate crypto process (main_crypto.py)
        sharing this Alpaca account: half the account's total value if
        crypto is actually enabled, else zero -- there is nothing to
        reserve for a pipeline that never trades. Mirrors
        CryptoRotationStrategy._account_half_value_dollars
        (crypto_strategy.py), which has no symmetric "is equity enabled"
        check since portfolio mode always runs (it is not optional).
        """
        if not bool(self.parameters.get("crypto_enabled", False)):
            return 0.0
        return self._account_total_value_dollars() * 0.5

    def _buy_portfolio_symbol(self, symbol: str, price: float, budget: float) -> str:
        """Buy a whole or fractional quantity within a stated portfolio budget."""
        with self._rotation_lock:
            if self._has_active_order(symbol, "buy"):
                return "working"
            spendable = min(float(self.get_cash()), budget) * (1.0 - self.CASH_BUFFER_FRACTION)
            spendable -= float(self.parameters.get("portfolio_cash_reserve_dollars", 0.0))
            spendable -= self._crypto_reserve_dollars()
            if spendable < float(self.parameters.get("portfolio_min_order_dollars", 1.0)):
                return "insufficient"
            if bool(self.parameters.get("fractional_shares", False)):
                quantity: Decimal | int = (Decimal(str(spendable)) / Decimal(str(price))).quantize(
                    Decimal("1.000000000"), rounding=ROUND_DOWN
                )
            else:
                quantity = math.floor(spendable / price)
            if quantity <= 0:
                return "insufficient"
            buy_order = self.create_order(
                symbol,
                quantity=quantity,
                side="buy",
                order_type="market",
                time_in_force="day",
            )
            if not self._submit_order_checked(buy_order, f"{symbol} portfolio buy"):
                return "rejected"
            self.log_message(
                f"Portfolio submitted buy of {quantity} {symbol} shares using up to ${budget:.2f}.",
                color="green",
            )
            return "submitted"

    _walk_forward_net_returns = staticmethod(decision_math.walk_forward_net_returns)

    def _uses_alpaca_market_data(self) -> bool:
        source = getattr(getattr(getattr(self, "broker", None), "data_source", None), "SOURCE", "")
        return str(source).upper() == "ALPACA"

    def _get_quote_price_and_bid_ask(
        self, symbol: str
    ) -> tuple[float | None, tuple[float, float] | None]:
        """Read Alpaca's signal price and spread inputs from one quote.

        Cached per iteration: exit-reason evaluation and signal computation
        both quote every held symbol within the same pass, moments apart, and
        a quote a few seconds stale is no less valid for either decision than
        refetching it would be.
        """
        cache = getattr(self, "_quote_cache", None)
        if cache is None:
            cache = {}
            self._quote_cache = cache
        if symbol in cache:
            return cache[symbol]
        result = self._fetch_quote_price_and_bid_ask(symbol)
        # Only cache a usable price. A transient fetch failure (result[0] is
        # None) must not poison every later call site for this symbol this
        # iteration -- each of those deserves its own chance at a fresh quote.
        if result[0] is not None:
            cache[symbol] = result
        return result

    def _fetch_quote_price_and_bid_ask(
        self, symbol: str
    ) -> tuple[float | None, tuple[float, float] | None]:
        try:
            quote = self.get_quote(symbol)
        except Exception:
            return None, None
        if quote is None:
            return None, None
        price: float | None = None
        raw_price = getattr(quote, "price", None)
        if raw_price is not None:
            try:
                candidate = float(raw_price)
            except (TypeError, ValueError):
                pass
            else:
                if math.isfinite(candidate) and candidate > 0:
                    price = candidate
        try:
            bid = float(getattr(quote, "bid", 0.0) or 0.0)
            ask = float(getattr(quote, "ask", 0.0) or 0.0)
        except (TypeError, ValueError, AttributeError):
            return price, None
        if price is None:
            fallback = bid if bid > 0 else ask
            price = fallback if math.isfinite(fallback) and fallback > 0 else None
        bid_ask = (
            (bid, ask)
            if math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask > bid
            else None
        )
        return price, bid_ask

    def _get_bid_ask(self, symbol: str) -> tuple[float, float] | None:
        """Live (bid, ask) for symbol, or None on any invalid quote."""
        return self._get_quote_price_and_bid_ask(symbol)[1]

    def _live_spread_percent(self, symbol: str) -> float | None:
        """Best-effort live bid/ask spread, as a round-trip cost estimate.

        Returns None on any missing/invalid/one-sided quote so the caller can
        fail open to the configured flat PORTFOLIO_ROUND_TRIP_COST_PERCENT.
        """
        bid_ask = self._get_bid_ask(symbol)
        if bid_ask is None:
            return None
        return self._spread_percent_from_bid_ask(bid_ask)

    def _spread_percent_from_bid_ask(self, bid_ask: tuple[float, float]) -> float:
        bid, ask = bid_ask
        mid = (bid + ask) / 2.0
        spread_percent = ((ask - bid) / mid) * 100.0
        return min(spread_percent, self._PORTFOLIO_LIVE_SPREAD_CAP_PERCENT)

    def _realizable_sale_price(self, symbol: str) -> float | None:
        """Price a market sell of this symbol would actually realize.

        A market sell fills against the live bid, not the last trade --
        which can sit anywhere inside the spread and make an exit threshold
        look closer than it really is. Falls back to get_last_price on any
        missing/invalid quote so a data hiccup can't block an exit.
        """
        bid_ask = self._get_bid_ask(symbol)
        if bid_ask is not None:
            return bid_ask[0]
        last_price = self.get_last_price(symbol)
        if last_price is None:
            return None
        try:
            value = float(last_price)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value > 0 else None

    def _portfolio_exit_reasons(
        self,
        held: dict[str, Decimal],
        entry_prices: dict[str, float],
        holding_dates: dict[str, str],
        today: date_type,
    ) -> dict[str, str]:
        """Decide which holdings exit today, and why, without submitting anything.

        Take-profit and stop-loss compare the broker's cost basis against the
        realizable (bid-side) price; a holding between those bounds is left to
        run. The holding-horizon backstop catches everything else -- including
        a holding whose cost basis or price is unavailable -- so a stagnant or
        unpriceable symbol can't occupy a slot forever.
        """
        take_profit_percent = float(self.parameters.get("portfolio_take_profit_percent", 1.0))
        stop_loss_percent = float(self.parameters.get("portfolio_stop_loss_percent", 0.5))
        backstop_days = int(self.parameters.get("portfolio_holding_horizon_max_days", 15))
        exit_reasons: dict[str, str] = {}
        for symbol in held:
            entry_price = entry_prices.get(symbol)
            current_price = self._realizable_sale_price(symbol)
            if entry_price is not None and current_price is not None:
                unrealized_percent = ((current_price - entry_price) / entry_price) * 100.0
                if unrealized_percent >= take_profit_percent:
                    exit_reasons[symbol] = f"take-profit reached ({unrealized_percent:+.2f}%)"
                    continue
                if unrealized_percent <= -stop_loss_percent:
                    exit_reasons[symbol] = f"stop-loss reached ({unrealized_percent:+.2f}%)"
                    continue
            if self._holding_is_due(holding_dates.get(symbol, today.isoformat()), today, backstop_days):
                exit_reasons[symbol] = f"{backstop_days}-day holding backstop reached"
        return exit_reasons

    def _portfolio_signal(
        self, symbol: str, bars: Any = None, *, bars_prefetched: bool = False
    ) -> dict[str, Any] | None:
        """Compute today's per-symbol trade context, whether or not it qualifies.

        Returns ``None`` only when there isn't enough data to say anything
        useful about `symbol` at all: no bars, no price, or it fails the
        discovery liquidity floor -- a real trade-blocking guard applied to
        the whole evaluated universe, unchanged here. Otherwise this always
        returns a context dict so every watched/held symbol contributes daily
        learning facts to `portfolio_memory.py`, regardless of whether
        today's dip clears `dip_threshold_percent`. The `qualifies` field is
        the sole gate `_run_portfolio_iteration` uses for trading eligibility
        (its `eligible` filter and `held_signals` construction both re-check
        it) -- everything else here is learning context, not a trading signal
        by itself.
        """
        if not bars_prefetched:
            bars = self.get_historical_prices(
                symbol, int(self.parameters["portfolio_analysis_days"]), "day"
            )
        if bars is None or bars.df is None or bars.df.empty or not {"high", "close"}.issubset(bars.df.columns):
            # Distinct from the liquidity-floor Nones below: no bars at all
            # means Alpaca has no price history for this symbol, ever -- a
            # discovery-sourced candidate like this can never qualify, so
            # _run_portfolio_iteration persists it as permanently unpriceable
            # instead of re-fetching and re-warning about it every time the
            # discovery rotation cursor comes back around.
            with self._unpriceable_symbols_guard():
                if not hasattr(self, "_unpriceable_symbols_this_iteration"):
                    self._unpriceable_symbols_this_iteration = set()
                self._unpriceable_symbols_this_iteration.add(symbol)
            return None
        frame = bars.df[["high", "close"]].dropna()
        values = frame.to_numpy(dtype=np.float64, copy=False)
        valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
        values = values[valid]
        dates = frame.index[valid]
        lookback = int(self.parameters["recent_high_lookback_days"])
        if len(values) <= lookback:
            return None
        alpaca_quote = self._uses_alpaca_market_data()
        if alpaca_quote:
            price, bid_ask = self._get_quote_price_and_bid_ask(symbol)
        else:
            price = self.get_last_price(symbol)
            bid_ask = None
        if price is None or not math.isfinite(float(price)) or float(price) <= 0:
            return None
        # A low-priced or thin-volume symbol can still clear the profit/OOS
        # filters below on a small backtest sample; for a sub-$100 account
        # neither the static watchlist nor a held position should be
        # re-evaluated once it falls under a sane liquidity floor. Checked
        # here (not just at discovery) since it reuses the bars/price already
        # fetched above rather than a second data pass, and it re-applies
        # every day rather than only at first discovery.
        min_price = float(self.parameters.get("portfolio_discovery_min_price_dollars", 0.0))
        if min_price > 0 and float(price) < min_price:
            return None
        recent_avg_volume: float | None = None
        volume_available = "volume" in bars.df.columns
        if volume_available:
            recent_volume = bars.df["volume"].dropna().tail(lookback)
            if not recent_volume.empty:
                recent_avg_volume = float(recent_volume.mean())
        min_avg_volume = float(self.parameters.get("portfolio_discovery_min_avg_volume", 0.0))
        if min_avg_volume > 0 and volume_available and (recent_avg_volume is None or recent_avg_volume < min_avg_volume):
            return None
        threshold = float(self.parameters["dip_threshold_percent"])
        dips, historical_returns = decision_math.historical_dip_returns(
            values[:, 0], values[:, 1], lookback
        )
        selected = dips >= threshold
        returns = historical_returns[selected].tolist()
        memory_history = [
            (
                str(date.date() if hasattr(date, "date") else date),
                float(dip),
                float(next_return),
            )
            for date, dip, next_return in zip(
                dates[lookback:-1][selected], dips[selected], historical_returns[selected]
            )
        ]
        recent_high = float(values[-lookback:, 0].max())
        current_dip = ((recent_high - float(price)) / recent_high) * 100.0
        configured_round_trip_cost = float(
            self.parameters.get("portfolio_round_trip_cost_percent", 0.20)
        )
        # The configured value is one flat guess applied to every symbol
        # regardless of liquidity. A live bid/ask spread is a per-symbol
        # floor under it -- never a full replacement, since Alpaca's free
        # quote feed is IEX-only -- so a thinly traded discovered symbol
        # can't look cheaper to trade than it actually is on a sub-$100
        # order where the spread is a much larger share of the target edge.
        # Fetched for every symbol reaching this point, not just a qualifying
        # dip, so it becomes one of the daily learning facts too.
        if alpaca_quote:
            live_spread = (
                self._spread_percent_from_bid_ask(bid_ask)
                if bid_ask is not None
                else None
            )
        else:
            live_spread = self._live_spread_percent(symbol)
        round_trip_cost = (
            max(configured_round_trip_cost, live_spread)
            if live_spread is not None
            else configured_round_trip_cost
        )
        expected_profit: float | None = None
        observations = 0
        oos_expected_profit: float | None = None
        oos_observations = 0
        return_stdev: float | None = None
        win_probability: float | None = None
        if returns:
            net_returns = np.asarray(returns, dtype=np.float64) - round_trip_cost
            walk_forward_returns = self._walk_forward_net_returns(
                returns,
                round_trip_cost,
                int(self.parameters.get("portfolio_oos_min_observations", 10)),
                float(self.parameters["portfolio_min_expected_profit_percent"]),
            )
            mean_net_return = float(net_returns.mean())
            variance = float(net_returns.var())
            wins = int(np.count_nonzero(net_returns > 0))
            expected_profit = mean_net_return
            observations = len(returns)
            oos_expected_profit = (
                sum(walk_forward_returns) / len(walk_forward_returns) if walk_forward_returns else None
            )
            oos_observations = len(walk_forward_returns)
            return_stdev = math.sqrt(variance)
            win_probability = (wins + 1) / (len(net_returns) + 2)
        return {
            "symbol": symbol,
            "price": float(price),
            "dip": current_dip,
            # Today's dip must clear the threshold *and* there must be at
            # least one historical comparable dip to estimate an edge from.
            "qualifies": current_dip >= threshold and bool(returns),
            # This net historical mean is a coarse current estimate. It is
            # never enough by itself: _run_portfolio_iteration also requires
            # the chronological walk-forward result below. None when there is
            # no historical comparable dip yet.
            "expected_profit": expected_profit,
            "observations": observations,
            "oos_expected_profit": oos_expected_profit,
            "oos_observations": oos_observations,
            # Feed the risky/conservative reasoning pattern in
            # _posture_adjusted_edge: how spread out this symbol's past
            # dip-signal outcomes were, and a Laplace-smoothed win rate
            # (matches TradeMemory.opportunity_probability's convention).
            "return_stdev": return_stdev,
            "win_probability": win_probability,
            # Exposed so callers (PortfolioMemory blending) can net a learned
            # edge against the same per-symbol cost basis expected_profit
            # already used, instead of the one flat configured guess.
            "round_trip_cost": round_trip_cost,
            # Daily learning facts beyond dip/news: liquidity/cost context
            # that PortfolioMemory now records for every evaluated symbol.
            "live_spread_percent": live_spread,
            "recent_avg_volume": recent_avg_volume,
            "_memory_history": memory_history,
        }

    def _portfolio_signals(
        self, symbols: list[str]
    ) -> list[dict[str, float | int | str | None] | None]:
        """Fetch independent symbol histories through a small bounded pool."""
        if not symbols:
            return []
        with self._unpriceable_symbols_guard():
            self._unpriceable_symbols_this_iteration = set()
        prefetched: dict[str, Any] | None = None
        try:
            batch = self.get_historical_prices_for_assets(
                symbols,
                int(self.parameters["portfolio_analysis_days"]),
                "day",
                chunk_size=100,
                max_workers=self._PORTFOLIO_HISTORY_WORKERS,
            )
            prefetched = {
                str(getattr(asset, "symbol", asset)).upper(): bars
                for asset, bars in batch.items()
            }
        except Exception as exc:
            # Preserve compatibility with alternate/backtesting brokers that
            # do not implement the multi-asset path.
            log_message = getattr(self, "log_message", None)
            if callable(log_message) and getattr(self, "logger", None) is not None:
                log_message(
                    "Portfolio history batch unavailable; using per-symbol requests: "
                    f"{type(exc).__name__}: {exc}",
                    color="yellow",
                )
            prefetched = None
        with ThreadPoolExecutor(
            max_workers=min(self._PORTFOLIO_HISTORY_WORKERS, len(symbols)),
            thread_name_prefix="portfolio-history",
        ) as executor:
            if prefetched is not None:
                return list(
                    executor.map(
                        lambda symbol: self._portfolio_signal(
                            symbol, prefetched[symbol], bars_prefetched=True
                        )
                        if symbol in prefetched
                        else self._portfolio_signal(symbol),
                        symbols,
                    )
                )
            return list(executor.map(self._portfolio_signal, symbols))

    def _symbol_news_scores(
        self, news_context: NewsContext, candidates: set[str]
    ) -> dict[str, int]:
        """Per-symbol news severity, cross-checked against the local symbol reference.

        Starts from NewsContext.per_symbol_scores (built from Alpaca's own
        article symbol tags), drops any tag the local reference has never
        seen from either source (catching a spurious tag), then extends
        coverage using scan_text_for_symbols for a company mentioned by name
        but missed by Alpaca's tagging -- bounded to today's evaluated
        `candidates`, never the whole market. A symbol with neither an
        Alpaca tag nor a text match is intentionally absent here -- callers
        fall back to the market-wide score for it, exactly as before this
        feature.
        """
        if not news_context.available:
            return {}
        scores = dict(news_context.per_symbol_scores)
        if not bool(self.parameters.get("symbol_reference_enabled", True)):
            return scores
        try:
            reference = self._symbol_reference()
            verified = reference.verified_symbols()
            if verified:
                scores = {symbol: value for symbol, value in scores.items() if symbol in verified}
            aliases = reference.aliases_for_symbols(candidates)
            for article in news_context.per_article:
                tagged = {str(symbol) for symbol in article.get("symbols", [])}
                untagged_candidates = candidates - tagged
                if not untagged_candidates:
                    continue
                text = f"{article.get('headline', '')} {article.get('summary', '')}"
                eligible_aliases = {
                    symbol: aliases[symbol]
                    for symbol in untagged_candidates
                    if symbol in aliases
                }
                for symbol in reference.scan_text_for_aliases(text, eligible_aliases):
                    scores[symbol] = scores.get(symbol, 0) + int(article.get("score", 0))
            return scores
        except Exception as exc:
            self.log_message(
                f"Symbol-aware news scoring failed safely: {type(exc).__name__}: {exc}",
                color="yellow",
            )
            return dict(news_context.per_symbol_scores)

    @staticmethod
    def _articles_for_symbol(news_context: NewsContext, symbol: str) -> list[dict]:
        """Articles Alpaca itself tagged with this symbol -- a conservative
        subset of _symbol_news_scores' coverage (which also credits untagged
        text-scan matches), but enough to ground an LLM explanation without
        re-running that bounded scan here."""
        if not news_context.available:
            return []
        return [
            {
                "headline": article.get("headline", ""),
                "summary": article.get("summary", ""),
                "url": article.get("url", ""),
            }
            for article in news_context.per_article
            if symbol in article.get("symbols", [])
        ]

    def _generate_daily_narrative(self, report: dict[str, Any]) -> str:
        """A short plain-English recap of this iteration for the daily email,
        from the local model. Purely descriptive summarization of decisions
        already made elsewhere in this method -- never a new decision itself,
        and never consumed by any trading logic. Fails open to an empty
        string, in which case the email simply omits the section."""
        if not bool(self.parameters.get("llm_news_enabled", False)):
            return ""
        try:
            actions = report.get("portfolio_actions") or ["none"]
            context_lines = [
                f"Result: {report.get('status', 'unavailable')}",
                f"Holdings: {report.get('portfolio_holdings', 'unavailable')}",
                f"News risk: {report.get('news_risk_level', 'unavailable')} "
                f"(score {report.get('news_score', 'unavailable')})",
                f"LLM risk assessment: {report.get('llm_risk_level', 'unavailable')} "
                f"- {report.get('llm_reasoning', '')}",
                f"Learned return forecast: {report.get('learned_forecast', 'not ready')}",
                f"Opportunistic Opportunity: {report.get('opportunistic_opportunity_status', 'unavailable')}",
                "Actions taken this iteration:",
                *[f"- {action}" for action in actions],
            ]
            analyzer = LLMNewsAnalyzer(
                model=str(self.parameters["llm_news_model"]),
                base_url=str(self.parameters.get("llm_news_base_url", "")),
            )
            return analyzer.summarize_day("\n".join(context_lines))
        except Exception as exc:
            self.log_message(
                f"Daily narrative failed safely: {type(exc).__name__}: {exc}",
                color="yellow",
            )
            return ""

    def _generate_exit_narrative(
        self, symbol: str, price_reason: str, news_context: NewsContext
    ) -> str:
        """One human-readable line connecting a just-submitted exit to that
        symbol's own headlines, when it has dedicated coverage today. Purely
        descriptive -- the exit already fired on price alone; this cannot
        change or delay it. Fails open to an empty string."""
        if not bool(self.parameters.get("llm_news_enabled", False)) or not news_context.available:
            return ""
        articles = self._articles_for_symbol(news_context, symbol)
        if not articles:
            return ""
        try:
            analyzer = LLMNewsAnalyzer(
                model=str(self.parameters["llm_news_model"]),
                base_url=str(self.parameters.get("llm_news_base_url", "")),
            )
            return analyzer.explain_exit(symbol, price_reason, articles)
        except Exception as exc:
            self.log_message(
                f"Exit narrative failed safely for {symbol}: {type(exc).__name__}: {exc}",
                color="yellow",
            )
            return ""

    def _check_discovery_red_flags(
        self,
        discovered_only: list[str],
        news_context: NewsContext,
        symbol_news_scores: dict[str, int],
        report: dict[str, Any],
    ) -> set[str]:
        """Exclude newly-discovered candidates with severe company red flags.

        The check runs before order decisions. It skips symbols without
        dedicated negative coverage rather than spending a model call on
        nothing to screen. A failed check remains fail-open for that symbol.
        """
        if not bool(self.parameters.get("llm_news_enabled", False)) or not news_context.available:
            return set()
        flags: dict[str, str] = {}
        excluded: set[str] = set()
        for symbol in discovered_only:
            score = symbol_news_scores.get(symbol)
            if score is None or score >= 0:
                continue
            articles = self._articles_for_symbol(news_context, symbol)
            if not articles:
                continue
            try:
                analyzer = LLMNewsAnalyzer(
                    model=str(self.parameters["llm_news_model"]),
                    base_url=str(self.parameters.get("llm_news_base_url", "")),
                )
                result: RedFlagCheck = analyzer.check_red_flag(symbol, articles)
            except Exception as exc:
                self.log_message(
                    f"Discovery red-flag check failed safely for {symbol}: "
                    f"{type(exc).__name__}: {exc}",
                    color="yellow",
                )
                continue
            if result.available and result.flagged:
                flags[symbol] = result.reason
                self.log_message(
                    f"Discovery red flag: {symbol} - {result.reason} (excluded today)",
                    color="yellow",
                )
                excluded.add(symbol)
        if flags:
            report["discovery_red_flags"] = "; ".join(
                f"{symbol}: {reason}" for symbol, reason in flags.items()
            )
        return excluded

    def _check_discovery_article_context(
        self,
        discovered_only: list[str],
        news_context: NewsContext,
        symbol_news_scores: dict[str, int],
        report: dict[str, Any],
        *,
        require_negative_score: bool = True,
    ) -> set[str]:
        """For the same negative-coverage discovery candidates
        `_check_discovery_red_flags` screens, fetch that symbol's own
        highest-signal article in full (not just Alpaca's headline/summary)
        and ask the local model for a structured sentiment/risk verdict.
        A bearish verdict excludes the symbol before order decisions. Skips a
        symbol with no article URL rather than spending a fetch+call on
        nothing.
        `article_filter.extract_financial_context` never raises, so no
        per-symbol try/except is needed here.

        `require_negative_score=False` (used by `_run_nightly_preevaluation`
        to pre-warm the cache for every candidate symbol overnight, not just
        discovery's negative-news ones) skips the score filter below and
        checks every symbol passed in.
        """
        if not bool(self.parameters.get("llm_news_enabled", False)) or not news_context.available:
            return set()
        verdicts: dict[str, str] = {}
        excluded: set[str] = set()
        for symbol in discovered_only:
            if require_negative_score:
                score = symbol_news_scores.get(symbol)
                if score is None or score >= 0:
                    continue
            url = next(
                (
                    str(article.get("url") or "").strip()
                    for article in self._articles_for_symbol(news_context, symbol)
                    if str(article.get("url") or "").strip()
                ),
                "",
            )
            if not url:
                continue
            context = article_filter.extract_financial_context(url, [symbol])
            if not context:
                continue
            sentiment = str(context.get("sentiment", "unknown"))
            catalyst = str(context.get("catalyst_type", "other"))
            risks = ", ".join(context.get("key_risks") or []) or "no specific risks cited"
            verdicts[symbol] = f"{sentiment} ({catalyst}): {risks}"
            self.log_message(
                f"Discovery article context: {symbol} - {verdicts[symbol]}",
                color="yellow" if sentiment == "bearish" else "blue",
            )
            if sentiment == "bearish":
                excluded.add(symbol)
        if verdicts:
            report["discovery_article_context"] = "; ".join(
                f"{symbol}: {verdict}" for symbol, verdict in verdicts.items()
            )
        return excluded

    def _run_pretrade_discovery_analysis(
        self,
        discovered_only: list[str],
        news_context: NewsContext,
        symbol_news_scores: dict[str, int],
        report: dict[str, Any],
    ) -> set[str]:
        """Run every decision-changing discovery check before order decisions."""
        if (
            not discovered_only
            or not bool(self.parameters.get("llm_news_enabled", False))
            or not news_context.available
        ):
            return set()
        negative_candidates = sorted(
            (
                symbol
                for symbol in discovered_only
                if symbol_news_scores.get(symbol) is not None
                and int(symbol_news_scores[symbol]) < 0
            ),
            key=lambda symbol: (int(symbol_news_scores[symbol]), symbol),
        )
        excluded = self._check_discovery_red_flags(
            negative_candidates, news_context, symbol_news_scores, report
        )
        excluded.update(
            self._check_discovery_article_context(
                negative_candidates, news_context, symbol_news_scores, report
            )
        )
        report["discovery_analysis_status"] = (
            f"Pre-trade LLM analysis checked {len(negative_candidates)} negative-news symbol(s)"
        )
        return excluded

    def _queue_exit_narratives(
        self, requests: list[tuple[str, str, NewsContext]]
    ) -> None:
        """Queue descriptive exit notes without delaying any order phase."""
        if not requests:
            return
        lock = getattr(self, "_exit_narrative_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._exit_narrative_lock = lock
        with lock:
            pending = getattr(self, "_pending_exit_narratives", None)
            if pending is None:
                pending = []
                self._pending_exit_narratives = pending
            pending.extend(requests)

    def _start_deferred_exit_narratives(self) -> None:
        """Generate descriptive exit notes after the decision path completes."""
        lock = getattr(self, "_exit_narrative_lock", None)
        if lock is None:
            return
        with lock:
            if getattr(self, "_exit_narrative_worker_running", False):
                return
            if not getattr(self, "_pending_exit_narratives", []):
                return
            self._exit_narrative_worker_running = True

        def analyze_in_background() -> None:
            while True:
                with lock:
                    exit_payloads = list(
                        getattr(self, "_pending_exit_narratives", [])
                    )
                    self._pending_exit_narratives = []
                    if not exit_payloads:
                        self._exit_narrative_worker_running = False
                        return
                try:
                    for symbol, price_reason, context in exit_payloads:
                        narrative = self._generate_exit_narrative(
                            symbol, price_reason, context
                        )
                        if narrative:
                            self.log_message(
                                f"Exit note: {symbol} - {narrative}", color="blue"
                            )
                except Exception as exc:
                    self.log_message(
                        "Deferred exit narrative failed safely: "
                        f"{type(exc).__name__}: {exc}",
                        color="yellow",
                    )

        threading.Thread(
            target=analyze_in_background,
            name="deferred-exit-narratives",
            daemon=True,
        ).start()

    def _run_nightly_preevaluation(self) -> dict[str, Any]:
        """One off-hours pass, meant to run once a night well after midnight:
        every managed/held symbol gets its per-symbol LLM article verdict
        pre-computed and cached (article_filter.py's `.article_verdicts.duckdb`,
        keyed by calendar day), so the live market-open/midday iteration finds
        a same-day cache hit instead of paying the Ollama round-trip live.

        Deliberately uses `_managed_portfolio_symbols` rather than
        `_portfolio_symbols` -- the latter calls `AutonomousUniverse.next_batch`,
        which mutates discovery batch-rotation state on every call, and this
        must never consume a discovery batch the live morning iteration
        should get instead. Fails open like every other news/LLM path: any
        problem here must never delay or block trading.
        """
        report: dict[str, Any] = {}
        if not bool(self.parameters.get("llm_news_enabled", False)) or not bool(
            self.parameters.get("portfolio_nightly_preeval_enabled", True)
        ):
            return report
        managed_symbols = self._managed_portfolio_symbols()
        positions = self._portfolio_held_positions(managed_symbols)
        held = positions[0] if positions is not None else {}
        all_symbols = sorted(managed_symbols | set(held))
        news_context = self._get_news_context()
        symbol_news_scores = self._symbol_news_scores(news_context, set(all_symbols))
        self._check_discovery_article_context(
            all_symbols,
            news_context,
            symbol_news_scores,
            report,
            require_negative_score=False,
        )
        self._save_nightly_preeval_state(
            report.get("discovery_article_context", ""), len(all_symbols)
        )
        return report

    _posture_adjusted_edge = staticmethod(decision_math.posture_adjusted_edge)
    _optimal_position_count = staticmethod(decision_math.optimal_position_count)

    def _autonomous_universe(self) -> AutonomousUniverse:
        """Instance-cached like _portfolio_memory(): called up to 4x/iteration
        and each fresh instance used to reopen its own DuckDB connection."""
        key = (
            str(self.parameters["portfolio_universe_database_file"]),
            int(self.parameters["portfolio_discovery_refresh_days"]),
            int(self.parameters["portfolio_discovery_batch_size"]),
            os.environ.get("ALPACA_IS_PAPER", "true").strip().lower() != "false",
        )
        cached = getattr(self, "_autonomous_universe_instance", None)
        if cached is None or getattr(self, "_autonomous_universe_key", None) != key:
            cached = AutonomousUniverse(
                Path(key[0]), key[1], key[2], paper=key[3]
            )
            self._autonomous_universe_instance = cached
            self._autonomous_universe_key = key
        return cached

    def _symbol_reference(self) -> SymbolReference:
        """Instance-cached like _portfolio_memory(); see _autonomous_universe()."""
        key = (
            str(self.parameters["symbol_reference_database_file"]),
            int(self.parameters["symbol_reference_refresh_days"]),
            os.environ.get("ALPACA_IS_PAPER", "true").strip().lower() != "false",
        )
        cached = getattr(self, "_symbol_reference_instance", None)
        if cached is None or getattr(self, "_symbol_reference_key", None) != key:
            cached = SymbolReference(Path(key[0]), key[1], paper=key[2])
            self._symbol_reference_instance = cached
            self._symbol_reference_key = key
        return cached

    def _refresh_symbol_reference(self, symbols: list[str]) -> None:
        """Queue a daemon refresh without delaying the trading iteration.

        A refresh failure must not affect trading: it only ever narrows or
        widens which per-symbol news attributions are trusted, never
        creates a trade or veto.
        """
        if not bool(self.parameters.get("symbol_reference_enabled", True)):
            return
        normalized = sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})
        if not normalized:
            return
        refresh_lock = getattr(self, "_symbol_reference_refresh_lock", None)
        if refresh_lock is None:
            refresh_lock = threading.Lock()
            self._symbol_reference_refresh_lock = refresh_lock
        with refresh_lock:
            pending = getattr(self, "_symbol_reference_pending_symbols", None)
            if pending is None:
                pending = set()
                self._symbol_reference_pending_symbols = pending
            pending.update(normalized)
            if bool(getattr(self, "_symbol_reference_refresh_running", False)):
                return
            self._symbol_reference_refresh_running = True

        def refresh_in_background() -> None:
            while True:
                with refresh_lock:
                    if not self._symbol_reference_pending_symbols:
                        self._symbol_reference_refresh_running = False
                        return
                    batch = sorted(self._symbol_reference_pending_symbols)
                    self._symbol_reference_pending_symbols.clear()
                try:
                    refreshed = self._symbol_reference().refresh(
                        batch,
                        os.environ.get("ALPACA_API_KEY", ""),
                        os.environ.get("ALPACA_API_SECRET", ""),
                    )
                    if refreshed:
                        self.log_message(
                            f"Symbol reference refreshed for {len(batch)} symbols.", color="blue"
                        )
                except Exception as exc:
                    self.log_message(
                        f"Symbol reference refresh failed safely: {type(exc).__name__}: {exc}",
                        color="yellow",
                    )

        threading.Thread(
            target=refresh_in_background,
            name="symbol-reference-refresh",
            daemon=True,
        ).start()

    def _portfolio_symbols(
        self,
        report: dict[str, Any],
        held: dict[str, Decimal],
        managed_symbols: set[str],
    ) -> list[str]:
        """Combine the watchlist, current holdings, and one discovery batch.

        Held symbols are always part of the universe so an existing position
        keeps getting a signal (and stays eligible for rotation) even after
        the discovery batch that surfaced it has moved on.
        """
        symbols = list(dict.fromkeys(sorted(managed_symbols) + sorted(held)))
        if not bool(self.parameters.get("portfolio_autonomous_discovery", False)):
            return symbols
        try:
            discovered = self._autonomous_universe().next_batch(
                os.environ.get("ALPACA_API_KEY", ""),
                os.environ.get("ALPACA_API_SECRET", ""),
            )
            report["discovered_symbols"] = ", ".join(discovered) or "none"
            return list(dict.fromkeys(symbols + discovered))
        except Exception as exc:
            # Discovery cannot turn a provider outage into a trade decision.
            report["discovery_status"] = f"unavailable: {type(exc).__name__}"
            self.log_message(
                f"Autonomous discovery failed safely: {type(exc).__name__}: {exc}",
                color="yellow",
            )
            return symbols

    def _guarded_universe_call(self, action: Callable[[], None], error_message: str) -> None:
        """Shared guard for every AutonomousUniverse write: no-op unless
        portfolio_autonomous_discovery is on, and any failure is logged,
        never raised, since discovery bookkeeping must never block a
        trading decision."""
        if not bool(self.parameters.get("portfolio_autonomous_discovery", False)):
            return
        try:
            action()
        except Exception as exc:
            self.log_message(f"{error_message}: {type(exc).__name__}: {exc}", color="yellow")

    def _remember_discovered_symbols(self, symbols: list[str]) -> None:
        self._guarded_universe_call(
            lambda: self._autonomous_universe().remember(symbols),
            "Could not persist learned symbols",
        )

    def _exclude_unpriceable_discovered_symbols(self, symbols: list[str]) -> None:
        """Stop a discovery-only symbol with no Alpaca price history from resurfacing."""
        self._guarded_universe_call(
            lambda: self._autonomous_universe().exclude_unpriceable(symbols),
            "Could not persist unpriceable symbols",
        )

    def _remember_confirmed_portfolio_symbol(self, symbol: str) -> None:
        """Grant management permission only after a broker-confirmed buy fill."""
        self._guarded_universe_call(
            lambda: self._autonomous_universe().remember_owned([str(symbol).upper()]),
            "Could not persist strategy ownership",
        )

    def _forget_confirmed_portfolio_symbol(self, symbol: str) -> None:
        """Revoke discovery management permission after the position is sold."""
        self._guarded_universe_call(
            lambda: self._autonomous_universe().forget_owned([str(symbol).upper()]),
            "Could not revoke strategy ownership",
        )

    def _submit_portfolio_builds(
        self,
        desired: list[dict[str, Any]],
        held_working: dict[str, Decimal],
        claimed_symbols: set[str],
        effective_max_positions: int,
        actions: list[str],
    ) -> int:
        """Submit empty-slot buys without reusing cash reserved earlier in the pass."""
        submitted = 0
        remaining_cash = max(0.0, float(self.get_cash()))
        for candidate in desired:
            symbol = str(candidate["symbol"])
            if symbol in held_working or symbol in claimed_symbols:
                continue
            if len(held_working) + submitted >= effective_max_positions:
                break
            slots_remaining = max(
                1, effective_max_positions - (len(held_working) + submitted)
            )
            budget = remaining_cash / slots_remaining
            outcome = self._buy_portfolio_symbol(
                symbol, float(candidate["price"]), budget
            )
            if outcome == "insufficient":
                break
            if outcome == "rejected":
                actions.append(
                    f"Portfolio build rejected: {symbol} purchase was not accepted"
                )
                continue
            remaining_cash = max(0.0, remaining_cash - budget)
            claimed_symbols.add(symbol)
            submitted += 1
            actions.append(f"Portfolio build: {symbol} purchase {outcome}")
        return submitted

    def _submit_portfolio_replacements(
        self,
        remaining_candidates: list[dict[str, Any]],
        signals: list[dict[str, Any]],
        held_working: dict[str, Decimal],
        claimed_symbols: set[str],
        minimum_profit: float,
        actions: list[str],
    ) -> int:
        """Replace weak unclaimed holdings with materially stronger candidates."""
        held_signals = {
            str(signal["symbol"]): signal
            for signal in signals
            if signal["symbol"] in held_working and bool(signal.get("qualifies"))
        }
        submitted = 0
        for candidate in remaining_candidates:
            target_symbol = str(candidate["symbol"])
            if target_symbol in held_working or target_symbol in claimed_symbols:
                continue
            unclaimed_held = [
                symbol for symbol in held_working if symbol not in claimed_symbols
            ]
            if not unclaimed_held:
                break
            source = min(
                unclaimed_held,
                key=lambda symbol: float(
                    held_signals.get(
                        symbol, {"posture_adjusted_edge": 0.0}
                    )["posture_adjusted_edge"]
                ),
            )
            source_score = float(
                held_signals.get(source, {"posture_adjusted_edge": 0.0})[
                    "posture_adjusted_edge"
                ]
            )
            advantage = float(candidate["posture_adjusted_edge"]) - source_score
            if advantage < minimum_profit:
                continue
            source_price = self.get_last_price(source)
            if (
                source_price is None
                or float(source_price) <= 0
                or self._has_active_order(source, "sell")
            ):
                continue
            budget = float(source_price) * float(held_working[source])
            if self._submit_portfolio_rotation_sell(
                source,
                target_symbol,
                held_working[source],
                budget,
                kind="replacement",
            ):
                held_working.pop(source, None)
                claimed_symbols.update({source, target_symbol})
                submitted += 1
                actions.append(
                    f"Portfolio rotation submitted: {source} to {target_symbol} "
                    f"(expected advantage {advantage:+.2f}%)"
                )
        return submitted

    def _maybe_top_up_portfolio(
        self,
        desired: list[dict[str, Any]],
        held_working: dict[str, Decimal],
        actions: list[str],
    ) -> None:
        """Put a usable residual deposit into the best already-held candidate."""
        top_up_candidate = next(
            (
                signal
                for signal in desired
                if str(signal["symbol"]) in held_working
            ),
            None,
        )
        cash = float(self.get_cash())
        minimum_cash = float(
            self.parameters.get("portfolio_cash_reserve_dollars", 0.0)
        ) + float(self.parameters.get("portfolio_min_order_dollars", 1.0))
        if top_up_candidate is not None and cash >= minimum_cash:
            symbol = str(top_up_candidate["symbol"])
            outcome = self._buy_portfolio_symbol(
                symbol, float(top_up_candidate["price"]), cash
            )
            actions.append(f"Portfolio top-up: {symbol} purchase {outcome}")

    def _submit_due_portfolio_exits(
        self,
        held: dict[str, Decimal],
        entry_prices: dict[str, float],
        news_context: NewsContext,
        actions: list[str],
        claimed_symbols: set[str],
        held_working: dict[str, Decimal],
    ) -> list[tuple[str, str, NewsContext]]:
        """Submit every due exit before queuing any descriptive LLM work."""
        holding_dates = dict(self.vars.portfolio_holding_dates)
        today = self.get_datetime().date()
        new_dates = False
        for symbol in held:
            if symbol not in holding_dates:
                holding_dates[symbol] = today.isoformat()
                new_dates = True
        for symbol in list(holding_dates):
            if symbol not in held:
                holding_dates.pop(symbol)
                new_dates = True
        if new_dates:
            self._set_portfolio_holding_dates(holding_dates)

        narrative_requests: list[tuple[str, str, NewsContext]] = []
        exit_reasons = self._portfolio_exit_reasons(
            held, entry_prices, holding_dates, today
        )
        for source in sorted(exit_reasons):
            if source in claimed_symbols:
                continue
            if self._has_active_order(source, "sell"):
                actions.append(f"Portfolio exit pending: waiting for {source} sale")
                continue
            reason = exit_reasons[source]
            exit_order = self.create_order(
                source,
                quantity=held[source],
                side="sell",
                order_type="market",
                time_in_force="day",
            )
            if not self._submit_order_checked(
                exit_order, f"{source} exit sell ({reason})"
            ):
                actions.append(f"Portfolio exit rejected: {source} sale was not accepted")
                continue
            actions.append(f"Portfolio exit submitted: {source} {reason}")
            narrative_requests.append((source, reason, news_context))
            held_working.pop(source, None)
            claimed_symbols.add(source)
        return narrative_requests

    def _reconcile_pending_portfolio_rotations(
        self, held: dict[str, Decimal]
    ) -> tuple[list[str], set[str]]:
        """Reconcile restart-safe sale/buy pairs before new decisions.

        This transaction phase is intentionally isolated from the large
        portfolio orchestrator so restart recovery can be integration-tested
        with broker-shaped state without running news and signal analysis.
        """
        actions: list[str] = []
        # Iterate a snapshot so removals made mid-loop do not disturb it.
        for source in sorted(self.vars.portfolio_pending_rotation):
            entry = self.vars.portfolio_pending_rotation.get(source)
            if entry is None:
                continue
            target = str(entry["to"])
            budget = float(entry["budget"])
            kind = str(entry["kind"])
            if held.get(source, Decimal("0")) > 0:
                if self._has_active_order(source, "sell"):
                    actions.append(f"Portfolio pending: waiting for {source} sale")
                    continue
                self._remove_portfolio_rotation(source)
                actions.append(f"Portfolio rotation reset: {source} sale did not fill")
                continue
            if held.get(target, Decimal("0")) > 0 and not self._has_active_order(
                target, "buy"
            ):
                self._remove_portfolio_rotation(source)
                actions.append(
                    f"Portfolio rotation complete ({kind}): the {target} purchase filled"
                )
                continue
            price = self.get_last_price(target)
            if price is None or not math.isfinite(float(price)) or float(price) <= 0:
                actions.append(f"Portfolio pending: no valid {target} price")
                continue
            outcome = self._buy_portfolio_symbol(target, float(price), budget)
            if outcome == "insufficient":
                self._remove_portfolio_rotation(source)
                actions.append(
                    f"Portfolio rotation finished: cash is below the minimum {target} order"
                )
            elif outcome == "working":
                actions.append(f"Portfolio pending: waiting for the {target} purchase to fill")
            elif outcome == "rejected":
                actions.append(
                    f"Portfolio pending: broker rejected the {target} purchase; retrying next cycle"
                )
            else:
                # Clear only after the replacement fill is confirmed.
                actions.append(f"Portfolio {target} purchase submitted after {source} sale")
        claimed_symbols = set(self.vars.portfolio_pending_rotation) | {
            str(entry["to"])
            for entry in self.vars.portfolio_pending_rotation.values()
        }
        return actions, claimed_symbols

    def _run_portfolio_iteration(self, report: dict[str, Any]) -> None:
        """Build or rotate a bounded portfolio from the explicit symbol list."""
        self._invalidate_orders_cache()
        self._quote_cache = {}
        minimum_observations = int(self.parameters["portfolio_min_signal_observations"])
        minimum_profit = float(self.parameters["portfolio_min_expected_profit_percent"])
        oos_minimum_observations = int(
            self.parameters.get("portfolio_oos_min_observations", 10)
        )
        oos_minimum_profit = float(
            self.parameters.get("portfolio_oos_min_net_profit_percent", 0.0)
        )
        max_positions = int(self.parameters["portfolio_max_positions"])

        managed_symbols = self._managed_portfolio_symbols()
        positions_result = self._portfolio_held_positions(managed_symbols)
        if positions_result is None:
            report["status"] = "No portfolio trade: account positions were unavailable"
            return
        held, entry_prices = positions_result
        report["portfolio_holdings"] = (
            ", ".join(f"{symbol}={quantity}" for symbol, quantity in sorted(held.items())) or "none"
        )
        symbols = self._portfolio_symbols(report, held, managed_symbols)
        self._refresh_symbol_reference(symbols)

        news_context = self._get_news_context()
        report.update(
            news_risk_level=news_context.risk_level,
            news_score=news_context.score if news_context.available else "unavailable",
            news_article_count=(
                news_context.article_count if news_context.available else "unavailable"
            ),
            news_explanation=news_context.explanation,
            news_headlines=news_context.headlines,
        )
        # Computed once here and reused by the ranking step below (symbol_news_scores
        # at its original call site) so the LLM sees exactly the same cross-checked,
        # deduped per-symbol coverage that ranking trusts, not the rawer
        # NewsContext.per_symbol_scores, and so the pipeline doesn't scan for
        # untagged symbol mentions twice in one iteration.
        symbol_news_scores = self._symbol_news_scores(news_context, set(symbols))
        llm_assessment = self._llm_assessment_for_iteration(
            news_context, symbols, set(held), symbol_news_scores
        )
        report.update(
            llm_risk_level=llm_assessment.risk_level,
            llm_score=llm_assessment.score if llm_assessment.available else "unavailable",
            llm_reasoning=(
                llm_assessment.reasoning
                if llm_assessment.available
                else llm_assessment.explanation
            ),
        )
        nightly_learnings = self._load_nightly_preeval_learnings()
        if nightly_learnings:
            report["nightly_learned_summary"] = (
                nightly_learnings.get("summary") or "no notable verdicts"
            )
            report["nightly_learned_symbol_count"] = nightly_learnings.get("symbol_count", 0)

        # Screen only symbols discovery itself just surfaced -- never a held
        # or statically-configured symbol -- for a severe, company-specific
        # red flag the liquidity/price floor can't see. Advisory by default;
        # see _check_discovery_red_flags.
        discovered_only = sorted(set(symbols) - managed_symbols - set(held))
        excluded_symbols = self._run_pretrade_discovery_analysis(
            discovered_only, news_context, symbol_news_scores, report
        )
        if excluded_symbols:
            symbols = [symbol for symbol in symbols if symbol not in excluded_symbols]

        # The adaptive model keeps learning from the configured market proxy
        # (Asset B) so its forecast can veto portfolio trades exactly as it
        # vetoes A/B rotations. A missing proxy price fails open.
        learning_result: LearningResult | None = None
        proxy_price = self.get_last_price(str(self.parameters["asset_b"]).upper())
        if proxy_price is not None and math.isfinite(float(proxy_price)) and float(proxy_price) > 0:
            learning_result = self._update_adaptive_learning(float(proxy_price), news_context)
            report.update(
                learning_observations=learning_result.observations,
                learned_forecast=(
                    f"{learning_result.predicted_return_percent:+.2f}%"
                    if learning_result.ready
                    and learning_result.predicted_return_percent is not None
                    else "not ready"
                ),
                learning_explanation=learning_result.explanation,
            )
            # Same regression, trained on the LLM's score instead, so the two
            # signals' predictive value can be compared once both have
            # enough history -- see _update_llm_adaptive_learning.
            llm_learning_result = self._update_llm_adaptive_learning(
                float(proxy_price), llm_assessment
            )
            report.update(
                llm_learning_observations=llm_learning_result.observations,
                llm_learned_forecast=(
                    f"{llm_learning_result.predicted_return_percent:+.2f}%"
                    if llm_learning_result.ready
                    and llm_learning_result.predicted_return_percent is not None
                    else "not ready"
                ),
                llm_learning_explanation=llm_learning_result.explanation,
            )
        veto_reason = self._market_veto_reason(news_context, llm_assessment, learning_result)

        # A/B is no longer an alternate strategy mode. It is a separately
        # labelled opportunity inside portfolio mode, trained only on the
        # settled A-versus-B observations already kept in decision memory.
        asset_a = str(self.parameters["asset_a"]).upper()
        asset_b = str(self.parameters["asset_b"]).upper()
        # proxy_price is this same asset_b, fetched moments ago for learning;
        # reusing it keeps the forecast and the learning update on one price
        # snapshot instead of two reads that could straddle a tick. Same
        # reasoning for asset_a_price: it's reused below for swap sizing if
        # the opportunity is eligible, instead of a second read.
        asset_a_price = self.get_last_price(asset_a)
        opportunity = self._opportunistic_opportunity(
            asset_a, asset_b, asset_a_price, proxy_price, news_context
        )
        probability = opportunity.get("probability")
        report.update(
            decision_memory_recorded=opportunity.get("status") != "unavailable",
            opportunistic_opportunity_status=opportunity.get("status"),
            opportunistic_opportunity_probability=(
                f"{float(probability):.1%}" if probability is not None else "not ready"
            ),
            opportunistic_opportunity_explanation=opportunity.get(
                "forecast_explanation", "A/B data was unavailable."
            ),
        )

        # Every phase below accumulates into `actions` and takes at most
        # max_positions worth of trades in this one call, instead of the one
        # trade per day the old waterfall-of-early-returns allowed. `held_working`
        # is popped as sells are submitted so later phases in this same pass see
        # an up-to-date view without waiting for a broker round-trip.
        # `claimed_symbols` is the single source of truth preventing any symbol
        # from being touched twice in one pass; it starts from every symbol
        # already referenced by a surviving pending rotation.
        actions, claimed_symbols = self._reconcile_pending_portfolio_rotations(held)
        held_working = dict(held)

        # Phase 1: manage every current holding for profit-taking and loss
        # containment, not just the first one. This is a plain single-leg
        # sell -- no paired buy, no rotation-slot usage -- and is never
        # vetoed, exactly like completing a pending rotation above. The
        # decision itself (take-profit / stop-loss against the broker's own
        # cost basis and the realizable bid-side price, with the holding-
        # horizon backstop) lives in _portfolio_exit_reasons. A configured
        # managed holding with no fill record is conservatively dated today
        # on first observation rather than sold immediately.
        narrative_requests = self._submit_due_portfolio_exits(
            held,
            entry_prices,
            news_context,
            actions,
            claimed_symbols,
            held_working,
        )
        self._queue_exit_narratives(narrative_requests)

        # Alpaca requests for separate symbols are independent. A small fixed
        # pool removes the observed serial latency without creating an
        # unbounded burst against the broker API.
        signals = self._portfolio_signals(symbols)
        # Only a discovery-sourced candidate is safe to permanently exclude --
        # a config-listed watchlist symbol (SPY/QQQ) hitting a transient data
        # outage must stay eligible for re-evaluation, not be blacklisted.
        discovery_only_symbols = set(symbols) - managed_symbols - set(held)
        unpriceable_this_iteration = getattr(self, "_unpriceable_symbols_this_iteration", set())
        unpriceable_discovered = sorted(unpriceable_this_iteration & discovery_only_symbols)
        if unpriceable_discovered:
            self._exclude_unpriceable_discovered_symbols(unpriceable_discovered)
        signals = [signal for signal in signals if signal is not None]
        # The risky/conservative reasoning pattern only reshapes ranking and
        # tie-breaking below; the eligibility floor two lines down still
        # gates on the raw historical expected_profit, unaffected by posture.
        risk_posture = str(self.parameters.get("portfolio_risk_posture", "conservative"))
        market_wide_news_score = news_context.score if news_context.available else None
        llm_purchase_score = llm_assessment.score if llm_assessment.available else None
        pending_backfill = {
            str(signal["symbol"]): signal.get("_memory_history", [])
            for signal in signals
            if str(signal["symbol"]) not in self.vars.portfolio_memory_backfilled_symbols
        }
        if pending_backfill and bool(self.parameters.get("portfolio_memory_enabled", True)):
            self.vars.portfolio_memory_backfilled_symbols.update(pending_backfill)
            try:
                inserted = self._portfolio_memory().backfill_many(pending_backfill)
                if inserted:
                    self.log_message(
                        f"Portfolio-memory historical backfill added {inserted} pooled observations.",
                        color="blue",
                    )
            except Exception as exc:
                self.log_message(
                    f"Portfolio-memory batch backfill failed safely: {type(exc).__name__}: {exc}",
                    color="yellow",
                )
        forecasts = self._update_portfolio_memories(
            signals, symbol_news_scores, market_wide_news_score
        )
        for signal in signals:
            symbol = str(signal["symbol"])
            # A symbol with dedicated coverage today (even a genuinely
            # neutral 0) is trusted over the market-wide score; only a
            # symbol with no coverage at all falls back to it.
            news_score = symbol_news_scores.get(symbol, market_wide_news_score)
            qualifies = bool(signal.get("qualifies"))
            # Every symbol reaching this point has usable bars/price/liquidity
            # (see _portfolio_signal), so every one of them -- not just those
            # with a qualifying dip today -- contributes a daily learning
            # observation; this is what "learn from every watched/held
            # symbol" means. signal_present keeps the pooled regression
            # trained on decision-specific dip days only, exactly like
            # trade_memory.py's own signal_present column, so broader daily
            # coverage doesn't dilute the forecast. All database writes and
            # the pooled fit were performed once above for the full batch.
            forecast = forecasts[symbol]
            # Ranking/eligibility fields are only meaningful -- and only ever
            # read downstream -- for a symbol that qualifies today; leaving
            # them absent/None for the rest reproduces exactly how these
            # symbols behaved before they started appearing in `signals` at
            # all (the eligible filter and held_signals below both re-check
            # `qualifies` for the same reason).
            if not qualifies:
                signal["learned_edge_ready"] = False
                signal["learned_edge"] = None
                signal["posture_adjusted_edge"] = None
                continue
            signal["learned_edge_ready"] = forecast.ready
            signal["learned_edge"] = (
                forecast.predicted_edge_percent - float(signal["round_trip_cost"])
                if forecast.ready and forecast.predicted_edge_percent is not None
                else None
            )
            signal["posture_adjusted_edge"] = self._posture_adjusted_edge(
                signal, risk_posture, news_score, llm_purchase_score
            )
        report["portfolio_risk_posture"] = risk_posture
        signal_snapshot.write_snapshot(
            str(self.parameters.get("portfolio_signal_snapshot_file", "")),
            self.get_datetime().isoformat(),
            risk_posture,
            signal_snapshot.build_snapshot_entries(signals, held),
        )
        eligible = [
            signal
            for signal in signals
            if bool(signal.get("qualifies"))
            and int(signal["observations"]) >= minimum_observations
            and float(signal["expected_profit"]) >= minimum_profit
            and int(signal["oos_observations"]) >= oos_minimum_observations
            and signal["oos_expected_profit"] is not None
            and float(signal["oos_expected_profit"]) >= oos_minimum_profit
        ]
        eligible.sort(key=lambda signal: (float(signal["posture_adjusted_edge"]), float(signal["dip"])), reverse=True)
        opportunity_probability = opportunity.get("probability")
        opportunity_edge = opportunity.get("predicted_edge")
        opportunity_is_eligible = (
            asset_a in held_working
            and asset_b not in held_working
            and asset_a not in claimed_symbols
            and asset_b not in claimed_symbols
            and opportunity.get("status") == "ready"
            and float(opportunity.get("dip") or 0.0) >= float(self.parameters["dip_threshold_percent"])
            and opportunity_probability is not None
            and float(opportunity_probability) >= float(self.parameters["portfolio_opportunistic_min_probability"])
            and opportunity_edge is not None
            and float(opportunity_edge) >= minimum_profit
            and not self.vars.portfolio_iteration_state.get("opportunistic_swap_done", False)
        )
        # Persist only positions the strategy actually owns. Merely qualifying
        # a discovered ticker must not grant permission to manage a manual
        # account position in that symbol. New buys are remembered on fill.
        self._remember_discovered_symbols(sorted(held))
        report["portfolio_candidates"] = ", ".join(
            f"{s['symbol']} net {s['expected_profit']:+.2f}%/{s['observations']} "
            f"(posture {s['posture_adjusted_edge']:+.2f}%); "
            f"OOS {float(s['oos_expected_profit']):+.2f}%/{s['oos_observations']}"
            for s in eligible
        ) or "none"

        signal_present = bool(eligible) or opportunity_is_eligible
        if veto_reason and signal_present:
            self.log_message(
                f"Portfolio signal present, but the trade was vetoed: {veto_reason}",
                color="red",
            )

        # Phase 2: the Opportunistic Opportunity is evaluated exactly once, as
        # a single non-looped decision, before Phase 3 gets to pick from
        # `eligible`. Reserving both legs here (via claimed_symbols) is what
        # structurally keeps it distinct from -- never folded into or
        # competing for a slot within -- the up-to-max_positions batch below,
        # even though PORTFOLIO_SYMBOLS defaults to include both assets. Now
        # that a trading day can run this function up to twice (see
        # _due_portfolio_iteration_window), "at most one swap per day" is no
        # longer structurally guaranteed by call count alone -- enforced
        # instead by opportunity_is_eligible's opportunistic_swap_done check
        # above, persisted in .portfolio_iteration_state.json so it survives
        # a restart between the day's two windows.
        if opportunity_is_eligible and not veto_reason:
            if self._has_active_order(asset_a, "sell"):
                actions.append("Opportunistic Opportunity pending: waiting for Asset A sale")
            else:
                source_price = asset_a_price
                if source_price is None or float(source_price) <= 0:
                    actions.append("No Opportunistic Opportunity: Asset A price was unavailable")
                else:
                    budget = float(source_price) * float(held_working[asset_a])
                    if self._submit_portfolio_rotation_sell(
                        asset_a,
                        asset_b,
                        held_working[asset_a],
                        budget,
                        kind="opportunistic",
                    ):
                        self.vars.portfolio_iteration_state["opportunistic_swap_done"] = True
                        self._save_portfolio_iteration_state()
                        held_working.pop(asset_a, None)
                        claimed_symbols.update({asset_a, asset_b})
                        actions.append(
                            f"Opportunistic Opportunity submitted: {asset_a} to {asset_b} "
                            f"({float(opportunity_probability):.1%} historical win probability, "
                            f"{float(opportunity_edge):+.2f}% predicted edge)"
                        )

        # Phase 3: build empty slots, then replace weak holdings, then top up
        # -- looping over every remaining ranked candidate this iteration
        # instead of acting on just the single best one and waiting until
        # tomorrow for the next.
        if not veto_reason:
            remaining_candidates = [
                signal for signal in eligible if str(signal["symbol"]) not in claimed_symbols
            ]
            # Narrow the configured ceiling to what today's capital (this
            # pipeline's own share of the shared account -- the full account
            # if crypto is disabled, half of it otherwise -- existing
            # holdings plus spendable cash) and candidate quality actually
            # support -- see _optimal_position_count, _account_total_value_dollars,
            # and _crypto_reserve_dollars.
            total_capital = self._account_total_value_dollars() - self._crypto_reserve_dollars()
            min_order_dollars = float(self.parameters.get("portfolio_min_order_dollars", 1.0))
            candidate_edges = [
                (float(signal["expected_profit"]), float(signal.get("return_stdev") or 0.0))
                for signal in remaining_candidates
            ]
            effective_max_positions = self._optimal_position_count(
                float(total_capital), min_order_dollars, candidate_edges, max_positions
            )
            report["portfolio_effective_max_positions"] = effective_max_positions
            desired = remaining_candidates[:effective_max_positions]

            builds_submitted = self._submit_portfolio_builds(
                desired,
                held_working,
                claimed_symbols,
                effective_max_positions,
                actions,
            )

            replacements_submitted = 0
            if len(held_working) + builds_submitted >= effective_max_positions:
                # A holding with no current dip signal is scored neutral (0%
                # expected edge), not punished: rotation happens only when the
                # target's posture-adjusted edge beats holding by the
                # configured margin. The old -100% default force-rotated any
                # recovered holding every time some other symbol dipped,
                # churning the portfolio. The posture lens only changes which
                # holding looks weakest and by how much; the
                # PORTFOLIO_MIN_EXPECTED_PROFIT_PERCENT floor is unchanged.
                replacements_submitted = self._submit_portfolio_replacements(
                    remaining_candidates,
                    signals,
                    held_working,
                    claimed_symbols,
                    minimum_profit,
                    actions,
                )

            if builds_submitted == 0 and replacements_submitted == 0:
                # A recurring small deposit should grow the highest-ranked
                # current holding instead of remaining idle once the portfolio
                # is full and every top candidate is already held.
                self._maybe_top_up_portfolio(desired, held_working, actions)

        report["portfolio_actions"] = actions
        report["status"] = self._summarize_portfolio_actions(actions, signal_present, veto_reason)

    @staticmethod
    def _summarize_portfolio_actions(
        actions: list[str], signal_present: bool, veto_reason: str | None
    ) -> str:
        """Compose the single top-line status CLAUDE.md's email report needs.

        Falls back to the historical single-sentence messages when at most
        one thing happened this iteration (the common case, byte-identical to
        the pre-rework behavior); composes a short multi-action summary
        otherwise.
        """
        if not actions:
            if veto_reason and signal_present:
                return veto_reason
            if not signal_present:
                return "No portfolio trade: no portfolio signal or Opportunistic Opportunity met its thresholds"
            return "No portfolio trade: current holdings match top signals and cash is below the minimum order"
        if len(actions) == 1:
            return actions[0]
        return f"Portfolio: {len(actions)} actions this iteration -- " + "; ".join(actions)

    def _due_iteration_window_now(self) -> str | None:
        """Return the due window label, or None to skip this poll cheaply.

        Polled every sleeptime tick (see initialize) but the full pipeline
        below should only actually run at the day's "open" and "midday"
        windows -- see _due_portfolio_iteration_window. Fails open to
        "skip, retry next poll" (never "run unconditionally") if today's
        market open can't be determined right now.
        """
        now = self.get_datetime()
        today = now.date().isoformat()
        state = self.vars.portfolio_iteration_state
        if state.get("date") != today:
            state = self._default_portfolio_iteration_state()
            state["date"] = today
            self.vars.portfolio_iteration_state = state
        try:
            market_open = self.broker.market_hours(close=False, next=False)
            offset_minutes = int(self.parameters.get("portfolio_second_iteration_offset_minutes", 0))
            window = self._due_portfolio_iteration_window(
                now, market_open, offset_minutes, state["windows_completed"]
            )
        except Exception as exc:
            self.log_message(
                f"Could not evaluate the iteration schedule safely, skipping this poll: "
                f"{type(exc).__name__}: {exc}",
                color="yellow",
            )
            return None
        return window

    def _mark_iteration_window_completed(self, window: str) -> None:
        """Persist a window only after its trading pipeline returns successfully."""
        state = self.vars.portfolio_iteration_state
        if window in state["windows_completed"]:
            return
        state["windows_completed"] = [*state["windows_completed"], window]
        self._save_portfolio_iteration_state()

    def on_trading_iteration(self) -> None:
        """Evaluate today's portfolio dip signals up to twice a trading day.

        Polled every sleeptime tick; _due_iteration_window_now gates the
        actual pipeline below to the day's "open" and "midday" windows so
        the news layer (and everything else) gets a second, independent
        daily read without doubling every tick into a trade decision.
        """
        window = self._due_iteration_window_now()
        if window is None:
            return
        report = {
            "threshold": float(self.parameters["dip_threshold_percent"]),
            "status": "Evaluation started",
        }
        try:
            self._run_portfolio_iteration(report)
            self._mark_iteration_window_completed(window)
        except Exception as exc:
            report["status"] = f"Evaluation error: {type(exc).__name__}: {exc}"
            # Active-order checks and persisted rotation intent make the
            # pipeline retry-safe. Do not consume this window: a transient
            # network or broker failure should retry on the next poll.
            self.log_message(
                f"Trading iteration failed safely: {type(exc).__name__}: {exc}",
                color="red",
            )
        finally:
            try:
                self._record_memory_decision(report)
                report["daily_narrative"] = self._generate_daily_narrative(report)
                self._send_daily_email(report)
            finally:
                # Exit notes describe completed sells and cannot affect the
                # decision that produced them. All decision-changing LLM work
                # has already completed synchronously inside the iteration.
                self._start_deferred_exit_narratives()

    def on_filled_order(
        self,
        position: Any,
        order: Any,
        price: float,
        quantity: float,
        multiplier: float,
    ) -> None:
        """Record broker-confirmed executions in the Lumibot log."""
        # This callback runs on the broker's own event thread, independent of
        # _run_portfolio_iteration's cadence -- it can fire long after the
        # iteration that populated self._orders_cache returned. A fill or
        # cancellation is itself proof the order book just changed, so any
        # snapshot from a prior iteration must not be trusted here.
        self._invalidate_orders_cache()
        symbol = str(getattr(getattr(order, "asset", None), "symbol", "unknown")).upper()
        side = getattr(order, "side", "unknown")
        # `quantity`/`price` here are the broker trade-update event's own fields,
        # which for an order that fills across multiple partial executions are
        # only the size/price of the LAST individual execution, not the order's
        # total. Use the order's total requested quantity and weighted-average
        # fill price instead so the log and journal reflect the whole trade.
        total_quantity = getattr(order, "quantity", None)
        fill_price = getattr(order, "get_fill_price", lambda: None)()
        if total_quantity is None:
            total_quantity = quantity
        if fill_price is None:
            fill_price = price
        self.log_message(
            f"Filled {side} order: {total_quantity} shares of {symbol} at ${fill_price:.2f}.",
            color="green",
        )
        try:
            self._decision_memory(1, 1).record_execution(
                self.get_datetime().date().isoformat(),
                str(symbol),
                str(side),
                float(fill_price),
                float(total_quantity),
            )
        except Exception as exc:
            self.log_message(
                f"Could not journal execution: {type(exc).__name__}: {exc}",
                color="red",
            )

        # Continue the rotation immediately after Alpaca confirms the sale.
        # The next daily iteration remains a fallback if this callback cannot
        # obtain fresh account or price data during a temporary outage.
        side_text = str(side).lower()

        # Buy the replacement as soon as the source sale fills (instead of
        # waiting a full day for the next iteration), and clear the pending
        # flag only when the replacement purchase itself fills.
        portfolio_pending = self.vars.portfolio_pending_rotation
        if side_text == "buy":
            self._record_portfolio_entry(str(symbol))
            self._remember_confirmed_portfolio_symbol(str(symbol))
        elif side_text == "sell":
            self._remove_portfolio_entry(str(symbol))
            self._forget_confirmed_portfolio_symbol(str(symbol))
        if portfolio_pending:
            if side_text == "buy":
                # Buy-fills are matched by scanning targets: N is bounded by
                # portfolio_max_positions (small), and there is no order-id
                # correlation to key off instead.
                completed_source = next(
                    (source for source, entry in portfolio_pending.items() if entry["to"] == symbol),
                    None,
                )
                if completed_source is not None:
                    kind = portfolio_pending[completed_source]["kind"]
                    self._remove_portfolio_rotation(completed_source)
                    self.log_message(
                        f"Portfolio rotation complete ({kind}): the {symbol} purchase filled.",
                        color="green",
                    )
                    return
            elif side_text == "sell" and symbol in portfolio_pending:
                entry = portfolio_pending[symbol]
                target = str(entry["to"])
                try:
                    target_price = self.get_last_price(target)
                    if (
                        target_price is None
                        or not math.isfinite(float(target_price))
                        or float(target_price) <= 0
                    ):
                        self.log_message(
                            f"The {symbol} sale filled, but {target} has no valid "
                            "price; the purchase will be retried next cycle.",
                            color="yellow",
                        )
                        return
                    outcome = self._buy_portfolio_symbol(
                        target, float(target_price), float(entry["budget"])
                    )
                    if outcome == "insufficient":
                        # Proceeds may not have settled yet; the next daily
                        # iteration retries with confirmed balances.
                        self.log_message(
                            f"The {target} purchase will be retried next cycle in "
                            "case the sale proceeds have not settled yet.",
                            color="yellow",
                        )
                    elif outcome == "rejected":
                        self.log_message(
                            f"The broker rejected the {target} purchase; the pending "
                            "rotation remains recorded for the next cycle.",
                            color="red",
                        )
                except Exception as exc:
                    self.log_message(
                        f"Portfolio post-sale purchase failed safely and will be "
                        f"retried: {type(exc).__name__}: {exc}",
                        color="red",
                    )
                return

    def on_canceled_order(self, order: Any) -> None:
        """Keep the rotation state truthful when the broker kills an order."""
        self._invalidate_orders_cache()  # see on_filled_order's comment
        symbol = str(getattr(getattr(order, "asset", None), "symbol", "unknown")).upper()
        side = str(getattr(order, "side", "unknown")).lower()
        self.log_message(
            f"Order canceled or rejected by the broker: {side} {symbol}.",
            color="red",
        )

        portfolio_pending = self.vars.portfolio_pending_rotation
        if side == "sell" and symbol in portfolio_pending:
            # Nothing was sold, so that portfolio rotation never started.
            kind = portfolio_pending[symbol]["kind"]
            self._remove_portfolio_rotation(symbol)
            self.log_message(
                f"The {symbol} sale was canceled; the {kind} rotation is "
                "reset and will be re-evaluated next cycle.",
                color="yellow",
            )
        # A canceled portfolio buy keeps its entry pending so the next
        # iteration retries the purchase with the cash still on hand.
