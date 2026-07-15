"""Daily dip-buying and asset-rotation strategy for Lumibot."""

import html
import json
import math
import os
import smtplib
import ssl
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date as date_type, timedelta
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from lumibot.strategies import Strategy

from adaptive_news_model import AdaptiveNewsModel, LearningResult
from autonomous_universe import AutonomousUniverse
from llm_news import LLMNewsAnalyzer, LLMNewsAssessment
from news_context import NewsContext, WorldEventAnalyzer
from congress_context import CongressContext, CongressTradeAnalyzer
from wsb_context import WSBContext, WallStreetBetsAnalyzer, WallStreetBetsSnapshot
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
        "congress_context_enabled": False,
        "wsb_context_enabled": False,
        "wsb_discovery_enabled": False,
        "llm_news_enabled": False,
        "decision_memory_enabled": True,
        "decision_memory_block_enabled": False,
        "portfolio_enabled": True,
        "portfolio_oos_min_observations": 10,
        "portfolio_oos_min_net_profit_percent": 0.0,
        "portfolio_round_trip_cost_percent": 0.20,
        "portfolio_max_holding_days": 1,
        "portfolio_risk_posture": "conservative",
    }

    # Fraction of cash withheld from the Asset B buy so the market order is not
    # rejected (or filled into a deficit) if the price moves before execution.
    CASH_BUFFER_FRACTION = 0.01

    # How strongly the risky/conservative reasoning pattern reshapes a
    # symbol's historical edge in _posture_adjusted_edge: conservative leans
    # on consistency (penalizing variance and bad-news days harder, ignoring
    # WSB hype), risky leans on raw edge (barely discounting variance or
    # negative news, and leaning into WSB-bullish momentum). These never
    # change PORTFOLIO_MIN_EXPECTED_PROFIT_PERCENT itself, only which
    # already-qualifying candidate looks best and which holding looks
    # weakest.
    _POSTURE_VARIANCE_PENALTY = {"conservative": 0.6, "risky": 0.15}
    _POSTURE_CONSISTENCY_WEIGHT = {"conservative": 1.0, "risky": 0.25}
    _POSTURE_NEWS_DISCOUNT_PER_POINT = {"conservative": 0.15, "risky": 0.05}
    _POSTURE_WSB_BULLISH_BONUS = {"conservative": 0.0, "risky": 0.25}
    _POSTURE_WSB_BEARISH_PENALTY = {"conservative": 0.20, "risky": 0.05}
    _POSTURE_MAX_ADJUSTMENT_PERCENT = 3.0

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

    def initialize(self) -> None:
        """Configure one evaluation per trading day."""
        self.sleeptime = "1D"
        self._rotation_lock = threading.Lock()
        self._portfolio_state_lock = threading.RLock()
        self._symbol_reference_refresh_lock = threading.Lock()
        self._symbol_reference_pending_symbols: set[str] = set()
        self._symbol_reference_refresh_running = False
        self.vars.pending_rotation = self._load_pending_rotation()
        # Historical bars are fetched during the first evaluation, after the
        # broker has supplied current market data.
        self.vars.decision_memory_backfill_attempted = False
        self.vars.portfolio_pending_rotation = self._load_portfolio_rotation()
        self.vars.portfolio_holding_dates = self._load_portfolio_holding_dates()
        if self.vars.pending_rotation and bool(self.parameters.get("portfolio_enabled", False)):
            # The A/B rotation flag is meaningless in portfolio mode and the
            # portfolio branch never reconciles it; clear it so a later switch
            # back to A/B mode starts from a truthful state.
            self._set_pending_rotation(False)
            self.log_message(
                "Cleared a stale A/B rotation flag; portfolio mode is active.",
                color="yellow",
            )
        if self.vars.pending_rotation:
            self.log_message(
                "Restored an in-progress rotation from disk; it will be "
                "reconciled on the next trading iteration.",
                color="yellow",
            )
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
        self._refresh_wsb_snapshot_before_trading()

    def _rotation_state_path(self) -> Path | None:
        raw = self.parameters.get("rotation_state_file")
        return Path(str(raw)) if raw else None

    def _load_pending_rotation(self) -> bool:
        """Restore the in-progress rotation flag after a restart."""
        path = self._rotation_state_path()
        if path is None or not path.exists():
            return False
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            return bool(state.get("pending_rotation", False))
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    def _set_pending_rotation(self, value: bool) -> bool:
        """Update the rotation flag in memory and persist it atomically."""
        previous = bool(getattr(self.vars, "pending_rotation", False))
        self.vars.pending_rotation = bool(value)
        path = self._rotation_state_path()
        if path is None:
            return True
        try:
            temporary_path = path.with_suffix(path.suffix + ".tmp")
            temporary_path.write_text(
                json.dumps({"pending_rotation": bool(value)}) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(path)
        except OSError as exc:
            self.vars.pending_rotation = previous
            self.log_message(f"Could not persist rotation state: {exc}", color="red")
            return False
        return True

    def _portfolio_rotation_state_path(self) -> Path | None:
        raw = self.parameters.get("portfolio_rotation_state_file")
        return Path(str(raw)) if raw else None

    def _portfolio_state_guard(self) -> threading.RLock:
        """Return the shared lock protecting callback/iteration state swaps."""
        lock = getattr(self, "_portfolio_state_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._portfolio_state_lock = lock
        return lock

    def _load_portfolio_rotation(self) -> dict[str, dict[str, Any]]:
        """Restore every staged portfolio rotation after a restart.

        Keyed by source ("from") symbol so a sell-fill callback can look its
        entry up in O(1). Transparently migrates the old single-record shape
        ({"from", "to", "budget"}) written by earlier versions into the new
        keyed shape, defaulting its kind to "replacement" since that format
        couldn't distinguish an Opportunistic Opportunity swap.
        """
        path = self._portfolio_rotation_state_path()
        if path is None or not path.exists():
            return {}
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
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
        """Persist the whole rotation collection atomically (whole-file swap).

        Callers always pass a freshly rebuilt dict rather than mutating the
        live one in place, matching the same discipline already used for
        portfolio_holding_dates, so a concurrent read from the broker
        callback thread never observes a partially updated collection.
        """
        with self._portfolio_state_guard():
            previous = self.vars.portfolio_pending_rotation
            self.vars.portfolio_pending_rotation = state
            path = self._portfolio_rotation_state_path()
            if path is None:
                return True
            try:
                if not state:
                    path.unlink(missing_ok=True)
                    return True
                temporary_path = path.with_suffix(path.suffix + ".tmp")
                temporary_path.write_text(json.dumps(state) + "\n", encoding="utf-8")
                temporary_path.replace(path)
            except OSError as exc:
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
        path = self._portfolio_holding_state_path()
        if path is None or not path.exists():
            return {}
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                return {}
            return {
                str(symbol).upper(): value
                for symbol, value in state.items()
                if isinstance(value, str)
                and str(symbol).strip()
                and self._valid_iso_date(value)
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
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
            path = self._portfolio_holding_state_path()
            if path is None:
                return
            try:
                temporary_path = path.with_suffix(path.suffix + ".tmp")
                temporary_path.write_text(
                    json.dumps(dates, sort_keys=True) + "\n", encoding="utf-8"
                )
                temporary_path.replace(path)
            except OSError as exc:
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

    @staticmethod
    def _holding_is_due(entry_date: str, today: date_type, maximum_days: int) -> bool:
        """Return whether a confirmed entry has reached its configured horizon."""
        try:
            return today - date_type.fromisoformat(entry_date) >= timedelta(days=maximum_days)
        except ValueError:
            return False

    def _has_active_order(self, symbol: str, side: str) -> bool:
        """Best-effort check for a working order; unknown states count as active."""
        try:
            orders = self.get_orders() or []
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
            state_file = Path(str(self.parameters["email_state_file"]))
            if state_file.exists() and state_file.read_text(encoding="utf-8").strip() == report_date:
                return

            message = EmailMessage()
            message["Subject"] = (
                f"Trading Agent Daily Report - {report_date} - {report['status']}"
            )
            message["From"] = str(self.parameters["email_from_address"])
            message["To"] = str(self.parameters["email_to_address"])
            portfolio_mode = report.get("portfolio_mode") == "enabled"
            lines = [
                "Raspberry Pi Trading Agent Daily Summary",
                "",
                f"Date: {report_date}",
                f"Evaluation time: {self.get_datetime().isoformat()}",
            ]
            if portfolio_mode:
                lines += [
                    "Mode: portfolio",
                    f"Risk posture: {report.get('portfolio_risk_posture', 'unavailable')}",
                    f"Holdings: {report.get('portfolio_holdings', 'unavailable')}",
                    f"Signal candidates: {report.get('portfolio_candidates', 'unavailable')}",
                    f"Discovered symbols: {report.get('discovered_symbols', 'none')}",
                    f"WSB discovery symbols: {report.get('wsb_discovered_symbols', 'none')}",
                    f"Discovery status: {report.get('discovery_status', 'ok')}",
                    f"Dip threshold: {report['threshold']:.2f}%",
                ]
            else:
                lines += [
                    f"Asset A: {report['asset_a']}",
                    f"Asset B: {report['asset_b']}",
                    f"Asset A price: {report.get('price_a', 'unavailable')}",
                    f"Asset B price: {report.get('price_b', 'unavailable')}",
                    f"Asset A quantity: {report.get('quantity_a', 'unavailable')}",
                    f"Asset B quantity: {report.get('quantity_b', 'unavailable')}",
                    f"Recent high: {report.get('recent_high', 'unavailable')}",
                    f"Calculated dip: {report.get('dip_percent', 'unavailable')}",
                    f"Dip threshold: {report['threshold']:.2f}%",
                ]
            lines += [
                f"News risk level: {report.get('news_risk_level', 'unavailable')}",
                f"News score: {report.get('news_score', 'unavailable')}",
                f"News articles checked: {report.get('news_article_count', 'unavailable')}",
                f"News explanation: {report.get('news_explanation', 'unavailable')}",
                f"LLM risk level: {report.get('llm_risk_level', 'unavailable')}",
                f"LLM score: {report.get('llm_score', 'unavailable')}",
                f"LLM reasoning: {report.get('llm_reasoning', 'unavailable')}",
                f"Learning observations: {report.get('learning_observations', 'unavailable')}",
                f"Learned return forecast: {report.get('learned_forecast', 'not ready')}",
                f"Learning explanation: {report.get('learning_explanation', 'unavailable')}",
                f"Congressional-trading context: {report.get('congress_explanation', 'unavailable')}",
                f"WallStreetBets context: {report.get('wsb_explanation', 'unavailable')}",
                f"Opportunistic Opportunity: {report.get('opportunistic_opportunity_status', 'unavailable')}",
                f"Opportunistic Opportunity probability: {report.get('opportunistic_opportunity_probability', 'unavailable')}",
                f"Opportunistic Opportunity evidence: {report.get('opportunistic_opportunity_explanation', 'unavailable')}",
            ]
            if not portfolio_mode:
                # Decision memory models the A/B rotation edge specifically.
                lines += [
                    f"Decision-memory observations: {report.get('decision_memory_observations', 'unavailable')}",
                    f"Predicted rotation edge: {report.get('rotation_edge_forecast', 'not ready')}",
                    f"Decision-memory explanation: {report.get('decision_memory_explanation', 'unavailable')}",
                ]
            if portfolio_mode:
                lines += [
                    "Portfolio actions this iteration:",
                    *[f"- {action}" for action in report.get("portfolio_actions", [])],
                ]
            lines += [
                "Notable scored headlines:",
                *[f"- {headline}" for headline in report.get("news_headlines", [])],
                "Congressional-trading highlights:",
                *[f"- {highlight}" for highlight in report.get("congress_highlights", [])],
                "WallStreetBets highlights:",
                *[f"- {highlight}" for highlight in report.get("wsb_highlights", [])],
                f"Result: {report['status']}",
                "",
                "Review all orders and positions in the Alpaca dashboard.",
                "This automated message is not financial advice.",
            ]
            message.set_content("\n".join(lines))
            message.add_alternative(
                self._render_email_html(report, report_date, portfolio_mode),
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

            state_file.write_text(report_date + "\n", encoding="utf-8")
            self.log_message(f"Daily email report sent for {report_date}.", color="green")
        except Exception as exc:
            self.log_message(
                f"Daily email report failed safely: {type(exc).__name__}: {exc}",
                color="red",
            )

    @staticmethod
    def _email_status_theme(status: str) -> tuple[str, str]:
        """Map a free-text status line to a (background, text) banner color."""
        lowered = status.lower()
        if "block" in lowered or "error" in lowered or "failed" in lowered:
            return "#fdecea", "#b3261e"
        if "pending" in lowered or "waiting" in lowered:
            return "#fff4e5", "#8a5300"
        if any(term in lowered for term in ("submitted", "complete", "filled", "finished", "top-up", "build")):
            return "#e6f4ea", "#1e7e34"
        return "#eceff1", "#455a64"

    @staticmethod
    def _email_value(value: Any, *, money: bool = False) -> str:
        if value is None:
            return "unavailable"
        if money and isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"${float(value):,.2f}"
        return str(value)

    @classmethod
    def _email_kv_section(cls, title: str, rows: list[tuple[str, Any]]) -> str:
        """Render a titled two-column table of label/value rows."""
        body_rows = []
        for index, (label, value) in enumerate(rows):
            shade = "#ffffff" if index % 2 == 0 else "#f8f9fb"
            body_rows.append(
                '<tr style="background-color:{shade};">'
                '<td style="padding:8px 12px;font-size:13px;color:#5f6368;width:44%;'
                'border-bottom:1px solid #eceff1;vertical-align:top;">{label}</td>'
                '<td style="padding:8px 12px;font-size:13px;color:#1a1a2e;font-weight:500;'
                'border-bottom:1px solid #eceff1;vertical-align:top;">{value}</td>'
                "</tr>".format(
                    shade=shade,
                    label=html.escape(label),
                    value=html.escape(cls._email_value(value)),
                )
            )
        return (
            '<div style="font-size:12px;font-weight:700;color:#8a8f98;'
            'text-transform:uppercase;letter-spacing:0.05em;margin:20px 0 6px;">'
            f"{html.escape(title)}</div>"
            '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
            'style="border-collapse:collapse;">' + "".join(body_rows) + "</table>"
        )

    @classmethod
    def _email_bullet_section(cls, title: str, items: list[str]) -> str:
        """Render a titled bullet list, or a muted placeholder when empty."""
        heading = (
            '<div style="font-size:12px;font-weight:700;color:#8a8f98;'
            'text-transform:uppercase;letter-spacing:0.05em;margin:20px 0 6px;">'
            f"{html.escape(title)}</div>"
        )
        if not items:
            return heading + (
                '<div style="font-size:13px;color:#8a8f98;font-style:italic;">'
                "None reported</div>"
            )
        list_items = "".join(
            f'<li style="margin-bottom:4px;">{html.escape(str(item))}</li>' for item in items
        )
        return heading + (
            '<ul style="margin:0;padding-left:18px;color:#333333;font-size:13px;line-height:1.6;">'
            + list_items
            + "</ul>"
        )

    def _render_email_html(
        self, report: dict[str, Any], report_date: str, portfolio_mode: bool
    ) -> str:
        """Build a styled HTML alternative body mirroring the plain-text report."""
        status = str(report["status"])
        badge_bg, badge_fg = self._email_status_theme(status)
        mode_label = "Portfolio mode" if portfolio_mode else "Asset rotation mode"

        if portfolio_mode:
            snapshot_rows = [
                ("Risk posture", report.get("portfolio_risk_posture", "unavailable")),
                ("Holdings", report.get("portfolio_holdings", "unavailable")),
                ("Signal candidates", report.get("portfolio_candidates", "unavailable")),
                ("Discovered symbols", report.get("discovered_symbols", "none")),
                ("WSB discovery symbols", report.get("wsb_discovered_symbols", "none")),
                ("Discovery status", report.get("discovery_status", "ok")),
                ("Dip threshold", f"{report['threshold']:.2f}%"),
            ]
        else:
            snapshot_rows = [
                ("Asset A", report.get("asset_a", "unavailable")),
                ("Asset B", report.get("asset_b", "unavailable")),
                ("Asset A price", self._email_value(report.get("price_a"), money=True)),
                ("Asset B price", self._email_value(report.get("price_b"), money=True)),
                ("Asset A quantity", report.get("quantity_a", "unavailable")),
                ("Asset B quantity", report.get("quantity_b", "unavailable")),
                ("Recent high", report.get("recent_high", "unavailable")),
                ("Calculated dip", report.get("dip_percent", "unavailable")),
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
            ("Congressional-trading context", report.get("congress_explanation", "unavailable")),
            ("WallStreetBets context", report.get("wsb_explanation", "unavailable")),
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
        if not portfolio_mode:
            forecast_rows[3:3] = [
                (
                    "Decision-memory observations",
                    report.get("decision_memory_observations", "unavailable"),
                ),
                ("Predicted rotation edge", report.get("rotation_edge_forecast", "not ready")),
                (
                    "Decision-memory explanation",
                    report.get("decision_memory_explanation", "unavailable"),
                ),
            ]

        sections = "".join(
            [
                self._email_kv_section("Snapshot", snapshot_rows),
                *(
                    [self._email_bullet_section("Portfolio actions", report.get("portfolio_actions", []))]
                    if portfolio_mode
                    else []
                ),
                self._email_kv_section("News & Risk Signals", signal_rows),
                self._email_kv_section("Learning & Forecasts", forecast_rows),
                self._email_bullet_section(
                    "Notable scored headlines", report.get("news_headlines", [])
                ),
                self._email_bullet_section(
                    "Congressional-trading highlights", report.get("congress_highlights", [])
                ),
                self._email_bullet_section(
                    "WallStreetBets highlights", report.get("wsb_highlights", [])
                ),
            ]
        )

        return f"""\
<!doctype html>
<html>
<body style="margin:0;padding:0;background-color:#f2f4f6;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f2f4f6;">
<tr><td align="center" style="padding:24px 12px;">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;background-color:#ffffff;border-radius:8px;overflow:hidden;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
<tr><td style="background-color:#1a1a2e;padding:20px 24px;">
<div style="color:#ffffff;font-size:18px;font-weight:600;">Raspberry Pi Trading Agent</div>
<div style="color:#b8bcc8;font-size:13px;margin-top:4px;">Daily Summary &middot; {html.escape(report_date)} &middot; {html.escape(mode_label)}</div>
</td></tr>
<tr><td style="padding:20px 24px 0;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:{badge_bg};border-radius:6px;">
<tr><td style="padding:14px 16px;color:{badge_fg};font-size:14px;font-weight:600;">{html.escape(status)}</td></tr>
</table>
</td></tr>
<tr><td style="padding:0 24px 8px;">
{sections}
</td></tr>
<tr><td style="padding:16px 24px 20px;color:#8a8f98;font-size:12px;line-height:1.5;border-top:1px solid #eceff1;">
Review all orders and positions in the Alpaca dashboard.<br>
This automated message is not financial advice.
</td></tr>
</table>
</td></tr>
</table>
</body>
</html>
"""

    def _get_news_context(self) -> NewsContext:
        """Return recent headline context, failing open on data problems."""
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
                f"News context unavailable; price strategy will continue: "
                f"{type(exc).__name__}: {exc}",
                color="red",
            )
            return NewsContext(
                available=False,
                risk_level="unavailable",
                explanation=f"News retrieval failed: {type(exc).__name__}: {exc}",
            )

    def _get_congress_context(self, symbols: list[str]) -> CongressContext:
        """Return delayed public-disclosure context without affecting orders."""
        if not bool(self.parameters.get("congress_context_enabled", False)):
            return CongressContext(
                available=False,
                explanation="Congressional-trading context is disabled in config.json.",
            )
        context = CongressTradeAnalyzer(
            timeout_seconds=float(self.parameters.get("congress_context_timeout_seconds", 10.0))
        ).analyze(symbols)
        self.log_message(context.explanation, color="blue")
        for highlight in context.highlights:
            self.log_message(f"Congressional-trading evidence: {highlight}", color="blue")
        return context

    def _get_wsb_context(self, symbols: list[str]) -> WSBContext:
        """Return public WSB context, failing open when the tracker is unavailable."""
        if not (
            bool(self.parameters.get("wsb_context_enabled", False))
            or bool(self.parameters.get("wsb_discovery_enabled", False))
        ):
            return WSBContext(
                available=False,
                explanation="WallStreetBets context and discovery are disabled in config.json.",
            )
        context = self._wsb_snapshot().context(symbols)
        self.log_message(context.explanation, color="blue")
        for highlight in context.highlights:
            self.log_message(f"WallStreetBets evidence: {highlight}", color="blue")
        return context

    def _wsb_snapshot(self) -> WallStreetBetsSnapshot:
        return WallStreetBetsSnapshot(
            Path(str(self.parameters["wsb_context_state_file"])),
            WallStreetBetsAnalyzer(
                timeout_seconds=float(self.parameters.get("wsb_context_timeout_seconds", 10.0))
            ),
        )

    def _refresh_wsb_snapshot_before_trading(self) -> None:
        """Refresh the single WSB snapshot before the day's trade evaluations."""
        if not (
            bool(self.parameters.get("wsb_context_enabled", False))
            or bool(self.parameters.get("wsb_discovery_enabled", False))
        ):
            return
        try:
            refreshed = self._wsb_snapshot().refresh_if_due()
            self.log_message(
                "WallStreetBets snapshot refreshed before trading."
                if refreshed
                else "WallStreetBets snapshot is within its 24-hour refresh window.",
                color="blue",
            )
        except Exception as exc:
            self.log_message(
                f"WallStreetBets pre-trade refresh failed safely: {type(exc).__name__}: {exc}",
                color="yellow",
            )

    def _get_llm_news_assessment(self, news_context: NewsContext) -> LLMNewsAssessment:
        """Ask Claude to assess today's headlines, failing open on problems."""
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
                provider=str(self.parameters["llm_news_provider"]),
                model=str(self.parameters["llm_news_model"]),
                base_url=str(self.parameters.get("llm_news_base_url", "")),
            )
            assessment = analyzer.assess(news_context.articles)
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

    def _update_adaptive_learning(
        self,
        price_b: float,
        news_context: NewsContext,
    ) -> LearningResult:
        """Update the persistent model and return its explainable forecast."""
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
                state_path=Path(str(self.parameters["news_learning_state_file"])),
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
                news_score=news_context.score if news_context.available else None,
            )
            self.log_message(result.explanation, color="blue")
            return result
        except Exception as exc:
            self.log_message(
                f"Adaptive news learning failed safely: {type(exc).__name__}: {exc}",
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
            memory = TradeMemory(
                database_path=Path(
                    str(self.parameters["decision_memory_database_file"])
                ),
                minimum_observations=int(
                    self.parameters["decision_memory_min_observations"]
                ),
                maximum_observations=int(
                    self.parameters["decision_memory_max_observations"]
                ),
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
            probability = TradeMemory(
                Path(str(self.parameters["decision_memory_database_file"])), 1, 1
            ).opportunity_probability()
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
            history = []
            for position, (date, close_b, high_b) in enumerate(b_rows):
                # Full windows only, excluding the event day's own high, so
                # backfilled dips match how the live path measures them.
                if position < lookback:
                    continue
                close_a = a_closes.get(date)
                if close_a is None:
                    continue
                recent_high = max(row[2] for row in b_rows[position - lookback : position])
                dip = ((recent_high - close_b) / recent_high) * 100.0
                history.append((date, close_a, close_b, dip, dip >= threshold))
            inserted = TradeMemory(
                Path(str(self.parameters["decision_memory_database_file"])),
                1,
                int(self.parameters["decision_memory_max_observations"]),
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

    def _record_memory_decision(self, report: dict[str, Any]) -> None:
        """Persist the final decision label after an observation was recorded."""
        if not report.get("decision_memory_recorded"):
            return
        try:
            TradeMemory(
                Path(str(self.parameters["decision_memory_database_file"])), 1, 1
            ).record_decision(
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
        if (
            bool(self.parameters.get("portfolio_autonomous_discovery", False))
            or bool(self.parameters.get("wsb_discovery_enabled", False))
        ):
            try:
                symbols.update(self._autonomous_universe().managed_symbols())
            except Exception as exc:
                self.log_message(
                    f"Could not read managed discovery symbols: {type(exc).__name__}: {exc}",
                    color="yellow",
                )
        for source, entry in self.vars.portfolio_pending_rotation.items():
            symbols.update((str(source).upper(), str(entry["to"]).upper()))
        return symbols

    def _portfolio_held_positions(
        self, managed_symbols: set[str]
    ) -> dict[str, Decimal] | None:
        """Return managed long stock positions, or None on broker-read failure."""
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
        return held

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
        if (
            news_context.available
            and bool(self.parameters["news_block_on_high_risk"])
            and news_context.score <= int(self.parameters["news_high_risk_score"])
        ):
            return f"Trade blocked: high world-event risk score {news_context.score}"
        if (
            llm_assessment.available
            and bool(self.parameters["llm_news_block_on_high_risk"])
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

    def _buy_asset_b_with_available_cash(self, asset_b: str, price_b: float) -> str:
        """Submit the largest Asset B buy the cash safely supports.

        Buys fractional shares when PORTFOLIO_FRACTIONAL_SHARES is enabled
        (the default), otherwise whole shares. Returns "submitted" when a new
        order was placed, "working" when a buy order is already in flight,
        "rejected" when the broker synchronously refuses it, or "insufficient"
        when the spendable cash is below the minimum order (or below one whole
        share in whole-share mode).
        """
        with self._rotation_lock:
            if self._has_active_order(asset_b, "buy"):
                self.log_message(
                    f"A buy order for {asset_b} is already working; "
                    "not submitting another.",
                    color="yellow",
                )
                return "working"

            cash = float(self.get_cash())
            spendable = cash * (1.0 - self.CASH_BUFFER_FRACTION) - float(
                self.parameters.get("portfolio_cash_reserve_dollars", 0.0)
            )
            if spendable < float(self.parameters.get("portfolio_min_order_dollars", 1.0)):
                self.log_message(
                    f"No purchase submitted: spendable cash ${spendable:.2f} "
                    "is below the configured minimum order amount.",
                    color="yellow",
                )
                return "insufficient"
            if bool(self.parameters.get("fractional_shares", False)):
                quantity: Decimal | int = (Decimal(str(spendable)) / Decimal(str(price_b))).quantize(
                    Decimal("1.000000000"), rounding=ROUND_DOWN
                )
            else:
                quantity = math.floor(spendable / price_b)
            if quantity <= 0:
                return "insufficient"

            buy_order = self.create_order(
                asset_b,
                quantity=quantity,
                side="buy",
                order_type="market",
                time_in_force="day",
            )
            if not self._submit_order_checked(buy_order, f"{asset_b} buy"):
                return "rejected"
            self.log_message(
                f"Submitted market buy for {quantity} shares of {asset_b}, the "
                f"largest supported quantity from ${cash:.2f} cash "
                f"after a {self.CASH_BUFFER_FRACTION:.0%} safety buffer.",
                color="green",
            )
            return "submitted"

    def _buy_portfolio_symbol(self, symbol: str, price: float, budget: float) -> str:
        """Buy a whole or fractional quantity within a stated portfolio budget."""
        with self._rotation_lock:
            if self._has_active_order(symbol, "buy"):
                return "working"
            spendable = min(float(self.get_cash()), budget) * (1.0 - self.CASH_BUFFER_FRACTION)
            spendable -= float(self.parameters.get("portfolio_cash_reserve_dollars", 0.0))
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

    @staticmethod
    def _walk_forward_net_returns(
        returns: list[float],
        round_trip_cost_percent: float,
        minimum_observations: int,
        entry_threshold_percent: float,
    ) -> list[float]:
        """Evaluate only trades selected from information available beforehand.

        Each validation result uses a historical mean formed strictly before
        that event.  This prevents a candidate's realised return from helping
        select itself, unlike an in-sample average.
        """
        outcomes: list[float] = []
        for index in range(minimum_observations, len(returns)):
            prior_net_mean = (
                sum(value - round_trip_cost_percent for value in returns[:index])
                / index
            )
            if prior_net_mean >= entry_threshold_percent:
                outcomes.append(returns[index] - round_trip_cost_percent)
        return outcomes

    def _portfolio_signal(
        self, symbol: str
    ) -> dict[str, float | int | str | None] | None:
        """Estimate next-session return from this symbol's prior comparable dips.

        This is a historical average, not a prediction or a promised profit.
        Requiring prior observations prevents a freshly listed symbol from being
        selected purely because it happened to dip today.
        """
        bars = self.get_historical_prices(
            symbol, int(self.parameters["portfolio_analysis_days"]), "day"
        )
        if bars is None or bars.df is None or bars.df.empty or not {"high", "close"}.issubset(bars.df.columns):
            return None
        rows = [
            (float(row["high"]), float(row["close"]))
            for _, row in bars.df[["high", "close"]].dropna().iterrows()
            if math.isfinite(float(row["high"])) and math.isfinite(float(row["close"]))
            and float(row["high"]) > 0 and float(row["close"]) > 0
        ]
        lookback = int(self.parameters["recent_high_lookback_days"])
        if len(rows) <= lookback:
            return None
        price = self.get_last_price(symbol)
        if price is None or not math.isfinite(float(price)) or float(price) <= 0:
            return None
        threshold = float(self.parameters["dip_threshold_percent"])
        returns: list[float] = []
        # Historical dips are measured against the *previous* lookback bars,
        # excluding the event day's own high, to match the live check below
        # (which compares today's price against already-completed bars).
        for index in range(lookback, len(rows) - 1):
            recent_high = max(high for high, _ in rows[index - lookback : index])
            dip = ((recent_high - rows[index][1]) / recent_high) * 100.0
            if dip >= threshold:
                returns.append(((rows[index + 1][1] - rows[index][1]) / rows[index][1]) * 100.0)
        recent_high = max(high for high, _ in rows[-lookback:])
        current_dip = ((recent_high - float(price)) / recent_high) * 100.0
        if current_dip < threshold or not returns:
            return None
        round_trip_cost = float(
            self.parameters.get("portfolio_round_trip_cost_percent", 0.20)
        )
        net_returns = [value - round_trip_cost for value in returns]
        walk_forward_returns = self._walk_forward_net_returns(
            returns,
            round_trip_cost,
            int(self.parameters.get("portfolio_oos_min_observations", 10)),
            float(self.parameters["portfolio_min_expected_profit_percent"]),
        )
        mean_net_return = sum(net_returns) / len(net_returns)
        variance = sum((value - mean_net_return) ** 2 for value in net_returns) / len(net_returns)
        wins = sum(1 for value in net_returns if value > 0)
        return {
            "symbol": symbol,
            "price": float(price),
            "dip": current_dip,
            # This net historical mean is a coarse current estimate. It is
            # never enough by itself: _run_portfolio_iteration also requires
            # the chronological walk-forward result below.
            "expected_profit": mean_net_return,
            "observations": len(returns),
            "oos_expected_profit": (
                sum(walk_forward_returns) / len(walk_forward_returns)
                if walk_forward_returns
                else None
            ),
            "oos_observations": len(walk_forward_returns),
            # Feed the risky/conservative reasoning pattern in
            # _posture_adjusted_edge: how spread out this symbol's past
            # dip-signal outcomes were, and a Laplace-smoothed win rate
            # (matches TradeMemory.opportunity_probability's convention).
            "return_stdev": math.sqrt(variance),
            "win_probability": (wins + 1) / (len(net_returns) + 2),
        }

    def _portfolio_signals(
        self, symbols: list[str]
    ) -> list[dict[str, float | int | str | None] | None]:
        """Fetch independent symbol histories through a small bounded pool."""
        if not symbols:
            return []
        with ThreadPoolExecutor(
            max_workers=min(self._PORTFOLIO_HISTORY_WORKERS, len(symbols)),
            thread_name_prefix="portfolio-history",
        ) as executor:
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
            for article in news_context.per_article:
                tagged = {str(symbol) for symbol in article.get("symbols", [])}
                untagged_candidates = candidates - tagged
                if not untagged_candidates:
                    continue
                text = f"{article.get('headline', '')} {article.get('summary', '')}"
                for symbol in reference.scan_text_for_symbols(text, untagged_candidates):
                    scores[symbol] = scores.get(symbol, 0) + int(article.get("score", 0))
            return scores
        except Exception as exc:
            self.log_message(
                f"Symbol-aware news scoring failed safely: {type(exc).__name__}: {exc}",
                color="yellow",
            )
            return dict(news_context.per_symbol_scores)

    @staticmethod
    def _posture_adjusted_edge(
        signal: dict[str, float | int | str | None],
        posture: str,
        news_score: float | int | None,
        wsb_sentiment: str | None,
    ) -> float:
        """Reshape a symbol's historical edge through a risky or conservative lens.

        Conservative leans on consistency: it penalizes return variance and a
        negative news day harder, and ignores WSB mentions as noise. Risky
        leans on raw edge: it barely discounts variance or bad news, and
        leans into WSB-bullish momentum while shrugging off bearish chatter.
        This never changes PORTFOLIO_MIN_EXPECTED_PROFIT_PERCENT itself; it
        only reweights which already-qualifying candidate looks best and
        which current holding looks weakest.
        """
        posture = posture if posture in ("conservative", "risky") else "conservative"
        expected_profit = float(signal["expected_profit"])
        stdev = float(signal.get("return_stdev") or 0.0)
        win_probability = float(signal.get("win_probability") or 0.5)
        adjustment = -AssetRotationStrategy._POSTURE_VARIANCE_PENALTY[posture] * stdev
        adjustment += (
            (win_probability - 0.5) * 2.0 * AssetRotationStrategy._POSTURE_CONSISTENCY_WEIGHT[posture]
        )
        if news_score is not None:
            capped_score = max(-10.0, min(10.0, float(news_score)))
            adjustment -= max(0.0, -capped_score) * AssetRotationStrategy._POSTURE_NEWS_DISCOUNT_PER_POINT[posture]
        if wsb_sentiment == "bullish":
            adjustment += AssetRotationStrategy._POSTURE_WSB_BULLISH_BONUS[posture]
        elif wsb_sentiment == "bearish":
            adjustment -= AssetRotationStrategy._POSTURE_WSB_BEARISH_PENALTY[posture]
        max_adjustment = AssetRotationStrategy._POSTURE_MAX_ADJUSTMENT_PERCENT
        adjustment = max(-max_adjustment, min(max_adjustment, adjustment))
        return expected_profit + adjustment

    def _autonomous_universe(self) -> AutonomousUniverse:
        return AutonomousUniverse(
            Path(str(self.parameters["portfolio_universe_state_file"])),
            int(self.parameters["portfolio_discovery_refresh_days"]),
            int(self.parameters["portfolio_discovery_batch_size"]),
            paper=os.environ.get("ALPACA_IS_PAPER", "true").strip().lower() != "false",
        )

    def _symbol_reference(self) -> SymbolReference:
        return SymbolReference(
            Path(str(self.parameters["symbol_reference_database_file"])),
            int(self.parameters["symbol_reference_refresh_days"]),
            paper=os.environ.get("ALPACA_IS_PAPER", "true").strip().lower() != "false",
        )

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

    def _remember_discovered_symbols(self, symbols: list[str]) -> None:
        if not (
            bool(self.parameters.get("portfolio_autonomous_discovery", False))
            or bool(self.parameters.get("wsb_discovery_enabled", False))
        ):
            return
        try:
            self._autonomous_universe().remember(symbols)
        except Exception as exc:
            self.log_message(f"Could not persist learned symbols: {type(exc).__name__}: {exc}", color="yellow")

    def _run_portfolio_iteration(self, report: dict[str, Any]) -> None:
        """Build or rotate a bounded portfolio from the explicit symbol list."""
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
        held = self._portfolio_held_positions(managed_symbols)
        if held is None:
            report["status"] = "No portfolio trade: account positions were unavailable"
            return
        report["portfolio_holdings"] = (
            ", ".join(f"{symbol}={quantity}" for symbol, quantity in sorted(held.items())) or "none"
        )
        symbols = self._portfolio_symbols(report, held, managed_symbols)
        wsb_context = self._get_wsb_context(symbols)
        if bool(self.parameters.get("wsb_discovery_enabled", False)) and wsb_context.available:
            wsb_symbols = [
                item.symbol
                for item in wsb_context.mentions[: int(self.parameters["wsb_discovery_max_symbols"])]
            ]
            symbols = list(dict.fromkeys(symbols + wsb_symbols))
            report["wsb_discovered_symbols"] = ", ".join(wsb_symbols) or "none"
        report.update(
            wsb_explanation=wsb_context.explanation,
            wsb_highlights=wsb_context.highlights,
        )
        self._refresh_symbol_reference(symbols)
        congress_context = self._get_congress_context(symbols)
        report.update(
            congress_symbols_matched=(
                congress_context.matched_symbols if congress_context.available else "unavailable"
            ),
            congress_explanation=congress_context.explanation,
            congress_highlights=congress_context.highlights,
        )

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
        llm_assessment = self._get_llm_news_assessment(news_context)
        report.update(
            llm_risk_level=llm_assessment.risk_level,
            llm_score=llm_assessment.score if llm_assessment.available else "unavailable",
            llm_reasoning=(
                llm_assessment.reasoning
                if llm_assessment.available
                else llm_assessment.explanation
            ),
        )

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
        veto_reason = self._market_veto_reason(news_context, llm_assessment, learning_result)

        # A/B is no longer an alternate strategy mode. It is a separately
        # labelled opportunity inside portfolio mode, trained only on the
        # settled A-versus-B observations already kept in decision memory.
        asset_a = str(self.parameters["asset_a"]).upper()
        asset_b = str(self.parameters["asset_b"]).upper()
        opportunity = self._opportunistic_opportunity(
            asset_a, asset_b, self.get_last_price(asset_a), self.get_last_price(asset_b), news_context
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
        actions: list[str] = []
        held_working = dict(held)

        # Phase 0: reconcile every existing pending rotation first. Completing
        # an in-flight rotation is never vetoed: the sale already happened and
        # leaving the proceeds in cash is its own risk. Iterate a snapshot so
        # removals made mid-loop don't disturb iteration.
        for source in sorted(self.vars.portfolio_pending_rotation):
            entry = self.vars.portfolio_pending_rotation.get(source)
            if entry is None:
                continue
            target, budget, kind = str(entry["to"]), float(entry["budget"]), str(entry["kind"])
            if held.get(source, Decimal("0")) > 0:
                if self._has_active_order(source, "sell"):
                    actions.append(f"Portfolio pending: waiting for {source} sale")
                    continue
                self._remove_portfolio_rotation(source)
                actions.append(f"Portfolio rotation reset: {source} sale did not fill")
                continue
            if held.get(target, Decimal("0")) > 0 and not self._has_active_order(target, "buy"):
                # The buy filled but the fill callback was lost (restart).
                self._remove_portfolio_rotation(source)
                actions.append(f"Portfolio rotation complete ({kind}): the {target} purchase filled")
                continue
            price = self.get_last_price(target)
            if price is None or not math.isfinite(float(price)) or float(price) <= 0:
                actions.append(f"Portfolio pending: no valid {target} price")
                continue
            outcome = self._buy_portfolio_symbol(target, float(price), budget)
            if outcome == "insufficient":
                # Balances are confirmed by now; the rotation ends here.
                self._remove_portfolio_rotation(source)
                actions.append(f"Portfolio rotation finished: cash is below the minimum {target} order")
            elif outcome == "working":
                actions.append(f"Portfolio pending: waiting for the {target} purchase to fill")
            elif outcome == "rejected":
                actions.append(
                    f"Portfolio pending: broker rejected the {target} purchase; retrying next cycle"
                )
            else:
                # The entry clears when the buy fills (on_filled_order), never
                # on submission, so a rejected order is retried next cycle.
                actions.append(f"Portfolio {target} purchase submitted after {source} sale")
        claimed_symbols: set[str] = set(self.vars.portfolio_pending_rotation.keys()) | {
            str(entry["to"]) for entry in self.vars.portfolio_pending_rotation.values()
        }

        # Phase 1: exit every holding that reached its holding horizon, not
        # just the first one. This is a plain single-leg sell -- no paired
        # buy, no rotation-slot usage -- and is never vetoed, exactly like
        # completing a pending rotation above.
        #
        # The portfolio signal is validated against next-session returns. Keep
        # actual holding duration aligned with that measured horizon, including
        # after restarts. A configured managed holding with no fill record is
        # conservatively dated today on first observation rather than sold
        # immediately.
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
        maximum_holding_days = int(self.parameters.get("portfolio_max_holding_days", 1))
        due_symbols = sorted(
            symbol
            for symbol in held
            if self._holding_is_due(holding_dates[symbol], today, maximum_holding_days)
        )
        for source in due_symbols:
            if source in claimed_symbols:
                continue
            if self._has_active_order(source, "sell"):
                actions.append(f"Portfolio exit pending: waiting for {source} sale")
                continue
            exit_order = self.create_order(
                source,
                quantity=held[source],
                side="sell",
                order_type="market",
                time_in_force="day",
            )
            if not self._submit_order_checked(exit_order, f"{source} holding-horizon sell"):
                actions.append(f"Portfolio exit rejected: {source} sale was not accepted")
                continue
            actions.append(
                f"Portfolio exit submitted: {source} reached its "
                f"{maximum_holding_days}-day holding horizon"
            )
            held_working.pop(source, None)
            claimed_symbols.add(source)

        # Alpaca requests for separate symbols are independent. A small fixed
        # pool removes the observed serial latency without creating an
        # unbounded burst against the broker API.
        signals = self._portfolio_signals(symbols)
        signals = [signal for signal in signals if signal is not None]
        # The risky/conservative reasoning pattern only reshapes ranking and
        # tie-breaking below; the eligibility floor two lines down still
        # gates on the raw historical expected_profit, unaffected by posture.
        risk_posture = str(self.parameters.get("portfolio_risk_posture", "conservative"))
        market_wide_news_score = news_context.score if news_context.available else None
        symbol_news_scores = self._symbol_news_scores(news_context, set(symbols))
        wsb_sentiment_by_symbol = (
            {item.symbol: item.sentiment for item in wsb_context.mentions} if wsb_context.available else {}
        )
        for signal in signals:
            symbol = str(signal["symbol"])
            # A symbol with dedicated coverage today (even a genuinely
            # neutral 0) is trusted over the market-wide score; only a
            # symbol with no coverage at all falls back to it.
            news_score = symbol_news_scores.get(symbol, market_wide_news_score)
            signal["posture_adjusted_edge"] = self._posture_adjusted_edge(
                signal, risk_posture, news_score, wsb_sentiment_by_symbol.get(symbol)
            )
        report["portfolio_risk_posture"] = risk_posture
        eligible = [
            signal
            for signal in signals
            if int(signal["observations"]) >= minimum_observations
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
        )
        # Remember holdings alongside today's qualifiers so a held symbol is
        # never trimmed out of the learned universe while it is still owned.
        self._remember_discovered_symbols(
            list(dict.fromkeys([str(signal["symbol"]) for signal in eligible] + sorted(held)))
        )
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
        # even though PORTFOLIO_SYMBOLS defaults to include both assets. Since
        # this function already runs at most once per trading day
        # (sleeptime="1D"), that structural isolation is sufficient on its own
        # to guarantee at most one Opportunistic Opportunity swap per day; no
        # separate persisted rate limit is needed.
        if opportunity_is_eligible and not veto_reason:
            if self._has_active_order(asset_a, "sell"):
                actions.append("Opportunistic Opportunity pending: waiting for Asset A sale")
            else:
                source_price = self.get_last_price(asset_a)
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
            desired = remaining_candidates[:max_positions]

            builds_submitted = 0
            for candidate in desired:
                symbol = str(candidate["symbol"])
                if symbol in held_working or symbol in claimed_symbols:
                    continue
                if len(held_working) + builds_submitted >= max_positions:
                    break
                slots_remaining = max(1, max_positions - (len(held_working) + builds_submitted))
                budget = float(self.get_cash()) / slots_remaining
                outcome = self._buy_portfolio_symbol(symbol, float(candidate["price"]), budget)
                if outcome == "insufficient":
                    # Cash is exhausted for new positions this pass; a
                    # self-funded replacement below is unaffected.
                    break
                if outcome == "rejected":
                    actions.append(f"Portfolio build rejected: {symbol} purchase was not accepted")
                    continue
                claimed_symbols.add(symbol)
                builds_submitted += 1
                actions.append(f"Portfolio build: {symbol} purchase {outcome}")

            replacements_submitted = 0
            if len(held_working) + builds_submitted >= max_positions:
                # A holding with no current dip signal is scored neutral (0%
                # expected edge), not punished: rotation happens only when the
                # target's posture-adjusted edge beats holding by the
                # configured margin. The old -100% default force-rotated any
                # recovered holding every time some other symbol dipped,
                # churning the portfolio. The posture lens only changes which
                # holding looks weakest and by how much; the
                # PORTFOLIO_MIN_EXPECTED_PROFIT_PERCENT floor is unchanged.
                held_signals = {
                    str(signal["symbol"]): signal for signal in signals if signal["symbol"] in held_working
                }
                for candidate in remaining_candidates:
                    target_symbol = str(candidate["symbol"])
                    if target_symbol in held_working or target_symbol in claimed_symbols:
                        continue
                    unclaimed_held = [symbol for symbol in held_working if symbol not in claimed_symbols]
                    if not unclaimed_held:
                        break
                    source = min(
                        unclaimed_held,
                        key=lambda symbol: float(
                            held_signals.get(symbol, {"posture_adjusted_edge": 0.0})["posture_adjusted_edge"]
                        ),
                    )
                    source_score = float(
                        held_signals.get(source, {"posture_adjusted_edge": 0.0})["posture_adjusted_edge"]
                    )
                    advantage = float(candidate["posture_adjusted_edge"]) - source_score
                    if advantage < minimum_profit:
                        continue
                    source_price = self.get_last_price(source)
                    if source_price is None or float(source_price) <= 0 or self._has_active_order(source, "sell"):
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
                        replacements_submitted += 1
                        actions.append(
                            f"Portfolio rotation submitted: {source} to {target_symbol} "
                            f"(expected advantage {advantage:+.2f}%)"
                        )

            if builds_submitted == 0 and replacements_submitted == 0:
                # A recurring small deposit should grow the highest-ranked
                # current holding instead of remaining idle once the portfolio
                # is full and every top candidate is already held.
                top_up_candidate = next(
                    (signal for signal in desired if str(signal["symbol"]) in held_working), None
                )
                cash = float(self.get_cash())
                minimum_cash = float(self.parameters.get("portfolio_cash_reserve_dollars", 0.0)) + float(
                    self.parameters.get("portfolio_min_order_dollars", 1.0)
                )
                if top_up_candidate is not None and cash >= minimum_cash:
                    symbol = str(top_up_candidate["symbol"])
                    outcome = self._buy_portfolio_symbol(symbol, float(top_up_candidate["price"]), cash)
                    actions.append(f"Portfolio top-up: {symbol} purchase {outcome}")

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

    def on_trading_iteration(self) -> None:
        """Evaluate the dip and safely advance any required portfolio rotation."""
        report = {
            "asset_a": str(self.parameters["asset_a"]).upper(),
            "asset_b": str(self.parameters["asset_b"]).upper(),
            "threshold": float(self.parameters["dip_threshold_percent"]),
            "status": "Evaluation started",
        }
        try:
            if bool(self.parameters.get("portfolio_enabled", False)):
                report["portfolio_mode"] = "enabled"
                self._run_portfolio_iteration(report)
                return
            asset_a = str(self.parameters["asset_a"]).upper()
            asset_b = str(self.parameters["asset_b"]).upper()
            threshold = float(self.parameters["dip_threshold_percent"])
            lookback = int(self.parameters["recent_high_lookback_days"])
            congress_context = self._get_congress_context([asset_a, asset_b])
            report.update(
                congress_symbols_matched=(
                    congress_context.matched_symbols if congress_context.available else "unavailable"
                ),
                congress_explanation=congress_context.explanation,
                congress_highlights=congress_context.highlights,
            )
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

            llm_assessment = self._get_llm_news_assessment(news_context)
            report.update(
                llm_risk_level=llm_assessment.risk_level,
                llm_score=(
                    llm_assessment.score if llm_assessment.available else "unavailable"
                ),
                llm_reasoning=(
                    llm_assessment.reasoning
                    if llm_assessment.available
                    else llm_assessment.explanation
                ),
            )

            price_a = self.get_last_price(asset_a)
            price_b = self.get_last_price(asset_b)
            position_a = self.get_position(asset_a)
            position_b = self.get_position(asset_b)
            quantity_a = self._quantity(position_a)
            quantity_b = self._quantity(position_b)
            report.update(
                price_a=price_a,
                price_b=price_b,
                quantity_a=str(quantity_a),
                quantity_b=str(quantity_b),
            )

            if price_a is None or price_b is None:
                report["status"] = "No trade: current price data was unavailable"
                self.log_message(
                    f"Price data unavailable for {asset_a} or {asset_b}; retrying next cycle.",
                    color="red",
                )
                return
            price_a = float(price_a)
            price_b = float(price_b)
            if not math.isfinite(price_a) or not math.isfinite(price_b) or price_a <= 0 or price_b <= 0:
                report["status"] = "No trade: invalid non-positive price received"
                self.log_message("Invalid non-positive market price received; no trade made.", color="red")
                return

            self._backfill_decision_memory(asset_a, asset_b)

            learning_result = self._update_adaptive_learning(price_b, news_context)
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

            # A previous iteration may have submitted the sale. Reconcile the
            # persisted rotation flag against live positions and open orders.
            if self.vars.pending_rotation:
                if quantity_a > 0:
                    if self._has_active_order(asset_a, "sell"):
                        report["status"] = f"Pending: waiting for the {asset_a} sale to fill"
                        self.log_message(
                            f"Waiting for the {asset_a} sale to fill before buying {asset_b}.",
                            color="yellow",
                        )
                        return
                    # The sale died without a callback (canceled, rejected, or
                    # lost across a restart). Re-evaluate the signal fresh.
                    self._set_pending_rotation(False)
                    self.log_message(
                        f"The pending {asset_a} sale is no longer working; "
                        "re-evaluating the dip signal from scratch.",
                        color="yellow",
                    )
                else:
                    outcome = self._buy_asset_b_with_available_cash(asset_b, price_b)
                    if outcome == "submitted":
                        report["status"] = (
                            f"Submitted purchase of {asset_b} after completed sale"
                        )
                    elif outcome == "working":
                        report["status"] = (
                            f"Pending: waiting for the open {asset_b} purchase to fill"
                        )
                    elif outcome == "rejected":
                        report["status"] = (
                            f"Pending: broker rejected the {asset_b} purchase; retrying next cycle"
                        )
                    else:
                        # Below the minimum purchasable amount: the rotation
                        # is complete by design.
                        self._set_pending_rotation(False)
                        report["status"] = (
                            f"Rotation finished: remaining cash is below the "
                            f"minimum {asset_b} purchase"
                        )
                    return

            bars = self.get_historical_prices(asset_b, lookback, "day")
            if bars is None or bars.df is None or bars.df.empty or "high" not in bars.df:
                report["status"] = "No trade: historical price data was unavailable"
                self.log_message(
                    f"Historical data unavailable for {asset_b}; retrying next cycle.",
                    color="red",
                )
                return

            valid_highs = bars.df["high"].dropna()
            if valid_highs.empty:
                report["status"] = "No trade: historical data contained no valid highs"
                self.log_message(f"No valid daily highs returned for {asset_b}.", color="red")
                return

            recent_high = float(valid_highs.max())
            if not math.isfinite(recent_high) or recent_high <= 0:
                report["status"] = "No trade: historical data contained an invalid high"
                self.log_message("Invalid recent high received; no trade made.", color="red")
                return
            dip_percent = ((recent_high - price_b) / recent_high) * 100.0
            report.update(
                recent_high=f"${recent_high:.2f}",
                dip_percent=f"{dip_percent:.2f}%",
            )
            self.log_message(
                f"{asset_a}=${price_a:.2f} ({quantity_a} shares), "
                f"{asset_b}=${price_b:.2f} ({quantity_b} shares), "
                f"{lookback}-day high=${recent_high:.2f}, dip={dip_percent:.2f}%.",
                color="blue",
            )

            decision_memory = self._update_decision_memory(
                price_a, price_b, dip_percent, news_context
            )
            report.update(
                decision_memory_recorded=True,
                decision_memory_observations=decision_memory.observations,
                rotation_edge_forecast=(
                    f"{decision_memory.predicted_edge_percent:+.2f}%"
                    if decision_memory.ready and decision_memory.predicted_edge_percent is not None
                    else "not ready"
                ),
                decision_memory_explanation=decision_memory.explanation,
            )

            if dip_percent < threshold:
                report["status"] = "No trade: Asset B did not meet the dip threshold"
                return
            if quantity_a <= 0:
                report["status"] = f"No trade: no long {asset_a} position was available"
                self.log_message(
                    f"{asset_b} meets the {threshold:.2f}% dip threshold, but no "
                    f"long {asset_a} position is available to rotate.",
                    color="yellow",
                )
                return

            should_block_for_news = (
                news_context.available
                and bool(self.parameters["news_block_on_high_risk"])
                and news_context.score <= int(self.parameters["news_high_risk_score"])
            )
            if should_block_for_news:
                report["status"] = (
                    f"Trade blocked: high world-event risk score {news_context.score}"
                )
                self.log_message(
                    f"Dip signal met, but rotation was blocked by the configured "
                    f"world-event risk guard (score {news_context.score}).",
                    color="red",
                )
                return

            should_block_for_llm = (
                llm_assessment.available
                and bool(self.parameters["llm_news_block_on_high_risk"])
                and llm_assessment.score <= int(self.parameters["llm_news_block_score"])
            )
            if should_block_for_llm:
                report["status"] = (
                    f"Trade blocked: LLM news assessment score "
                    f"{llm_assessment.score:+d}"
                )
                self.log_message(
                    f"Dip signal met, but rotation was blocked by the LLM news "
                    f"assessment (score {llm_assessment.score:+d}): "
                    f"{llm_assessment.reasoning}",
                    color="red",
                )
                return

            learned_risk_block = (
                bool(self.parameters["news_learning_block_enabled"])
                and learning_result.ready
                and learning_result.predicted_return_percent is not None
                and learning_result.correlation is not None
                and abs(learning_result.correlation)
                >= float(self.parameters["news_learning_min_correlation"])
                and learning_result.predicted_return_percent
                <= float(self.parameters["news_predicted_return_block_percent"])
            )
            if learned_risk_block:
                forecast = learning_result.predicted_return_percent
                report["status"] = (
                    f"Trade blocked: adaptive model forecast {forecast:+.2f}%"
                )
                self.log_message(
                    f"Dip signal met, but rotation was blocked by the mature adaptive "
                    f"news model forecast of {forecast:+.2f}% for the next session.",
                    color="red",
                )
                return

            should_block_for_decision_memory = (
                bool(self.parameters.get("decision_memory_block_enabled", False))
                and decision_memory.ready
                and decision_memory.predicted_edge_percent is not None
                and decision_memory.correlation is not None
                and abs(decision_memory.correlation)
                >= float(self.parameters["decision_memory_min_correlation"])
                and decision_memory.predicted_edge_percent
                <= float(self.parameters["decision_memory_edge_block_percent"])
            )
            if should_block_for_decision_memory:
                forecast = decision_memory.predicted_edge_percent
                report["status"] = (
                    f"Trade blocked: decision-memory edge forecast {forecast:+.2f}%"
                )
                self.log_message(
                    f"Dip signal met, but decision memory forecast that holding {asset_a} "
                    f"should outperform {asset_b} by {-forecast:+.2f}% next session.",
                    color="red",
                )
                return

            sell_order = self.create_order(
                asset_a,
                quantity=quantity_a,
                side="sell",
                order_type="market",
                time_in_force="day",
            )
            if not self._set_pending_rotation(True):
                report["status"] = (
                    f"No trade: could not persist the pending {asset_a} rotation safely"
                )
                return
            try:
                accepted = self._submit_order_checked(sell_order, f"{asset_a} rotation sell")
            except Exception:
                self._set_pending_rotation(False)
                raise
            if not accepted:
                self._set_pending_rotation(False)
                report["status"] = f"No trade: broker rejected the {asset_a} rotation sale"
                return
            report["status"] = f"Submitted market sale of {asset_a} to rotate into {asset_b}"
            self.log_message(
                f"Dip signal triggered. Submitted market sale of {quantity_a} "
                f"shares of {asset_a}; {asset_b} will be bought after the fill.",
                color="green",
            )
        except Exception as exc:
            report["status"] = f"Evaluation error: {type(exc).__name__}: {exc}"
            # Network and broker failures are logged and retried on the next
            # scheduled iteration instead of terminating the service.
            self.log_message(
                f"Trading iteration failed safely: {type(exc).__name__}: {exc}",
                color="red",
            )
        finally:
            self._record_memory_decision(report)
            self._send_daily_email(report)

    def on_filled_order(
        self,
        position: Any,
        order: Any,
        price: float,
        quantity: float,
        multiplier: float,
    ) -> None:
        """Record broker-confirmed executions in the Lumibot log."""
        symbol = getattr(getattr(order, "asset", None), "symbol", "unknown")
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
            TradeMemory(
                Path(str(self.parameters["decision_memory_database_file"])), 1, 1
            ).record_execution(
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
        asset_a = str(self.parameters["asset_a"]).upper()
        asset_b = str(self.parameters["asset_b"]).upper()
        side_text = str(side).lower()

        # Portfolio rotations follow the same two-phase pattern as A/B: buy
        # the replacement as soon as the source sale fills (instead of waiting
        # a full day for the next iteration), and clear the pending flag only
        # when the replacement purchase itself fills.
        portfolio_pending = self.vars.portfolio_pending_rotation
        if bool(self.parameters.get("portfolio_enabled", False)):
            if side_text == "buy":
                self._record_portfolio_entry(str(symbol))
            elif side_text == "sell":
                self._remove_portfolio_entry(str(symbol))
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

        # The Asset B purchase filled: the rotation is complete.
        if self.vars.pending_rotation and symbol == asset_b and side_text == "buy":
            self._set_pending_rotation(False)
            self.log_message(
                f"Rotation complete: the {asset_b} purchase filled.", color="green"
            )
            return

        if not self.vars.pending_rotation or symbol != asset_a or side_text != "sell":
            return

        try:
            price_b = self.get_last_price(asset_b)
            if price_b is None or float(price_b) <= 0:
                self.log_message(
                    f"The {asset_a} sale filled, but {asset_b} has no valid price; "
                    "the purchase will be retried next cycle.",
                    color="yellow",
                )
                return
            outcome = self._buy_asset_b_with_available_cash(asset_b, float(price_b))
            if outcome == "insufficient":
                # Cash may not have settled yet; keep the rotation pending so
                # the next iteration retries with confirmed balances and only
                # then declares the rotation finished.
                self.log_message(
                    f"The {asset_b} purchase will be retried next cycle in case "
                    "the sale proceeds have not settled yet.",
                    color="yellow",
                )
            elif outcome == "rejected":
                self.log_message(
                    f"The broker rejected the {asset_b} purchase; the pending "
                    "rotation remains recorded for the next cycle.",
                    color="red",
                )
        except Exception as exc:
            self.log_message(
                f"Post-sale purchase failed safely and will be retried: "
                f"{type(exc).__name__}: {exc}",
                color="red",
            )

    def on_canceled_order(self, order: Any) -> None:
        """Keep the rotation state truthful when the broker kills an order."""
        symbol = getattr(getattr(order, "asset", None), "symbol", "unknown")
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

        if not self.vars.pending_rotation:
            return

        asset_a = str(self.parameters["asset_a"]).upper()
        if symbol == asset_a and side == "sell":
            # Nothing was sold, so the rotation never started. Clear the flag
            # and let the next iteration evaluate the dip signal fresh.
            self._set_pending_rotation(False)
            self.log_message(
                f"The {asset_a} sale was canceled; the rotation is reset and "
                "the signal will be re-evaluated next cycle.",
                color="yellow",
            )
        # A canceled Asset B buy keeps pending_rotation set, so the next
        # iteration retries the purchase with the cash still on hand.
