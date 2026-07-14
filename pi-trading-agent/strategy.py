"""Daily dip-buying and asset-rotation strategy for Lumibot."""

import json
import math
import os
import smtplib
import ssl
import threading
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from lumibot.strategies import Strategy

from adaptive_news_model import AdaptiveNewsModel, LearningResult
from autonomous_universe import AutonomousUniverse
from llm_news import LLMNewsAnalyzer, LLMNewsAssessment
from news_context import NewsContext, WorldEventAnalyzer
from trade_memory import RotationForecast, TradeMemory


class AssetRotationStrategy(Strategy):
    """Run either the original A/B rotation or an opt-in dip-signal portfolio."""

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
        "portfolio_enabled": False,
    }

    # Fraction of cash withheld from the Asset B buy so the market order is not
    # rejected (or filled into a deficit) if the price moves before execution.
    CASH_BUFFER_FRACTION = 0.01

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

    def initialize(self) -> None:
        """Configure one evaluation per trading day."""
        self.sleeptime = "1D"
        self._rotation_lock = threading.Lock()
        self.vars.pending_rotation = self._load_pending_rotation()
        # Historical bars are fetched during the first evaluation, after the
        # broker has supplied current market data.
        self.vars.decision_memory_backfill_attempted = False
        self.vars.portfolio_pending_rotation = self._load_portfolio_rotation()
        if self.vars.pending_rotation:
            self.log_message(
                "Restored an in-progress rotation from disk; it will be "
                "reconciled on the next trading iteration.",
                color="yellow",
            )
        if self.vars.portfolio_pending_rotation:
            pending = self.vars.portfolio_pending_rotation
            self.log_message(
                f"Restored portfolio rotation {pending['from']} to {pending['to']}; it will be reconciled next cycle.",
                color="yellow",
            )

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

    def _set_pending_rotation(self, value: bool) -> None:
        """Update the rotation flag in memory and persist it atomically."""
        self.vars.pending_rotation = bool(value)
        path = self._rotation_state_path()
        if path is None:
            return
        try:
            temporary_path = path.with_suffix(path.suffix + ".tmp")
            temporary_path.write_text(
                json.dumps({"pending_rotation": bool(value)}) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(path)
        except OSError as exc:
            self.log_message(f"Could not persist rotation state: {exc}", color="red")

    def _portfolio_rotation_state_path(self) -> Path | None:
        raw = self.parameters.get("portfolio_rotation_state_file")
        return Path(str(raw)) if raw else None

    def _load_portfolio_rotation(self) -> dict[str, Any] | None:
        """Restore a single staged portfolio replacement after a restart."""
        path = self._portfolio_rotation_state_path()
        if path is None or not path.exists():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if all(isinstance(state.get(key), str) and state[key] for key in ("from", "to")):
                budget = float(state.get("budget", 0))
                if math.isfinite(budget) and budget > 0:
                    return {"from": state["from"].upper(), "to": state["to"].upper(), "budget": budget}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return None

    def _set_portfolio_rotation(self, state: dict[str, Any] | None) -> None:
        self.vars.portfolio_pending_rotation = state
        path = self._portfolio_rotation_state_path()
        if path is None:
            return
        try:
            if state is None:
                path.unlink(missing_ok=True)
                return
            temporary_path = path.with_suffix(path.suffix + ".tmp")
            temporary_path.write_text(json.dumps(state) + "\n", encoding="utf-8")
            temporary_path.replace(path)
        except OSError as exc:
            self.log_message(f"Could not persist portfolio rotation state: {exc}", color="red")

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
            message.set_content(
                "\n".join(
                    [
                        "Raspberry Pi Trading Agent Daily Summary",
                        "",
                        f"Date: {report_date}",
                        f"Evaluation time: {self.get_datetime().isoformat()}",
                        f"Asset A: {report['asset_a']}",
                        f"Asset B: {report['asset_b']}",
                        f"Asset A price: {report.get('price_a', 'unavailable')}",
                        f"Asset B price: {report.get('price_b', 'unavailable')}",
                        f"Asset A quantity: {report.get('quantity_a', 'unavailable')}",
                        f"Asset B quantity: {report.get('quantity_b', 'unavailable')}",
                        f"Recent high: {report.get('recent_high', 'unavailable')}",
                        f"Calculated dip: {report.get('dip_percent', 'unavailable')}",
                        f"Dip threshold: {report['threshold']:.2f}%",
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
                        f"Decision-memory observations: {report.get('decision_memory_observations', 'unavailable')}",
                        f"Predicted rotation edge: {report.get('rotation_edge_forecast', 'not ready')}",
                        f"Decision-memory explanation: {report.get('decision_memory_explanation', 'unavailable')}",
                        "Notable scored headlines:",
                        *[
                            f"- {headline}"
                            for headline in report.get("news_headlines", [])
                        ],
                        f"Result: {report['status']}",
                        "",
                        "Review all orders and positions in the Alpaca dashboard.",
                        "This automated message is not financial advice.",
                    ]
                )
            )

            host = str(self.parameters["email_smtp_host"])
            port = int(self.parameters["email_smtp_port"])
            # The password comes from the environment so it never travels
            # through Lumibot's parameters dict, which may be logged.
            password = os.environ.get("EMAIL_SMTP_PASSWORD") or str(
                self.parameters.get("email_smtp_password", "")
            )
            with smtplib.SMTP(host, port, timeout=30) as smtp:
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
                close_a = a_closes.get(date)
                if close_a is None:
                    continue
                recent_high = max(row[2] for row in b_rows[max(0, position - lookback + 1) : position + 1])
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

    def _buy_asset_b_with_available_cash(self, asset_b: str, price_b: float) -> str:
        """Submit the largest whole-share Asset B buy the cash safely supports.

        Returns "submitted" when a new order was placed, "working" when a buy
        order is already in flight, or "insufficient" when the spendable cash
        cannot purchase one whole share.
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
            )
            self.submit_order(buy_order)
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
            self.submit_order(self.create_order(symbol, quantity=quantity, side="buy", order_type="market"))
            self.log_message(
                f"Portfolio submitted buy of {quantity} {symbol} shares using up to ${budget:.2f}.",
                color="green",
            )
            return "submitted"

    def _portfolio_signal(self, symbol: str) -> dict[str, float | int | str] | None:
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
        for index in range(lookback - 1, len(rows) - 1):
            recent_high = max(high for high, _ in rows[index - lookback + 1 : index + 1])
            dip = ((recent_high - rows[index][1]) / recent_high) * 100.0
            if dip >= threshold:
                returns.append(((rows[index + 1][1] - rows[index][1]) / rows[index][1]) * 100.0)
        recent_high = max(high for high, _ in rows[-lookback:])
        current_dip = ((recent_high - float(price)) / recent_high) * 100.0
        if current_dip < threshold or not returns:
            return None
        return {
            "symbol": symbol,
            "price": float(price),
            "dip": current_dip,
            "expected_profit": sum(returns) / len(returns),
            "observations": len(returns),
        }

    def _portfolio_symbols(self, report: dict[str, Any]) -> list[str]:
        """Combine the static watchlist with one bounded discovery batch."""
        symbols = [str(symbol).upper() for symbol in self.parameters["portfolio_symbols"]]
        if not bool(self.parameters.get("portfolio_autonomous_discovery", False)):
            return symbols
        try:
            discovered = AutonomousUniverse(
                Path(str(self.parameters["portfolio_universe_state_file"])),
                int(self.parameters["portfolio_discovery_refresh_days"]),
                int(self.parameters["portfolio_discovery_batch_size"]),
            ).next_batch(
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
        if not bool(self.parameters.get("portfolio_autonomous_discovery", False)):
            return
        try:
            AutonomousUniverse(
                Path(str(self.parameters["portfolio_universe_state_file"])),
                int(self.parameters["portfolio_discovery_refresh_days"]),
                int(self.parameters["portfolio_discovery_batch_size"]),
            ).remember(symbols)
        except Exception as exc:
            self.log_message(f"Could not persist learned symbols: {type(exc).__name__}: {exc}", color="yellow")

    def _run_portfolio_iteration(self, report: dict[str, Any]) -> None:
        """Build or rotate a bounded portfolio from the explicit symbol list."""
        symbols = self._portfolio_symbols(report)
        minimum_observations = int(self.parameters["portfolio_min_signal_observations"])
        minimum_profit = float(self.parameters["portfolio_min_expected_profit_percent"])
        max_positions = int(self.parameters["portfolio_max_positions"])
        news_context = self._get_news_context()
        report.update(news_risk_level=news_context.risk_level, news_score=news_context.score if news_context.available else "unavailable")
        if news_context.available and bool(self.parameters["news_block_on_high_risk"]) and news_context.score <= int(self.parameters["news_high_risk_score"]):
            report["status"] = f"Portfolio trade blocked: high world-event risk score {news_context.score}"
            return

        pending = self.vars.portfolio_pending_rotation
        if pending:
            source, target, budget = pending["from"], pending["to"], float(pending["budget"])
            source_quantity = self._quantity(self.get_position(source))
            if source_quantity > 0:
                if self._has_active_order(source, "sell"):
                    report["status"] = f"Portfolio pending: waiting for {source} sale"
                    return
                self._set_portfolio_rotation(None)
                report["status"] = f"Portfolio rotation reset: {source} sale did not fill"
                return
            price = self.get_last_price(target)
            if price is None or float(price) <= 0:
                report["status"] = f"Portfolio pending: no valid {target} price"
                return
            outcome = self._buy_portfolio_symbol(target, float(price), budget)
            if outcome != "working":
                self._set_portfolio_rotation(None)
            report["status"] = f"Portfolio {target} purchase {outcome} after {source} sale"
            return

        signals = [self._portfolio_signal(symbol) for symbol in symbols]
        signals = [signal for signal in signals if signal is not None]
        eligible = [signal for signal in signals if int(signal["observations"]) >= minimum_observations and float(signal["expected_profit"]) >= minimum_profit]
        eligible.sort(key=lambda signal: (float(signal["expected_profit"]), float(signal["dip"])), reverse=True)
        self._remember_discovered_symbols([str(signal["symbol"]) for signal in eligible])
        report["portfolio_candidates"] = ", ".join(f"{s['symbol']} {s['expected_profit']:+.2f}%/{s['observations']}" for s in eligible) or "none"
        if not eligible:
            report["status"] = "No portfolio trade: no symbol met the historical-profit threshold"
            return

        held = {symbol: self._quantity(self.get_position(symbol)) for symbol in symbols}
        held = {symbol: quantity for symbol, quantity in held.items() if quantity > 0}
        desired = eligible[:max_positions]
        desired_symbols = {str(signal["symbol"]) for signal in desired}
        target = next((signal for signal in desired if signal["symbol"] not in held), None)
        if target is None:
            # A recurring small deposit should grow the highest-ranked current
            # holding instead of remaining idle once the portfolio is full.
            cash = float(self.get_cash())
            minimum_cash = float(self.parameters.get("portfolio_cash_reserve_dollars", 0.0)) + float(
                self.parameters.get("portfolio_min_order_dollars", 1.0)
            )
            if cash >= minimum_cash:
                target = desired[0]
                outcome = self._buy_portfolio_symbol(
                    str(target["symbol"]), float(target["price"]), cash
                )
                report["status"] = f"Portfolio top-up: {target['symbol']} purchase {outcome}"
                return
            report["status"] = "No portfolio trade: current holdings match top signals and cash is below the minimum order"
            return
        if len(held) < max_positions:
            slots_remaining = min(max_positions - len(held), len(desired_symbols.difference(held)))
            outcome = self._buy_portfolio_symbol(str(target["symbol"]), float(target["price"]), float(self.get_cash()) / slots_remaining)
            report["status"] = f"Portfolio build: {target['symbol']} purchase {outcome}"
            return

        held_signals = {str(signal["symbol"]): signal for signal in signals if signal["symbol"] in held}
        source = min(held, key=lambda symbol: float(held_signals.get(symbol, {"expected_profit": -100.0})["expected_profit"]))
        source_score = float(held_signals.get(source, {"expected_profit": -100.0})["expected_profit"])
        advantage = float(target["expected_profit"]) - source_score
        if advantage < minimum_profit:
            report["status"] = f"No portfolio rotation: {target['symbol']} advantage {advantage:+.2f}% is below threshold"
            return
        source_price = self.get_last_price(source)
        if source_price is None or float(source_price) <= 0 or self._has_active_order(source, "sell"):
            report["status"] = f"No portfolio rotation: {source} is unavailable or has a working order"
            return
        budget = float(source_price) * float(held[source])
        self.submit_order(self.create_order(source, quantity=held[source], side="sell", order_type="market"))
        self._set_portfolio_rotation({"from": source, "to": str(target["symbol"]), "budget": budget})
        report["status"] = f"Portfolio rotation submitted: {source} to {target['symbol']} (expected advantage {advantage:+.2f}%)"

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
                    else:
                        # Below one whole share: the rotation is complete by design.
                        self._set_pending_rotation(False)
                        report["status"] = (
                            f"Rotation finished: remaining cash cannot buy one "
                            f"{asset_b} share"
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
            )
            self.submit_order(sell_order)
            self._set_pending_rotation(True)
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
        self.log_message(
            f"Filled {side} order: {quantity} shares of {symbol} at ${price:.2f}.",
            color="green",
        )
        try:
            TradeMemory(
                Path(str(self.parameters["decision_memory_database_file"])), 1, 1
            ).record_execution(
                self.get_datetime().date().isoformat(),
                str(symbol),
                str(side),
                float(price),
                float(quantity),
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
