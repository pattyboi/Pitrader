"""Daily dip-buying and asset-rotation strategy for Lumibot."""

import json
import math
import os
import smtplib
import ssl
import threading
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
            self.log_message(
                f"Restored portfolio rotation {pending['from']} to {pending['to']}; it will be reconciled next cycle.",
                color="yellow",
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
        self.vars.portfolio_holding_dates = dates
        path = self._portfolio_holding_state_path()
        if path is None:
            return
        try:
            temporary_path = path.with_suffix(path.suffix + ".tmp")
            temporary_path.write_text(json.dumps(dates, sort_keys=True) + "\n", encoding="utf-8")
            temporary_path.replace(path)
        except OSError as exc:
            self.log_message(f"Could not persist portfolio holding dates: {exc}", color="red")

    def _record_portfolio_entry(self, symbol: str) -> None:
        dates = dict(self.vars.portfolio_holding_dates)
        dates[str(symbol).upper()] = self.get_datetime().date().isoformat()
        self._set_portfolio_holding_dates(dates)

    def _remove_portfolio_entry(self, symbol: str) -> None:
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
        pending = self.vars.portfolio_pending_rotation
        if pending:
            symbols.update((str(pending["from"]).upper(), str(pending["to"]).upper()))
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
        order was placed, "working" when a buy order is already in flight, or
        "insufficient" when the spendable cash is below the minimum order (or
        below one whole share in whole-share mode).
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
            self.submit_order(
                self.create_order(symbol, quantity=quantity, side="buy", order_type="market", time_in_force="day")
            )
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
        return {
            "symbol": symbol,
            "price": float(price),
            "dip": current_dip,
            # This net historical mean is a coarse current estimate. It is
            # never enough by itself: _run_portfolio_iteration also requires
            # the chronological walk-forward result below.
            "expected_profit": sum(net_returns) / len(net_returns),
            "observations": len(returns),
            "oos_expected_profit": (
                sum(walk_forward_returns) / len(walk_forward_returns)
                if walk_forward_returns
                else None
            ),
            "oos_observations": len(walk_forward_returns),
        }

    def _autonomous_universe(self) -> AutonomousUniverse:
        return AutonomousUniverse(
            Path(str(self.parameters["portfolio_universe_state_file"])),
            int(self.parameters["portfolio_discovery_refresh_days"]),
            int(self.parameters["portfolio_discovery_batch_size"]),
            paper=os.environ.get("ALPACA_IS_PAPER", "true").strip().lower() != "false",
        )

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

        # Completing an in-flight rotation is never vetoed: the sale already
        # happened and leaving the proceeds in cash is its own risk.
        pending = self.vars.portfolio_pending_rotation
        if pending:
            source, target, budget = pending["from"], pending["to"], float(pending["budget"])
            if held.get(source, Decimal("0")) > 0:
                if self._has_active_order(source, "sell"):
                    report["status"] = f"Portfolio pending: waiting for {source} sale"
                    return
                self._set_portfolio_rotation(None)
                report["status"] = f"Portfolio rotation reset: {source} sale did not fill"
                return
            if held.get(target, Decimal("0")) > 0 and not self._has_active_order(target, "buy"):
                # The buy filled but the fill callback was lost (restart).
                self._set_portfolio_rotation(None)
                report["status"] = f"Portfolio rotation complete: the {target} purchase filled"
                return
            price = self.get_last_price(target)
            if price is None or not math.isfinite(float(price)) or float(price) <= 0:
                report["status"] = f"Portfolio pending: no valid {target} price"
                return
            outcome = self._buy_portfolio_symbol(target, float(price), budget)
            if outcome == "insufficient":
                # Balances are confirmed by now; the rotation ends here.
                self._set_portfolio_rotation(None)
                report["status"] = (
                    f"Portfolio rotation finished: cash is below the minimum {target} order"
                )
            elif outcome == "working":
                report["status"] = f"Portfolio pending: waiting for the {target} purchase to fill"
            else:
                # The flag clears when the buy fills (on_filled_order), never
                # on submission, so a rejected order is retried next cycle.
                report["status"] = f"Portfolio {target} purchase submitted after {source} sale"
            return

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
        if due_symbols:
            source = due_symbols[0]
            if self._has_active_order(source, "sell"):
                report["status"] = f"Portfolio exit pending: waiting for {source} sale"
                return
            self.submit_order(
                self.create_order(
                    source, quantity=held[source], side="sell", order_type="market", time_in_force="day"
                )
            )
            report["status"] = (
                f"Portfolio exit submitted: {source} reached its "
                f"{maximum_holding_days}-day holding horizon"
            )
            return

        signals = [self._portfolio_signal(symbol) for symbol in symbols]
        signals = [signal for signal in signals if signal is not None]
        eligible = [
            signal
            for signal in signals
            if int(signal["observations"]) >= minimum_observations
            and float(signal["expected_profit"]) >= minimum_profit
            and int(signal["oos_observations"]) >= oos_minimum_observations
            and signal["oos_expected_profit"] is not None
            and float(signal["oos_expected_profit"]) >= oos_minimum_profit
        ]
        eligible.sort(key=lambda signal: (float(signal["expected_profit"]), float(signal["dip"])), reverse=True)
        opportunity_probability = opportunity.get("probability")
        opportunity_edge = opportunity.get("predicted_edge")
        opportunity_is_eligible = (
            asset_a in held
            and asset_b not in held
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
            f"{s['symbol']} net {s['expected_profit']:+.2f}%/{s['observations']}; "
            f"OOS {float(s['oos_expected_profit']):+.2f}%/{s['oos_observations']}"
            for s in eligible
        ) or "none"
        if not eligible and not opportunity_is_eligible:
            report["status"] = "No portfolio trade: no portfolio signal or Opportunistic Opportunity met its thresholds"
            return
        if veto_reason:
            report["status"] = veto_reason
            self.log_message(
                f"Portfolio signal present, but the trade was vetoed: {veto_reason}",
                color="red",
            )
            return

        if opportunity_is_eligible:
            if self._has_active_order(asset_a, "sell"):
                report["status"] = "Opportunistic Opportunity pending: waiting for Asset A sale"
                return
            source_price = self.get_last_price(asset_a)
            if source_price is None or float(source_price) <= 0:
                report["status"] = "No Opportunistic Opportunity: Asset A price was unavailable"
                return
            budget = float(source_price) * float(held[asset_a])
            self.submit_order(
                self.create_order(
                    asset_a, quantity=held[asset_a], side="sell", order_type="market", time_in_force="day"
                )
            )
            self._set_portfolio_rotation({"from": asset_a, "to": asset_b, "budget": budget})
            report["status"] = (
                f"Opportunistic Opportunity submitted: {asset_a} to {asset_b} "
                f"({float(opportunity_probability):.1%} historical win probability, "
                f"{float(opportunity_edge):+.2f}% predicted edge)"
            )
            return

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
        # A holding with no current dip signal is scored neutral (0% expected
        # edge), not punished: rotation happens only when the target's
        # historical edge beats holding by the configured margin. The old
        # -100% default force-rotated any recovered holding every time some
        # other symbol dipped, churning the portfolio.
        source = min(held, key=lambda symbol: float(held_signals.get(symbol, {"expected_profit": 0.0})["expected_profit"]))
        source_score = float(held_signals.get(source, {"expected_profit": 0.0})["expected_profit"])
        advantage = float(target["expected_profit"]) - source_score
        if advantage < minimum_profit:
            report["status"] = f"No portfolio rotation: {target['symbol']} advantage {advantage:+.2f}% is below threshold"
            return
        source_price = self.get_last_price(source)
        if source_price is None or float(source_price) <= 0 or self._has_active_order(source, "sell"):
            report["status"] = f"No portfolio rotation: {source} is unavailable or has a working order"
            return
        budget = float(source_price) * float(held[source])
        self.submit_order(
            self.create_order(source, quantity=held[source], side="sell", order_type="market", time_in_force="day")
        )
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
            if symbol == portfolio_pending["to"] and side_text == "buy":
                self._set_portfolio_rotation(None)
                self.log_message(
                    f"Portfolio rotation complete: the {symbol} purchase filled.",
                    color="green",
                )
                return
            if symbol == portfolio_pending["from"] and side_text == "sell":
                target = str(portfolio_pending["to"])
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
                        target, float(target_price), float(portfolio_pending["budget"])
                    )
                    if outcome == "insufficient":
                        # Proceeds may not have settled yet; the next daily
                        # iteration retries with confirmed balances.
                        self.log_message(
                            f"The {target} purchase will be retried next cycle in "
                            "case the sale proceeds have not settled yet.",
                            color="yellow",
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
        if portfolio_pending and symbol == portfolio_pending["from"] and side == "sell":
            # Nothing was sold, so the portfolio rotation never started.
            self._set_portfolio_rotation(None)
            self.log_message(
                f"The {symbol} sale was canceled; the portfolio rotation is "
                "reset and will be re-evaluated next cycle.",
                color="yellow",
            )
        # A canceled portfolio buy keeps the pending state so the next
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
