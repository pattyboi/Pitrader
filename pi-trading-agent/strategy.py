"""Daily dip-buying and asset-rotation strategy for Lumibot."""

import math
import smtplib
from decimal import Decimal, InvalidOperation
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from lumibot.strategies import Strategy

from adaptive_news_model import AdaptiveNewsModel, LearningResult
from news_context import NewsContext, WorldEventAnalyzer


class AssetRotationStrategy(Strategy):
    """Rotate Asset A into Asset B when Asset B falls from its recent high."""

    parameters = {
        "asset_a": "SPY",
        "asset_b": "QQQ",
        "dip_threshold_percent": 5.0,
        "recent_high_lookback_days": 20,
        "email_report_enabled": False,
        "news_context_enabled": True,
        "news_learning_enabled": True,
    }

    def initialize(self) -> None:
        """Configure one evaluation per trading day."""
        self.sleeptime = "1D"
        self.vars.pending_rotation = False

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
                        f"Learning observations: {report.get('learning_observations', 'unavailable')}",
                        f"Learned return forecast: {report.get('learned_forecast', 'not ready')}",
                        f"Learning explanation: {report.get('learning_explanation', 'unavailable')}",
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
            with smtplib.SMTP(host, port, timeout=30) as smtp:
                smtp.ehlo()
                if bool(self.parameters["email_use_tls"]):
                    smtp.starttls()
                    smtp.ehlo()
                smtp.login(
                    str(self.parameters["email_smtp_username"]),
                    str(self.parameters["email_smtp_password"]),
                )
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
                observations=0,
                ready=False,
                predicted_return_percent=None,
                slope=None,
                correlation=None,
                explanation=f"Learning update failed: {type(exc).__name__}: {exc}",
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

    def _buy_asset_b_with_available_cash(self, asset_b: str, price_b: float) -> bool:
        """Invest all available whole-share buying capacity in Asset B."""
        cash = float(self.get_cash())
        quantity = math.floor(cash / price_b)
        if quantity < 1:
            self.log_message(
                f"No purchase submitted: available cash ${cash:.2f} cannot buy one "
                f"share of {asset_b} at ${price_b:.2f}.",
                color="yellow",
            )
            return False

        buy_order = self.create_order(
            asset_b,
            quantity=quantity,
            side="buy",
            order_type="market",
        )
        self.submit_order(buy_order)
        self.log_message(
            f"Submitted market buy for {quantity} shares of {asset_b}, using "
            f"the maximum whole-share quantity supported by ${cash:.2f} cash.",
            color="green",
        )
        return True

    def on_trading_iteration(self) -> None:
        """Evaluate the dip and safely advance any required portfolio rotation."""
        report = {
            "asset_a": str(self.parameters["asset_a"]).upper(),
            "asset_b": str(self.parameters["asset_b"]).upper(),
            "threshold": float(self.parameters["dip_threshold_percent"]),
            "status": "Evaluation started",
        }
        try:
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
            if price_a <= 0 or price_b <= 0:
                report["status"] = "No trade: invalid non-positive price received"
                self.log_message("Invalid non-positive market price received; no trade made.", color="red")
                return

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

            # A previous iteration may have submitted the sale. Wait until the
            # position is actually gone before spending the resulting cash.
            if self.vars.pending_rotation:
                if quantity_a > 0:
                    report["status"] = f"Pending: waiting for the {asset_a} sale to fill"
                    self.log_message(
                        f"Waiting for the {asset_a} sale to fill before buying {asset_b}.",
                        color="yellow",
                    )
                    return
                if self._buy_asset_b_with_available_cash(asset_b, price_b):
                    self.vars.pending_rotation = False
                    report["status"] = f"Submitted purchase of {asset_b} after completed sale"
                else:
                    report["status"] = f"No purchase: insufficient cash for one {asset_b} share"
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

            sell_order = self.create_order(
                asset_a,
                quantity=quantity_a,
                side="sell",
                order_type="market",
            )
            self.submit_order(sell_order)
            self.vars.pending_rotation = True
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

        # Continue the rotation immediately after Alpaca confirms the sale.
        # The next daily iteration remains a fallback if this callback cannot
        # obtain fresh account or price data during a temporary outage.
        asset_a = str(self.parameters["asset_a"]).upper()
        asset_b = str(self.parameters["asset_b"]).upper()
        if not self.vars.pending_rotation or symbol != asset_a or str(side).lower() != "sell":
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
            if self._buy_asset_b_with_available_cash(asset_b, float(price_b)):
                self.vars.pending_rotation = False
        except Exception as exc:
            self.log_message(
                f"Post-sale purchase failed safely and will be retried: "
                f"{type(exc).__name__}: {exc}",
                color="red",
            )
