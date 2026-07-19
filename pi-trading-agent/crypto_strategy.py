"""Crypto dip-buying rotation strategy for Lumibot -- runs only while NYSE is closed.

Deliberately its own Strategy subclass, not built on top of AssetRotationStrategy
(strategy.py): the equity class is already large and its email/discovery/news
machinery is tuned for equities. This shares only the pure, asset-class-agnostic
math in decision_math.py; everything else here is a narrower, crypto-specific
reimplementation of the same shape (dip signal, take-profit/stop-loss/holding-
horizon exits, empty-slot buys), following strategy.py's small-testable-method
discipline.

Runs as a separate OS process from main.py (see main_crypto.py) because
Lumibot's Trader cannot run two live strategies in one process. Gated to NYSE-
closed hours by market_sessions.nyse_is_open -- not by self.broker's own
market-hours methods, which report "open" unconditionally once this strategy
calls self.set_market("24/7") (the standard Lumibot crypto pattern, needed so
Lumibot's own scheduler never blocks waiting for a stock-market open/close
that will never come for a 24/7 asset).
"""

import json
import math
import os
import re
import smtplib
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date as date_type, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable

from lumibot.entities import Asset
from lumibot.strategies import Strategy

import decision_math
import email_render
import signal_snapshot
from autonomous_universe import AutonomousUniverse
from market_sessions import is_next_calendar_day, nyse_is_open
from portfolio_memory import PortfolioMemory
from trade_memory import OpportunityProbability, RotationForecast, TradeMemory

_CRYPTO_BASE_SYMBOL = re.compile(r"^[A-Z0-9]{1,10}$")


def _crypto_asset_symbol_filter(item: dict[str, Any]) -> str | None:
    """Extract a bare base symbol (e.g. "BTC") from one of Alpaca's crypto
    asset listings.

    Unlike AutonomousUniverse's equity default (a plain ticker string),
    Alpaca's crypto assets come back as "BASE/QUOTE" pairs (e.g. "BTC/USD"),
    have no "fractionable" field to check, and include non-USD-quoted pairs
    (e.g. "ETH/BTC") this strategy can't trade since it always quotes in USD
    (self.quote_asset). Only USD-quoted pairs are kept.
    """
    if item.get("tradable") is not True:
        return None
    raw_symbol = str(item.get("symbol", "")).upper()
    if "/" not in raw_symbol:
        return None
    base, _, quote = raw_symbol.partition("/")
    if quote != "USD" or not _CRYPTO_BASE_SYMBOL.fullmatch(base):
        return None
    return base


class CryptoRotationStrategy(Strategy):
    """Run a dip-signal crypto rotation only while NYSE regular hours are closed."""

    parameters = {
        "crypto_enabled": False,
        "crypto_symbols": ["BTC", "ETH"],
        "crypto_max_positions": 1,
        "crypto_analysis_days": 252,
        "crypto_recent_high_lookback_days": 20,
        "crypto_min_signal_observations": 20,
        "crypto_dip_threshold_percent": 5.0,
        "crypto_min_expected_profit_percent": 1.0,
        "crypto_oos_min_observations": 10,
        "crypto_oos_min_net_profit_percent": 0.0,
        "crypto_round_trip_cost_percent": 0.50,
        "crypto_take_profit_percent": 1.5,
        "crypto_stop_loss_percent": 1.0,
        "crypto_holding_horizon_max_days": 15,
        "crypto_min_order_dollars": 5.0,
        "crypto_iteration_interval_minutes": 15,
        "crypto_risk_posture": "conservative",
        "crypto_memory_enabled": True,
        "crypto_memory_min_observations": 20,
        "crypto_memory_max_observations": 500,
        "crypto_email_report_enabled": False,
        "crypto_autonomous_discovery": False,
        "crypto_discovery_batch_size": 6,
        "crypto_discovery_refresh_days": 7,
        "crypto_asset_a": "BTC",
        "crypto_asset_b": "ETH",
        "crypto_opportunistic_min_probability": 0.55,
    }

    # Fraction of cash withheld from a buy so the market order is not rejected
    # (or filled into a deficit) if the price moves before execution. Same
    # value and rationale as AssetRotationStrategy.CASH_BUFFER_FRACTION.
    CASH_BUFFER_FRACTION = 0.01

    # Order statuses that mean an order can no longer fill. Mirrors
    # AssetRotationStrategy's constants of the same name (strategy.py) --
    # duplicated rather than imported since CryptoRotationStrategy
    # deliberately does not inherit from that class.
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
    _CRYPTO_HISTORY_WORKERS = 4
    # Crypto spreads run structurally wider than large-cap equity ETFs, so
    # this cap is looser than AssetRotationStrategy's
    # _PORTFOLIO_LIVE_SPREAD_CAP_PERCENT (5.0) -- it exists for the same
    # reason: Alpaca's free quote feed is IEX/exchange-limited, not full
    # market depth, so a bad or thin print should never make a symbol's live
    # spread reading swamp the flat cost estimate it is only meant to floor.
    _CRYPTO_LIVE_SPREAD_CAP_PERCENT = 8.0

    def initialize(self) -> None:
        interval_minutes = int(self.parameters.get("crypto_iteration_interval_minutes", 15))
        self.sleeptime = f"{interval_minutes}M"
        # 24/7 so Lumibot's own scheduler never waits for a stock-market
        # open/close that will never come; NYSE-closed gating is done
        # ourselves in on_trading_iteration via market_sessions.nyse_is_open.
        self.set_market("24/7")
        self._rotation_lock = threading.Lock()
        self._crypto_state_lock = threading.RLock()
        self._last_logged_nyse_open: bool | None = None
        self._unpriceable_symbols_lock = threading.Lock()
        self._unpriceable_symbols_this_iteration: set[str] = set()
        self.vars.crypto_holding_dates = self._load_crypto_holding_dates()
        self.vars.crypto_memory_backfilled_symbols = set()
        self.vars.crypto_pending_rotation = self._load_crypto_rotation()
        self.vars.crypto_opportunistic_swap_date = self._load_crypto_opportunistic_swap_date()
        self.vars.crypto_decision_memory_backfill_attempted = False
        if self.vars.crypto_pending_rotation is not None:
            entry = self.vars.crypto_pending_rotation
            self.log_message(
                f"Restored an in-progress crypto rotation: {entry['from']} to {entry['to']}; "
                "reconciling next cycle.",
                color="yellow",
            )

    def _crypto_state_guard(self) -> threading.RLock:
        return self._crypto_state_lock

    # -- Holding-date state (restart-safe, mirrors AssetRotationStrategy's
    # portfolio_holding_dates in strategy.py) --------------------------------

    def _crypto_holding_state_path(self) -> Path | None:
        raw = self.parameters.get("crypto_holding_state_file")
        return Path(str(raw)) if raw else None

    def _load_crypto_holding_dates(self) -> dict[str, str]:
        path = self._crypto_holding_state_path()
        if path is None or not path.exists():
            return {}
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                return {}
            return {
                str(symbol).upper(): value
                for symbol, value in state.items()
                if isinstance(value, str) and str(symbol).strip() and self._valid_iso_date(value)
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

    def _set_crypto_holding_dates(self, dates: dict[str, str]) -> None:
        with self._crypto_state_guard():
            previous = self.vars.crypto_holding_dates
            self.vars.crypto_holding_dates = dates
            path = self._crypto_holding_state_path()
            if path is None:
                return
            try:
                temporary_path = path.with_suffix(path.suffix + ".tmp")
                temporary_path.write_text(json.dumps(dates, sort_keys=True) + "\n", encoding="utf-8")
                temporary_path.replace(path)
            except OSError as exc:
                self.vars.crypto_holding_dates = previous
                self.log_message(f"Could not persist crypto holding dates: {exc}", color="red")

    def _record_crypto_entry(self, symbol: str) -> None:
        with self._crypto_state_guard():
            dates = dict(self.vars.crypto_holding_dates)
            dates[str(symbol).upper()] = datetime.now(timezone.utc).date().isoformat()
            self._set_crypto_holding_dates(dates)

    def _remove_crypto_entry(self, symbol: str) -> None:
        with self._crypto_state_guard():
            dates = dict(self.vars.crypto_holding_dates)
            dates.pop(str(symbol).upper(), None)
            self._set_crypto_holding_dates(dates)

    @staticmethod
    def _holding_is_due(entry_date: str, today: date_type, maximum_days: int) -> bool:
        """Return whether a confirmed entry has reached its configured horizon.

        Plain calendar days, unlike AssetRotationStrategy's NYSE-trading-day
        variant would need to be -- crypto trades every day, so there is no
        "next session" concept to anchor to.
        """
        try:
            return today - date_type.fromisoformat(entry_date) >= timedelta(days=maximum_days)
        except ValueError:
            return False

    # -- Opportunistic Opportunity rotation state (restart-safe sell-then-buy
    # staging, scoped to a single pair since crypto only ever has one
    # CRYPTO_ASSET_A/CRYPTO_ASSET_B swap in flight at a time -- simpler than
    # AssetRotationStrategy's keyed-by-source dict, which supports many
    # simultaneous portfolio replacement rotations, strategy.py) -----------

    def _crypto_rotation_state_path(self) -> Path | None:
        raw = self.parameters.get("crypto_rotation_state_file")
        return Path(str(raw)) if raw else None

    def _load_crypto_rotation(self) -> dict[str, Any] | None:
        path = self._crypto_rotation_state_path()
        if path is None or not path.exists():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        if not isinstance(state, dict):
            return None
        source = state.get("from")
        target = state.get("to")
        if not isinstance(source, str) or not isinstance(target, str) or not source.strip() or not target.strip():
            return None
        try:
            budget = float(state.get("budget", 0))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(budget) or budget <= 0:
            return None
        return {"from": source.upper(), "to": target.upper(), "budget": budget}

    def _set_crypto_rotation(self, state: dict[str, Any] | None) -> None:
        with self._crypto_state_guard():
            previous = self.vars.crypto_pending_rotation
            self.vars.crypto_pending_rotation = state
            path = self._crypto_rotation_state_path()
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
                self.vars.crypto_pending_rotation = previous
                self.log_message(f"Could not persist crypto rotation state: {exc}", color="red")

    def _crypto_opportunistic_swap_state_path(self) -> Path | None:
        raw = self.parameters.get("crypto_opportunistic_swap_state_file")
        return Path(str(raw)) if raw else None

    def _load_crypto_opportunistic_swap_date(self) -> str:
        """Restore the calendar date the day's swap (if any) was already done on.

        Persisted (not just in self.vars) so a restart between two crypto
        iterations on the same NYSE-closed day can't allow a second swap.
        """
        path = self._crypto_opportunistic_swap_state_path()
        if path is None or not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _mark_crypto_opportunistic_swap_done(self, today: date_type) -> None:
        self.vars.crypto_opportunistic_swap_date = today.isoformat()
        path = self._crypto_opportunistic_swap_state_path()
        if path is None:
            return
        try:
            path.write_text(today.isoformat() + "\n", encoding="utf-8")
        except OSError as exc:
            self.log_message(f"Could not persist crypto opportunistic-swap state: {exc}", color="red")

    # -- Order helpers (broker-generic; mirrors AssetRotationStrategy's
    # methods of the same name in strategy.py) -------------------------------

    def _has_active_order(self, symbol: str, side: str) -> bool:
        try:
            orders = self.get_orders() or []
        except Exception as exc:
            self.log_message(
                f"Could not read orders ({type(exc).__name__}: {exc}); assuming one may still be working.",
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
        status = getattr(order, "status", "")
        value = getattr(status, "value", status)
        return str(value).strip().lower()

    def _submit_order_checked(self, order: Any, description: str) -> bool:
        submitted = self.submit_order(order)
        if submitted is None:
            self.log_message(f"Broker did not accept {description}: submission returned no order.", color="red")
            return False
        status = self._order_status(submitted)
        if status in self._FAILED_ORDER_STATUSES:
            error = str(getattr(submitted, "error_message", "") or "").strip()
            suffix = f": {error}" if error else ""
            self.log_message(f"Broker rejected {description}{suffix}.", color="red")
            return False
        return True

    # -- Pricing helpers (Asset-wrapped so the quote resolves for crypto,
    # unlike the bare-string calls the equity pipeline uses) -----------------

    @staticmethod
    def _crypto_asset(symbol: str) -> Asset:
        return Asset(symbol=symbol, asset_type="crypto")

    def _get_crypto_bid_ask(self, symbol: str) -> tuple[float, float] | None:
        try:
            quote = self.get_quote(self._crypto_asset(symbol), quote=self.quote_asset)
        except Exception:
            return None
        if quote is None:
            return None
        try:
            bid = float(quote.bid)
            ask = float(quote.ask)
        except (TypeError, ValueError, AttributeError):
            return None
        if not (math.isfinite(bid) and math.isfinite(ask)) or bid <= 0 or ask <= bid:
            return None
        return bid, ask

    def _crypto_live_spread_percent(self, symbol: str) -> float | None:
        bid_ask = self._get_crypto_bid_ask(symbol)
        if bid_ask is None:
            return None
        bid, ask = bid_ask
        mid = (bid + ask) / 2.0
        spread_percent = ((ask - bid) / mid) * 100.0
        return min(spread_percent, self._CRYPTO_LIVE_SPREAD_CAP_PERCENT)

    def _crypto_realizable_sale_price(self, symbol: str) -> float | None:
        """Price a market sell would actually realize (live bid, not last trade)."""
        bid_ask = self._get_crypto_bid_ask(symbol)
        if bid_ask is not None:
            return bid_ask[0]
        last_price = self.get_last_price(self._crypto_asset(symbol), quote=self.quote_asset)
        if last_price is None:
            return None
        try:
            value = float(last_price)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value > 0 else None

    # -- Positions -------------------------------------------------------

    @staticmethod
    def _quantity(position: Any) -> Decimal:
        if position is None:
            return Decimal("0")
        try:
            return max(Decimal(str(position.quantity)), Decimal("0"))
        except (AttributeError, InvalidOperation, TypeError, ValueError):
            return Decimal("0")

    def _crypto_held_positions(
        self, managed_symbols: set[str]
    ) -> tuple[dict[str, Decimal], dict[str, float]] | None:
        """Return (quantities, avg entry prices) for managed crypto positions.

        None on broker-read failure. Filters to asset_type == "crypto" -- the
        equity pipeline's own _portfolio_held_positions (strategy.py) already
        excludes crypto positions the opposite way, so the two never double-
        count the same shared Alpaca account's holdings.
        """
        try:
            positions = self.get_positions() or []
        except Exception as exc:
            self.log_message(
                f"Could not read account positions ({type(exc).__name__}: {exc}); skipping this crypto evaluation.",
                color="red",
            )
            return None
        held: dict[str, Decimal] = {}
        entry_prices: dict[str, float] = {}
        for position in positions:
            asset = getattr(position, "asset", None)
            symbol = getattr(asset, "symbol", None)
            asset_type = str(getattr(asset, "asset_type", "") or "").lower()
            normalized_symbol = str(symbol).upper() if symbol else ""
            if not normalized_symbol or normalized_symbol not in managed_symbols or asset_type != "crypto":
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

    def _crypto_deployed_dollars(
        self, held: dict[str, Decimal], signals_by_symbol: dict[str, dict[str, Any]]
    ) -> float:
        """Current mark-to-market value of managed crypto holdings.

        Reuses a signal's already-fetched price when available instead of a
        second data call; only re-fetches for a held symbol that fell out of
        crypto_symbols (so it no longer has a fresh signal this iteration).
        """
        total = 0.0
        for symbol, quantity in held.items():
            signal = signals_by_symbol.get(symbol)
            price = signal["price"] if signal is not None else self.get_last_price(
                self._crypto_asset(symbol), quote=self.quote_asset
            )
            if price is None:
                continue
            try:
                total += float(quantity) * float(price)
            except (TypeError, ValueError):
                continue
        return total

    # -- Exits -------------------------------------------------------------

    def _crypto_exit_reasons(
        self,
        held: dict[str, Decimal],
        entry_prices: dict[str, float],
        holding_dates: dict[str, str],
        today: date_type,
    ) -> dict[str, str]:
        """Decide which holdings exit today, and why. Mirrors
        AssetRotationStrategy._portfolio_exit_reasons (strategy.py)."""
        take_profit_percent = float(self.parameters.get("crypto_take_profit_percent", 1.5))
        stop_loss_percent = float(self.parameters.get("crypto_stop_loss_percent", 1.0))
        backstop_days = int(self.parameters.get("crypto_holding_horizon_max_days", 15))
        exit_reasons: dict[str, str] = {}
        for symbol in held:
            entry_price = entry_prices.get(symbol)
            current_price = self._crypto_realizable_sale_price(symbol)
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

    # -- Signal --------------------------------------------------------------

    def _crypto_signal(self, symbol: str) -> dict[str, float | int | str | bool | None] | None:
        """Compute today's dip signal for symbol. Mirrors
        AssetRotationStrategy._portfolio_signal (strategy.py), minus the
        discovery-liquidity-floor checks that don't apply to crypto's tiny,
        already-curated Alpaca universe."""
        base = self._crypto_asset(symbol)
        bars = self.get_historical_prices(
            base, int(self.parameters["crypto_analysis_days"]), "day", quote=self.quote_asset
        )
        if bars is None or bars.df is None or bars.df.empty or not {"high", "close"}.issubset(bars.df.columns):
            # No bars at all means Alpaca has no price history for this pair
            # -- a discovery-sourced candidate like this can never qualify,
            # so _run_crypto_iteration persists it as permanently unpriceable
            # instead of re-fetching and re-warning every time the discovery
            # rotation cursor comes back around. Mirrors
            # AssetRotationStrategy._portfolio_signal (strategy.py).
            with self._unpriceable_symbols_lock:
                self._unpriceable_symbols_this_iteration.add(symbol)
            return None
        rows = [
            (float(row["high"]), float(row["close"]))
            for _, row in bars.df[["high", "close"]].dropna().iterrows()
            if math.isfinite(float(row["high"])) and math.isfinite(float(row["close"]))
            and float(row["high"]) > 0 and float(row["close"]) > 0
        ]
        lookback = int(self.parameters["crypto_recent_high_lookback_days"])
        if len(rows) <= lookback:
            return None
        price = self.get_last_price(base, quote=self.quote_asset)
        if price is None or not math.isfinite(float(price)) or float(price) <= 0:
            return None
        threshold = float(self.parameters["crypto_dip_threshold_percent"])
        returns: list[float] = []
        for index in range(lookback, len(rows) - 1):
            recent_high = max(high for high, _ in rows[index - lookback : index])
            dip = ((recent_high - rows[index][1]) / recent_high) * 100.0
            if dip >= threshold:
                returns.append(((rows[index + 1][1] - rows[index][1]) / rows[index][1]) * 100.0)
        recent_high = max(high for high, _ in rows[-lookback:])
        current_dip = ((recent_high - float(price)) / recent_high) * 100.0
        configured_round_trip_cost = float(self.parameters.get("crypto_round_trip_cost_percent", 0.50))
        live_spread = self._crypto_live_spread_percent(symbol)
        round_trip_cost = (
            max(configured_round_trip_cost, live_spread) if live_spread is not None else configured_round_trip_cost
        )
        expected_profit: float | None = None
        observations = 0
        oos_expected_profit: float | None = None
        oos_observations = 0
        return_stdev: float | None = None
        win_probability: float | None = None
        if returns:
            net_returns = [value - round_trip_cost for value in returns]
            walk_forward_returns = decision_math.walk_forward_net_returns(
                returns,
                round_trip_cost,
                int(self.parameters.get("crypto_oos_min_observations", 10)),
                float(self.parameters["crypto_min_expected_profit_percent"]),
            )
            mean_net_return = sum(net_returns) / len(net_returns)
            variance = sum((value - mean_net_return) ** 2 for value in net_returns) / len(net_returns)
            wins = sum(1 for value in net_returns if value > 0)
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
            "qualifies": current_dip >= threshold and bool(returns),
            "expected_profit": expected_profit,
            "observations": observations,
            "oos_expected_profit": oos_expected_profit,
            "oos_observations": oos_observations,
            "return_stdev": return_stdev,
            "win_probability": win_probability,
            "round_trip_cost": round_trip_cost,
            "live_spread_percent": live_spread,
        }

    def _crypto_signals(self, symbols: list[str]) -> list[dict[str, Any] | None]:
        if not symbols:
            return []
        with self._unpriceable_symbols_lock:
            self._unpriceable_symbols_this_iteration = set()
        with ThreadPoolExecutor(
            max_workers=min(self._CRYPTO_HISTORY_WORKERS, len(symbols)), thread_name_prefix="crypto-history"
        ) as executor:
            return list(executor.map(self._crypto_signal, symbols))

    # -- Pooled cross-symbol memory (mirrors AssetRotationStrategy's
    # _portfolio_memory/_update_portfolio_memory/_backfill_portfolio_memory
    # in strategy.py, using PortfolioMemory's crypto-appropriate
    # next_session_predicate instead of the NYSE-session-based default) -----

    def _crypto_memory(self) -> PortfolioMemory:
        return PortfolioMemory(
            database_path=Path(str(self.parameters["crypto_memory_database_file"])),
            minimum_observations=int(self.parameters["crypto_memory_min_observations"]),
            maximum_observations=int(self.parameters["crypto_memory_max_observations"]),
            next_session_predicate=is_next_calendar_day,
        )

    def _backfill_crypto_memory(self, symbol: str) -> None:
        """Seed one symbol's pooled memory from settled daily bars, once ever."""
        backfilled = self.vars.crypto_memory_backfilled_symbols
        if symbol in backfilled:
            return
        backfilled.add(symbol)
        if not bool(self.parameters.get("crypto_memory_enabled", True)):
            return
        try:
            bars = self.get_historical_prices(
                self._crypto_asset(symbol),
                int(self.parameters["crypto_analysis_days"]),
                "day",
                quote=self.quote_asset,
            )
            if bars is None or bars.df is None or bars.df.empty or not {"high", "close"}.issubset(bars.df.columns):
                return
            rows = [
                (str(index.date() if hasattr(index, "date") else index), float(row["high"]), float(row["close"]))
                for index, row in bars.df[["high", "close"]].dropna().iterrows()
                if math.isfinite(float(row["high"])) and math.isfinite(float(row["close"]))
                and float(row["high"]) > 0 and float(row["close"]) > 0
            ]
            lookback = int(self.parameters["crypto_recent_high_lookback_days"])
            threshold = float(self.parameters["crypto_dip_threshold_percent"])
            history = []
            for position in range(lookback, len(rows) - 1):
                recent_high = max(row[1] for row in rows[position - lookback : position])
                close = rows[position][2]
                dip = ((recent_high - close) / recent_high) * 100.0
                if dip < threshold:
                    continue
                next_close = rows[position + 1][2]
                next_return = ((next_close - close) / close) * 100.0
                history.append((rows[position][0], dip, next_return))
            inserted = self._crypto_memory().backfill_history(symbol, history)
            if inserted:
                self.log_message(
                    f"Crypto-memory historical backfill added {inserted} settled observations for {symbol}.",
                    color="blue",
                )
        except Exception as exc:
            self.log_message(
                f"Crypto-memory historical backfill failed safely for {symbol}: {type(exc).__name__}: {exc}",
                color="yellow",
            )

    def _update_crypto_memory(
        self,
        symbol: str,
        price: float,
        dip_percent: float,
        signal_present: bool,
        live_spread_percent: float | None = None,
        historical_expected_profit: float | None = None,
        historical_win_probability: float | None = None,
        historical_return_stdev: float | None = None,
    ) -> RotationForecast:
        """Record today's context for one symbol and forecast its next-session return."""
        if not bool(self.parameters.get("crypto_memory_enabled", True)):
            return RotationForecast(0, False, None, None, "Crypto memory is disabled in config.json.")
        try:
            return self._crypto_memory().update_and_forecast(
                evaluation_date=datetime.now(timezone.utc).date().isoformat(),
                symbol=symbol,
                price=price,
                dip_percent=dip_percent,
                # Crypto has no news-scoring layer yet (phase 4+), unlike
                # AssetRotationStrategy's news_score column.
                news_score=None,
                signal_present=signal_present,
                live_spread_percent=live_spread_percent,
                recent_avg_volume=None,
                historical_expected_profit=historical_expected_profit,
                historical_win_probability=historical_win_probability,
                historical_return_stdev=historical_return_stdev,
            )
        except Exception as exc:
            self.log_message(
                f"Crypto memory update failed safely for {symbol}: {type(exc).__name__}: {exc}",
                color="yellow",
            )
            return RotationForecast(0, False, None, None, f"Crypto memory failed: {type(exc).__name__}: {exc}")

    # -- Autonomous discovery (mirrors AssetRotationStrategy's
    # _autonomous_universe/_portfolio_symbols/_managed_portfolio_symbols and
    # the _remember_*/_forget_*/_guarded_universe_call family in strategy.py,
    # using AutonomousUniverse's asset_class="crypto" parameterization and
    # _crypto_asset_symbol_filter instead of the equity defaults) -----------

    def _crypto_autonomous_universe(self) -> AutonomousUniverse:
        return AutonomousUniverse(
            Path(str(self.parameters["crypto_universe_database_file"])),
            int(self.parameters["crypto_discovery_refresh_days"]),
            int(self.parameters["crypto_discovery_batch_size"]),
            paper=os.environ.get("ALPACA_IS_PAPER", "true").strip().lower() != "false",
            asset_class="crypto",
            symbol_filter=_crypto_asset_symbol_filter,
        )

    def _managed_crypto_symbols(self) -> set[str]:
        """Symbols this strategy is permitted to count or sell.

        A shared Alpaca account may contain manual crypto holdings; static
        CRYPTO_SYMBOLS are explicitly opted in, and discovered symbols
        become managed only once a buy fill has confirmed this strategy
        actually owns them.
        """
        symbols = {
            str(symbol).strip().upper()
            for symbol in self.parameters.get("crypto_symbols", [])
            if str(symbol).strip()
        }
        if bool(self.parameters.get("crypto_autonomous_discovery", False)):
            try:
                symbols.update(self._crypto_autonomous_universe().managed_symbols())
            except Exception as exc:
                self.log_message(
                    f"Could not read managed crypto discovery symbols: {type(exc).__name__}: {exc}",
                    color="yellow",
                )
        return symbols

    def _crypto_symbols(self, report: dict[str, Any], held: dict[str, Decimal]) -> list[str]:
        """Combine the static watchlist, current holdings, and one discovery batch."""
        managed = self._managed_crypto_symbols()
        symbols = list(dict.fromkeys(sorted(managed) + sorted(held)))
        if not bool(self.parameters.get("crypto_autonomous_discovery", False)):
            return symbols
        try:
            discovered = self._crypto_autonomous_universe().next_batch(
                os.environ.get("ALPACA_API_KEY", ""),
                os.environ.get("ALPACA_API_SECRET", ""),
            )
            report["discovered_crypto_symbols"] = ", ".join(discovered) or "none"
            return list(dict.fromkeys(symbols + discovered))
        except Exception as exc:
            report["discovery_status"] = f"unavailable: {type(exc).__name__}"
            self.log_message(f"Crypto autonomous discovery failed safely: {type(exc).__name__}: {exc}", color="yellow")
            return symbols

    def _guarded_crypto_universe_call(self, action: Callable[[], None], error_message: str) -> None:
        if not bool(self.parameters.get("crypto_autonomous_discovery", False)):
            return
        try:
            action()
        except Exception as exc:
            self.log_message(f"{error_message}: {type(exc).__name__}: {exc}", color="yellow")

    def _remember_discovered_crypto_symbols(self, symbols: list[str]) -> None:
        self._guarded_crypto_universe_call(
            lambda: self._crypto_autonomous_universe().remember(symbols),
            "Could not persist learned crypto symbols",
        )

    def _exclude_unpriceable_discovered_crypto_symbols(self, symbols: list[str]) -> None:
        self._guarded_crypto_universe_call(
            lambda: self._crypto_autonomous_universe().exclude_unpriceable(symbols),
            "Could not persist unpriceable crypto symbols",
        )

    def _remember_confirmed_crypto_symbol(self, symbol: str) -> None:
        self._guarded_crypto_universe_call(
            lambda: self._crypto_autonomous_universe().remember_owned([str(symbol).upper()]),
            "Could not persist crypto strategy ownership",
        )

    def _forget_confirmed_crypto_symbol(self, symbol: str) -> None:
        self._guarded_crypto_universe_call(
            lambda: self._crypto_autonomous_universe().forget_owned([str(symbol).upper()]),
            "Could not revoke crypto strategy ownership",
        )

    # -- Opportunistic Opportunity: a data-backed, at-most-once-per-day
    # CRYPTO_ASSET_A -> CRYPTO_ASSET_B swap. Mirrors
    # AssetRotationStrategy's _opportunistic_opportunity/
    # _backfill_decision_memory/_update_decision_memory (strategy.py), using
    # the same .crypto_trade_memory.duckdb TradeMemory instance execution
    # journaling already writes to (phase 3), with is_next_calendar_day
    # instead of TradeMemory's NYSE-session default. -----------------------

    def _crypto_decision_memory(self, maximum_observations: int = 1) -> TradeMemory:
        return TradeMemory(
            Path(str(self.parameters["crypto_trade_memory_database_file"])),
            1,
            maximum_observations,
            next_session_predicate=is_next_calendar_day,
        )

    def _backfill_crypto_decision_memory(self, asset_a: str, asset_b: str) -> None:
        """Seed crypto decision memory from settled daily bars, once per process start."""
        if self.vars.crypto_decision_memory_backfill_attempted:
            return
        self.vars.crypto_decision_memory_backfill_attempted = True
        try:
            bars_a = self.get_historical_prices(
                self._crypto_asset(asset_a), int(self.parameters["crypto_analysis_days"]), "day", quote=self.quote_asset
            )
            bars_b = self.get_historical_prices(
                self._crypto_asset(asset_b), int(self.parameters["crypto_analysis_days"]), "day", quote=self.quote_asset
            )
            if (
                bars_a is None or bars_b is None or bars_a.df is None or bars_b.df is None
                or bars_a.df.empty or bars_b.df.empty
                or not {"close"}.issubset(bars_a.df.columns)
                or not {"close", "high"}.issubset(bars_b.df.columns)
            ):
                self.log_message("Crypto decision-memory historical backfill unavailable; continuing normally.", color="yellow")
                return
            a_closes = {
                str(index.date() if hasattr(index, "date") else index): float(value)
                for index, value in bars_a.df["close"].dropna().items()
                if math.isfinite(float(value)) and float(value) > 0
            }
            b_rows = [
                (str(index.date() if hasattr(index, "date") else index), float(row["close"]), float(row["high"]))
                for index, row in bars_b.df[["close", "high"]].dropna().iterrows()
                if math.isfinite(float(row["close"])) and math.isfinite(float(row["high"]))
                and float(row["close"]) > 0 and float(row["high"]) > 0
            ]
            lookback = int(self.parameters["crypto_recent_high_lookback_days"])
            threshold = float(self.parameters["crypto_dip_threshold_percent"])
            history = []
            for position, (date, close_b, high_b) in enumerate(b_rows):
                if position < lookback:
                    continue
                close_a = a_closes.get(date)
                if close_a is None:
                    continue
                recent_high = max(row[2] for row in b_rows[position - lookback : position])
                dip = ((recent_high - close_b) / recent_high) * 100.0
                history.append((date, close_a, close_b, dip, dip >= threshold))
            inserted = self._crypto_decision_memory(
                int(self.parameters.get("crypto_memory_max_observations", 500))
            ).backfill_history(history)
            self.log_message(
                f"Crypto decision-memory historical backfill added {inserted} settled daily observations.",
                color="blue",
            )
        except Exception as exc:
            self.log_message(
                f"Crypto decision-memory historical backfill failed safely: {type(exc).__name__}: {exc}",
                color="yellow",
            )

    def _update_crypto_decision_memory(self, price_a: float, price_b: float, dip_percent: float) -> RotationForecast:
        try:
            memory = self._crypto_decision_memory(int(self.parameters.get("crypto_memory_max_observations", 500)))
            result = memory.update_and_forecast(
                evaluation_date=datetime.now(timezone.utc).date().isoformat(),
                price_a=price_a,
                price_b=price_b,
                dip_percent=dip_percent,
                # Crypto has no news-scoring layer yet, unlike
                # AssetRotationStrategy's news_score column.
                news_score=None,
                signal_present=dip_percent >= float(self.parameters["crypto_dip_threshold_percent"]),
            )
            self.log_message(result.explanation, color="blue")
            return result
        except Exception as exc:
            self.log_message(f"Crypto decision memory failed safely: {type(exc).__name__}: {exc}", color="red")
            return RotationForecast(0, False, None, None, f"Crypto decision memory failed: {type(exc).__name__}: {exc}")

    def _crypto_opportunistic_opportunity(
        self, asset_a: str, asset_b: str, price_a: float | None, price_b: float | None
    ) -> dict[str, float | int | str | None]:
        unavailable: dict[str, float | int | str | None] = {"status": "unavailable", "probability": None}
        if price_a is None or price_b is None or min(float(price_a), float(price_b)) <= 0:
            return unavailable
        bars = self.get_historical_prices(
            self._crypto_asset(asset_b),
            int(self.parameters["crypto_recent_high_lookback_days"]),
            "day",
            quote=self.quote_asset,
        )
        if bars is None or bars.df is None or bars.df.empty or "high" not in bars.df:
            return unavailable
        highs = [float(value) for value in bars.df["high"].dropna() if math.isfinite(float(value)) and float(value) > 0]
        if not highs:
            return unavailable
        recent_high = max(highs)
        dip = ((recent_high - float(price_b)) / recent_high) * 100.0
        self._backfill_crypto_decision_memory(asset_a, asset_b)
        forecast = self._update_crypto_decision_memory(float(price_a), float(price_b), dip)
        try:
            probability = self._crypto_decision_memory().opportunity_probability()
        except Exception as exc:
            self.log_message(f"Crypto opportunity probability lookup failed safely: {type(exc).__name__}: {exc}", color="red")
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

    def _submit_crypto_rotation_sell(self, source: str, target: str, quantity: Decimal, budget: float) -> bool:
        """Persist rotation intent before exposing its sell to the broker."""
        if self.vars.crypto_pending_rotation is not None:
            self.log_message(
                f"Refused to stage {source} to {target}: a crypto rotation is already in flight.",
                color="red",
            )
            return False
        self._set_crypto_rotation({"from": source, "to": target, "budget": budget})
        order = self.create_order(
            self._crypto_asset(source),
            quantity,
            "sell",
            quote=self.quote_asset,
            order_type="market",
            time_in_force="gtc",
        )
        try:
            accepted = self._submit_order_checked(order, f"{source} sell for opportunistic crypto swap")
        except Exception:
            self._set_crypto_rotation(None)
            raise
        if not accepted:
            self._set_crypto_rotation(None)
            return False
        return True

    def _reconcile_pending_crypto_rotation(self, held: dict[str, Decimal]) -> tuple[list[str], set[str]]:
        """Reconcile a restart-safe sale/buy pair before new decisions.

        Mirrors AssetRotationStrategy._reconcile_pending_portfolio_rotations
        (strategy.py), scoped to the single pending entry crypto ever has.
        """
        entry = self.vars.crypto_pending_rotation
        if entry is None:
            return [], set()
        source = str(entry["from"])
        target = str(entry["to"])
        budget = float(entry["budget"])
        if held.get(source, Decimal("0")) > 0:
            if self._has_active_order(source, "sell"):
                return [f"Crypto rotation pending: waiting for {source} sale"], {source, target}
            self._set_crypto_rotation(None)
            return [f"Crypto rotation reset: {source} sale did not fill"], set()
        if held.get(target, Decimal("0")) > 0 and not self._has_active_order(target, "buy"):
            self._set_crypto_rotation(None)
            return [f"Crypto rotation complete: the {target} purchase filled"], set()
        price = self.get_last_price(self._crypto_asset(target), quote=self.quote_asset)
        if price is None or not math.isfinite(float(price)) or float(price) <= 0:
            return [f"Crypto rotation pending: no valid {target} price"], {source, target}
        outcome = self._buy_crypto_symbol(target, float(price), budget)
        if outcome == "insufficient":
            self._set_crypto_rotation(None)
            return [f"Crypto rotation finished: cash is below the minimum {target} order"], set()
        if outcome == "working":
            return [f"Crypto rotation pending: waiting for the {target} purchase to fill"], {source, target}
        if outcome == "rejected":
            return [f"Crypto rotation pending: broker rejected the {target} purchase; retrying next cycle"], {source, target}
        return [f"Crypto {target} purchase submitted after {source} sale"], {source, target}

    # -- Buying --------------------------------------------------------------

    _ACCOUNT_VALUE_CACHE_SECONDS = 30.0

    def _account_half_value_dollars(self) -> float:
        """Half of the shared Alpaca account's total value (cash + net equity).

        Conceptually mirrors AssetRotationStrategy._crypto_reserve_dollars
        (strategy.py): equity and crypto run as separate processes against
        the same account and each independently targets a 50/50 split of
        it, rather than a fixed configured dollar figure, so the two sides
        converge on the same split without any direct coordination between
        the two processes. Unlike equity's reserve, this always halves --
        there is no "is the other side enabled" check here, since portfolio
        (equity) mode is not optional and always runs. Falls back to cash
        alone if a fresh broker equity read fails, and treats a non-finite
        (NaN/inf) or negative reading as zero available value -- all three
        failure modes can only push this pipeline's share more
        conservative, never let it overspend. Cached briefly
        (_ACCOUNT_VALUE_CACHE_SECONDS) since Lumibot's
        get_portfolio_value()/get_cash() each force their own fresh broker
        round-trip on every call with no caching between them.
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
        half_value = total_value * 0.5
        self._account_value_cache = (now, half_value)
        return half_value

    def _buy_crypto_symbol(self, symbol: str, price: float, budget: float) -> str:
        """Buy a fractional quantity within a stated crypto budget.

        `budget` already reflects the crypto cash allocation ceiling (see
        _run_crypto_iteration); min(get_cash(), budget) additionally ensures
        this never spends more than the account's actual, real-time cash --
        the same soft, per-call safeguard AssetRotationStrategy's
        _buy_portfolio_symbol (strategy.py) uses for its own budget.
        """
        with self._rotation_lock:
            if self._has_active_order(symbol, "buy"):
                return "working"
            spendable = min(float(self.get_cash()), budget) * (1.0 - self.CASH_BUFFER_FRACTION)
            if spendable < float(self.parameters.get("crypto_min_order_dollars", 5.0)):
                return "insufficient"
            quantity = (Decimal(str(spendable)) / Decimal(str(price))).quantize(
                Decimal("1.00000000"), rounding=ROUND_DOWN
            )
            if quantity <= 0:
                return "insufficient"
            buy_order = self.create_order(
                self._crypto_asset(symbol),
                quantity,
                "buy",
                quote=self.quote_asset,
                order_type="market",
                time_in_force="gtc",
            )
            if not self._submit_order_checked(buy_order, f"{symbol} crypto buy"):
                return "rejected"
            self.log_message(
                f"Crypto submitted buy of {quantity} {symbol} using up to ${budget:.2f}.", color="green"
            )
            return "submitted"

    # -- Iteration -------------------------------------------------------

    def _run_crypto_iteration(self, report: dict[str, Any]) -> None:
        actions: list[str] = []
        report["crypto_actions"] = actions
        managed = self._managed_crypto_symbols()
        if not managed:
            report["status"] = "No crypto symbols configured"
            return
        result = self._crypto_held_positions(managed)
        if result is None:
            report["status"] = "Could not read account positions"
            return
        held, entry_prices = result
        today = datetime.now(timezone.utc).date()

        rotation_actions, claimed_symbols = self._reconcile_pending_crypto_rotation(held)
        actions.extend(rotation_actions)

        holding_dates = dict(self.vars.crypto_holding_dates)
        dates_changed = False
        for symbol in held:
            if symbol not in holding_dates:
                holding_dates[symbol] = today.isoformat()
                dates_changed = True
        for symbol in list(holding_dates):
            if symbol not in held:
                holding_dates.pop(symbol)
                dates_changed = True
        if dates_changed:
            self._set_crypto_holding_dates(holding_dates)

        held_working = dict(held)
        # Symbols exited this pass are excluded from the build phase below --
        # without this, a take-profit sale whose signal still reads
        # "qualifies" (e.g. on a noisy print) could get bought straight back
        # in the same iteration. Mirrors AssetRotationStrategy's
        # claimed_symbols guard in _submit_due_portfolio_exits (strategy.py).
        exited_this_pass: set[str] = set()
        exit_reasons = self._crypto_exit_reasons(held, entry_prices, holding_dates, today)
        for source in sorted(exit_reasons):
            if source in claimed_symbols or self._has_active_order(source, "sell"):
                continue
            reason = exit_reasons[source]
            exit_order = self.create_order(
                self._crypto_asset(source),
                held[source],
                "sell",
                quote=self.quote_asset,
                order_type="market",
                time_in_force="gtc",
            )
            if self._submit_order_checked(exit_order, f"{source} crypto exit sell ({reason})"):
                self.log_message(f"Crypto exit submitted: {source} {reason}", color="yellow")
                actions.append(f"Crypto exit submitted: {source} {reason}")
                held_working.pop(source, None)
                exited_this_pass.add(source)

        symbols = self._crypto_symbols(report, held)
        signals = self._crypto_signals(symbols)
        # Only a discovery-sourced candidate is safe to permanently exclude --
        # a config-listed CRYPTO_SYMBOLS entry hitting a transient data outage
        # must stay eligible for re-evaluation, not be blacklisted. Mirrors
        # AssetRotationStrategy's discovery_only_symbols/unpriceable_discovered
        # handling in _run_portfolio_iteration (strategy.py).
        discovery_only_symbols = set(symbols) - managed - set(held)
        unpriceable_discovered = sorted(self._unpriceable_symbols_this_iteration & discovery_only_symbols)
        if unpriceable_discovered:
            self._exclude_unpriceable_discovered_crypto_symbols(unpriceable_discovered)
        signals_by_symbol = {str(signal["symbol"]): signal for signal in signals if signal is not None}
        # Persist only positions the strategy actually owns -- merely
        # qualifying a discovered pair must not grant permission to manage a
        # manual account position in that symbol. New buys are remembered on
        # fill (_remember_confirmed_crypto_symbol).
        self._remember_discovered_crypto_symbols(sorted(held))

        posture = str(self.parameters.get("crypto_risk_posture", "conservative"))
        report["crypto_risk_posture"] = posture
        min_profit = float(self.parameters.get("crypto_min_expected_profit_percent", 1.0))
        min_observations = int(self.parameters.get("crypto_min_signal_observations", 20))
        oos_min_observations = int(self.parameters.get("crypto_oos_min_observations", 10))
        oos_min_profit = float(self.parameters.get("crypto_oos_min_net_profit_percent", 0.0))
        eligible: list[dict[str, Any]] = []
        # Every evaluated symbol -- not just one clearing today's dip
        # threshold -- contributes a daily learning observation to the pooled
        # memory, mirroring AssetRotationStrategy's _run_portfolio_iteration.
        for symbol, signal in signals_by_symbol.items():
            self._backfill_crypto_memory(symbol)
            forecast = self._update_crypto_memory(
                symbol,
                float(signal["price"]),
                float(signal["dip"]),
                signal_present=bool(signal.get("qualifies")),
                live_spread_percent=signal.get("live_spread_percent"),
                historical_expected_profit=signal.get("expected_profit"),
                historical_win_probability=signal.get("win_probability"),
                historical_return_stdev=signal.get("return_stdev"),
            )
            if not bool(signal.get("qualifies")) or symbol in held_working or symbol in exited_this_pass:
                signal["learned_edge_ready"] = False
                signal["learned_edge"] = None
                continue
            if signal.get("expected_profit") is None or float(signal["expected_profit"]) < min_profit:
                continue
            if int(signal.get("observations", 0)) < min_observations:
                continue
            if int(signal.get("oos_observations", 0)) < oos_min_observations:
                continue
            if signal.get("oos_expected_profit") is None or float(signal["oos_expected_profit"]) < oos_min_profit:
                continue
            signal["learned_edge_ready"] = forecast.ready
            signal["learned_edge"] = (
                forecast.predicted_edge_percent - float(signal["round_trip_cost"])
                if forecast.ready and forecast.predicted_edge_percent is not None
                else None
            )
            signal["posture_adjusted_edge"] = decision_math.posture_adjusted_edge(signal, posture, None)
            eligible.append(signal)
        eligible.sort(key=lambda signal: float(signal["posture_adjusted_edge"]), reverse=True)
        report["crypto_candidates"] = len(eligible)
        report["crypto_holdings"] = ", ".join(sorted(held_working)) or "none"
        signal_snapshot.write_snapshot(
            str(self.parameters.get("crypto_signal_snapshot_file", "")),
            datetime.now(timezone.utc).isoformat(),
            posture,
            signal_snapshot.build_snapshot_entries(signals_by_symbol.values(), held),
        )

        # Opportunistic Opportunity: evaluated exactly once, as a single
        # non-looped decision, reserving both legs via claimed_symbols so it
        # never competes with the up-to-max-positions build loop below.
        # Mirrors AssetRotationStrategy's equivalent block in
        # _run_portfolio_iteration (strategy.py); "at most one swap per day"
        # is enforced by crypto_opportunistic_swap_date, persisted in
        # .crypto_opportunistic_swap_state.json so it survives a restart.
        asset_a = str(self.parameters.get("crypto_asset_a", "BTC")).upper()
        asset_b = str(self.parameters.get("crypto_asset_b", "ETH")).upper()
        price_a = self.get_last_price(self._crypto_asset(asset_a), quote=self.quote_asset)
        price_b = self.get_last_price(self._crypto_asset(asset_b), quote=self.quote_asset)
        opportunity = self._crypto_opportunistic_opportunity(asset_a, asset_b, price_a, price_b)
        opportunity_probability = opportunity.get("probability")
        opportunity_edge = opportunity.get("predicted_edge")
        report["crypto_opportunistic_status"] = opportunity.get("status")
        report["crypto_opportunistic_probability"] = (
            f"{float(opportunity_probability):.1%}" if opportunity_probability is not None else "not ready"
        )
        report["crypto_opportunistic_explanation"] = opportunity.get("forecast_explanation", "unavailable")

        swap_already_done_today = self.vars.crypto_opportunistic_swap_date == today.isoformat()
        opportunity_is_eligible = (
            asset_a in held_working
            and asset_b not in held_working
            and asset_a not in claimed_symbols
            and asset_b not in claimed_symbols
            and opportunity.get("status") == "ready"
            and float(opportunity.get("dip") or 0.0) >= float(self.parameters["crypto_dip_threshold_percent"])
            and opportunity_probability is not None
            and float(opportunity_probability) >= float(self.parameters.get("crypto_opportunistic_min_probability", 0.55))
            and opportunity_edge is not None
            and float(opportunity_edge) >= min_profit
            and not swap_already_done_today
        )
        if opportunity_is_eligible:
            if self._has_active_order(asset_a, "sell"):
                actions.append("Crypto Opportunistic Opportunity pending: waiting for Asset A sale")
            elif price_a is None or float(price_a) <= 0:
                actions.append("No crypto Opportunistic Opportunity: Asset A price was unavailable")
            else:
                budget = float(price_a) * float(held_working[asset_a])
                if self._submit_crypto_rotation_sell(asset_a, asset_b, held_working[asset_a], budget):
                    self._mark_crypto_opportunistic_swap_done(today)
                    held_working.pop(asset_a, None)
                    claimed_symbols.update({asset_a, asset_b})
                    actions.append(
                        f"Crypto Opportunistic Opportunity submitted: {asset_a} to {asset_b} "
                        f"({float(opportunity_probability):.1%} historical win probability, "
                        f"predicted edge {float(opportunity_edge):+.2f}%)"
                    )

        configured_max_positions = int(self.parameters.get("crypto_max_positions", 1))
        candidate_edges = [
            (float(signal["posture_adjusted_edge"]), float(signal.get("return_stdev") or 0.0))
            for signal in eligible
        ]
        crypto_allocation = self._account_half_value_dollars()
        report["crypto_cash_allocation_dollars"] = crypto_allocation
        min_order_dollars = float(self.parameters.get("crypto_min_order_dollars", 5.0))
        effective_max_positions = decision_math.optimal_position_count(
            crypto_allocation, min_order_dollars, candidate_edges, configured_max_positions
        )
        report["crypto_effective_max_positions"] = effective_max_positions

        deployed = self._crypto_deployed_dollars(held_working, signals_by_symbol)
        report["crypto_deployed_dollars"] = f"${deployed:.2f}"
        remaining_budget = max(0.0, crypto_allocation - deployed)
        submitted = 0
        for candidate in eligible:
            symbol = str(candidate["symbol"])
            if symbol in claimed_symbols:
                continue
            if len(held_working) + submitted >= effective_max_positions:
                break
            slots_remaining = max(1, effective_max_positions - (len(held_working) + submitted))
            budget = remaining_budget / slots_remaining
            outcome = self._buy_crypto_symbol(symbol, float(candidate["price"]), budget)
            if outcome == "insufficient":
                break
            if outcome == "rejected" or outcome == "working":
                continue
            remaining_budget = max(0.0, remaining_budget - budget)
            submitted += 1
            self.log_message(f"Crypto build: {symbol} purchase {outcome}", color="green")
            actions.append(f"Crypto build: {symbol} purchase {outcome}")

        report["status"] = f"Evaluation complete ({len(actions)} action(s))" if actions else "No action needed"

    def on_trading_iteration(self) -> None:
        """Polled every sleeptime tick, 24/7 (self.set_market("24/7") means
        Lumibot's own scheduler never blocks waiting for a market open/close
        that will never come). No-ops while NYSE is open or CRYPTO_ENABLED is
        false; otherwise runs the full crypto dip-signal pass every tick --
        unlike the equity pipeline's twice-a-day window, there is no need to
        throttle further here since crypto_iteration_interval_minutes (via
        self.sleeptime) already sets the desired evaluation cadence.
        """
        if not bool(self.parameters.get("crypto_enabled", False)):
            return
        market_open = nyse_is_open(datetime.now(timezone.utc))
        if market_open != self._last_logged_nyse_open:
            self._last_logged_nyse_open = market_open
            if market_open:
                self.log_message("NYSE is open; crypto trading paused until it closes.", color="blue")
            else:
                self.log_message("NYSE is closed; crypto trading resumed.", color="blue")
        if market_open:
            return
        report: dict[str, Any] = {
            "threshold": float(self.parameters.get("crypto_dip_threshold_percent", 5.0)),
            "status": "Evaluation started",
        }
        try:
            self._run_crypto_iteration(report)
        except Exception as exc:
            report["status"] = f"Evaluation error: {type(exc).__name__}: {exc}"
            self.log_message(f"Crypto iteration failed safely: {type(exc).__name__}: {exc}", color="red")
        finally:
            self._send_crypto_email(report)

    # -- Email report (uses email_render.py's shared HTML helpers; mirrors
    # AssetRotationStrategy's _send_daily_email/_render_email_html in
    # strategy.py at a smaller scale -- no news/LLM/discovery sections since
    # crypto doesn't have those yet) -----------------------------------------

    @staticmethod
    def _crypto_cash_allocation_display(report: dict[str, Any]) -> str:
        """Formats the dynamic cash allocation for the email, matching the
        'unavailable' convention every other report field in this email
        already uses -- crypto_cash_allocation_dollars is only set partway
        through _run_crypto_iteration, so an early return (no symbols
        configured, positions unavailable) or a caught exception leaves it
        absent from report; defaulting to 0.0 there would render a
        misleading '$0.00' instead of disclosing the value is unknown."""
        allocation = report.get("crypto_cash_allocation_dollars")
        return f"${float(allocation):.2f}" if allocation is not None else "unavailable"

    def _send_crypto_email(self, report: dict[str, Any]) -> None:
        """Send at most one crypto summary email per calendar day."""
        if not bool(self.parameters.get("crypto_email_report_enabled", False)):
            return
        try:
            report_date = datetime.now(timezone.utc).date().isoformat()
            state_file = Path(str(self.parameters["crypto_email_state_file"]))
            if state_file.exists() and state_file.read_text(encoding="utf-8").strip() == report_date:
                return

            message = EmailMessage()
            message["Subject"] = f"Trading Agent Crypto Report - {report_date} - {report.get('status', 'unknown')}"
            message["From"] = str(self.parameters["email_from_address"])
            message["To"] = str(self.parameters["email_to_address"])
            lines = [
                "Raspberry Pi Trading Agent Crypto Summary",
                "",
                f"Date: {report_date}",
                "Mode: crypto (active only while NYSE is closed)",
                f"Risk posture: {report.get('crypto_risk_posture', 'unavailable')}",
                f"Holdings: {report.get('crypto_holdings', 'unavailable')}",
                f"Signal candidates: {report.get('crypto_candidates', 'unavailable')}",
                f"Effective max positions today: {report.get('crypto_effective_max_positions', 'unavailable')} "
                f"(configured ceiling {self.parameters.get('crypto_max_positions', 'unavailable')})",
                f"Cash allocation: {self._crypto_cash_allocation_display(report)}",
                f"Deployed: {report.get('crypto_deployed_dollars', 'unavailable')}",
                f"Discovered symbols: {report.get('discovered_crypto_symbols', 'none')}",
                f"Discovery status: {report.get('discovery_status', 'ok')}",
                f"Dip threshold: {float(report.get('threshold', 0.0)):.2f}%",
                f"Opportunistic Opportunity: {report.get('crypto_opportunistic_status', 'unavailable')}",
                f"Opportunistic Opportunity probability: {report.get('crypto_opportunistic_probability', 'unavailable')}",
                f"Opportunistic Opportunity evidence: {report.get('crypto_opportunistic_explanation', 'unavailable')}",
                "Crypto actions this iteration:",
                *[f"- {action}" for action in report.get("crypto_actions", [])],
                f"Result: {report.get('status', 'unknown')}",
                "",
                "Review all orders and positions in the Alpaca dashboard.",
                "This automated message is not financial advice.",
            ]
            message.set_content("\n".join(lines))
            message.add_alternative(self._render_crypto_email_html(report, report_date), subtype="html")

            host = str(self.parameters["email_smtp_host"])
            port = int(self.parameters["email_smtp_port"])
            password = os.environ.get("EMAIL_SMTP_PASSWORD") or str(self.parameters.get("email_smtp_password", ""))
            with smtplib.SMTP(host, port, timeout=15) as smtp:
                smtp.ehlo()
                if bool(self.parameters["email_use_tls"]):
                    smtp.starttls(context=ssl.create_default_context())
                    smtp.ehlo()
                smtp.login(str(self.parameters["email_smtp_username"]), password)
                smtp.send_message(message)

            state_file.write_text(report_date + "\n", encoding="utf-8")
            self.log_message(f"Crypto daily email report sent for {report_date}.", color="green")
        except Exception as exc:
            self.log_message(f"Crypto daily email report failed safely: {type(exc).__name__}: {exc}", color="red")

    def _render_crypto_email_html(self, report: dict[str, Any], report_date: str) -> str:
        status = str(report.get("status", "unknown"))
        snapshot_rows = [
            ("Risk posture", report.get("crypto_risk_posture", "unavailable")),
            ("Holdings", report.get("crypto_holdings", "unavailable")),
            ("Signal candidates", report.get("crypto_candidates", "unavailable")),
            (
                "Effective max positions today",
                f"{report.get('crypto_effective_max_positions', 'unavailable')} "
                f"(configured ceiling {self.parameters.get('crypto_max_positions', 'unavailable')})",
            ),
            ("Cash allocation", self._crypto_cash_allocation_display(report)),
            ("Deployed", report.get("crypto_deployed_dollars", "unavailable")),
            ("Discovered symbols", report.get("discovered_crypto_symbols", "none")),
            ("Discovery status", report.get("discovery_status", "ok")),
            ("Dip threshold", f"{float(report.get('threshold', 0.0)):.2f}%"),
            ("Opportunistic Opportunity", report.get("crypto_opportunistic_status", "unavailable")),
            ("Opportunistic Opportunity probability", report.get("crypto_opportunistic_probability", "unavailable")),
            ("Opportunistic Opportunity evidence", report.get("crypto_opportunistic_explanation", "unavailable")),
        ]
        sections = "".join(
            [
                email_render.email_kv_section("Snapshot", snapshot_rows),
                email_render.email_bullet_section("Crypto actions", report.get("crypto_actions", [])),
            ]
        )
        return email_render.render_email_shell(
            report_date=report_date,
            mode_label="Crypto mode",
            status=status,
            narrative="",
            sections_html=sections,
        )

    def on_filled_order(self, position: Any, order: Any, price: float, quantity: float, multiplier: float) -> None:
        symbol = str(getattr(getattr(order, "asset", None), "symbol", "unknown")).upper()
        side = str(getattr(order, "side", "unknown")).lower()
        total_quantity = getattr(order, "quantity", None) or quantity
        fill_price = getattr(order, "get_fill_price", lambda: None)() or price
        self.log_message(f"Filled crypto {side} order: {total_quantity} {symbol} at ${fill_price}.", color="green")
        try:
            TradeMemory(
                Path(str(self.parameters["crypto_trade_memory_database_file"])), 1, 1
            ).record_execution(
                datetime.now(timezone.utc).date().isoformat(),
                symbol,
                side,
                float(fill_price),
                float(total_quantity),
            )
        except Exception as exc:
            self.log_message(f"Could not journal crypto execution: {type(exc).__name__}: {exc}", color="red")
        # Continue an in-flight Opportunistic Opportunity rotation immediately
        # after Alpaca confirms the fill, instead of waiting up to
        # crypto_iteration_interval_minutes for the next iteration. Mirrors
        # AssetRotationStrategy's on_filled_order rotation handling
        # (strategy.py).
        pending = self.vars.crypto_pending_rotation
        if side == "buy":
            self._record_crypto_entry(symbol)
            self._remember_confirmed_crypto_symbol(symbol)
            if pending is not None and str(pending["to"]) == symbol:
                self._set_crypto_rotation(None)
                self.log_message(f"Crypto rotation complete: the {symbol} purchase filled.", color="green")
        elif side == "sell":
            self._remove_crypto_entry(symbol)
            self._forget_confirmed_crypto_symbol(symbol)
            if pending is not None and str(pending["from"]) == symbol:
                target = str(pending["to"])
                try:
                    target_price = self.get_last_price(self._crypto_asset(target), quote=self.quote_asset)
                    if target_price is None or not math.isfinite(float(target_price)) or float(target_price) <= 0:
                        self.log_message(
                            f"The {symbol} sale filled, but {target} has no valid price; "
                            "the purchase will be retried next cycle.",
                            color="yellow",
                        )
                    else:
                        outcome = self._buy_crypto_symbol(target, float(target_price), float(pending["budget"]))
                        if outcome == "insufficient":
                            self.log_message(
                                f"The {target} purchase will be retried next cycle in case "
                                "the sale proceeds have not settled yet.",
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
                        f"Crypto post-sale purchase failed safely and will be retried: "
                        f"{type(exc).__name__}: {exc}",
                        color="red",
                    )

    def on_canceled_order(self, order: Any) -> None:
        symbol = str(getattr(getattr(order, "asset", None), "symbol", "unknown")).upper()
        side = str(getattr(order, "side", "unknown")).lower()
        self.log_message(f"Crypto order canceled or rejected by the broker: {side} {symbol}.", color="red")

        pending = self.vars.crypto_pending_rotation
        if side == "sell" and pending is not None and str(pending["from"]) == symbol:
            self._set_crypto_rotation(None)
            self.log_message(
                f"The {symbol} sale was canceled; the crypto rotation is reset and will be re-evaluated next cycle.",
                color="yellow",
            )
