"""Shared, asset-agnostic mechanics for the equity and crypto strategies.

The trading strategies deliberately keep market-specific decision logic in
their own modules. This mixin contains only broker/runtime behavior whose
semantics are identical for both processes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

from portfolio_memory import PortfolioMemory, PortfolioMemoryInput
from runtime_state import DuckDBStateStore
from trade_memory import RotationForecast


@dataclass(slots=True)
class IterationContext:
    """Complete decision context produced before ranking or order execution."""

    report: dict[str, Any]
    managed_symbols: set[str]
    held: dict[str, Decimal]
    entry_prices: dict[str, float]
    symbols: list[str]
    news_context: Any
    llm_assessment: Any
    symbol_news_scores: dict[str, int]
    veto_reason: str | None = None
    signals: list[dict[str, Any]] = field(default_factory=list)


def build_memory_inputs(
    signals: list[dict[str, Any]],
    llm_score: int | None,
    *,
    include_recent_volume: bool = False,
) -> list[PortfolioMemoryInput]:
    """Project evaluated signals into the shared pooled-memory schema."""
    return [
        PortfolioMemoryInput(
            symbol=str(signal["symbol"]),
            price=float(signal["price"]),
            dip_percent=float(signal["dip"]),
            llm_score=llm_score,
            signal_present=bool(signal.get("qualifies")),
            live_spread_percent=signal.get("live_spread_percent"),
            recent_avg_volume=(
                signal.get("recent_avg_volume") if include_recent_volume else None
            ),
            historical_expected_profit=signal.get("expected_profit"),
            historical_win_probability=signal.get("win_probability"),
            historical_return_stdev=signal.get("return_stdev"),
        )
        for signal in signals
    ]


def update_memory_forecasts(
    *,
    signals: list[dict[str, Any]],
    evaluation_date: str,
    llm_score: int | None,
    enabled: bool,
    memory_factory: Callable[[], PortfolioMemory],
    disabled_explanation: str,
    failure_label: str,
    log_message: Callable[..., Any],
    include_recent_volume: bool = False,
) -> dict[str, RotationForecast]:
    """Batch one asset pipeline's observations and fail open consistently."""
    if not signals:
        return {}
    if not enabled:
        disabled = RotationForecast(0, False, None, None, disabled_explanation)
        return {str(signal["symbol"]): disabled for signal in signals}
    inputs = build_memory_inputs(
        signals, llm_score, include_recent_volume=include_recent_volume
    )
    try:
        return memory_factory().update_many_and_forecast(evaluation_date, inputs)
    except Exception as exc:
        log_message(
            f"{failure_label} batch update failed safely: {type(exc).__name__}: {exc}",
            color="yellow",
        )
        failed = RotationForecast(
            0,
            False,
            None,
            None,
            f"{failure_label} failed: {type(exc).__name__}: {exc}",
        )
        return {item.symbol: failed for item in inputs}


class BrokerRuntimeSupport:
    """Common order, position, and restart-safe persistence helpers."""

    _RUNTIME_STATE_DATABASE_PARAMETER = "runtime_state_database_file"
    _DELETE_EMPTY_ONLY_WHEN_NONE = False

    def _runtime_state(self) -> DuckDBStateStore | None:
        raw = self.parameters.get(self._RUNTIME_STATE_DATABASE_PARAMETER)
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
            # Explicit empty values are tombstones: deleting the key would
            # allow a stale legacy file to be imported on the next restart.
            store.set(key, value)
            return
        if legacy_path is None:
            return
        empty = value is None if self._DELETE_EMPTY_ONLY_WHEN_NONE else not value
        if delete_empty and empty:
            legacy_path.unlink(missing_ok=True)
            return
        temporary_path = legacy_path.with_suffix(legacy_path.suffix + ".tmp")
        serialized = str(value) if plain_text else json.dumps(value, sort_keys=True)
        temporary_path.write_text(serialized + "\n", encoding="utf-8")
        temporary_path.replace(legacy_path)

    @staticmethod
    def _valid_iso_date(value: str) -> bool:
        try:
            date.fromisoformat(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def _quantity(position: Any) -> Decimal:
        """Return a safe, non-negative quantity for a broker position."""
        if position is None:
            return Decimal("0")
        try:
            return max(Decimal(str(position.quantity)), Decimal("0"))
        except (AttributeError, InvalidOperation, TypeError, ValueError):
            return Decimal("0")

    def _cached_orders(self) -> list[Any]:
        """Fetch the broker order book at most once per iteration."""
        cached = getattr(self, "_orders_cache", None)
        if cached is None:
            cached = self.get_orders() or []
            self._orders_cache = cached
        return cached

    def _invalidate_orders_cache(self) -> None:
        self._orders_cache = None

    def _has_active_order(self, symbol: str, side: str) -> bool:
        """Best-effort check; an unreadable order book fails safely as active."""
        try:
            orders = self._cached_orders()
        except Exception as exc:
            self.log_message(
                f"Could not read orders ({type(exc).__name__}: {exc}); "
                "assuming one may still be working.",
                color="yellow",
            )
            return True
        normalized_side = side.lower()
        for order in orders:
            order_symbol = getattr(getattr(order, "asset", None), "symbol", None)
            order_side = str(getattr(order, "side", "")).lower()
            if order_symbol != symbol or order_side != normalized_side:
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
        """Submit an order and detect Lumibot's non-raising error results."""
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
        self._invalidate_orders_cache()
        return True
