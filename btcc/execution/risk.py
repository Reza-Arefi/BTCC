"""Execution risk gate — infrastructure checks separate from strategy HALT.

PAPER mode may use paper assumptions.
REAL mode fails closed when exchange information is unknown/unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from btcc.execution.account import AccountSnapshot
from btcc.execution.intent import DurableTradeIntent
from btcc.execution.kill_switch import ExecutionKillSwitch
from btcc.execution.modes import ExecutionMode
from btcc.execution.order_state import OrderJournal
from btcc.execution.symbols import SymbolMeta, floor_quantity_for_symbol
from btcc.safety.no_trading import TradingForbiddenError


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskContext:
    """Optional market/account inputs. Missing REAL data → fail closed for REAL."""

    account: AccountSnapshot | None = None
    symbol_meta: SymbolMeta | None = None
    open_position_count: int | None = None
    current_exposure_pct: float | None = None  # 0..1 of equity/balance reserved
    strategy_allows_new_entries: bool = True
    market_data_stale: bool | None = None
    account_data_stale: bool | None = None
    unexpected_existing_position: bool = False
    api_rate_limited: bool = False
    reconciliation_mismatch: bool = False


class ExecutionRiskGate:
    """Validate intents before OMS may attempt submission."""

    def __init__(
        self,
        *,
        mode: ExecutionMode,
        allow_trading: bool = False,
        kill_switch: ExecutionKillSwitch | None = None,
        order_journal: OrderJournal | None = None,
        max_positions: int = 4,
        max_total_exposure: float = 1.0,
        allocation_pct: float = 0.25,
    ) -> None:
        self.mode = mode
        self.allow_trading = bool(allow_trading)
        self.kill_switch = kill_switch or ExecutionKillSwitch()
        self.order_journal = order_journal
        # Hard safety caps for bounded 6h — reject construction, not just intents.
        if mode == ExecutionMode.REAL_BOUNDED_6H:
            if int(max_positions) > 4:
                raise TradingForbiddenError("REAL_BOUNDED_6H max_positions cannot exceed 4")
            if float(allocation_pct) > 0.25 + 1e-12:
                raise TradingForbiddenError("REAL_BOUNDED_6H allocation cannot exceed 25%")
            if float(max_total_exposure) > 1.0 + 1e-12:
                raise TradingForbiddenError("REAL_BOUNDED_6H max_total_exposure cannot exceed 100%")
        self.max_positions = int(max_positions)
        self.max_total_exposure = float(max_total_exposure)
        self.allocation_pct = float(allocation_pct)

    def validate_intent(
        self,
        intent: DurableTradeIntent | Any,
        ctx: RiskContext | None = None,
    ) -> RiskDecision:
        ctx = ctx or RiskContext()

        # Unrestricted REAL must never flip allow_trading=true.
        # REAL_CANARY_SINGLE_SHOT / REAL_BOUNDED_6H may set allow_trading only at
        # explicit enablement of their respective runtimes.
        if self.allow_trading and self.mode not in (
            ExecutionMode.REAL_CANARY_SINGLE_SHOT,
            ExecutionMode.REAL_BOUNDED_6H,
        ):
            raise TradingForbiddenError(
                "allow_trading must remain false except explicit REAL_CANARY_SINGLE_SHOT "
                "or REAL_BOUNDED_6H enablement"
            )

        if self.mode == ExecutionMode.REAL:
            return self._validate_real(intent, ctx)

        if self.mode == ExecutionMode.REAL_CANARY_SINGLE_SHOT:
            return self._validate_canary(intent, ctx)

        if self.mode == ExecutionMode.REAL_BOUNDED_6H:
            return self._validate_bounded_6h(intent, ctx)

        if self.mode == ExecutionMode.REAL_SHADOW:
            return RiskDecision(
                False,
                "REAL_SHADOW_READ_ONLY",
                {"note": "shadow observes MEXC only; never approves submissions"},
            )

        if self.mode in (ExecutionMode.PAPER, ExecutionMode.TEST):
            return self._validate_paper(intent, ctx)

        return RiskDecision(False, "UNKNOWN_EXECUTION_MODE")

    def _validate_paper(self, intent: Any, ctx: RiskContext) -> RiskDecision:
        qty = float(getattr(intent, "requested_quantity", getattr(intent, "position_btc", 0.0)) or 0.0)
        if qty <= 0:
            return RiskDecision(False, "INVALID_QUANTITY")

        if not ctx.strategy_allows_new_entries:
            return RiskDecision(False, "STRATEGY_HALT")

        # Kill switch: blocks new submissions; reconciliation still allowed elsewhere
        if not self.kill_switch.can_submit_new_order():
            return RiskDecision(False, "EXECUTION_KILL_SWITCH", {"permissions": self.kill_switch.permissions})

        # Portfolio caps (25% × 4 = 100%)
        if ctx.open_position_count is not None and ctx.open_position_count >= self.max_positions:
            return RiskDecision(
                False,
                "MAX_POSITIONS",
                {"open": ctx.open_position_count, "max": self.max_positions},
            )

        req_alloc = float(
            getattr(intent, "requested_allocation_pct", self.allocation_pct) or self.allocation_pct
        )
        if ctx.current_exposure_pct is not None:
            projected = float(ctx.current_exposure_pct) + req_alloc
            if projected > self.max_total_exposure + 1e-12:
                return RiskDecision(
                    False,
                    "MAX_TOTAL_EXPOSURE",
                    {
                        "current": ctx.current_exposure_pct,
                        "requested": req_alloc,
                        "projected": projected,
                        "max": self.max_total_exposure,
                    },
                )

        if ctx.unexpected_existing_position:
            return RiskDecision(False, "UNEXPECTED_EXISTING_POSITION")

        coid = getattr(intent, "client_order_id", None)
        if coid and self.order_journal is not None and self.order_journal.has_unresolved(str(coid)):
            return RiskDecision(False, "UNRESOLVED_PREVIOUS_ORDER", {"client_order_id": coid})

        if ctx.api_rate_limited:
            return RiskDecision(False, "API_RATE_LIMIT")

        if ctx.reconciliation_mismatch:
            return RiskDecision(False, "RECONCILIATION_MISMATCH")

        return RiskDecision(True, None)

    def _validate_real(self, intent: Any, ctx: RiskContext) -> RiskDecision:
        # Stage 2: REAL path must fail closed on every unknown.
        if not self.kill_switch.can_submit_new_order():
            return RiskDecision(False, "EXECUTION_KILL_SWITCH")

        if self.mode == ExecutionMode.REAL and not self.allow_trading:
            return RiskDecision(False, "REAL_TRADING_DISABLED")

        if ctx.account is None:
            return RiskDecision(False, "REAL_ACCOUNT_UNKNOWN")
        if ctx.account.is_paper:
            return RiskDecision(False, "PAPER_EQUITY_USED_FOR_REAL")
        if ctx.account_data_stale is True or ctx.account_data_stale is None:
            # Unknown staleness → fail closed for REAL
            return RiskDecision(False, "REAL_ACCOUNT_STALE_OR_UNKNOWN")

        if ctx.symbol_meta is None:
            return RiskDecision(False, "REAL_SYMBOL_METADATA_UNKNOWN")
        if not ctx.symbol_meta.is_trading:
            return RiskDecision(False, "SYMBOL_NOT_TRADING", {"status": ctx.symbol_meta.status})

        if ctx.market_data_stale is True or ctx.market_data_stale is None:
            return RiskDecision(False, "REAL_MARKET_DATA_STALE_OR_UNKNOWN")

        qty = float(getattr(intent, "requested_quantity", 0.0) or 0.0)
        if qty <= 0:
            return RiskDecision(False, "INVALID_QUANTITY")
        floored = floor_quantity_for_symbol(qty, ctx.symbol_meta)
        if floored <= 0:
            return RiskDecision(False, "QUANTITY_BELOW_MIN_AFTER_FLOOR")
        if floored + 1e-15 < ctx.symbol_meta.min_quantity:
            return RiskDecision(False, "QUANTITY_BELOW_MINIMUM")
        if floored > qty + 1e-15:
            return RiskDecision(False, "ROUNDING_INCREASED_SIZE")

        notional = floored  # caller supplies quote conversion later; structural check
        if ctx.symbol_meta.min_notional > 0 and notional + 1e-15 < ctx.symbol_meta.min_notional:
            # Without price, REAL must fail closed rather than guess
            if ctx.symbol_meta.min_notional > 0:
                return RiskDecision(False, "MIN_NOTIONAL_UNVERIFIED_OR_TOO_SMALL")

        if ctx.account.available_balance + 1e-15 < floored:
            return RiskDecision(False, "INSUFFICIENT_AVAILABLE_BALANCE")

        if ctx.open_position_count is not None and ctx.open_position_count >= self.max_positions:
            return RiskDecision(False, "MAX_POSITIONS")

        req_alloc = float(getattr(intent, "requested_allocation_pct", self.allocation_pct) or self.allocation_pct)
        if ctx.current_exposure_pct is not None:
            if float(ctx.current_exposure_pct) + req_alloc > self.max_total_exposure + 1e-12:
                return RiskDecision(False, "MAX_TOTAL_EXPOSURE")

        if ctx.unexpected_existing_position:
            return RiskDecision(False, "UNEXPECTED_EXISTING_POSITION")

        coid = getattr(intent, "client_order_id", None)
        if coid and self.order_journal is not None and self.order_journal.has_unresolved(str(coid)):
            return RiskDecision(False, "UNRESOLVED_PREVIOUS_ORDER")

        if ctx.api_rate_limited:
            return RiskDecision(False, "API_RATE_LIMIT")
        if ctx.reconciliation_mismatch:
            return RiskDecision(False, "RECONCILIATION_MISMATCH")

        # Unrestricted REAL path remains disabled — use canary or bounded 6h only.
        return RiskDecision(False, "REAL_UNRESTRICTED_DISABLED")

    def _validate_bounded_6h(self, intent: Any, ctx: RiskContext) -> RiskDecision:
        """Bounded 6h: same fail-closed REAL checks; caps enforced at risk layer."""
        if not self.allow_trading:
            return RiskDecision(False, "BOUNDED_6H_ALLOW_TRADING_FALSE")
        # Reuse canary structural path (T1-only, floor qty, exposure, unresolved).
        return self._validate_canary(intent, ctx)

    def _validate_canary(self, intent: Any, ctx: RiskContext) -> RiskDecision:
        """Single-shot canary: same fail-closed checks; approves only when allow_trading."""
        if not self.allow_trading:
            return RiskDecision(False, "CANARY_ALLOW_TRADING_FALSE")

        # Reuse REAL structural checks but allow approval at the end.
        if not self.kill_switch.can_submit_new_order():
            return RiskDecision(False, "EXECUTION_KILL_SWITCH")

        if ctx.account is None:
            return RiskDecision(False, "REAL_ACCOUNT_UNKNOWN")
        if ctx.account.is_paper:
            return RiskDecision(False, "PAPER_EQUITY_USED_FOR_REAL")
        if ctx.account_data_stale is True or ctx.account_data_stale is None:
            return RiskDecision(False, "REAL_ACCOUNT_STALE_OR_UNKNOWN")

        if ctx.symbol_meta is None:
            return RiskDecision(False, "REAL_SYMBOL_METADATA_UNKNOWN")
        if not ctx.symbol_meta.is_trading:
            return RiskDecision(False, "SYMBOL_NOT_TRADING", {"status": ctx.symbol_meta.status})

        if ctx.market_data_stale is True or ctx.market_data_stale is None:
            return RiskDecision(False, "REAL_MARKET_DATA_STALE_OR_UNKNOWN")

        qty = float(getattr(intent, "requested_quantity", 0.0) or 0.0)
        if qty <= 0:
            return RiskDecision(False, "INVALID_QUANTITY")
        floored = floor_quantity_for_symbol(qty, ctx.symbol_meta)
        if floored <= 0:
            return RiskDecision(False, "QUANTITY_BELOW_MIN_AFTER_FLOOR")
        if floored + 1e-15 < ctx.symbol_meta.min_quantity:
            return RiskDecision(False, "QUANTITY_BELOW_MINIMUM")
        if floored > qty + 1e-15:
            return RiskDecision(False, "ROUNDING_INCREASED_SIZE")

        intent_meta = getattr(intent, "meta", {}) or {}
        notional = float(intent_meta.get("notional_btc") or 0.0)
        if ctx.symbol_meta.min_notional > 0:
            if notional <= 0:
                return RiskDecision(False, "MIN_NOTIONAL_UNVERIFIED_OR_TOO_SMALL")
            if notional + 1e-15 < ctx.symbol_meta.min_notional:
                return RiskDecision(False, "BELOW_MIN_NOTIONAL")

        if notional > 0 and ctx.account.available_balance + 1e-15 < notional:
            return RiskDecision(False, "INSUFFICIENT_AVAILABLE_BALANCE")

        # Canary hard-cap: max_positions from gate (typically 1)
        if ctx.open_position_count is not None and ctx.open_position_count >= self.max_positions:
            return RiskDecision(False, "MAX_POSITIONS")

        req_alloc = float(getattr(intent, "requested_allocation_pct", self.allocation_pct) or self.allocation_pct)
        if req_alloc > self.allocation_pct + 1e-12:
            return RiskDecision(False, "ALLOCATION_EXCEEDS_CONFIG")
        if ctx.current_exposure_pct is not None:
            if float(ctx.current_exposure_pct) + req_alloc > self.max_total_exposure + 1e-12:
                return RiskDecision(False, "MAX_TOTAL_EXPOSURE")

        if ctx.unexpected_existing_position:
            return RiskDecision(False, "UNEXPECTED_EXISTING_POSITION")

        coid = getattr(intent, "client_order_id", None)
        if coid and self.order_journal is not None and self.order_journal.has_unresolved(str(coid)):
            return RiskDecision(False, "UNRESOLVED_PREVIOUS_ORDER")

        if ctx.api_rate_limited:
            return RiskDecision(False, "API_RATE_LIMIT")
        if ctx.reconciliation_mismatch:
            return RiskDecision(False, "RECONCILIATION_MISMATCH")

        selected = str(getattr(intent, "selected_exit", "") or "")
        if selected and selected not in {"trail_1", "T1", "T1_sl0p75_act0p75_dist0p25"}:
            return RiskDecision(False, "NON_T1_EXIT_FORBIDDEN")

        return RiskDecision(True, None)

    def assert_broker_allowed(self, broker_mode: ExecutionMode) -> None:
        if broker_mode == ExecutionMode.REAL:
            raise TradingForbiddenError("REAL unrestricted broker trading disabled")
        if broker_mode == ExecutionMode.REAL_CANARY_SINGLE_SHOT and self.mode != ExecutionMode.REAL_CANARY_SINGLE_SHOT:
            raise TradingForbiddenError("Canary broker only allowed in REAL_CANARY_SINGLE_SHOT mode")
        if broker_mode == ExecutionMode.REAL_BOUNDED_6H and self.mode != ExecutionMode.REAL_BOUNDED_6H:
            raise TradingForbiddenError("Bounded-6h broker only allowed in REAL_BOUNDED_6H mode")
        if broker_mode == ExecutionMode.REAL_SHADOW and self.mode != ExecutionMode.REAL_SHADOW:
            raise TradingForbiddenError("ShadowBroker only allowed in REAL_SHADOW mode")
        if broker_mode == ExecutionMode.TEST and self.mode != ExecutionMode.TEST:
            raise TradingForbiddenError("SimulatedBroker only allowed in TEST mode")
