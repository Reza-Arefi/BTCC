"""Account state interfaces — PAPER equity vs REAL exchange balance must never mix."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from btcc.execution.modes import ExecutionMode
from btcc.sim.paper_equity import PaperEquityTracker


@dataclass(frozen=True)
class AccountSnapshot:
    mode: str  # PAPER | REAL
    available_balance: float
    locked_balance: float
    total_balance: float
    currency: str = "BTC"
    open_positions: int = 0
    reserved_order_exposure: float = 0.0
    realized_fees: float | None = None
    unrealized_value: float | None = None
    source: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_paper(self) -> bool:
        return self.mode == ExecutionMode.PAPER.value

    @property
    def is_real(self) -> bool:
        return self.mode in {
            ExecutionMode.REAL.value,
            ExecutionMode.REAL_CANARY_SINGLE_SHOT.value,
            ExecutionMode.REAL_SHADOW.value,
        }


@runtime_checkable
class AccountStateProvider(Protocol):
    def get_account_snapshot(self) -> AccountSnapshot | None: ...


class UnavailableRealAccountProvider:
    """REAL account data unavailable in Stage 2 — fail closed (returns None)."""

    def get_account_snapshot(self) -> AccountSnapshot | None:
        return None


class PaperAccountProvider:
    """Adapts PaperEquityTracker — must never be used as REAL available balance."""

    def __init__(
        self,
        equity: PaperEquityTracker,
        *,
        open_opps_provider: Any | None = None,
    ) -> None:
        self._equity = equity
        self._open_opps_provider = open_opps_provider

    def get_account_snapshot(self) -> AccountSnapshot:
        opens = []
        if self._open_opps_provider is not None:
            try:
                opens = list(self._open_opps_provider() or [])
            except Exception:
                opens = []
        reserved = float(self._equity.reserved_btc(opens))
        available = float(self._equity.available_btc(opens))
        total = float(self._equity.equity_btc)
        n_open = sum(1 for o in opens if o.get("status") in ("OPEN", "PENDING_ENTRY"))
        return AccountSnapshot(
            mode=ExecutionMode.PAPER.value,
            available_balance=available,
            locked_balance=reserved,
            total_balance=total,
            currency="BTC",
            open_positions=n_open,
            reserved_order_exposure=reserved,
            source="paper_equity",
            meta={
                "allocation_pct": self._equity.allocation_pct,
                "starting_equity_btc": self._equity.starting_equity_btc,
                "warning": "PAPER_EQUITY_NOT_REAL_BALANCE",
            },
        )


def real_sizing_quantity(
    *,
    available_balance: float,
    allocation_pct: float,
    fee_reserve_pct: float = 0.0,
) -> float:
    """REAL sizing precursor: allocation of available balance after fee reserve.

    Does NOT use paper equity. Precision floor applied by caller via symbols.floor_*.
    """
    if available_balance <= 0 or allocation_pct <= 0:
        return 0.0
    spendable = available_balance * (1.0 - max(0.0, fee_reserve_pct))
    return max(0.0, spendable * float(allocation_pct))
