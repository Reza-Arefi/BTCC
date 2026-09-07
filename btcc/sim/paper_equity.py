"""BTC-denominated paper account with compounding position sizing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PaperEquityTracker:
    """Track simulated BTC equity and compounding position allocations."""

    starting_equity_btc: float
    allocation_pct: float
    equity_btc: float

    @classmethod
    def from_config(cls, sim: dict[str, Any]) -> PaperEquityTracker:
        start = float(sim.get("starting_equity_btc", 0.01311845))
        pct = float(sim.get("position_allocation_pct", 0.25))
        return cls(starting_equity_btc=start, allocation_pct=pct, equity_btc=start)

    @classmethod
    def from_state(cls, sim: dict[str, Any], state: dict[str, Any]) -> PaperEquityTracker:
        tr = cls.from_config(sim)
        if "equity_btc" in state:
            tr.equity_btc = float(state["equity_btc"])
        return tr

    def to_state(self) -> dict[str, Any]:
        return {
            "starting_equity_btc": self.starting_equity_btc,
            "equity_btc": self.equity_btc,
            "allocation_pct": self.allocation_pct,
            "cumulative_return_pct": self.cumulative_return_pct(),
        }

    def cumulative_return_pct(self) -> float:
        if self.starting_equity_btc <= 0:
            return 0.0
        return 100.0 * (self.equity_btc / self.starting_equity_btc - 1.0)

    def reserved_btc(self, open_opps: list[dict[str, Any]]) -> float:
        total = 0.0
        for opp in open_opps:
            if opp.get("status") not in ("OPEN", "PENDING_ENTRY"):
                continue
            pos_btc = opp.get("position_btc")
            if pos_btc is not None:
                total += float(pos_btc)
                continue
            for item in opp.get("leg_items") or []:
                if item.get("is_counterfactual"):
                    continue
                leg = item.get("leg")
                pos = getattr(leg, "position", None) if leg is not None else None
                if isinstance(pos, dict):
                    total += float(pos.get("entry_btc_spent") or 0.0)
        return total

    def available_btc(self, open_opps: list[dict[str, Any]]) -> float:
        return max(0.0, self.equity_btc - self.reserved_btc(open_opps))

    def sizing_for_new_trade(self, open_opps: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Size a new trade as allocation_pct × current equity, capped by free exposure."""
        equity = float(self.equity_btc)
        if equity <= 0 or self.allocation_pct <= 0:
            return None
        requested_btc = equity * float(self.allocation_pct)
        available = self.available_btc(open_opps)
        if available <= 0:
            return None
        position_btc = min(requested_btc, available)
        actual_pct = position_btc / equity
        remaining_exposure_pct = available / equity
        return {
            "position_btc": position_btc,
            "requested_allocation_pct": float(self.allocation_pct),
            "actual_allocation_pct": actual_pct,
            "remaining_exposure_pct": remaining_exposure_pct,
            "exposure_limited": position_btc + 1e-15 < requested_btc,
        }

    def position_btc_for_new_trade(self, open_opps: list[dict[str, Any]]) -> float | None:
        sizing = self.sizing_for_new_trade(open_opps)
        if sizing is None:
            return None
        return float(sizing["position_btc"])

    def notional_usd(self, position_btc: float, btc_usdt: float) -> float:
        if position_btc <= 0 or btc_usdt <= 0:
            raise ValueError("Invalid position_btc or btc_usdt for notional conversion")
        return float(position_btc) * float(btc_usdt)

    def apply_close(self, pnl_btc: float) -> float:
        self.equity_btc += float(pnl_btc)
        return self.equity_btc
