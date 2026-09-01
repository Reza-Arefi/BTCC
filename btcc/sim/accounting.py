"""BTC-denominated virtual accounting for LONG ALT/BTC trades.

Economic model (spot, long-only, no leverage):

  Entry (BTC → ALT):
    spend virtual USD notional converted via BTCUSDT to BTC budget,
    buy ALT at ALT/BTC ask (entry_price * (1 + slippage)), pay fee in BTC.

  Exit (ALT → BTC):
    sell ALT at ALT/BTC bid (exit_price * (1 - slippage)), pay fee in BTC.

PnL_BTC = exit_btc_received - entry_btc_spent
PnL_USD_equiv = PnL_BTC * btc_usdt_at_exit (display only)

All exit strategies share the same entry economics and cost rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CostModel:
    fee_rate_per_side: float = 0.0010
    slippage_rate_per_side: float = 0.0005

    def buy_fill_price(self, mid: float) -> float:
        """Pay more ALT/BTC to buy ALT (adverse slippage)."""
        return float(mid) * (1.0 + self.slippage_rate_per_side)

    def sell_fill_price(self, mid: float) -> float:
        """Receive less ALT/BTC when selling ALT."""
        return float(mid) * (1.0 - self.slippage_rate_per_side)


def open_long_alt_btc(
    *,
    alt_btc_mid: float,
    btc_usdt: float,
    notional_usd: float,
    costs: CostModel,
) -> dict[str, Any]:
    """Open LONG ALT/BTC using ``notional_usd`` worth of BTC at ``btc_usdt``."""
    if alt_btc_mid <= 0 or btc_usdt <= 0 or notional_usd <= 0:
        raise ValueError("Invalid prices/notional for open_long_alt_btc")

    entry_btc_budget = float(notional_usd) / float(btc_usdt)
    fill = costs.buy_fill_price(alt_btc_mid)
    # Fee charged on BTC notional spent
    fee_btc = entry_btc_budget * costs.fee_rate_per_side
    btc_spent = entry_btc_budget  # full budget deployed
    btc_for_alts = entry_btc_budget - fee_btc
    alt_qty = btc_for_alts / fill
    slip_btc = entry_btc_budget * costs.slippage_rate_per_side  # informational

    return {
        "side": "LONG_ALT_BTC",
        "entry_alt_btc_mid": float(alt_btc_mid),
        "entry_fill_price": float(fill),
        "btc_usdt_entry": float(btc_usdt),
        "notional_usd": float(notional_usd),
        "entry_btc_spent": float(btc_spent),
        "entry_fee_btc": float(fee_btc),
        "entry_slippage_btc_approx": float(slip_btc),
        "alt_qty": float(alt_qty),
    }


def close_long_alt_btc(
    *,
    position: dict[str, Any],
    alt_btc_mid: float,
    btc_usdt: float,
    costs: CostModel,
) -> dict[str, Any]:
    """Close LONG ALT/BTC; return BTC proceeds and PnL."""
    fill = costs.sell_fill_price(alt_btc_mid)
    alt_qty = float(position["alt_qty"])
    gross_btc = alt_qty * fill
    fee_btc = gross_btc * costs.fee_rate_per_side
    exit_btc = gross_btc - fee_btc
    entry_btc = float(position["entry_btc_spent"])
    pnl_btc = exit_btc - entry_btc
    pnl_pct = pnl_btc / entry_btc if entry_btc else 0.0
    pnl_usd = pnl_btc * float(btc_usdt)
    gross_pnl_btc = (alt_qty * float(alt_btc_mid)) - entry_btc  # mid-to-mid rough
    total_fees_btc = float(position["entry_fee_btc"]) + fee_btc
    total_slip_btc = float(position.get("entry_slippage_btc_approx", 0.0)) + (
        entry_btc * costs.slippage_rate_per_side
    )

    return {
        "exit_alt_btc_mid": float(alt_btc_mid),
        "exit_fill_price": float(fill),
        "btc_usdt_exit": float(btc_usdt),
        "exit_btc_received": float(exit_btc),
        "exit_fee_btc": float(fee_btc),
        "pnl_btc": float(pnl_btc),
        "pnl_pct": float(pnl_pct),
        "pnl_usd_equiv": float(pnl_usd),
        "gross_pnl_btc_approx": float(gross_pnl_btc),
        "fees_btc": float(total_fees_btc),
        "slippage_btc_approx": float(total_slip_btc),
        "net_pnl_btc": float(pnl_btc),
    }
