"""BTC-equivalent account equity — never use USDT alone as total equity.

Definitions used by the bot (Stage-6 readiness):

  btc_free / btc_locked
      Spot BTC wallet balances from GET /api/v3/account.

  available_btc
      = btc_free
      Quote currency available to spend on MARKET BUY *BTC pairs.
      Locked BTC (open orders / OCO) is NOT available for new entries.

  usdt_as_btc
      = usdt_free / btc_usdt   (and similarly for USDC/FDUSD if present)
      Converted via BTCUSDT last price. Never treated as total equity alone.

  alt_balances_btc_value
      Sum of non-BTC, non-stable balances valued through ALT/BTC last prices
      when the pair exists in the trading universe. Illiquid/unmapped alts are
      excluded and reported separately (conservative — they do not inflate
      sizing equity).

  total_equity_btc  (BTC-equivalent equity)
      = btc_free + btc_locked + usdt_as_btc + alt_balances_btc_value

  trading_capital_btc
      = total_equity_btc
      Used for allocation (12.5%) and risk ceiling (0.5%) sizing.
      Intentionally NOT equal to available_btc alone: locked inventory and
      stablecoin reserves count toward risk budget / allocation base.

FORBIDDEN:
  Using available USDT (or any single stablecoin) as total account equity.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Mapping

from binance_btc_bot.exchange.base import AccountSnapshot

STABLES = frozenset({"USDT", "USDC", "FDUSD", "BUSD", "TUSD", "DAI"})


@dataclass(frozen=True)
class EquitySnapshot:
    total_equity_btc: float
    available_btc: float
    trading_capital_btc: float
    btc_free: float
    btc_locked: float
    usdt_free: float
    usdt_as_btc: float
    other_stables_as_btc: float
    alt_balances_btc_value: float
    unmapped_alts: tuple[str, ...] = ()
    components: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["unmapped_alts"] = list(self.unmapped_alts)
        d["notes"] = list(self.notes)
        return d


def compute_equity_btc(
    account: AccountSnapshot,
    *,
    btc_usdt: float,
    alt_prices_btc: Mapping[str, float] | None = None,
    include_stables: bool = True,
    include_alts: bool = True,
) -> EquitySnapshot:
    """Compute BTC-equivalent equity from a Binance account snapshot.

    ``alt_prices_btc`` maps base asset → ALT/BTC last price (e.g. ETH → 0.05).
    """
    alt_prices_btc = dict(alt_prices_btc or {})
    notes: list[str] = []
    btc = account.balances.get("BTC")
    btc_free = float(btc.free) if btc else 0.0
    btc_locked = float(btc.locked) if btc else 0.0

    usdt_free = 0.0
    usdt_as_btc = 0.0
    other_stables_as_btc = 0.0
    if include_stables:
        if btc_usdt <= 0:
            notes.append("BTCUSDT_PRICE_MISSING — stablecoins not converted")
        else:
            for asset, bal in account.balances.items():
                total = float(bal.free) + float(bal.locked)
                if total <= 0:
                    continue
                if asset == "USDT":
                    usdt_free = float(bal.free)
                    usdt_as_btc += total / float(btc_usdt)
                elif asset in STABLES:
                    other_stables_as_btc += total / float(btc_usdt)

    alt_value = 0.0
    unmapped: list[str] = []
    if include_alts:
        for asset, bal in account.balances.items():
            if asset in {"BTC"} or asset in STABLES:
                continue
            qty = float(bal.free) + float(bal.locked)
            if qty <= 0:
                continue
            px = alt_prices_btc.get(asset)
            if px is None or px <= 0:
                unmapped.append(asset)
                continue
            alt_value += qty * float(px)

    total = btc_free + btc_locked + usdt_as_btc + other_stables_as_btc + alt_value
    notes.append(
        "trading_capital_btc = total_equity_btc (BTC+locked+stables+mapped alts); "
        "available_btc = btc_free only"
    )
    if unmapped:
        notes.append(f"unmapped_alts_excluded={','.join(sorted(unmapped))}")

    return EquitySnapshot(
        total_equity_btc=total,
        available_btc=btc_free,
        trading_capital_btc=total,
        btc_free=btc_free,
        btc_locked=btc_locked,
        usdt_free=usdt_free,
        usdt_as_btc=usdt_as_btc,
        other_stables_as_btc=other_stables_as_btc,
        alt_balances_btc_value=alt_value,
        unmapped_alts=tuple(sorted(unmapped)),
        components={
            "btc_free": btc_free,
            "btc_locked": btc_locked,
            "usdt_as_btc": usdt_as_btc,
            "other_stables_as_btc": other_stables_as_btc,
            "alt_balances_btc_value": alt_value,
        },
        notes=tuple(notes),
    )


def worked_risk_example(
    *,
    equity_btc: float = 1.0,
    max_loss_per_trade: float = 0.005,
    allocation_pct: float = 0.125,
    fee_buffer_pct: float = 0.002,
    arm_sl: float = 0.0075,
) -> dict[str, Any]:
    """Numerical proof that 0.5% is a loss budget, not stop distance."""
    risk_budget = equity_btc * max_loss_per_trade
    notional_by_risk = risk_budget / arm_sl
    notional_by_alloc = equity_btc * allocation_pct * (1.0 - fee_buffer_pct)
    binding = min(notional_by_risk, notional_by_alloc)
    planned_loss = binding * arm_sl
    return {
        "equity_btc": equity_btc,
        "max_loss_per_trade": max_loss_per_trade,
        "meaning": "max planned monetary loss at hard SL <= 0.5% of equity",
        "NOT_meaning": "stop distance = 0.5%",
        "t1_hard_sl_pct": arm_sl,
        "risk_budget_btc": risk_budget,
        "max_notional_by_risk_btc": notional_by_risk,
        "max_notional_by_allocation_btc": notional_by_alloc,
        "binding_notional_btc": binding,
        "binding_constraint": "ALLOCATION" if notional_by_alloc <= notional_by_risk else "RISK",
        "planned_loss_btc": planned_loss,
        "planned_loss_pct_of_equity": planned_loss / equity_btc if equity_btc else None,
        "eight_trades_max_allocation": 8 * allocation_pct,
        "fee_buffer_pct": fee_buffer_pct,
    }
