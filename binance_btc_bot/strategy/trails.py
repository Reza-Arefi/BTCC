"""Frozen trail strategy definitions + Binance-native parameter mapping.

Research geometry is preserved exactly. Binance constraints are reported
explicitly — never silently rewritten into the research config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from binance_btc_bot.config_loader import FROZEN_STRATEGIES
from binance_btc_bot.exchange.base import SymbolInfo, TrailingOcoRequest


@dataclass(frozen=True)
class TrailStrategy:
    key: str
    arm_sl_activation_trail: float
    activation: float
    trail_distance: float

    @property
    def stop_loss_pct(self) -> float:
        return self.arm_sl_activation_trail

    @property
    def trail_distance_bips(self) -> int:
        """Binance trailingDelta is integer BIPS (1 BIP = 0.01%)."""
        return int(round(self.trail_distance * 10_000))


def get_strategy(key: str, strategies_cfg: Mapping[str, Any] | None = None) -> TrailStrategy:
    k = str(key).upper()
    if strategies_cfg and k in strategies_cfg:
        raw = strategies_cfg[k]
        return TrailStrategy(
            key=k,
            arm_sl_activation_trail=float(raw["arm_sl_activation_trail"]),
            activation=float(raw["activation"]),
            trail_distance=float(raw["trail_distance"]),
        )
    if k not in FROZEN_STRATEGIES:
        raise KeyError(f"unknown strategy {key}")
    sl, act, dist = FROZEN_STRATEGIES[k]
    return TrailStrategy(key=k, arm_sl_activation_trail=sl, activation=act, trail_distance=dist)


def all_strategies(strategies_cfg: Mapping[str, Any] | None = None) -> dict[str, TrailStrategy]:
    keys = list(FROZEN_STRATEGIES.keys())
    return {k: get_strategy(k, strategies_cfg) for k in keys}


@dataclass(frozen=True)
class BinanceTrailMapping:
    """Explicit mapping from research trail → Binance OCO trailing parameters."""

    strategy_key: str
    mode: str
    above_type: str
    below_type: str
    activation_price: float
    initial_stop_price: float
    trailing_delta_bips: int
    requested_trail_pct: float
    allowed: bool
    constraint_notes: tuple[str, ...]
    request: TrailingOcoRequest | None = None


def map_trail_to_binance_oco(
    *,
    strategy: TrailStrategy,
    symbol: str,
    entry_price: float,
    quantity: float,
    symbol_info: SymbolInfo | None = None,
    prefer_market_contingent: bool = True,
    list_client_order_id: str | None = None,
    side: str = "SELL",
) -> BinanceTrailMapping:
    """Map T1–T10 geometry onto Binance native OCO trailing exit.

    Mapping (LONG ALT/BTC exit = SELL OCO):
      aboveType = TAKE_PROFIT
      aboveStopPrice = entry * (1 + activation)     # activation gate
      aboveTrailingDelta = trail_distance in BIPS   # Binance owns trail
      belowType = STOP_LOSS
      belowStopPrice = entry * (1 - arm_sl)         # hard SL before activation

    Notes / limitations (documented, not silently changed):
    - Research ratchets the stop after activation and effectively replaces the
      initial SL with the trail. Binance OCO keeps both legs until one fills;
      after activation the trail stop sits above the hard SL, so the trail leg
      is the economic exit while the hard SL remains a crash safety net.
    - Contingent MARKET types (TAKE_PROFIT / STOP_LOSS) avoid limit slippage
      buffers; LIMIT variants would require an explicit price buffer.
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")
    if quantity <= 0:
        raise ValueError("quantity must be > 0")

    notes: list[str] = []
    above_type = "TAKE_PROFIT" if prefer_market_contingent else "TAKE_PROFIT_LIMIT"
    below_type = "STOP_LOSS" if prefer_market_contingent else "STOP_LOSS_LIMIT"
    activation_price = float(entry_price) * (1.0 + float(strategy.activation))
    initial_stop = float(entry_price) * (1.0 - float(strategy.arm_sl_activation_trail))
    trail_bips = strategy.trail_distance_bips
    allowed = True

    if symbol_info is not None:
        if not symbol_info.oco_allowed:
            allowed = False
            notes.append("ocoAllowed=false on symbol")
        if above_type not in symbol_info.order_types or below_type not in symbol_info.order_types:
            allowed = False
            notes.append(f"required order types missing: need {above_type}+{below_type}")
        # TAKE_PROFIT SELL uses "above" trailing filter band
        mn = symbol_info.min_trailing_above_delta
        mx = symbol_info.max_trailing_above_delta
        if mn is not None and trail_bips < mn:
            allowed = False
            notes.append(
                f"Requested trail {strategy.trail_distance*100:.2f}% ({trail_bips} BIPS) "
                f"< Binance minTrailingAboveDelta={mn} BIPS — NOT silently raised"
            )
        if mx is not None and trail_bips > mx:
            allowed = False
            notes.append(
                f"Requested trail {strategy.trail_distance*100:.2f}% ({trail_bips} BIPS) "
                f"> Binance maxTrailingAboveDelta={mx} BIPS — NOT silently reduced"
            )
        if symbol_info.price_tick > 0:
            activation_price = _floor_to_tick(activation_price, symbol_info.price_tick)
            initial_stop = _floor_to_tick(initial_stop, symbol_info.price_tick)
            # For SL, floor is conservative (trigger slightly earlier/safer for SELL stop).
            # Activation floor may delay arming by <1 tick — documented.
            notes.append(
                f"prices snapped to tickSize={symbol_info.price_tick} "
                f"(activation={activation_price}, stop={initial_stop})"
            )

    if trail_bips != int(strategy.trail_distance * 10_000 + 1e-12):
        notes.append(
            f"trail_distance {strategy.trail_distance} rounded to {trail_bips} BIPS "
            f"(Binance requires integer BIPS)"
        )

    req = None
    if allowed:
        req = TrailingOcoRequest(
            symbol=symbol.upper(),
            side=side.upper(),
            quantity=float(quantity),
            above_type=above_type,
            above_stop_price=float(activation_price),
            above_trailing_delta=int(trail_bips),
            below_type=below_type,
            below_stop_price=float(initial_stop),
            list_client_order_id=list_client_order_id,
        )

    return BinanceTrailMapping(
        strategy_key=strategy.key,
        mode="OCO_TAKE_PROFIT_TRAIL_PLUS_STOP_LOSS",
        above_type=above_type,
        below_type=below_type,
        activation_price=float(activation_price),
        initial_stop_price=float(initial_stop),
        trailing_delta_bips=int(trail_bips),
        requested_trail_pct=float(strategy.trail_distance),
        allowed=allowed,
        constraint_notes=tuple(notes),
        request=req,
    )


def _floor_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    # integer division floor
    units = int(price / tick)
    return units * tick
