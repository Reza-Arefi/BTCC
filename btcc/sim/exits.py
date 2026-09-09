"""Parallel virtual exit strategies for one LONG ALT/BTC opportunity.

Strategies share entry timestamp/price/notional/signal; differ only in exits.

Same-candle conflict (no lower-TF data):
  If both SL and TP are touched in the same 15m OHLC bar, assume SL first.
  If trail activation and initial hard SL are both touched in the same bar
  (trail not yet active), assume hard SL first — do not activate trail.
  Documented in sim_config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from btcc.sim.accounting import CostModel, close_long_alt_btc, open_long_alt_btc


@dataclass
class StrategySpec:
    key: str
    name: str
    stop_loss_pct: float
    take_profit_pct: float | None = None
    trail_activation_pct: float | None = None
    trail_distance_pct: float | None = None
    # Optional adaptive trailing (research T11–T20). None => fixed trail.
    adaptive_mode: str | None = None
    adaptive_cfg: dict[str, Any] | None = None


def specs_from_config(sim: dict[str, Any]) -> list[StrategySpec]:
    out: list[StrategySpec] = []
    for key, raw in (sim.get("strategies") or {}).items():
        trail = raw.get("trailing") or {}
        out.append(
            StrategySpec(
                key=key,
                name=str(raw.get("name", key)),
                stop_loss_pct=float(raw["stop_loss_pct"]),
                take_profit_pct=(
                    float(raw["take_profit_pct"])
                    if raw.get("take_profit_pct") is not None
                    else None
                ),
                trail_activation_pct=(
                    float(trail["activation_pct"]) if trail.get("activation_pct") is not None else None
                ),
                trail_distance_pct=(
                    float(trail["distance_pct"]) if trail.get("distance_pct") is not None else None
                ),
                adaptive_mode=(str(raw["adaptive_mode"]) if raw.get("adaptive_mode") else None),
                adaptive_cfg=(dict(raw["adaptive_cfg"]) if raw.get("adaptive_cfg") else None),
            )
        )
    return out


@dataclass
class StrategyLegState:
    spec: StrategySpec
    position: dict[str, Any]
    entry_ts: Any
    entry_mid: float
    initial_sl: float
    tp: float | None
    trailing_active: bool = False
    highest_since_activation: float | None = None
    trailing_stop: float | None = None
    closed: bool = False
    exit_reason: str | None = None
    exit_ts: Any = None
    exit_result: dict[str, Any] | None = None
    # MFE / MAE excursion tracking (ALT/BTC return vs entry fill)
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    peak_price: float | None = None
    trough_price: float | None = None
    trail_activation_ts: Any = None
    activation_return_pct: float | None = None
    peak_before_exit_pct: float = 0.0
    bars_held: int = 0
    # Diagnostics for adaptive / MFE analysis
    time_to_activation_min: float | None = None
    time_to_peak_min: float | None = None
    milestone_times_min: dict[str, float] | None = None
    current_trail_distance_pct: float | None = None
    stop_floor_pct: float | None = None  # min stop as return from entry (profit-lock)

    def active_stop(self) -> float:
        if self.trailing_active and self.trailing_stop is not None:
            return float(self.trailing_stop)
        return float(self.initial_sl)


def open_opportunity_legs(
    *,
    alt_btc_entry_mid: float,
    btc_usdt: float,
    notional_usd: float,
    costs: CostModel,
    specs: list[StrategySpec],
    entry_ts: Any,
) -> list[StrategyLegState]:
    legs: list[StrategyLegState] = []
    for spec in specs:
        pos = open_long_alt_btc(
            alt_btc_mid=alt_btc_entry_mid,
            btc_usdt=btc_usdt,
            notional_usd=notional_usd,
            costs=costs,
        )
        fill = float(pos["entry_fill_price"])
        initial_sl = fill * (1.0 - spec.stop_loss_pct)
        tp = fill * (1.0 + spec.take_profit_pct) if spec.take_profit_pct is not None else None
        legs.append(
            StrategyLegState(
                spec=spec,
                position=pos,
                entry_ts=entry_ts,
                entry_mid=float(alt_btc_entry_mid),
                initial_sl=initial_sl,
                tp=tp,
            )
        )
    return legs


def _close_leg(
    leg: StrategyLegState,
    *,
    exit_mid: float,
    btc_usdt: float,
    costs: CostModel,
    exit_ts: Any,
    reason: str,
) -> None:
    result = close_long_alt_btc(
        position=leg.position,
        alt_btc_mid=exit_mid,
        btc_usdt=btc_usdt,
        costs=costs,
    )
    leg.closed = True
    leg.exit_reason = reason
    leg.exit_ts = exit_ts
    leg.exit_result = result


def _minutes_since(entry_ts: Any, ts: Any) -> float:
    try:
        return max(0.0, (pd.Timestamp(ts) - pd.Timestamp(entry_ts)).total_seconds() / 60.0)
    except Exception:
        return 0.0


def _update_milestones(leg: StrategyLegState, entry_fill: float, high: float, ts: Any) -> None:
    if entry_fill <= 0:
        return
    if leg.milestone_times_min is None:
        leg.milestone_times_min = {}
    mins = _minutes_since(leg.entry_ts, ts)
    peak_ret = (high / entry_fill) - 1.0
    for label, thr in (("p1", 0.01), ("p1_5", 0.015), ("p2", 0.02), ("p3", 0.03)):
        if peak_ret >= thr and label not in leg.milestone_times_min:
            leg.milestone_times_min[label] = mins
    # peak time: update whenever we make a new peak
    if leg.peak_price is not None and abs(high - float(leg.peak_price)) < 1e-15:
        leg.time_to_peak_min = mins


def _adaptive_trail_distance(leg: StrategyLegState, entry_fill: float, ts: Any) -> float:
    """Return current trail distance fraction; never loosens vs prior distance on leg."""
    mode = leg.spec.adaptive_mode
    cfg = leg.spec.adaptive_cfg or {}
    peak = float(leg.highest_since_activation or entry_fill)
    peak_ret = (peak / entry_fill) - 1.0 if entry_fill > 0 else 0.0
    base = float(leg.spec.trail_distance_pct or 0.0)

    if mode is None:
        dist = base
    elif mode in ("fixed",):
        dist = base
    elif mode == "two_stage":
        # T13
        if peak_ret >= 0.02:
            dist = float(cfg.get("trail_at_2pct", 0.0025))
        elif peak_ret >= 0.015:
            dist = float(cfg.get("trail_at_1_5pct", 0.005))
        else:
            dist = float(cfg.get("trail_initial", 0.0075))
    elif mode == "profit_lock":
        # T14 — distance stays initial; floors handled separately
        dist = float(cfg.get("trail_initial", 0.0075))
    elif mode == "progressive":
        # T15
        if peak_ret >= 0.03:
            dist = 0.0025
        elif peak_ret >= 0.02:
            dist = 0.0035
        elif peak_ret >= 0.015:
            dist = 0.0045
        elif peak_ret >= 0.01:
            dist = 0.006
        else:
            dist = 0.0075
    elif mode == "break_even_then_tight":
        # T18 after BE: stages by peak
        if peak_ret >= 0.02:
            dist = 0.0025
        elif peak_ret >= 0.0125:
            dist = 0.005
        else:
            dist = float(cfg.get("trail_after_be", 0.0075))
    elif mode == "mfe_responsive":
        # T19
        if peak_ret >= 0.03:
            dist = 0.0025
        elif peak_ret >= 0.02:
            dist = 0.004
        elif peak_ret >= 0.01:
            dist = 0.006
        else:
            dist = 0.0075
    elif mode == "time_adaptive":
        # T20
        mins = _minutes_since(leg.entry_ts, ts)
        if mins <= 3.0:
            dist = 0.0075
        elif mins <= 7.0:
            dist = 0.005
        else:
            dist = 0.0025
    else:
        dist = base

    # Never loosen trail distance once tightened
    prev = leg.current_trail_distance_pct
    if prev is not None and dist > float(prev) + 1e-15:
        dist = float(prev)
    leg.current_trail_distance_pct = dist
    return dist


def _apply_stop_floors(leg: StrategyLegState, entry_fill: float) -> None:
    """Raise trailing_stop to profit-lock floors when configured (never lower)."""
    mode = leg.spec.adaptive_mode
    cfg = leg.spec.adaptive_cfg or {}
    if mode != "profit_lock" or entry_fill <= 0:
        return
    peak = float(leg.highest_since_activation or entry_fill)
    peak_ret = (peak / entry_fill) - 1.0
    floor_pct = None
    if peak_ret >= 0.02:
        floor_pct = float(cfg.get("floor_at_2pct", 0.0075))
    elif peak_ret >= 0.015:
        floor_pct = float(cfg.get("floor_at_1_5pct", 0.0025))
    if floor_pct is None:
        return
    leg.stop_floor_pct = max(float(leg.stop_floor_pct or -1.0), floor_pct)
    floor_px = entry_fill * (1.0 + float(leg.stop_floor_pct))
    leg.trailing_stop = max(float(leg.trailing_stop or 0.0), floor_px)


def process_bar_on_leg(
    leg: StrategyLegState,
    *,
    bar: dict[str, Any],
    btc_usdt: float,
    costs: CostModel,
    same_candle_conflict: str = "assume_sl_first",
) -> bool:
    """Update one leg with a new ALT/BTC OHLC bar. Returns True if closed this bar."""
    if leg.closed:
        return False

    high = float(bar["high"])
    low = float(bar["low"])
    close = float(bar["close"])
    ts = bar["timestamp"]
    leg.bars_held += 1
    entry_fill = float(leg.position["entry_fill_price"])
    if entry_fill > 0:
        bar_mfe = (high / entry_fill) - 1.0
        bar_mae = (low / entry_fill) - 1.0
        leg.mfe_pct = max(leg.mfe_pct, bar_mfe)
        leg.mae_pct = min(leg.mae_pct, bar_mae)
        new_peak = leg.peak_price is None or high >= float(leg.peak_price)
        leg.peak_price = high if leg.peak_price is None else max(leg.peak_price, high)
        leg.trough_price = low if leg.trough_price is None else min(leg.trough_price, low)
        leg.peak_before_exit_pct = max(leg.peak_before_exit_pct, bar_mfe)
        if new_peak:
            _update_milestones(leg, entry_fill, high, ts)
    stop = leg.active_stop()

    # Trailing activation / ratchet
    #
    # Conservative trail-vs-SL rule (before trail is active):
    # If this bar touches BOTH trail activation (high) AND the initial hard SL
    # (low), assume SL occurred first — close as STOP_LOSS and do not activate.
    was_trailing = leg.trailing_active
    has_trail = leg.spec.trail_activation_pct is not None and (
        leg.spec.trail_distance_pct is not None or leg.spec.adaptive_mode is not None
    )
    if has_trail and not leg.trailing_active:
        activation = float(leg.position["entry_fill_price"]) * (1.0 + float(leg.spec.trail_activation_pct))
        initial_sl = float(leg.initial_sl)
        if high >= activation and low <= initial_sl:
            _close_leg(
                leg,
                exit_mid=initial_sl,
                btc_usdt=btc_usdt,
                costs=costs,
                exit_ts=ts,
                reason="STOP_LOSS_TRAIL_ACTIVATION_SAME_CANDLE",
            )
            return True
        if high >= activation:
            leg.trailing_active = True
            leg.highest_since_activation = high
            leg.trail_activation_ts = ts
            leg.activation_return_pct = (high / entry_fill - 1.0) if entry_fill > 0 else None
            leg.time_to_activation_min = _minutes_since(leg.entry_ts, ts)
            mode = leg.spec.adaptive_mode
            if mode == "break_even_then_tight":
                # T18: at activation move stop to ~break-even (entry fill)
                leg.trailing_stop = float(entry_fill)
                leg.current_trail_distance_pct = float((leg.spec.adaptive_cfg or {}).get("trail_after_be", 0.0075))
            else:
                dist = _adaptive_trail_distance(leg, entry_fill, ts)
                if dist <= 0 and leg.spec.trail_distance_pct is not None:
                    dist = float(leg.spec.trail_distance_pct)
                leg.trailing_stop = high * (1.0 - dist)
            _apply_stop_floors(leg, entry_fill)
            stop = leg.active_stop()

    if leg.trailing_active:
        if high > (leg.highest_since_activation or 0.0):
            leg.highest_since_activation = high
        dist = _adaptive_trail_distance(leg, entry_fill, ts)
        if dist <= 0 and leg.spec.trail_distance_pct is not None:
            dist = float(leg.spec.trail_distance_pct)
        peak = float(leg.highest_since_activation or high)
        new_stop = peak * (1.0 - dist)
        leg.trailing_stop = max(float(leg.trailing_stop or 0.0), new_stop)
        _apply_stop_floors(leg, entry_fill)
        stop = leg.active_stop()

    hit_sl = low <= stop
    hit_tp = leg.tp is not None and high >= float(leg.tp)
    trail_exit = leg.trailing_active and hit_sl

    if hit_sl and hit_tp:
        if trail_exit:
            reason = "TRAILING_STOP_SAME_CANDLE_CONFLICT"
        else:
            reason = "STOP_LOSS_SAME_CANDLE_CONFLICT"
        _close_leg(leg, exit_mid=stop, btc_usdt=btc_usdt, costs=costs, exit_ts=ts, reason=reason)
        return True

    if hit_sl:
        reason = "TRAILING_STOP" if (leg.trailing_active or was_trailing) else "STOP_LOSS"
        _close_leg(leg, exit_mid=stop, btc_usdt=btc_usdt, costs=costs, exit_ts=ts, reason=reason)
        return True

    if hit_tp:
        _close_leg(
            leg,
            exit_mid=float(leg.tp),
            btc_usdt=btc_usdt,
            costs=costs,
            exit_ts=ts,
            reason="TAKE_PROFIT",
        )
        return True

    return False


def process_bars_until_closed(
    legs: list[StrategyLegState],
    bars: pd.DataFrame,
    *,
    btc_usdt_series: pd.Series | None,
    default_btc_usdt: float,
    costs: CostModel,
    same_candle_conflict: str = "assume_sl_first",
) -> list[StrategyLegState]:
    """Walk forward bars (after entry) until all legs closed or bars exhausted."""
    if bars is None or bars.empty:
        return legs
    for _, row in bars.iterrows():
        if all(leg.closed for leg in legs):
            break
        ts = row["timestamp"]
        if btc_usdt_series is not None and ts in btc_usdt_series.index:
            btc_usdt = float(btc_usdt_series.loc[ts])
        else:
            btc_usdt = default_btc_usdt
        bar = {
            "timestamp": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for leg in legs:
            process_bar_on_leg(
                leg,
                bar=bar,
                btc_usdt=btc_usdt,
                costs=costs,
                same_candle_conflict=same_candle_conflict,
            )
    return legs


def rebuild_leg_from_snapshot(snap: dict[str, Any], specs: list[StrategySpec]) -> StrategyLegState | None:
    """Rebuild an open StrategyLegState from a persisted snapshot (restart safety)."""
    if snap.get("closed"):
        return None
    key = snap.get("strategy_key")
    spec = next((s for s in specs if s.key == key), None)
    if spec is None:
        return None
    pos = {
        "side": "LONG_ALT_BTC",
        "entry_alt_btc_mid": snap.get("entry_alt_btc_mid"),
        "entry_fill_price": snap.get("entry_fill_price"),
        "btc_usdt_entry": None,
        "notional_usd": snap.get("notional_usd"),
        "entry_btc_spent": snap.get("entry_btc_spent"),
        "entry_fee_btc": float(snap.get("entry_btc_spent") or 0) * 0.0,  # informational only after restore
        "entry_slippage_btc_approx": 0.0,
        "alt_qty": snap.get("alt_qty"),
    }
    # Re-derive fee from notional if possible is skipped; PnL uses entry_btc_spent + exit
    if pos["entry_fee_btc"] == 0 and pos.get("entry_btc_spent") and pos.get("entry_fill_price") and pos.get("alt_qty"):
        # Reconstruct fee as spent - alt_qty*fill
        try:
            pos["entry_fee_btc"] = max(
                0.0,
                float(pos["entry_btc_spent"]) - float(pos["alt_qty"]) * float(pos["entry_fill_price"]),
            )
        except Exception:
            pos["entry_fee_btc"] = 0.0
    return StrategyLegState(
        spec=spec,
        position=pos,
        entry_ts=snap.get("entry_ts"),
        entry_mid=float(snap.get("entry_alt_btc_mid") or 0),
        initial_sl=float(snap["initial_sl"]) if snap.get("initial_sl") is not None else 0.0,
        tp=float(snap["take_profit"]) if snap.get("take_profit") is not None else None,
        trailing_active=bool(snap.get("trailing_active")),
        highest_since_activation=None,
        trailing_stop=float(snap["trailing_stop"]) if snap.get("trailing_stop") is not None else None,
        closed=False,
    )


def leg_to_record(leg: StrategyLegState, opportunity_id: str) -> dict[str, Any]:
    pos = leg.position
    res = leg.exit_result or {}
    hold = None
    if leg.exit_ts is not None and leg.entry_ts is not None:
        try:
            hold = (pd.Timestamp(leg.exit_ts) - pd.Timestamp(leg.entry_ts)).total_seconds() / 3600.0
        except Exception:
            hold = None
    realized = float(res.get("pnl_pct") or 0.0)
    mfe = float(leg.mfe_pct)
    mfe_frac = (realized / mfe) if mfe > 1e-9 else None
    return {
        "opportunity_id": opportunity_id,
        "strategy_key": leg.spec.key,
        "strategy_name": leg.spec.name,
        "entry_ts": str(leg.entry_ts),
        "entry_alt_btc_mid": pos.get("entry_alt_btc_mid"),
        "entry_fill_price": pos.get("entry_fill_price"),
        "entry_btc_spent": pos.get("entry_btc_spent"),
        "alt_qty": pos.get("alt_qty"),
        "notional_usd": pos.get("notional_usd"),
        "initial_sl": leg.initial_sl,
        "take_profit": leg.tp,
        "trailing_active": leg.trailing_active,
        "trailing_stop": leg.trailing_stop,
        "closed": leg.closed,
        "exit_ts": str(leg.exit_ts) if leg.exit_ts is not None else None,
        "exit_reason": leg.exit_reason,
        "exit_alt_btc_mid": res.get("exit_alt_btc_mid"),
        "exit_fill_price": res.get("exit_fill_price"),
        "exit_btc_received": res.get("exit_btc_received"),
        "pnl_btc": res.get("pnl_btc"),
        "pnl_pct": res.get("pnl_pct"),
        "pnl_usd_equiv": res.get("pnl_usd_equiv"),
        "gross_pnl_btc_approx": res.get("gross_pnl_btc_approx"),
        "fees_btc": res.get("fees_btc"),
        "slippage_btc_approx": res.get("slippage_btc_approx"),
        "net_pnl_btc": res.get("net_pnl_btc"),
        "holding_hours": hold,
        "bars_held": leg.bars_held,
        "mfe_pct": mfe,
        "mae_pct": float(leg.mae_pct),
        "peak_price": leg.peak_price,
        "trough_price": leg.trough_price,
        "trail_activation_ts": str(leg.trail_activation_ts) if leg.trail_activation_ts else None,
        "trail_activated": bool(leg.trailing_active or leg.trail_activation_ts),
        "activation_return_pct": leg.activation_return_pct,
        "peak_before_exit_pct": float(leg.peak_before_exit_pct),
        "realized_fraction_of_mfe": mfe_frac,
        "time_to_activation_min": leg.time_to_activation_min,
        "time_to_peak_min": leg.time_to_peak_min,
        "time_to_p1_min": (leg.milestone_times_min or {}).get("p1"),
        "time_to_p1_5_min": (leg.milestone_times_min or {}).get("p1_5"),
        "time_to_p2_min": (leg.milestone_times_min or {}).get("p2"),
        "time_to_p3_min": (leg.milestone_times_min or {}).get("p3"),
        "adaptive_mode": leg.spec.adaptive_mode,
        "final_trail_distance_pct": leg.current_trail_distance_pct,
        "stop_floor_pct": leg.stop_floor_pct,
    }
