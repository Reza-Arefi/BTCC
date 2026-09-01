"""Load E-memory walk-forward experiment configuration."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from btcc.sim.config import load_sim_config

ROOT = Path(__file__).resolve().parents[2]
MEMORY_CFG_PATH = ROOT / "configs" / "experiments" / "selector_E_memory_walkforward.yaml"
SELECTOR_CFG_PATH = ROOT / "configs" / "selector_experiment_config.yaml"

MEMORY_ARM_LABELS = ("E-3", "E-7", "E-14", "E-30", "E-60", "E-90")
CF_ARM_LABELS = tuple(f"T{i}" for i in range(1, 11))
CF_STRATEGY_KEYS = tuple(f"trail_{i}" for i in range(1, 11))


def compute_memory_window(
    *,
    eval_days: int,
    warmup_days: int,
    warmup_bars: int,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return (data_start, cf_warmup_start, eval_start, eval_end)."""
    eval_end = pd.Timestamp(datetime.now(timezone.utc))
    eval_start = eval_end - pd.Timedelta(days=int(eval_days))
    cf_warmup_start = eval_start - pd.Timedelta(days=int(warmup_days))
    data_start = cf_warmup_start - pd.Timedelta(minutes=15 * int(warmup_bars))
    return data_start, cf_warmup_start, eval_start, eval_end


def load_selector_memory_config(path: Path | None = None) -> dict[str, Any]:
    p = path or MEMORY_CFG_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    me = dict(raw.get("selector_memory_experiment") or {})
    sim = load_sim_config()

    src_path = ROOT / str(me.get("strategies_source", "configs/selector_experiment_config.yaml"))
    src_raw = yaml.safe_load(src_path.read_text(encoding="utf-8")) or {}
    all_strategies = dict((src_raw.get("selector_experiment") or {}).get("strategies") or {})

    cf_keys = list(me.get("counterfactual_strategy_keys") or CF_STRATEGY_KEYS)
    strategies = {k: deepcopy(all_strategies[k]) for k in cf_keys if k in all_strategies}

    merged = deepcopy(sim)
    merged["experiment_kind"] = "selector_E_memory_walkforward_v2"
    merged["long_threshold"] = float(me.get("long_threshold", 0.60))
    upper = me.get("upper_threshold")
    merged["upper_threshold"] = float(upper) if upper is not None else None
    merged["starting_capital_usd"] = float(me.get("starting_capital_usd", 1000.0))
    merged["notional_usd"] = float(me.get("notional_usd", 100.0))
    merged["compound_portfolio"] = bool(me.get("compound_portfolio", True))
    merged["max_open_opportunities"] = int(me.get("max_open_opportunities", 10))
    merged["one_opportunity_per_pair"] = bool(me.get("one_opportunity_per_pair", True))
    merged["strategies"] = strategies
    merged["disable_weight_updates"] = True
    merged["disable_late_entry_rejection"] = bool(me.get("disable_late_entry_rejection", True))
    merged["weight_mode"] = "static"
    merged["entry_policies"] = {"enabled": []}
    merged["allow_trading"] = False
    merged["telegram"] = str(me.get("telegram", "OFF"))
    bd = merged.get("btc_d_health") or {}
    bd["require_for_new_trades"] = False
    merged["btc_d_health"] = bd
    btc_d = dict(merged.get("btc_d") or {})
    exp_btc = me.get("btc_d") or {}
    if "enabled" in exp_btc:
        btc_d["enabled"] = bool(exp_btc["enabled"])
    merged["btc_d"] = btc_d
    merged["selector_memory_experiment"] = me
    merged["_memory_config_path"] = str(p)
    return merged


def validate_selector_memory(sim: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    me = sim.get("selector_memory_experiment") or {}
    strategies = sim.get("strategies") or {}
    cf_keys = list(me.get("counterfactual_strategy_keys") or CF_STRATEGY_KEYS)
    if set(strategies.keys()) != set(cf_keys):
        errors.append(f"Expected strategies {cf_keys}, got {sorted(strategies.keys())}")
    if float(sim.get("long_threshold", -1)) != 0.60:
        errors.append("long_threshold must be 0.60")
    if sim.get("upper_threshold") is not None:
        errors.append("upper_threshold must be null")
    if bool(me.get("allow_trading", False)) or bool(me.get("auto_live_handoff", False)):
        errors.append("allow_trading and auto_live_handoff must be false")
    lb = me.get("lookbacks_days") or {}
    for label in MEMORY_ARM_LABELS:
        if label not in lb:
            errors.append(f"Missing lookback for {label}")
    eval_days = int(me.get("eval_days", 365))
    warmup_days = int(me.get("warmup_days", 365))
    max_lb = max(int(v) for v in lb.values()) if lb else 365
    if warmup_days < max_lb:
        errors.append(f"warmup_days ({warmup_days}) must be >= max lookback ({max_lb})")
    if eval_days < 1:
        errors.append("eval_days must be >= 1")
    return errors


def pre_run_memory_summary(sim: dict[str, Any] | None = None) -> str:
    sim = sim or load_selector_memory_config()
    me = sim.get("selector_memory_experiment") or {}
    lb = me.get("lookbacks_days") or {}
    lines = [
        "E-MEMORY WALK-FORWARD EXPERIMENT",
        f"  eval_days = {me.get('eval_days', 365)}",
        f"  warmup_days = {me.get('warmup_days', 365)}",
        f"  EWMA half-life = {me.get('ewma_half_life_days', 7)}d (fixed)",
        f"  min_hold = {(me.get('switching') or {}).get('minimum_selection_duration_hours', 6)}h",
        "",
        "E VARIANTS (lookback only):",
    ]
    for label in MEMORY_ARM_LABELS:
        lines.append(f"  {label}: {lb.get(label)}d history window")
    lines += [
        "",
        "COUNTERFACTUALS: T1–T10",
        f"  entry S >= {float(sim['long_threshold']):.2f}",
        f"  live = {bool(sim.get('allow_trading', False))}",
    ]
    return "\n".join(lines)
