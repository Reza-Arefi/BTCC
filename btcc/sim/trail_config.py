"""Load trailing-exit experiment configuration merged with base sim settings."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from btcc.sim.config import load_sim_config

ROOT = Path(__file__).resolve().parents[2]
TRAIL_CFG_PATH = ROOT / "configs" / "trail_experiment_config.yaml"

TRAIL_STRATEGY_KEYS = tuple(f"trail_{i}" for i in range(1, 11))

# Frozen T1–T10 specification (percent as decimals)
TRAIL_SPEC = {
    "trail_1": (0.0075, 0.0075, 0.0025),
    "trail_2": (0.0100, 0.0100, 0.0025),
    "trail_3": (0.0100, 0.0100, 0.0050),
    "trail_4": (0.0150, 0.0100, 0.0050),
    "trail_5": (0.0150, 0.0150, 0.0050),
    "trail_6": (0.0200, 0.0150, 0.0075),
    "trail_7": (0.0200, 0.0200, 0.0075),
    "trail_8": (0.0250, 0.0200, 0.0100),
    "trail_9": (0.0300, 0.0300, 0.0100),
    "trail_10": (0.0300, 0.0300, 0.0150),
}


def load_trail_experiment_config(path: Path | None = None) -> dict[str, Any]:
    p = path or TRAIL_CFG_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    te = dict(raw.get("trail_experiment") or {})
    sim = load_sim_config()
    merged = deepcopy(sim)
    merged["experiment_kind"] = "trail_exit_v1"
    merged["long_threshold"] = float(te.get("long_threshold", 0.65))
    merged["upper_threshold"] = float(te.get("upper_threshold", 0.85))
    merged["starting_capital_usd"] = float(te.get("starting_capital_usd", 1000.0))
    merged["notional_usd"] = float(te.get("notional_usd", 100.0))
    merged["compound_portfolio"] = bool(te.get("compound_portfolio", True))
    merged["max_open_opportunities"] = int(te.get("max_open_opportunities", 10))
    merged["one_opportunity_per_pair"] = bool(te.get("one_opportunity_per_pair", True))
    merged["strategies"] = deepcopy(te.get("strategies") or {})
    merged["benchmark_strategy_key"] = str(te.get("benchmark_strategy_key", "trail_3"))
    merged["disable_weight_updates"] = True
    merged["disable_late_entry_rejection"] = True
    merged["weight_mode"] = "static"
    merged["entry_policies"] = {"enabled": []}
    bd = merged.get("btc_d_health") or {}
    bd["require_for_new_trades"] = False
    merged["btc_d_health"] = bd
    merged["trail_experiment"] = te
    merged["_trail_config_path"] = str(p)
    return merged


def validate_trail_strategies(sim: dict[str, Any]) -> list[str]:
    """Return list of validation errors (empty if OK)."""
    errors: list[str] = []
    strategies = sim.get("strategies") or {}
    if set(strategies.keys()) != set(TRAIL_STRATEGY_KEYS):
        errors.append(f"Expected strategies {TRAIL_STRATEGY_KEYS}, got {sorted(strategies.keys())}")
    for key, (sl, act, dist) in TRAIL_SPEC.items():
        raw = strategies.get(key)
        if not raw:
            errors.append(f"Missing {key}")
            continue
        if abs(float(raw["stop_loss_pct"]) - sl) > 1e-9:
            errors.append(f"{key} stop_loss_pct mismatch")
        if raw.get("take_profit_pct") is not None:
            errors.append(f"{key} must not have take_profit_pct")
        trail = raw.get("trailing") or {}
        if abs(float(trail.get("activation_pct", -1)) - act) > 1e-9:
            errors.append(f"{key} activation_pct mismatch")
        if abs(float(trail.get("distance_pct", -1)) - dist) > 1e-9:
            errors.append(f"{key} distance_pct mismatch")
    return errors
