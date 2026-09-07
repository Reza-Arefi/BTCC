"""Load trailing-exit experiment configuration merged with base sim settings."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from btcc.sim.config import load_sim_config

ROOT = Path(__file__).resolve().parents[2]
TRAIL_CFG_PATH = ROOT / "configs" / "trail_experiment_config.yaml"

TRAIL_STRATEGY_KEYS = tuple(f"trail_{i}" for i in range(1, 15))

# Frozen T1–T14 specification: (stop_loss_pct, activation_pct, distance_pct)
TRAIL_SPEC = {
    "trail_1": (0.0075, 0.0075, 0.0025),
    "trail_2": (0.0075, 0.0075, 0.0050),
    "trail_3": (0.0075, 0.0075, 0.0075),
    "trail_4": (0.0100, 0.0100, 0.0025),
    "trail_5": (0.0100, 0.0100, 0.0050),  # benchmark = previous S3
    "trail_6": (0.0100, 0.0100, 0.0100),
    "trail_7": (0.0150, 0.0100, 0.0050),
    "trail_8": (0.0150, 0.0100, 0.0100),
    "trail_9": (0.0150, 0.0150, 0.0050),
    "trail_10": (0.0150, 0.0150, 0.0100),
    "trail_11": (0.0150, 0.0150, 0.0150),
    "trail_12": (0.0200, 0.0200, 0.0050),
    "trail_13": (0.0200, 0.0200, 0.0100),
    "trail_14": (0.0200, 0.0200, 0.0200),
}

BENCHMARK_KEY = "trail_5"
ENTRY_S_MIN = 0.55
ENTRY_S_MAX = 0.85  # exclusive


def load_trail_experiment_config(path: Path | None = None) -> dict[str, Any]:
    p = path or TRAIL_CFG_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    te = dict(raw.get("trail_experiment") or {})
    sim = load_sim_config()
    merged = deepcopy(sim)
    merged["experiment_kind"] = "trail_exit_v2"
    merged["long_threshold"] = float(te.get("long_threshold", ENTRY_S_MIN))
    merged["upper_threshold"] = float(te.get("upper_threshold", ENTRY_S_MAX))
    merged["starting_capital_usd"] = float(te.get("starting_capital_usd", 1000.0))
    merged["notional_usd"] = float(te.get("notional_usd", 100.0))
    merged["compound_portfolio"] = bool(te.get("compound_portfolio", True))
    merged["max_open_opportunities"] = int(te.get("max_open_opportunities", 10))
    merged["one_opportunity_per_pair"] = bool(te.get("one_opportunity_per_pair", True))
    merged["strategies"] = deepcopy(te.get("strategies") or {})
    merged["benchmark_strategy_key"] = str(te.get("benchmark_strategy_key", BENCHMARK_KEY))
    merged["disable_weight_updates"] = True
    merged["disable_late_entry_rejection"] = True
    merged["weight_mode"] = "static"
    merged["entry_policies"] = {"enabled": []}
    bd = merged.get("btc_d_health") or {}
    bd["require_for_new_trades"] = False
    merged["btc_d_health"] = bd
    btc_d = dict(merged.get("btc_d") or {})
    exp_btc = te.get("btc_d") or {}
    if "enabled" in exp_btc:
        btc_d["enabled"] = bool(exp_btc["enabled"])
    merged["btc_d"] = btc_d
    merged["trail_experiment"] = te
    merged["_trail_config_path"] = str(p)
    return merged


def validate_trail_strategies(sim: dict[str, Any]) -> list[str]:
    """Return list of validation errors (empty if OK)."""
    errors: list[str] = []
    strategies = sim.get("strategies") or {}
    if set(strategies.keys()) != set(TRAIL_STRATEGY_KEYS):
        errors.append(
            f"Expected strategies {TRAIL_STRATEGY_KEYS}, got {sorted(strategies.keys())}"
        )
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
    if abs(float(sim.get("long_threshold", -1)) - ENTRY_S_MIN) > 1e-9:
        errors.append(f"long_threshold must be {ENTRY_S_MIN}")
    if abs(float(sim.get("upper_threshold", -1)) - ENTRY_S_MAX) > 1e-9:
        errors.append(f"upper_threshold must be {ENTRY_S_MAX}")
    if str(sim.get("benchmark_strategy_key")) != BENCHMARK_KEY:
        errors.append(f"benchmark_strategy_key must be {BENCHMARK_KEY}")
    if not bool(sim.get("disable_late_entry_rejection", False)):
        errors.append("disable_late_entry_rejection must be True")
    if (sim.get("btc_d_health") or {}).get("require_for_new_trades", True):
        errors.append("btc_d_health.require_for_new_trades must be False")
    return errors


def pre_run_config_summary(sim: dict[str, Any] | None = None) -> str:
    """Human-readable pre-run configuration block."""
    sim = sim or load_trail_experiment_config()
    lines = [
        "ENTRY:",
        f"  S_MIN = {float(sim['long_threshold']):.2f}",
        f"  S_MAX = {float(sim['upper_threshold']):.2f}  (exclusive upper → band [{sim['long_threshold']:.2f}, {sim['upper_threshold']:.2f}))",
        f"  STATIC_WEIGHTS = {sim.get('weight_mode') == 'static' and sim.get('disable_weight_updates')}",
        f"  EXHAUSTION_REJECTION = {not bool(sim.get('disable_late_entry_rejection'))}",
        "",
        "STRATEGIES:",
    ]
    for i, key in enumerate(TRAIL_STRATEGY_KEYS, 1):
        sl, act, dist = TRAIL_SPEC[key]
        mark = "  <-- BENCHMARK" if key == BENCHMARK_KEY else ""
        lines.append(
            f"  T{i}: SL=-{sl*100:.2f}% Act=+{act*100:.2f}% Dist={dist*100:.2f}%{mark}"
        )
    lines += [
        "",
        "EXECUTION:",
        f"  capital = ${float(sim.get('starting_capital_usd', 1000)):.0f}",
        f"  max_open = {int(sim.get('max_open_opportunities', 10))}",
        f"  live = {bool(sim.get('allow_trading', False))}",
        f"  telegram = {sim.get('telegram', 'OFF')}",
        f"  auto_live_handoff = {bool((sim.get('trail_experiment') or {}).get('auto_live_handoff', False))}",
        "",
        "PERIOD:",
        f"  {(sim.get('trail_experiment') or {}).get('total_days', 365)} days",
        f"  experiment_kind = {sim.get('experiment_kind')}",
    ]
    return "\n".join(lines)
