"""Load dynamic trailing-selector experiment configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from btcc.sim.config import load_sim_config

ROOT = Path(__file__).resolve().parents[2]
SELECTOR_CFG_PATH = ROOT / "configs" / "selector_experiment_config.yaml"

FIXED_STRATEGY_KEYS = tuple(f"trail_{i}" for i in range(1, 13))
FIXED_ARM_LABELS = tuple(f"T{i}" for i in range(1, 13))
SELECTOR_ARM_LABELS = ("A", "B", "C", "D", "E", "F")
ALL_ARM_LABELS = FIXED_ARM_LABELS + SELECTOR_ARM_LABELS
SELECTOR_IDS = tuple(f"selector_{c.lower()}" for c in SELECTOR_ARM_LABELS)
BENCHMARK_KEY = "trail_4"
DEFAULT_STRATEGY_KEY = BENCHMARK_KEY


def _label_for_key(key: str) -> str:
    if key.startswith("trail_"):
        return f"T{key.split('_')[1]}"
    if key.startswith("selector_"):
        return key.replace("selector_", "").upper()
    return key


def load_selector_experiment_config(path: Path | None = None) -> dict[str, Any]:
    p = path or SELECTOR_CFG_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    se = dict(raw.get("selector_experiment") or {})
    sim = load_sim_config()
    merged = deepcopy(sim)
    merged["experiment_kind"] = "selector_experiment_v1"
    merged["long_threshold"] = float(se.get("long_threshold", 0.60))
    upper = se.get("upper_threshold")
    merged["upper_threshold"] = float(upper) if upper is not None else None
    merged["starting_capital_usd"] = float(se.get("starting_capital_usd", 1000.0))
    merged["notional_usd"] = float(se.get("notional_usd", 100.0))
    merged["compound_portfolio"] = bool(se.get("compound_portfolio", True))
    merged["max_open_opportunities"] = int(se.get("max_open_opportunities", 10))
    merged["one_opportunity_per_pair"] = bool(se.get("one_opportunity_per_pair", True))
    merged["strategies"] = deepcopy(se.get("strategies") or {})
    merged["benchmark_strategy_key"] = BENCHMARK_KEY
    merged["disable_weight_updates"] = True
    merged["disable_late_entry_rejection"] = bool(se.get("disable_late_entry_rejection", True))
    merged["weight_mode"] = "static"
    merged["entry_policies"] = {"enabled": []}
    merged["allow_trading"] = False
    merged["telegram"] = str(se.get("telegram", "OFF"))
    bd = merged.get("btc_d_health") or {}
    bd["require_for_new_trades"] = False
    merged["btc_d_health"] = bd
    btc_d = dict(merged.get("btc_d") or {})
    exp_btc = se.get("btc_d") or {}
    if "enabled" in exp_btc:
        btc_d["enabled"] = bool(exp_btc["enabled"])
    merged["btc_d"] = btc_d
    merged["selector_experiment"] = se
    merged["_selector_config_path"] = str(p)
    return merged


def validate_selector_experiment(sim: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    strategies = sim.get("strategies") or {}
    if set(strategies.keys()) != set(FIXED_STRATEGY_KEYS):
        errors.append(f"Expected strategies {FIXED_STRATEGY_KEYS}, got {sorted(strategies.keys())}")
    for key in FIXED_STRATEGY_KEYS:
        raw = strategies.get(key)
        if not raw:
            errors.append(f"Missing {key}")
            continue
        if raw.get("take_profit_pct") is not None:
            errors.append(f"{key} must not have take_profit_pct")
        trail = raw.get("trailing") or {}
        if trail.get("activation_pct") is None or trail.get("distance_pct") is None:
            errors.append(f"{key} trailing activation/distance required")
    if float(sim.get("long_threshold", -1)) != 0.60:
        errors.append("long_threshold must be 0.60")
    if sim.get("upper_threshold") is not None:
        errors.append("upper_threshold must be null (S >= 0.60 only)")
    if not bool(sim.get("disable_late_entry_rejection", False)):
        errors.append("disable_late_entry_rejection must be True")
    if (sim.get("btc_d_health") or {}).get("require_for_new_trades", True):
        errors.append("btc_d_health.require_for_new_trades must be False")
    se = sim.get("selector_experiment") or {}
    if bool(se.get("allow_trading", False)) or bool(se.get("auto_live_handoff", False)):
        errors.append("allow_trading and auto_live_handoff must be false")
    sw = se.get("switching") or {}
    if float(sw.get("minimum_selection_duration_hours", 0)) < 0:
        errors.append("minimum_selection_duration_hours must be >= 0")
    return errors


def pre_run_config_summary(sim: dict[str, Any] | None = None) -> str:
    sim = sim or load_selector_experiment_config()
    se = sim.get("selector_experiment") or {}
    lines = [
        "ENTRY:",
        f"  S >= {float(sim['long_threshold']):.2f}  (no upper cap)",
        f"  STATIC_WEIGHTS = True",
        f"  LATE_ENTRY_REJECTION = {not bool(sim.get('disable_late_entry_rejection'))}",
        "",
        "FIXED STRATEGIES T1-T12:",
    ]
    for i, key in enumerate(FIXED_STRATEGY_KEYS, 1):
        raw = (sim.get("strategies") or {}).get(key, {})
        sl = float(raw.get("stop_loss_pct", 0)) * 100
        act = float((raw.get("trailing") or {}).get("activation_pct", 0)) * 100
        dist = float((raw.get("trailing") or {}).get("distance_pct", 0)) * 100
        mark = "  <-- BENCHMARK" if key == BENCHMARK_KEY else ""
        lines.append(f"  T{i}: SL=-{sl:.2f}% Act=+{act:.2f}% Dist={dist:.2f}%{mark}")
    lines += [
        "",
        "DYNAMIC SELECTORS: A B C D E F",
        f"  min_selection_hours = {(se.get('switching') or {}).get('minimum_selection_duration_hours', 6)}",
        f"  switch_margin = {(se.get('switching') or {}).get('switch_margin', 0.0005)}",
        "",
        "EXECUTION:",
        f"  capital = ${float(sim.get('starting_capital_usd', 1000)):.0f} per arm (18 arms)",
        f"  max_open = {int(sim.get('max_open_opportunities', 10))}",
        f"  live = {bool(sim.get('allow_trading', False))}",
        f"  telegram = {sim.get('telegram', 'OFF')}",
        f"  total_days = {se.get('total_days', 365)}",
    ]
    return "\n".join(lines)
