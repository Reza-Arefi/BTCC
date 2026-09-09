"""Load dynamic trailing-selector experiment configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from btcc.sim.config import load_sim_config

ROOT = Path(__file__).resolve().parents[2]
SELECTOR_CFG_PATH = ROOT / "configs" / "selector_experiment_config.yaml"

FIXED_STRATEGY_KEYS = tuple(f"trail_{i}" for i in range(1, 21))
FIXED_ARM_LABELS = tuple(f"T{i}" for i in range(1, 21))
SELECTOR_ARM_LABELS = ("A", "B", "C", "D", "E", "F")
ALL_ARM_LABELS = FIXED_ARM_LABELS + SELECTOR_ARM_LABELS
SELECTOR_IDS = tuple(f"selector_{c.lower()}" for c in SELECTOR_ARM_LABELS)
BENCHMARK_KEY = "trail_4"
DEFAULT_STRATEGY_KEY = BENCHMARK_KEY
# 3y validation universe (T11/T12 excluded)
PERIOD_FIXED_STRATEGY_KEYS = tuple(f"trail_{i}" for i in range(1, 11))
PERIOD_FIXED_ARM_LABELS = tuple(f"T{i}" for i in range(1, 11))
SELECTOR_3Y_CFG_PATH = ROOT / "configs" / "selector_experiment_3y_config.yaml"
SELECTOR_5M_THR_SWEEP_CFG_PATH = ROOT / "configs" / "selector_experiment_5m_thr_sweep_config.yaml"

# Allowed S thresholds for the 5m strict-`>` sweep experiment
DEFAULT_5M_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)


def _thr_tag(thr: float) -> str:
    """0.60 -> thr0p60"""
    return f"thr{thr:.2f}".replace(".", "p")


def load_selector_experiment_5m_thr_sweep_config(path: Path | None = None) -> dict[str, Any]:
    return load_selector_experiment_config(path or SELECTOR_5M_THR_SWEEP_CFG_PATH)


def apply_threshold_override(sim: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Return a copy of sim with long_threshold and experiment_id_prefix for one thr arm."""
    out = deepcopy(sim)
    thr = float(threshold)
    out["long_threshold"] = thr
    se = dict(out.get("selector_experiment") or {})
    se["long_threshold"] = thr
    base_prefix = str(se.get("experiment_id_prefix") or "selector_compare_5m")
    if "_thr" in base_prefix:
        base_prefix = base_prefix.split("_thr")[0]
    se["experiment_id_prefix"] = f"{base_prefix}_{_thr_tag(thr)}"
    out["selector_experiment"] = se
    return out


def _label_for_key(key: str) -> str:
    if key.startswith("trail_"):
        return f"T{key.split('_')[1]}"
    if key.startswith("selector_"):
        return key.replace("selector_", "").upper()
    return key


def active_fixed_strategy_keys(sim: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Ordered trail_* keys present in the experiment config."""
    if sim is None:
        return FIXED_STRATEGY_KEYS
    strategies = sim.get("strategies") or {}
    keys = [k for k in FIXED_STRATEGY_KEYS if k in strategies]
    if not keys:
        # honor explicit list if strategies keyed differently
        se = sim.get("selector_experiment") or {}
        explicit = se.get("counterfactual_strategy_keys") or se.get("fixed_strategy_keys")
        if explicit:
            keys = [str(k) for k in explicit]
    # Include any trail_N beyond the built-in range that appear in strategies
    extras = [
        k
        for k in strategies
        if str(k).startswith("trail_") and k not in keys
    ]
    extras.sort(key=lambda x: int(str(x).split("_")[1]))
    return tuple(keys + extras)


def active_fixed_arm_labels(sim: dict[str, Any] | None = None) -> tuple[str, ...]:
    return tuple(_label_for_key(k) for k in active_fixed_strategy_keys(sim))


def active_all_arm_labels(sim: dict[str, Any] | None = None) -> tuple[str, ...]:
    return active_fixed_arm_labels(sim) + SELECTOR_ARM_LABELS


def load_selector_experiment_config(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else SELECTOR_CFG_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    se = dict(raw.get("selector_experiment") or {})
    sim = load_sim_config()
    merged = deepcopy(sim)
    merged["experiment_kind"] = str(se.get("version") or se.get("experiment_kind") or "selector_experiment_v1")
    merged["long_threshold"] = float(se.get("long_threshold", 0.60))
    upper = se.get("upper_threshold")
    merged["upper_threshold"] = float(upper) if upper is not None else None
    merged["starting_capital_usd"] = float(se.get("starting_capital_usd", 1000.0))
    merged["notional_usd"] = float(se.get("notional_usd", 100.0))
    merged["compound_portfolio"] = bool(se.get("compound_portfolio", True))
    merged["max_open_opportunities"] = int(se.get("max_open_opportunities", 10))
    merged["one_opportunity_per_pair"] = bool(se.get("one_opportunity_per_pair", True))
    strategies = deepcopy(se.get("strategies") or {})
    # Optional explicit exclusions (e.g. T11/T12 for 3y)
    exclude = set(se.get("exclude_strategy_keys") or [])
    if exclude:
        strategies = {k: v for k, v in strategies.items() if k not in exclude}
    # Optional allow-list
    allow = se.get("counterfactual_strategy_keys") or se.get("fixed_strategy_keys")
    if allow:
        allow_set = set(allow)
        strategies = {k: v for k, v in strategies.items() if k in allow_set}
    merged["strategies"] = strategies
    merged["benchmark_strategy_key"] = str(se.get("benchmark_strategy_key") or BENCHMARK_KEY)
    merged["disable_weight_updates"] = True
    merged["disable_late_entry_rejection"] = bool(se.get("disable_late_entry_rejection", True))
    merged["weight_mode"] = "static"
    merged["entry_policies"] = {"enabled": []}
    merged["allow_trading"] = False
    merged["telegram"] = str(se.get("telegram", "OFF"))
    merged["enable_period_analysis"] = bool(se.get("enable_period_analysis", False))
    merged["total_days"] = int(se.get("total_days", 365))
    merged["threshold_strict"] = bool(se.get("threshold_strict", False))
    merged["allow_threshold_override"] = bool(se.get("allow_threshold_override", False))
    merged["candle_interval"] = se.get("candle_interval") or se.get("interval")
    merged["min_warmup_bars"] = se.get("min_warmup_bars")
    if se.get("thresholds") is not None:
        merged["thresholds"] = [float(x) for x in se["thresholds"]]
    for key in (
        "eval_start",
        "eval_end",
        "candle_dir",
        "candle_venue",
        "offline_candles",
        "universe_bases",
        "compare_baseline",
    ):
        if key in se:
            merged[key] = se[key]
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


def load_selector_experiment_3y_config(path: Path | None = None) -> dict[str, Any]:
    return load_selector_experiment_config(path or SELECTOR_3Y_CFG_PATH)


def validate_selector_experiment(sim: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    strategies = sim.get("strategies") or {}
    keys = active_fixed_strategy_keys(sim)
    if not keys:
        errors.append("No fixed trail strategies configured")
    forbidden = {"trail_11", "trail_12"}
    se = sim.get("selector_experiment") or {}
    kind = str(sim.get("experiment_kind", ""))
    exclude_t11 = bool(se.get("exclude_t11_t12", False)) or kind.endswith("3y") or "5m_thr" in kind
    if exclude_t11:
        overlap = forbidden.intersection(strategies)
        if overlap:
            errors.append(f"T11/T12 must be excluded for this experiment, found {sorted(overlap)}")
    for key in keys:
        raw = strategies.get(key)
        if not raw:
            errors.append(f"Missing {key}")
            continue
        if raw.get("take_profit_pct") is not None:
            errors.append(f"{key} must not have take_profit_pct")
        trail = raw.get("trailing") or {}
        if trail.get("activation_pct") is None:
            errors.append(f"{key} trailing activation required")
        if trail.get("distance_pct") is None and not raw.get("adaptive_mode"):
            errors.append(f"{key} trailing distance required (or adaptive_mode)")
    allow_thr = bool(sim.get("allow_threshold_override") or se.get("allow_threshold_override"))
    thr = float(sim.get("long_threshold", -1))
    if not allow_thr and thr != 0.60:
        errors.append("long_threshold must be 0.60")
    if allow_thr and not (0.0 < thr < 1.0):
        errors.append(f"long_threshold out of range: {thr}")
    if sim.get("upper_threshold") is not None:
        errors.append("upper_threshold must be null (no upper S band)")
    if not bool(sim.get("disable_late_entry_rejection", False)):
        errors.append("disable_late_entry_rejection must be True")
    if (sim.get("btc_d_health") or {}).get("require_for_new_trades", True):
        errors.append("btc_d_health.require_for_new_trades must be False")
    if bool(se.get("allow_trading", False)) or bool(se.get("auto_live_handoff", False)):
        errors.append("allow_trading and auto_live_handoff must be false")
    sw = se.get("switching") or {}
    if float(sw.get("minimum_selection_duration_hours", 0)) < 0:
        errors.append("minimum_selection_duration_hours must be >= 0")
    return errors


def pre_run_config_summary(sim: dict[str, Any] | None = None) -> str:
    sim = sim or load_selector_experiment_config()
    se = sim.get("selector_experiment") or {}
    keys = active_fixed_strategy_keys(sim)
    thr = float(sim["long_threshold"])
    op = ">" if bool(sim.get("threshold_strict") or se.get("threshold_strict")) else ">="
    interval = sim.get("candle_interval") or se.get("candle_interval") or "15m"
    lines = [
        "ENTRY:",
        f"  S {op} {thr:.2f}  (no upper cap)",
        f"  threshold_strict = {bool(sim.get('threshold_strict') or se.get('threshold_strict'))}",
        f"  candle_interval = {interval}",
        f"  STATIC_WEIGHTS = True",
        f"  LATE_ENTRY_REJECTION = {not bool(sim.get('disable_late_entry_rejection'))}",
        "",
        f"FIXED STRATEGIES ({len(keys)}):",
    ]
    for key in keys:
        raw = (sim.get("strategies") or {}).get(key, {})
        i = int(key.split("_")[1])
        sl = float(raw.get("stop_loss_pct", 0)) * 100
        act = float((raw.get("trailing") or {}).get("activation_pct", 0)) * 100
        dist = float((raw.get("trailing") or {}).get("distance_pct", 0)) * 100
        mark = "  <-- BENCHMARK" if key == sim.get("benchmark_strategy_key", BENCHMARK_KEY) else ""
        lines.append(f"  T{i}: SL=-{sl:.2f}% Act=+{act:.2f}% Dist={dist:.2f}%{mark}")
    if set(FIXED_STRATEGY_KEYS) - set(keys):
        missing = sorted(set(FIXED_STRATEGY_KEYS) - set(keys))
        lines.append(f"  EXCLUDED: {', '.join(missing)}")
    lines += [
        "",
        "DYNAMIC SELECTORS: A B C D E F",
        f"  min_selection_hours = {(se.get('switching') or {}).get('minimum_selection_duration_hours', 6)}",
        f"  switch_margin = {(se.get('switching') or {}).get('switch_margin', 0.0005)}",
        "",
        "EXECUTION:",
        f"  capital = ${float(sim.get('starting_capital_usd', 1000)):.0f} per arm ({len(keys) + 6} arms)",
        f"  max_open = {int(sim.get('max_open_opportunities', 10))}",
        f"  live = {bool(sim.get('allow_trading', False))}",
        f"  telegram = {sim.get('telegram', 'OFF')}",
        f"  total_days = {se.get('total_days', sim.get('total_days', 365))}",
        f"  enable_period_analysis = {bool(sim.get('enable_period_analysis', False))}",
    ]
    return "\n".join(lines)
