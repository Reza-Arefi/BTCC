"""Load selector E-v1 live configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from btcc.sim.config import load_sim_config

ROOT = Path(__file__).resolve().parents[2]
SELECTOR_LIVE_CFG_PATH = ROOT / "configs" / "selector_live_config.yaml"
SELECTOR_EXP_CFG_PATH = ROOT / "configs" / "selector_experiment_config.yaml"

LIVE_TRAIL_KEYS = tuple(f"trail_{i}" for i in range(1, 11))
LIVE_ARM_LABELS = tuple(f"T{i}" for i in range(1, 11))


def load_selector_live_config(path: Path | None = None) -> dict[str, Any]:
    p = path or SELECTOR_LIVE_CFG_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    sl = dict(raw.get("selector_live") or {})
    sim = load_sim_config()
    merged = deepcopy(sim)
    merged["selector_live"] = sl
    merged["enabled"] = bool(sl.get("enabled", False))
    merged["experiment_kind"] = "selector_live_E_v1"
    merged["bot_version"] = str(sl.get("bot_version", "3.0.0-selector-E-v1"))
    merged["selector_version"] = str(sl.get("version", "E-v1"))
    merged["long_threshold"] = float(sl.get("long_threshold", 0.60))
    upper = sl.get("upper_threshold")
    merged["upper_threshold"] = float(upper) if upper is not None else None
    merged["starting_equity_btc"] = float(sl.get("starting_equity_btc", 0.01311845))
    merged["position_allocation_pct"] = float(sl.get("position_allocation_pct", 0.25))
    merged["compound_equity"] = bool(sl.get("compound_equity", True))
    merged["leverage"] = float(sl.get("leverage", 0))
    merged["paper_mode"] = bool(sl.get("paper_mode", True))
    merged["execution_mode"] = str(sl.get("execution_mode", "PAPER")).upper()
    merged["starting_capital_usd"] = float(sl.get("starting_capital_usd", 1000.0))
    merged["notional_usd"] = float(sl.get("notional_usd", 100.0))
    merged["max_open_opportunities"] = int(sl.get("max_open_opportunities", 4))
    merged["max_total_exposure"] = float(sl.get("max_total_exposure", 1.0))
    merged["one_opportunity_per_pair"] = bool(sl.get("one_opportunity_per_pair", True))
    merged["disable_late_entry_rejection"] = bool(sl.get("disable_late_entry_rejection", True))
    merged["disable_weight_updates"] = True
    merged["weight_mode"] = "static"
    merged["allow_trading"] = False
    bd = merged.get("btc_d_health") or {}
    bd["require_for_new_trades"] = False
    merged["btc_d_health"] = bd

    # Load T1–T10 strategy geometry from experiment config (unchanged baseline)
    exp_raw = yaml.safe_load(SELECTOR_EXP_CFG_PATH.read_text(encoding="utf-8")) or {}
    all_strategies = dict((exp_raw.get("selector_experiment") or {}).get("strategies") or {})
    keys = list(sl.get("counterfactual_strategy_keys") or LIVE_TRAIL_KEYS)
    merged["strategies"] = {k: deepcopy(all_strategies[k]) for k in keys if k in all_strategies}
    merged["storage"] = dict(sl.get("storage") or {})
    merged["analysis_windows"] = list(sl.get("analysis_windows") or [])
    merged["safety"] = dict(sl.get("safety") or {})
    merged["telegram_live"] = dict(sl.get("telegram") or {})
    merged["alerts"] = dict(sl.get("alerts") or {})
    merged["selector_live_raw"] = sl
    merged["_selector_live_config_path"] = str(p)
    return merged


def version_manifest(sim: dict[str, Any] | None = None) -> dict[str, Any]:
    sim = sim or load_selector_live_config()
    sl = sim.get("selector_live_raw") or {}
    strategies = sim.get("strategies") or {}
    if not isinstance(strategies, dict):
        strategies = {}
    return {
        "bot_version": sim.get("bot_version"),
        "selector_version": sim.get("selector_version"),
        "selector_kind": (sl.get("selector") or {}).get("kind"),
        "selector_params": sl.get("selector"),
        "switching": sl.get("switching"),
        "counterfactual_keys": list(strategies.keys()),
        "entry_long_threshold": sim.get("long_threshold"),
        "entry_upper_threshold": sim.get("upper_threshold"),
    }
