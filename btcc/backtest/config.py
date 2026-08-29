"""Backtest configuration — extends signal_config without modifying it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from btcc.config import load_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BACKTEST_CONFIG = ROOT / "configs" / "backtest_config.yaml"


def load_backtest_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_BACKTEST_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        bt = yaml.safe_load(f)

    signal_path = ROOT / bt["signal_config"]
    if not signal_path.is_absolute():
        signal_path = ROOT / bt["signal_config"]
    cfg = load_config(signal_path)

    # Merge backtest-specific settings
    cfg["backtest"] = bt["backtest"]
    cfg["backtest_data"] = bt["data"]
    cfg["backtest_output"] = bt["output"]
    cfg["_backtest_root"] = str(ROOT)

    for key in ("candle_dir", "dominance_cache"):
        rel = cfg["backtest_data"][key]
        p = Path(rel)
        if not p.is_absolute():
            cfg["backtest_data"][key] = str(ROOT / p)

    out_root = Path(cfg["backtest_output"]["results_root"])
    if not out_root.is_absolute():
        cfg["backtest_output"]["results_root"] = str(ROOT / out_root)

    return cfg
