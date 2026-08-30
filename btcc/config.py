"""BTCC config loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "signal_config.yaml"
DEFAULT_ADAPTIVE = ROOT / "configs" / "adaptive_config.yaml"
DEFAULT_SIM = ROOT / "configs" / "sim_config.yaml"


def load_adaptive_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_ADAPTIVE
    if not cfg_path.exists():
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("adaptive", data)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("candle_dir", "prediction_dir"):
        if "data" in cfg and key in cfg["data"]:
            p = Path(cfg["data"][key])
            if not p.is_absolute():
                cfg["data"][key] = str(ROOT / p)
    cfg["_root"] = str(ROOT)
    cfg["telegram"]["bot_token"] = os.getenv("BTCC_TELEGRAM_BOT_TOKEN", "")
    cfg["telegram"]["chat_id"] = os.getenv("BTCC_TELEGRAM_CHAT_ID", "")
    # Merge adaptive walk-forward settings (optional file)
    adaptive = load_adaptive_config()
    if adaptive:
        for key in ("models_dir", "predictions_path", "reports_dir", "initial_backtest"):
            if key in adaptive and adaptive[key] and not Path(adaptive[key]).is_absolute():
                adaptive[key] = str(ROOT / adaptive[key])
        cfg["adaptive"] = adaptive
    else:
        cfg.setdefault("adaptive", {"enabled": False})
    # Adaptive V2 simulation config (optional)
    if DEFAULT_SIM.exists():
        from btcc.sim.config import load_sim_config

        cfg["sim"] = load_sim_config(DEFAULT_SIM)
    else:
        cfg.setdefault("sim", {"enabled": False})
    return cfg


def usdt_symbols(cfg: dict[str, Any]) -> list[str]:
    from btcc.universe import usdt_symbols as _usdt

    return _usdt(cfg)
