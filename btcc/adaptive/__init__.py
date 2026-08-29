"""BTCC adaptive walk-forward learning package — SIGNAL ONLY."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTIVE = ROOT / "configs" / "adaptive_config.yaml"


def load_adaptive_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_ADAPTIVE
    with open(cfg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("adaptive", data)
