"""Load Adaptive V2 simulation config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIM = ROOT / "configs" / "sim_config.yaml"


def load_sim_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_SIM
    with open(cfg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    sim = data.get("sim", data)
    # Resolve relative storage paths against project root
    storage = sim.setdefault("storage", {})
    for key, val in list(storage.items()):
        if key.endswith("_path") or key == "root":
            p = Path(val)
            if not p.is_absolute():
                storage[key] = str(ROOT / p)
    sim["_root"] = str(ROOT)
    sim["_config_path"] = str(cfg_path)
    return sim
