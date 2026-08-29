"""Champion / Challenger model persistence — SIGNAL ONLY."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FACTOR_KEYS = (
    "momentum",
    "trend",
    "btc_regime",
    "volume",
    "volatility",
    "rsi",
    "structure",
)


@dataclass
class SignalModel:
    version: str
    model_type: str  # champion | challenger
    factor_weights: dict[str, float]
    probability_kind: str = "baseline_model_probability"
    calibration: dict[str, list] | None = None
    training_period: dict[str, str] | None = None
    validation_period: dict[str, str] | None = None
    sample_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    created_utc: str = ""
    promoted_from: str | None = None
    notes: str = ""
    source_tags: list[str] = field(default_factory=list)

    def normalized_weights(self) -> dict[str, float]:
        w = {k: float(self.factor_weights.get(k, 0.0)) for k in FACTOR_KEYS}
        s = sum(w.values())
        if s <= 0:
            n = len(FACTOR_KEYS)
            return {k: 1.0 / n for k in FACTOR_KEYS}
        return {k: v / s for k, v in w.items()}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_dir(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_model(model: SignalModel, root: Path, name: str | None = None) -> Path:
    name = name or f"{model.model_type}_{model.version}"
    d = model_dir(root, name)
    payload = asdict(model)
    (d / "model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return d


def load_model(path: Path) -> SignalModel:
    data = json.loads((path / "model.json").read_text(encoding="utf-8"))
    return SignalModel(**data)


def champion_pointer(root: Path) -> Path:
    return root / "CURRENT_CHAMPION.txt"


def get_champion(root: Path) -> SignalModel | None:
    ptr = champion_pointer(root)
    if not ptr.exists():
        return None
    name = ptr.read_text(encoding="utf-8").strip()
    d = root / name
    if not (d / "model.json").exists():
        return None
    return load_model(d)


def set_champion(root: Path, model: SignalModel, folder_name: str | None = None) -> Path:
    folder_name = folder_name or f"champion_{model.version}"
    model.model_type = "champion"
    d = save_model(model, root, folder_name)
    champion_pointer(root).write_text(folder_name, encoding="utf-8")
    return d


def create_champion_v1_from_config(cfg: dict[str, Any], root: Path) -> SignalModel:
    """Initial Champion = current fixed signal_config factor weights."""
    weights = dict(cfg["factors"]["weights"])
    model = SignalModel(
        version="v1",
        model_type="champion",
        factor_weights={k: float(weights[k]) for k in FACTOR_KEYS},
        probability_kind="baseline_model_probability",
        calibration=None,
        training_period={"source": "signal_config_baseline", "backtest": cfg.get("adaptive", {}).get("initial_backtest", "")},
        sample_count=0,
        metrics={},
        created_utc=_now(),
        notes="Initial Champion from live BTCC fixed weights. Not yet adapted.",
        source_tags=["HISTORICAL_INITIAL_DATA"],
    )
    set_champion(root, model, "champion_v1")
    return model


def next_version(root: Path, prefix: str = "v") -> str:
    """Return next version like v2, v3 based on existing champion_* folders."""
    n = 1
    for p in root.glob("champion_v*"):
        try:
            n = max(n, int(p.name.replace("champion_v", "")) + 1)
        except ValueError:
            continue
    for p in root.glob("challenger_v*"):
        try:
            n = max(n, int(p.name.replace("challenger_v", "")))
        except ValueError:
            continue
    return f"{prefix}{n}"


def promote_challenger(root: Path, challenger: SignalModel, decision: dict) -> SignalModel:
    """Promote Challenger → new Champion; keep prior Champion folder intact."""
    ver = challenger.version
    champion = SignalModel(
        version=ver,
        model_type="champion",
        factor_weights=challenger.normalized_weights(),
        probability_kind=challenger.probability_kind,
        calibration=challenger.calibration,
        training_period=challenger.training_period,
        validation_period=challenger.validation_period,
        sample_count=challenger.sample_count,
        metrics={**challenger.metrics, "promotion_decision": decision},
        created_utc=_now(),
        promoted_from=f"challenger_{ver}",
        notes="Promoted from Challenger after out-of-sample validation.",
        source_tags=list(challenger.source_tags),
    )
    set_champion(root, champion, f"champion_{ver}")
    # Also persist decision file
    save_model(challenger, root, f"challenger_{ver}")
    (root / f"challenger_{ver}" / "promotion.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    return champion


def copy_tree_safe(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    shutil.copytree(src, dst)
