"""Load FINAL_CHECKPOINT into live sim store (explicit historical→live handoff)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from btcc.sim.checkpoint import verify_final_checkpoint
from btcc.sim.score import normalize_weights

logger = logging.getLogger(__name__)


def load_final_checkpoint(final_dir: Path | str) -> dict[str, Any]:
    final_dir = Path(final_dir)
    verify = verify_final_checkpoint(final_dir)
    if not verify.get("ok"):
        raise RuntimeError(f"FINAL_CHECKPOINT invalid: {verify.get('errors')}")
    handoff = json.loads((final_dir / "handoff.json").read_text(encoding="utf-8"))
    weights = handoff.get("final_weights") or {}
    weights = normalize_weights(weights) if weights else {}
    return {
        "path": str(final_dir),
        "handoff": handoff,
        "verify": verify,
        "weights": weights,
        "weight_version": handoff.get("final_weight_version"),
        "git_commit": handoff.get("git_commit"),
        "last_processed_timestamp": handoff.get("last_processed_timestamp"),
        "last_processed_day": handoff.get("last_processed_day"),
    }


def apply_checkpoint_to_live_state(
    store,
    checkpoint: dict[str, Any],
    *,
    fallback_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Persist handoff metadata + weights into live SimStore state (no 365d rerun)."""
    weights = checkpoint.get("weights") or fallback_weights or {}
    weights = normalize_weights(weights)
    state = store.load_state() if hasattr(store, "load_state") else {}
    state = dict(state or {})
    state["initialized_from_historical_checkpoint"] = True
    state["historical_checkpoint_path"] = checkpoint.get("path")
    state["historical_checkpoint_git_commit"] = checkpoint.get("git_commit")
    state["historical_checkpoint_weight_version"] = checkpoint.get("weight_version")
    state["historical_checkpoint_last_day"] = checkpoint.get("last_processed_day")
    state["current_weights"] = weights
    state["weights_version"] = checkpoint.get("weight_version") or "final_checkpoint"
    if hasattr(store, "save_state"):
        store.save_state(state)
    logger.info(
        "Live store initialized from FINAL_CHECKPOINT path=%s version=%s",
        checkpoint.get("path"),
        checkpoint.get("weight_version"),
    )
    return state
