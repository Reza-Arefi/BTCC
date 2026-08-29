"""Daily adaptive checkpoint — train Challenger only when enough labels exist."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from btcc.adaptive.learner import build_challenger
from btcc.adaptive.metrics import evaluate_model_on_frame
from btcc.adaptive.model import (
    get_champion,
    next_version,
    promote_challenger,
    save_model,
)
from btcc.adaptive.promotion import compare_and_decide
from btcc.adaptive.rescore import rescore_frame
from btcc.adaptive.store import AdaptivePredictionStore

logger = logging.getLogger(__name__)


def write_adaptive_report(
    reports_dir: Path,
    champion,
    decision: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = reports_dir / f"adaptive_report_{day}.md"
    lines = [
        f"# Adaptive Learning Report — {day}",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Current Champion",
        f"- Version: `{champion.version}`",
        f"- Probability kind: {champion.probability_kind}",
        f"- Weights: `{champion.normalized_weights()}`",
        f"- Sample count (at creation): {champion.sample_count}",
        "",
        "## Checkpoint decision",
        f"- Action: **{decision.get('action', decision.get('decision', 'n/a'))}**",
        "",
        "```json",
        json.dumps({k: v for k, v in decision.items() if k not in ("train", "val", "holdout")}, indent=2, default=str),
        "```",
        "",
    ]
    if extra:
        lines += ["## Extra", "```json", json.dumps(extra, indent=2, default=str), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote adaptive report %s", path)
    return path


def run_checkpoint(
    store: AdaptivePredictionStore,
    models_dir: Path,
    reports_dir: Path,
    signal_cfg: dict[str, Any],
    adaptive_cfg: dict[str, Any],
) -> dict[str, Any]:
    """24h checkpoint: attempt Challenger only if sample requirements are met."""
    champion = get_champion(models_dir)
    if champion is None:
        return {"decision": "NO_UPDATE", "reason": "no_champion"}

    horizon = int(adaptive_cfg.get("primary_horizon", 4))
    labeled = store.labeled(horizon)
    n = len(labeled)
    result: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mature_4h_observations": n,
        "champion_version": champion.version,
        "champion_weights": champion.normalized_weights(),
    }

    # Always evaluate current champion on all labeled data for reporting
    champ_eval = evaluate_model_on_frame(rescore_frame(labeled, champion, signal_cfg)) if n else {}
    result["champion_eval_all_labeled"] = champ_eval.get("primary", {})

    challenger, meta = build_challenger(labeled, champion, adaptive_cfg)
    if challenger is None:
        result.update(meta)
        result["action"] = "KEEP_CHAMPION"
        write_adaptive_report(reports_dir, champion, result)
        return result

    challenger.version = next_version(models_dir)
    val = meta["val"]
    decision = compare_and_decide(champion, challenger, val, signal_cfg, adaptive_cfg)
    save_model(challenger, models_dir, f"challenger_{challenger.version}")

    result["challenger_version"] = challenger.version
    result["challenger_weights"] = challenger.normalized_weights()
    result["train_n"] = meta["train_n"]
    result["val_n"] = meta["val_n"]
    result["holdout_n"] = meta["holdout_n"]
    result["train_period"] = challenger.training_period
    result["validation_period"] = challenger.validation_period
    result.update(decision)

    # Gate: never promote until enough LIVE observations mature
    live_tag = adaptive_cfg.get("data_source_tag_live", "LIVE_OBSERVATION_DATA")
    live_n = 0
    if "data_source" in labeled.columns:
        live_n = int((labeled["data_source"] == live_tag).sum())
    result["live_mature_4h"] = live_n
    min_live = int(adaptive_cfg.get("min_live_samples_for_promotion", 200))
    if adaptive_cfg.get("require_live_for_promotion", True) and live_n < min_live:
        result["promote"] = False
        result["action"] = "KEEP_CHAMPION"
        result["promotion_blocked"] = (
            f"Need ≥{min_live} LIVE_OBSERVATION_DATA mature 4h labels "
            f"(have {live_n}). Historical Top-5 alone must not promote."
        )
        write_adaptive_report(reports_dir, champion, result)
        return result

    if decision.get("promote"):
        new_champ = promote_challenger(models_dir, challenger, decision)
        result["promoted_to"] = new_champ.version
        result["action"] = "PROMOTE_CHALLENGER"
        write_adaptive_report(reports_dir, new_champ, result)
    else:
        result["action"] = "KEEP_CHAMPION"
        write_adaptive_report(reports_dir, champion, result)

    return result
