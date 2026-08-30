"""Daily weight update at 23:00 America/Sao_Paulo (idempotent)."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from btcc.sim.store import SimStore
from btcc.sim.weights import estimate_weights_from_window, filter_rolling_window

logger = logging.getLogger(__name__)


def local_now(tz_name: str = "America/Sao_Paulo") -> datetime:
    return datetime.now(ZoneInfo(tz_name))


def should_run_daily_update(sim_cfg: dict[str, Any], store: SimStore, now_local: datetime | None = None) -> bool:
    wu = sim_cfg.get("weight_update") or {}
    tz = ZoneInfo(str(wu.get("timezone", "America/Sao_Paulo")))
    now_local = now_local or datetime.now(tz)
    hour = int(wu.get("hour", 23))
    minute = int(wu.get("minute", 0))
    # Run once the local clock is at/after scheduled time on a calendar day not yet processed
    if (now_local.hour, now_local.minute) < (hour, minute):
        return False
    today = now_local.date().isoformat()
    last = store.last_daily_update_date()
    return last != today


def run_daily_weight_update(
    sim_cfg: dict[str, Any],
    store: SimStore,
    *,
    now_utc: datetime | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Recalculate weights from rolling 90d window. Idempotent per local date."""
    wu = sim_cfg.get("weight_update") or {}
    tz_name = str(wu.get("timezone", "America/Sao_Paulo"))
    tz = ZoneInfo(tz_name)
    now_local = (now_utc.astimezone(tz) if now_utc else datetime.now(tz))
    local_date = now_local.date().isoformat()

    if not force and store.last_daily_update_date() == local_date:
        return {
            "status": "SKIPPED",
            "reason": "already_updated_today",
            "local_date": local_date,
        }

    update_id = f"wu_{local_date}_{uuid.uuid4().hex[:8]}"
    prev = store.get_current_weights(
        fallback=dict((sim_cfg.get("_fallback_factor_weights") or {}))
    )
    # If no fallback embedded, try empty → equal weights via estimate safeguards
    if not prev:
        from btcc.sim.score import FACTOR_KEYS
        prev = {k: 1.0 / len(FACTOR_KEYS) for k in FACTOR_KEYS}

    preds = store.load_predictions()
    end_ts = pd.Timestamp(now_local.astimezone(ZoneInfo("UTC")))
    window_days = int(wu.get("rolling_window_days", 90))
    window = filter_rolling_window(preds, end_ts, window_days) if not preds.empty else preds

    new_w, stats = estimate_weights_from_window(
        window,
        prev,
        half_life_days=float(wu.get("time_decay_half_life_days", 45)),
        weight_min=float(wu.get("weight_min", 0.05)),
        weight_max=float(wu.get("weight_max", 0.40)),
        min_obs_total=int(wu.get("min_observations_total", 200)),
        min_obs_per_factor=int(wu.get("min_observations_per_factor", 50)),
        stability_blend=float(wu.get("stability_blend", 0.25)),
        horizon=int(sim_cfg.get("primary_horizon_hours", 4)),
    )

    hist_rows = []
    if new_w is None:
        status = "FAILED"
        notes = stats.get("reason", "unknown")
        logger.warning("Daily weight update FAILED (%s) — keeping previous weights", notes)
        for ind in prev:
            hist_rows.append({
                "update_timestamp": now_local.isoformat(),
                "update_id": update_id,
                "timezone": tz_name,
                "learning_window_start": stats.get("learning_window_start"),
                "learning_window_end": stats.get("learning_window_end"),
                "n_samples": stats.get("n_samples"),
                "indicator": ind,
                "old_weight": prev.get(ind),
                "new_weight": prev.get(ind),
                "indicator_ic": (stats.get("ics") or {}).get(ind),
                "indicator_n": (stats.get("ns") or {}).get(ind),
                "update_status": status,
                "notes": notes,
            })
        store.append_weight_history(hist_rows)
        # Still mark the day to keep idempotency (failed attempts shouldn't loop forever
        # destroying state — but allow force retry). Spec: failed update must not destroy
        # last known-good; we keep prev. Marking the day prevents double-apply storms;
        # operators can --force.
        store.mark_daily_update(local_date, update_id)
        record = {
            "update_id": update_id,
            "local_date": local_date,
            "status": status,
            "notes": notes,
            "weights": prev,
            "stats": {k: v for k, v in stats.items() if k not in ("ics", "ns", "raw_weights")},
        }
        store.append_daily_update_log(record)
        return record

    for ind in new_w:
        hist_rows.append({
            "update_timestamp": now_local.isoformat(),
            "update_id": update_id,
            "timezone": tz_name,
            "learning_window_start": stats.get("learning_window_start"),
            "learning_window_end": stats.get("learning_window_end"),
            "n_samples": stats.get("n_samples"),
            "indicator": ind,
            "old_weight": prev.get(ind),
            "new_weight": new_w.get(ind),
            "indicator_ic": (stats.get("ics") or {}).get(ind),
            "indicator_n": (stats.get("ns") or {}).get(ind),
            "update_status": "OK",
            "notes": "always_update_rolling_90d",
        })
    store.append_weight_history(hist_rows)
    store.set_current_weights(
        new_w,
        meta={
            "update_id": update_id,
            "local_date": local_date,
            "learning_window_start": stats.get("learning_window_start"),
            "learning_window_end": stats.get("learning_window_end"),
            "n_samples": stats.get("n_samples"),
        },
    )
    store.mark_daily_update(local_date, update_id)
    record = {
        "update_id": update_id,
        "local_date": local_date,
        "status": "OK",
        "weights": new_w,
        "old_weights": prev,
        "stats": {
            "n_samples": stats.get("n_samples"),
            "learning_window_start": stats.get("learning_window_start"),
            "learning_window_end": stats.get("learning_window_end"),
            "ycol": stats.get("ycol"),
        },
    }
    store.append_daily_update_log(record)
    store.append_learning_metrics([{
        "update_id": update_id,
        "local_date": local_date,
        "n_samples": stats.get("n_samples"),
        "learning_window_start": stats.get("learning_window_start"),
        "learning_window_end": stats.get("learning_window_end"),
        **{f"w_{k}": new_w[k] for k in new_w},
        **{f"ic_{k}": (stats.get("ics") or {}).get(k) for k in new_w},
    }])
    logger.info(
        "Daily weight update OK id=%s n=%s window=%s→%s",
        update_id,
        stats.get("n_samples"),
        stats.get("learning_window_start"),
        stats.get("learning_window_end"),
    )
    return record
