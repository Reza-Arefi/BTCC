"""Bootstrap Selector E counterfactual history from historical backtest legs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from btcc.sim.selector_engine import CounterfactualHistory

logger = logging.getLogger(__name__)


def bootstrap_cf_history_from_backtest(
    cf_history: CounterfactualHistory,
    *,
    legs_path: Path,
    warmup_days: int = 10,
    asof: pd.Timestamp | None = None,
) -> int:
    """Seed CF history from closed counterfactual legs in a backtest CSV (last ``warmup_days``)."""
    path = Path(legs_path)
    if not path.exists() or path.stat().st_size == 0:
        logger.warning("CF warmup bootstrap skipped — legs file missing: %s", path)
        return 0

    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return 0

    is_cf = df.get("is_counterfactual", pd.Series()).astype(str).str.lower().isin(["true", "1"])
    closed = df.get("closed", pd.Series()).astype(str).str.lower().isin(["true", "1"])
    work = df[is_cf & closed].copy()
    if work.empty or "exit_ts" not in work.columns:
        return 0

    work["exit_ts"] = pd.to_datetime(work["exit_ts"], utc=True, errors="coerce")
    work = work.dropna(subset=["exit_ts"])
    if work.empty:
        return 0

    end = pd.Timestamp(asof).tz_convert("UTC") if asof is not None else work["exit_ts"].max()
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    cutoff = end - pd.Timedelta(days=int(warmup_days))
    work = work[(work["exit_ts"] >= cutoff) & (work["exit_ts"] < end)]
    if work.empty:
        return 0

    existing = {(t.opportunity_id, t.strategy_key) for t in cf_history.trades}
    added = 0
    for _, row in work.sort_values("exit_ts").iterrows():
        oid = str(row.get("opportunity_id"))
        sk = str(row.get("strategy_key"))
        key = (oid, sk)
        if key in existing:
            continue
        cf_history.record(
            opportunity_id=oid,
            strategy_key=sk,
            exit_ts=row["exit_ts"],
            pnl_pct=float(row.get("pnl_pct") or 0.0),
            regime=row.get("regime") if pd.notna(row.get("regime")) else None,
        )
        existing.add(key)
        added += 1

    logger.info(
        "CF warmup bootstrap: +%d trades from %s (last %dd before %s)",
        added,
        path,
        warmup_days,
        end,
    )
    return added


def resolve_warmup_source(sim: dict[str, Any], root: Path | None = None) -> Path | None:
    sl = sim.get("selector_live_raw") or {}
    src = sl.get("cf_warmup_source")
    if not src:
        return None
    p = Path(str(src))
    if not p.is_absolute():
        base = root or Path(__file__).resolve().parents[2]
        p = base / p
    return p if p.exists() else None


def maybe_bootstrap_cf_history(
    cf_history: CounterfactualHistory,
    sim: dict[str, Any],
    *,
    force: bool = False,
) -> int:
    """Bootstrap CF history when empty (or when ``force``)."""
    sl = sim.get("selector_live_raw") or {}
    warmup_days = int(sl.get("cf_warmup_days", 10))
    if not force and cf_history.trades:
        return 0
    src = resolve_warmup_source(sim)
    if src is None:
        logger.warning("CF warmup bootstrap skipped — no cf_warmup_source configured/found")
        return 0
    return bootstrap_cf_history_from_backtest(
        cf_history,
        legs_path=src,
        warmup_days=warmup_days,
    )
