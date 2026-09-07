"""Rolling analytics, dashboard, and plots for Selector E-v1 live engine."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from btcc.monitoring.selector_windows import (
    build_window_dashboard,
    counterfactual_performance_by_arm,
    selection_concentration_report,
)
from btcc.sim.selector_live_config import load_selector_live_config, version_manifest
from btcc.sim.selector_live_store import SelectorLiveStore

logger = logging.getLogger(__name__)


def update_selector_live_analytics(sim_cfg: dict[str, Any] | None = None) -> Path:
    sim = sim_cfg or load_selector_live_config()
    store = SelectorLiveStore(sim)
    analytics_dir = store.analytics_dir
    plots_dir = analytics_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    legs = store.read_legs()
    selection = store.read_selection_audit()
    sl_raw = sim.get("selector_live_raw") or {}
    windows = list(sl_raw.get("analysis_windows") or [])
    starting_btc = float(sim.get("starting_equity_btc", 0.01311845))
    state = store.load_state()
    equity_btc = float((state.get("paper_equity") or {}).get("equity_btc", starting_btc))

    concentration = selection_concentration_report(selection, selector="E")
    dashboard = build_window_dashboard(
        legs,
        selection,
        windows=windows,
        selector="E",
        starting_capital=starting_btc * 70000,  # display-scale for pct metrics
    )

    closed = legs[legs.get("closed").astype(str).str.lower().isin(["true", "1"])] if not legs.empty and "closed" in legs.columns else legs
    is_cf = closed["is_counterfactual"].astype(str).str.lower().isin(["true", "1"]) if not closed.empty and "is_counterfactual" in closed.columns else pd.Series(dtype=bool)
    cf_perf = counterfactual_performance_by_arm(closed[is_cf] if not closed.empty else pd.DataFrame())

    manifest = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "version": version_manifest(sim),
        "E_concentration": concentration,
        "window_dashboard": dashboard,
        "counterfactual_avg_trade_pct": cf_perf,
        "safety_state": store.load_state().get("safety_state", "NORMAL"),
        "paper_equity_btc": equity_btc,
        "starting_equity_btc": starting_btc,
    }
    analytics_dir.mkdir(parents=True, exist_ok=True)
    (analytics_dir / "live_metrics.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (analytics_dir / "window_dashboard.json").write_text(json.dumps(dashboard, indent=2, default=str), encoding="utf-8")

    _write_dashboard_csv(analytics_dir / "window_dashboard.csv", dashboard)
    _plot_concentration(plots_dir, concentration)
    logger.info("Selector live analytics → %s", analytics_dir)
    return analytics_dir


def _write_dashboard_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_concentration(out_dir: Path, conc: dict[str, Any]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    freq = conc.get("frequency_pct") or {}
    if not freq:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(freq.keys()), list(freq.values()), color="steelblue")
    ax.set_ylabel("Selection frequency (%)")
    ax.set_title(
        f"E selection concentration | entropy={conc.get('selection_entropy')} | "
        f"unique={conc.get('n_unique_strategies')} | {conc.get('diagnosis', '')}"
    )
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "E_selection_concentration.png", dpi=120)
    plt.close(fig)
