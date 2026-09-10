"""Stage 3 — freeze entry=E2, compare selectors A–F vs best fixed exit (T7).

Replays A–F chronologically on Stage 1's E2 counterfactual T1–T10 legs
(no lookahead: selection uses only CF history with exit_ts < entry_ts).

Keep a selector only if it clearly beats fixed T7 on the 30d holdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_t_strategy_21d_1m_experiment import STARTING_BTC, strategy_metrics  # noqa: E402
from run_t_strategy_21d_1m_experiment import t1_t10_from_live  # noqa: E402
from btcc.sim.selector_engine import CounterfactualHistory, build_selector_group  # noqa: E402

ENTRY = "E2"
FIXED_BENCHMARK = "T7"
EXIT_LABELS = tuple(f"T{i}" for i in range(1, 11))
EXIT_KEYS = tuple(f"trail_{i}" for i in range(1, 11))
LABEL_TO_KEY = {f"T{i}": f"trail_{i}" for i in range(1, 11)}
KEY_TO_LABEL = {v: k for k, v in LABEL_TO_KEY.items()}
DEFAULT_STAGE1 = REPO / "results" / "stage1_entry_90d_1s_20260910_115513"
DEFAULT_STAGE2 = REPO / "results" / "stage2_exit_E2_90d_1s_20260910_155823"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage3_selector")


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _sim_for_selectors() -> dict[str, Any]:
    return {
        "strategies": t1_t10_from_live(),
        "benchmark_strategy_key": "trail_7",
        "selector_experiment": {
            "long_threshold": 0.65,
            "upper_threshold": None,
            "allow_threshold_override": True,
            "disable_late_entry_rejection": True,
            "switching": {
                "minimum_selection_duration_hours": 6,
                "switch_margin": 0.0005,
            },
            "selectors": {
                "A": {"kind": "ewma_7d", "half_life_days": 7},
                "B": {"kind": "multi_horizon_ewma"},
                "C": {"kind": "regime_conditional", "half_life_days": 7, "min_regime_observations": 5},
                "D": {
                    "kind": "recent_plus_regime",
                    "half_life_days": 7,
                    "min_regime_observations": 5,
                    "recent_weight": 0.65,
                    "regime_weight": 0.35,
                },
                "E": {"kind": "rank_ewma", "half_life_days": 7},
                "F": {"kind": "downside_aware", "half_life_days": 7, "downside_lambda": 0.5},
            },
        },
    }


def _summarize(legs: pd.DataFrame, label: str) -> dict[str, Any]:
    m = strategy_metrics(legs, starting_btc=STARTING_BTC, label=label)
    if not legs.empty and "pnl_usd_equiv" in legs.columns:
        usd = pd.to_numeric(legs["pnl_usd_equiv"], errors="coerce")
        net_usd = float(usd.fillna(0.0).sum()) if usd.notna().any() else None
    else:
        net_usd = None
    m["net_usd"] = net_usd
    m["net_return_pct_capital"] = None if net_usd is None else 100.0 * net_usd / 1000.0
    return m


def _plot_bars(summary: pd.DataFrame, col: str, title: str, path: Path, ylabel: str) -> None:
    if summary.empty or col not in summary.columns:
        return
    arms = summary["Strategy"].astype(str).tolist()
    vals = pd.to_numeric(summary[col], errors="coerce").fillna(0.0).tolist()
    colors = ["#2e8b57" if v >= 0 else "#d62728" for v in vals]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(np.arange(len(arms)), vals, color=colors, width=0.7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(arms)))
    ax.set_xticklabels(arms, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def replay_selectors(opps: pd.DataFrame, legs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (selector_legs, selection_audit)."""
    sim = _sim_for_selectors()
    selectors = build_selector_group(sim)
    cf = CounterfactualHistory()

    # Map opportunity -> {arm_label: row}
    leg_map: dict[str, dict[str, pd.Series]] = {}
    for _, r in legs.iterrows():
        leg_map.setdefault(str(r["opportunity_id"]), {})[str(r["arm_key"])] = r

    # Exit events for CF recording (all T legs)
    exit_events: list[tuple[pd.Timestamp, str, str, float, str | None]] = []
    for _, r in legs.iterrows():
        exit_events.append(
            (
                _utc(r["exit_ts"]),
                str(r["opportunity_id"]),
                LABEL_TO_KEY[str(r["arm_key"])],
                float(r["pnl_pct"] or 0.0),
                None,  # regime filled from opp at record time if needed
            )
        )
    exit_events.sort(key=lambda x: x[0])

    # Attach regime from opportunities
    regime_by_oid = {
        str(r["opportunity_id"]): (None if pd.isna(r.get("regime")) else str(r["regime"]))
        for _, r in opps.iterrows()
    }
    exit_events = [
        (ts, oid, sk, pnl, regime_by_oid.get(oid))
        for ts, oid, sk, pnl, _ in exit_events
    ]

    opps_sorted = opps.sort_values("entry_ts").reset_index(drop=True)
    exit_i = 0
    sel_legs: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    recorded: set[tuple[str, str]] = set()

    def drain_exits_before(asof: pd.Timestamp) -> None:
        nonlocal exit_i
        while exit_i < len(exit_events) and exit_events[exit_i][0] < asof:
            ts, oid, sk, pnl, regime = exit_events[exit_i]
            key = (oid, sk)
            if key not in recorded:
                cf.record(
                    opportunity_id=oid,
                    strategy_key=sk,
                    exit_ts=ts,
                    pnl_pct=pnl,
                    regime=regime,
                )
                recorded.add(key)
            exit_i += 1

    for _, opp in opps_sorted.iterrows():
        oid = str(opp["opportunity_id"])
        entry_ts = _utc(opp["entry_ts"])
        phase = str(opp.get("phase") or "")
        regime = None if pd.isna(opp.get("regime")) else str(opp["regime"])
        drain_exits_before(entry_ts)

        arms = leg_map.get(oid) or {}
        if FIXED_BENCHMARK not in arms:
            continue

        for sel in selectors.values():
            pick = sel.select(cf, entry_ts, regime, strategy_keys=EXIT_KEYS)
            chosen_label = str(pick["selected_arm_label"])
            chosen_row = arms.get(chosen_label)
            if chosen_row is None:
                # fallback to T7 if missing
                chosen_label = FIXED_BENCHMARK
                chosen_row = arms[FIXED_BENCHMARK]
            rec = chosen_row.to_dict()
            rec["arm_key"] = sel.arm_label
            rec["selected_T"] = chosen_label
            rec["selected_strategy_key"] = pick["selected_strategy_key"]
            rec["selector_id"] = sel.selector_id
            rec["switched"] = bool(pick.get("switched"))
            rec["selected_score"] = pick.get("selected_score")
            rec["phase"] = phase
            rec["variant"] = ENTRY
            rec["is_selector"] = True
            sel_legs.append(rec)
            audit.append(
                {
                    "opportunity_id": oid,
                    "entry_ts": str(entry_ts),
                    "phase": phase,
                    "regime": regime,
                    "selector": sel.arm_label,
                    "selected_T": chosen_label,
                    "selected_strategy_key": pick["selected_strategy_key"],
                    "selected_score": pick.get("selected_score"),
                    "switched": bool(pick.get("switched")),
                    "pnl_btc": float(chosen_row.get("pnl_btc") or 0.0),
                    "pnl_pct": float(chosen_row.get("pnl_pct") or 0.0),
                }
            )

    # Drain remaining exits (not required for selection after last entry)
    drain_exits_before(_utc("2100-01-01"))

    return pd.DataFrame(sel_legs), pd.DataFrame(audit)


def run(*, stage1_dir: Path, stage2_dir: Path | None, fixed_exit: str = FIXED_BENCHMARK) -> Path:
    legs_all = pd.read_csv(stage1_dir / "strategy_legs.csv", low_memory=False)
    opps_all = pd.read_csv(stage1_dir / "opportunities.csv", low_memory=False)
    meta1 = json.loads((stage1_dir / "run_meta.json").read_text(encoding="utf-8"))

    legs = legs_all[legs_all["variant"].astype(str) == ENTRY].copy()
    opps = opps_all[opps_all["variant"].astype(str) == ENTRY].copy()
    opps["entry_ts"] = pd.to_datetime(opps["entry_ts"], utc=True)
    legs["entry_ts"] = pd.to_datetime(legs["entry_ts"], utc=True)
    legs["exit_ts"] = pd.to_datetime(legs["exit_ts"], utc=True)

    logger.info("Replaying selectors on %d E2 opportunities (%d T-legs)", len(opps), len(legs))
    sel_legs, audit = replay_selectors(opps, legs)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / f"stage3_selector_{ENTRY}_{fixed_exit}_90d_1s_{run_id}"
    plots = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    # Fixed T7 legs + selector legs summaries
    rows: list[dict[str, Any]] = []
    fixed = legs[legs["arm_key"] == fixed_exit].copy()
    for phase in ("full", "select", "holdout"):
        sub = fixed if phase == "full" else fixed[fixed["phase"] == phase]
        m = _summarize(sub, fixed_exit)
        m["arm"] = fixed_exit
        m["phase"] = phase
        m["kind"] = "fixed"
        rows.append(m)

    for sel_label in ("A", "B", "C", "D", "E", "F"):
        sub_all = sel_legs[sel_legs["arm_key"] == sel_label] if not sel_legs.empty else pd.DataFrame()
        for phase in ("full", "select", "holdout"):
            sub = sub_all if phase == "full" else sub_all[sub_all["phase"] == phase]
            m = _summarize(sub, sel_label)
            m["arm"] = sel_label
            m["phase"] = phase
            m["kind"] = "selector"
            if not audit.empty:
                a = audit[audit["selector"] == sel_label]
                if phase != "full":
                    a = a[a["phase"] == phase]
                m["n_switches"] = int(a["switched"].astype(bool).sum()) if not a.empty else 0
                if not a.empty:
                    dist = a["selected_T"].value_counts(normalize=True)
                    m["top_selected_T"] = str(dist.index[0])
                    m["top_selected_T_share"] = float(dist.iloc[0])
            rows.append(m)

    matrix = pd.DataFrame(rows)
    select = matrix[matrix["phase"] == "select"].copy()
    holdout = matrix[matrix["phase"] == "holdout"].copy()

    # Rank selectors+fixed on select by Net BTC
    rank_sel = select.sort_values(by=["Net BTC", "Expectancy"], ascending=[False, False]).reset_index(drop=True)
    fixed_sel = select[select["arm"] == fixed_exit]
    fixed_hold = holdout[holdout["arm"] == fixed_exit]
    fixed_sel_net = float(fixed_sel.iloc[0]["Net BTC"]) if not fixed_sel.empty else None
    fixed_hold_net = float(fixed_hold.iloc[0]["Net BTC"]) if not fixed_hold.empty else None

    # Best selector on select among A-F only
    sel_only = rank_sel[rank_sel["kind"] == "selector"]
    best_sel = str(sel_only.iloc[0]["Strategy"]) if not sel_only.empty else None
    best_sel_net = float(sel_only.iloc[0]["Net BTC"]) if not sel_only.empty else None
    best_sel_hold = holdout[holdout["arm"] == best_sel]
    best_sel_hold_net = float(best_sel_hold.iloc[0]["Net BTC"]) if not best_sel_hold.empty else None

    beats_select = (
        best_sel_net is not None and fixed_sel_net is not None and best_sel_net > fixed_sel_net
    )
    beats_holdout = (
        best_sel_hold_net is not None and fixed_hold_net is not None and best_sel_hold_net > fixed_hold_net
    )
    keep_selector = bool(beats_select and beats_holdout)

    decision = {
        "stage": 3,
        "frozen_entry": ENTRY,
        "fixed_benchmark_exit": fixed_exit,
        "source_stage1": str(stage1_dir),
        "source_stage2": str(stage2_dir) if stage2_dir else None,
        "best_selector_on_select": best_sel,
        "best_selector_select_net_btc": best_sel_net,
        "best_selector_holdout_net_btc": best_sel_hold_net,
        "fixed_T7_select_net_btc": fixed_sel_net,
        "fixed_T7_holdout_net_btc": fixed_hold_net,
        "beats_fixed_on_select": beats_select,
        "beats_fixed_on_holdout": beats_holdout,
        "keep_selector": keep_selector,
        "live_candidate": {
            "entry": ENTRY,
            "exit": best_sel if keep_selector else fixed_exit,
            "exit_kind": "selector" if keep_selector else "fixed",
        },
        "rule": "Keep selector only if it beats fixed T7 on BOTH 60d select and 30d holdout.",
        "next_stage": "Stage 4 sizing/capacity only after live candidate is frozen.",
    }

    meta = {
        "frozen_entry": ENTRY,
        "fixed_benchmark_exit": fixed_exit,
        "eval_start": meta1.get("eval_start"),
        "eval_end": meta1.get("eval_end"),
        "select_end": meta1.get("select_end"),
        "n_opportunities": int(len(opps)),
        "decision": decision,
        "notes": [
            "Stage 3 selector replay on E2 entry stream",
            "CF history = T1–T10 outcomes only; no lookahead",
            "Compare A–F vs frozen Stage-2 winner T7",
        ],
    }

    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    sel_legs.to_csv(out_dir / "selector_legs.csv", index=False)
    audit.to_csv(out_dir / "selection_audit.csv", index=False)
    matrix.to_csv(out_dir / "selector_vs_fixed_matrix.csv", index=False)
    select.to_csv(out_dir / "summary_select60d.csv", index=False)
    holdout.to_csv(out_dir / "summary_holdout30d.csv", index=False)
    rank_sel.to_csv(out_dir / "ranking_select60d.csv", index=False)

    # Plots
    _plot_bars(
        rank_sel,
        "Net BTC",
        f"Stage3 — Net BTC select 60d (entry={ENTRY}, benchmark={fixed_exit})",
        plots / "select_net_btc_AF_vs_T7.png",
        "Net BTC",
    )
    _plot_bars(
        holdout.sort_values("Net BTC", ascending=False),
        "Net BTC",
        f"Stage3 — Net BTC holdout 30d (entry={ENTRY}, benchmark={fixed_exit})",
        plots / "holdout_net_btc_AF_vs_T7.png",
        "Net BTC",
    )

    # Selection frequency heat for each selector
    if not audit.empty:
        for sel in ("A", "B", "C", "D", "E", "F"):
            a = audit[audit["selector"] == sel]
            if a.empty:
                continue
            counts = a["selected_T"].value_counts().reindex(EXIT_LABELS, fill_value=0)
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.bar(range(len(EXIT_LABELS)), counts.values, color="#4c78a8")
            ax.set_xticks(range(len(EXIT_LABELS)))
            ax.set_xticklabels(EXIT_LABELS)
            ax.set_title(f"Selector {sel} — chosen T frequency (all phases)")
            ax.set_ylabel("Trades")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(plots / f"selection_frequency_{sel}.png", dpi=120)
            plt.close(fig)

    cols = ["Strategy", "kind", "Trades", "Win %", "Net BTC", "Return %", "Max DD", "Expectancy"]
    for optional in ("n_switches", "top_selected_T", "top_selected_T_share"):
        if optional in rank_sel.columns:
            cols.append(optional)

    report = [
        f"# Stage 3 — Selectors A–F vs fixed {fixed_exit} (entry={ENTRY})",
        "",
        f"- Source Stage 1: `{stage1_dir}`",
        f"- Window: `{meta1.get('eval_start')}` → `{meta1.get('eval_end')}`",
        f"- Rule: keep selector only if it beats **{fixed_exit}** on select AND holdout",
        "",
        "## Decision",
        f"- Best selector on select: `{best_sel}`",
        f"- Beats {fixed_exit} on select: `{beats_select}`",
        f"- Beats {fixed_exit} on holdout: `{beats_holdout}`",
        f"- **Keep selector:** `{keep_selector}`",
        f"- **Live candidate:** `{decision['live_candidate']}`",
        "",
        "## Select ranking (60d)",
        "```",
        rank_sel[[c for c in cols if c in rank_sel.columns]].to_string(index=False),
        "```",
        "",
        "## Holdout confirmation (30d)",
        "```",
        holdout.sort_values("Net BTC", ascending=False)[[c for c in cols if c in holdout.columns]].to_string(
            index=False
        ),
        "```",
        "",
        "## Next",
        "Stage 4: sizing / capacity on the frozen live candidate only.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(str(x) for x in report), encoding="utf-8")

    logger.info(
        "Stage3 done → %s | keep_selector=%s live=%s",
        out_dir,
        keep_selector,
        decision["live_candidate"],
    )
    print(rank_sel[[c for c in ("Strategy", "kind", "Trades", "Net BTC", "Return %", "Max DD") if c in rank_sel.columns]].to_string(index=False))
    print(f"\nKeep selector: {keep_selector}")
    print(f"Live candidate: {decision['live_candidate']}")
    print(f"Results: {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1)
    ap.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2)
    ap.add_argument("--fixed-exit", default=FIXED_BENCHMARK)
    args = ap.parse_args()
    run(
        stage1_dir=args.stage1_dir.resolve(),
        stage2_dir=args.stage2_dir.resolve() if args.stage2_dir else None,
        fixed_exit=str(args.fixed_exit).upper(),
    )


if __name__ == "__main__":
    main()
