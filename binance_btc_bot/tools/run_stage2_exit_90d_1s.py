"""Stage 2 — freeze winning entry (default E2), re-rank T1–T10 exits.

Uses the same entry stream / 1s exits as Stage 1.
By default loads E2 legs from a Stage 1 results dir (no re-sim).
Optional --resim to run a fresh E2-only backtest (hours).

Results → results/stage2_exit_E2_90d_1s_<timestamp>/
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

EXIT_LABELS = tuple(f"T{i}" for i in range(1, 11))
DEFAULT_ENTRY = "E2"
DEFAULT_STAGE1 = REPO / "results" / "stage1_entry_90d_1s_20260910_115513"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage2_exit")


def _summarize(legs: pd.DataFrame, label: str) -> dict[str, Any]:
    m = strategy_metrics(legs, starting_btc=STARTING_BTC, label=label)
    if not legs.empty and "pnl_usd_equiv" in legs.columns:
        usd = pd.to_numeric(legs["pnl_usd_equiv"], errors="coerce")
        net_usd = float(usd.fillna(0.0).sum()) if usd.notna().any() else None
    else:
        net_usd = None
        if not legs.empty and "pnl_pct" in legs.columns:
            net_usd = float(pd.to_numeric(legs["pnl_pct"], errors="coerce").fillna(0.0).sum() * 125.0)
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


def load_entry_legs(stage1_dir: Path, entry: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    legs_path = stage1_dir / "strategy_legs.csv"
    meta_path = stage1_dir / "run_meta.json"
    if not legs_path.exists():
        raise FileNotFoundError(f"missing {legs_path}")
    legs = pd.read_csv(legs_path, low_memory=False)
    legs = legs[legs["variant"].astype(str) == entry].copy()
    if legs.empty:
        raise RuntimeError(f"no legs for entry={entry} in {stage1_dir}")
    legs["entry_ts"] = pd.to_datetime(legs["entry_ts"], utc=True)
    legs["exit_ts"] = pd.to_datetime(legs["exit_ts"], utc=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return legs, meta


def run_from_stage1(*, stage1_dir: Path, entry: str = DEFAULT_ENTRY) -> Path:
    legs, stage1_meta = load_entry_legs(stage1_dir, entry)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / f"stage2_exit_{entry}_90d_1s_{run_id}"
    plots = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for arm in EXIT_LABELS:
        sub = legs[legs["arm_key"] == arm]
        for phase_name, phase_df in (
            ("full", sub),
            ("select", sub[sub["phase"] == "select"] if "phase" in sub.columns else pd.DataFrame()),
            ("holdout", sub[sub["phase"] == "holdout"] if "phase" in sub.columns else pd.DataFrame()),
        ):
            m = _summarize(phase_df, arm)
            m["arm"] = arm
            m["phase"] = phase_name
            m["entry"] = entry
            rows.append(m)
    matrix = pd.DataFrame(rows)

    select = matrix[matrix["phase"] == "select"].copy()
    holdout = matrix[matrix["phase"] == "holdout"].copy()
    full = matrix[matrix["phase"] == "full"].copy()

    rank_sel = select.copy()
    rank_sel["_dd"] = pd.to_numeric(rank_sel["Max DD"], errors="coerce").fillna(-1)
    rank_sel = rank_sel.sort_values(
        by=["Net BTC", "Expectancy", "_dd"], ascending=[False, False, False]
    ).drop(columns=["_dd"]).reset_index(drop=True)
    winner = str(rank_sel.iloc[0]["Strategy"]) if not rank_sel.empty else None
    hold_row = holdout[holdout["Strategy"] == winner]
    hold = hold_row.iloc[0].to_dict() if not hold_row.empty else {}

    decision = {
        "stage": 2,
        "frozen_entry": entry,
        "exit_arms": list(EXIT_LABELS),
        "source_stage1": str(stage1_dir),
        "primary_rule": "Net BTC on 60d select (E2 entry stream); confirm on 30d holdout",
        "winner_on_select": winner,
        "winner_select_net_btc": None if not winner else float(rank_sel.iloc[0]["Net BTC"] or 0),
        "winner_select_return_pct": None if not winner else rank_sel.iloc[0].get("Return %"),
        "winner_select_max_dd": None if not winner else rank_sel.iloc[0].get("Max DD"),
        "winner_select_trades": None if not winner else int(rank_sel.iloc[0].get("Trades") or 0),
        "winner_holdout_net_btc": hold.get("Net BTC"),
        "winner_holdout_return_pct": hold.get("Return %"),
        "winner_holdout_max_dd": hold.get("Max DD"),
        "winner_holdout_trades": hold.get("Trades"),
        "next_stage": "Stage 3: compare selectors A–F vs best fixed T on frozen E2 entry",
        "rule": "Do not pick exit on full 90d. Stage-1 best_T is only a hint; Stage-2 select ranking is authoritative for the fixed entry.",
    }

    meta = {
        "frozen_entry": entry,
        "eval_start": stage1_meta.get("eval_start"),
        "eval_end": stage1_meta.get("eval_end"),
        "select_end": stage1_meta.get("select_end"),
        "days": stage1_meta.get("days", 90),
        "select_days": stage1_meta.get("select_days", 60),
        "holdout_days": stage1_meta.get("holdout_days", 30),
        "signal_interval": stage1_meta.get("signal_interval", "15m"),
        "exit_interval": stage1_meta.get("exit_interval", "1s"),
        "n_e2_legs": int(len(legs)),
        "n_opportunities_approx": int(legs["opportunity_id"].nunique()) if not legs.empty else 0,
        "decision": decision,
        "notes": [
            f"Stage 2 exit ranking on frozen entry={entry}",
            "Reused Stage 1 counterfactual T1–T10 legs on the E2 entry stream (same candles/costs)",
            "No selectors in this stage",
        ],
    }

    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    legs.to_csv(out_dir / "strategy_legs_E2.csv", index=False)
    matrix.to_csv(out_dir / "exit_T_matrix.csv", index=False)
    select.to_csv(out_dir / "exit_summary_select60d.csv", index=False)
    holdout.to_csv(out_dir / "exit_summary_holdout30d.csv", index=False)
    full.to_csv(out_dir / "exit_summary_full90d.csv", index=False)
    rank_sel.to_csv(out_dir / "exit_ranking_select60d.csv", index=False)

    _plot_bars(
        rank_sel,
        "Net BTC",
        f"Stage2 exits — Net BTC (60d select, entry={entry})",
        plots / "select_net_btc_by_T.png",
        "Net BTC",
    )
    _plot_bars(
        holdout.sort_values("Strategy"),
        "Net BTC",
        f"Stage2 exits — Net BTC (30d holdout, entry={entry})",
        plots / "holdout_net_btc_by_T.png",
        "Net BTC",
    )
    _plot_bars(
        rank_sel,
        "Return %",
        f"Stage2 exits — Return % (60d select, entry={entry})",
        plots / "select_return_by_T.png",
        "Return %",
    )
    _plot_bars(
        holdout.sort_values("Strategy"),
        "Max DD",
        f"Stage2 exits — Max DD (30d holdout, entry={entry})",
        plots / "holdout_max_dd_by_T.png",
        "Max DD",
    )

    # Select vs holdout Net BTC paired bars
    try:
        fig, ax = plt.subplots(figsize=(11, 4.8))
        x = np.arange(len(EXIT_LABELS))
        s_map = {r["Strategy"]: float(r["Net BTC"] or 0) for _, r in select.iterrows()}
        h_map = {r["Strategy"]: float(r["Net BTC"] or 0) for _, r in holdout.iterrows()}
        s_vals = [s_map.get(a, 0.0) for a in EXIT_LABELS]
        h_vals = [h_map.get(a, 0.0) for a in EXIT_LABELS]
        ax.bar(x - 0.2, s_vals, width=0.4, label="select 60d")
        ax.bar(x + 0.2, h_vals, width=0.4, label="holdout 30d")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(EXIT_LABELS)
        ax.set_title(f"Stage2 T1–T10 Net BTC — select vs holdout (entry={entry})")
        ax.set_ylabel("Net BTC")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots / "select_vs_holdout_net_btc.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        logger.warning("paired plot failed: %s", e)

    cols = ["Strategy", "Trades", "Win %", "Net BTC", "Return %", "Max DD", "Expectancy", "Profit Factor"]
    report = [
        f"# Stage 2 — Exit selection (freeze entry={entry})",
        "",
        f"- Source Stage 1: `{stage1_dir}`",
        f"- Window: `{meta.get('eval_start')}` → `{meta.get('eval_end')}`",
        f"- Select 60d / Holdout 30d | exits on 1s tape",
        "",
        "## Decision",
        f"- **Winner on select:** `{winner}`",
        f"- Select Net BTC: `{decision['winner_select_net_btc']}`",
        f"- Select Return %: `{decision['winner_select_return_pct']}`",
        f"- Select Max DD: `{decision['winner_select_max_dd']}`",
        f"- Holdout Net BTC: `{decision['winner_holdout_net_btc']}`",
        f"- Holdout Return %: `{decision['winner_holdout_return_pct']}`",
        f"- Holdout Max DD: `{decision['winner_holdout_max_dd']}`",
        "",
        "## Select ranking (60d)",
        "```",
        rank_sel[cols].to_string(index=False) if not rank_sel.empty else "(empty)",
        "```",
        "",
        "## Holdout confirmation (30d)",
        "```",
        holdout.sort_values("Net BTC", ascending=False)[cols].to_string(index=False)
        if not holdout.empty
        else "(empty)",
        "```",
        "",
        "## Next",
        f"Freeze entry={entry} + exit={winner}; Stage 3 = selectors A–F vs that fixed T.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(str(x) for x in report), encoding="utf-8")

    logger.info("Wrote Stage2 → %s | winner=%s", out_dir, winner)
    print(rank_sel[cols].to_string(index=False))
    print(f"\nWinner on select: {winner}")
    print(f"Results: {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: freeze entry, rank T1–T10")
    ap.add_argument("--entry", default=DEFAULT_ENTRY, help="Frozen entry variant (default E2)")
    ap.add_argument(
        "--stage1-dir",
        type=Path,
        default=DEFAULT_STAGE1,
        help="Stage 1 results directory with strategy_legs.csv",
    )
    args = ap.parse_args()
    run_from_stage1(stage1_dir=args.stage1_dir.resolve(), entry=str(args.entry).upper())


if __name__ == "__main__":
    main()
