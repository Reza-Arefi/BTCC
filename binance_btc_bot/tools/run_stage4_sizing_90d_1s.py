"""Stage 4 — sizing / capacity on frozen live candidate E2 + T7.

Sweeps:
  - allocation_per_trade: 5% … 25%
  - max_open: 4 … 12
  - grid of both

Accounting (event-driven, no lookahead):
  requested_notional = fraction * starting_equity (1000 USD)
  actual_notional = min(requested, equity - reserved)  # 100% exposure cap
  skip if n_open >= max_open
  pnl_usd = pnl_pct * actual_notional

Rank on 60d select; confirm on 30d holdout.
Keep sizing only if holdout return stays positive and DD is not catastrophic vs baseline 12.5%/8.
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

ENTRY = "E2"
EXIT = "T7"
STARTING_CAPITAL_USD = 1000.0
BASELINE_FRAC = 0.125
BASELINE_MAX_OPEN = 8
DEFAULT_STAGE1 = REPO / "results" / "stage1_entry_90d_1s_20260910_115513"

ALLOC_FRACS = (0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25)
MAX_OPENS = (4, 6, 8, 10, 12)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage4_sizing")


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def load_candidate_trades(stage1_dir: Path) -> pd.DataFrame:
    legs = pd.read_csv(stage1_dir / "strategy_legs.csv", low_memory=False)
    t = legs[(legs["variant"].astype(str) == ENTRY) & (legs["arm_key"].astype(str) == EXIT)].copy()
    if t.empty:
        raise RuntimeError(f"no {ENTRY}+{EXIT} legs in {stage1_dir}")
    t["entry_ts"] = pd.to_datetime(t["entry_ts"], utc=True)
    t["exit_ts"] = pd.to_datetime(t["exit_ts"], utc=True)
    t["pnl_pct"] = pd.to_numeric(t["pnl_pct"], errors="coerce").fillna(0.0)
    t = t.sort_values("entry_ts").reset_index(drop=True)
    return t


def simulate_portfolio(
    trades: pd.DataFrame,
    *,
    fraction: float,
    max_open: int,
    starting_equity: float = STARTING_CAPITAL_USD,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Event-driven fixed-fraction portfolio with max_open + exposure cap."""
    if trades.empty:
        return pd.DataFrame(), {
            "arm": f"A{fraction*100:.1f}_M{max_open}",
            "fraction": fraction,
            "max_open": max_open,
            "trades_signal": 0,
            "trades_executed": 0,
            "skipped_max_open": 0,
            "exposure_limited": 0,
            "final_equity_usd": starting_equity,
            "net_pnl_usd": 0.0,
            "return_pct": 0.0,
            "max_dd": 0.0,
            "max_simultaneous": 0,
            "max_exposure": 0.0,
            "win_rate_pct": None,
            "profit_factor": None,
            "avg_pnl_usd": None,
        }

    df = trades.reset_index(drop=True)
    events: list[tuple[pd.Timestamp, int, int, str]] = []
    for i, r in df.iterrows():
        events.append((_utc(r["entry_ts"]), 0, int(i), "entry"))
        events.append((_utc(r["exit_ts"]), 1, int(i), "exit"))
    events.sort(key=lambda x: (x[0], x[1], x[2]))

    equity = float(starting_equity)
    reserved = 0.0
    open_book: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    peak = equity
    max_dd = 0.0
    max_exposure = 0.0
    max_simultaneous = 0
    skipped_max_open = 0
    exposure_limited = 0
    fully_available = 0

    for ts, _, i, kind in events:
        r = df.loc[i]
        oid = str(r["opportunity_id"])

        if kind == "exit":
            pos = open_book.pop(oid, None)
            if pos is None:
                continue
            reserved = max(0.0, reserved - float(pos["actual_notional"]))
            equity += float(pos["pnl_usd"])
            peak = max(peak, equity)
            dd = (equity - peak) / peak if peak else 0.0
            max_dd = min(max_dd, dd)
            rows.append({**pos, "equity_after": equity, "drawdown_pct": 100.0 * dd})
            continue

        if len(open_book) >= int(max_open):
            skipped_max_open += 1
            rows.append(
                {
                    "opportunity_id": oid,
                    "entry_ts": r["entry_ts"],
                    "exit_ts": r["exit_ts"],
                    "symbol": r.get("symbol"),
                    "phase": r.get("phase"),
                    "executed": False,
                    "skip_reason": "MAX_OPEN",
                    "requested_allocation": fraction,
                    "actual_allocation": 0.0,
                    "requested_notional": fraction * starting_equity,
                    "actual_notional": 0.0,
                    "pnl_usd": 0.0,
                    "pnl_pct": float(r["pnl_pct"]),
                    "equity_before": equity,
                    "n_open_before": len(open_book),
                }
            )
            continue

        req = float(fraction) * float(starting_equity)
        available = max(0.0, equity - reserved)
        actual = min(req, available)
        exp_lim = actual + 1e-9 < req
        if exp_lim:
            exposure_limited += 1
        else:
            fully_available += 1
        if actual <= 1e-9:
            skipped_max_open += 1  # treat as capacity skip
            rows.append(
                {
                    "opportunity_id": oid,
                    "entry_ts": r["entry_ts"],
                    "exit_ts": r["exit_ts"],
                    "symbol": r.get("symbol"),
                    "phase": r.get("phase"),
                    "executed": False,
                    "skip_reason": "NO_EQUITY",
                    "requested_allocation": fraction,
                    "actual_allocation": 0.0,
                    "requested_notional": req,
                    "actual_notional": 0.0,
                    "pnl_usd": 0.0,
                    "pnl_pct": float(r["pnl_pct"]),
                    "equity_before": equity,
                    "n_open_before": len(open_book),
                }
            )
            continue

        pnl_usd = float(r["pnl_pct"]) * actual
        reserved += actual
        n_open = len(open_book) + 1
        max_simultaneous = max(max_simultaneous, n_open)
        exposure_now = reserved / equity if equity else 0.0
        max_exposure = max(max_exposure, exposure_now)

        open_book[oid] = {
            "opportunity_id": oid,
            "entry_ts": r["entry_ts"],
            "exit_ts": r["exit_ts"],
            "symbol": r.get("symbol"),
            "phase": r.get("phase"),
            "executed": True,
            "skip_reason": "",
            "requested_allocation": fraction,
            "actual_allocation": actual / starting_equity,
            "requested_notional": req,
            "actual_notional": actual,
            "pnl_usd": pnl_usd,
            "pnl_pct": float(r["pnl_pct"]),
            "equity_before": equity,
            "n_open_after_entry": n_open,
            "exposure_after_entry": exposure_now,
            "exposure_limited": exp_lim,
        }

    if open_book:
        # Force-close remaining at recorded pnl (should not happen for closed legs)
        for oid, pos in list(open_book.items()):
            equity += float(pos["pnl_usd"])
            rows.append({**pos, "equity_after": equity, "force_closed": True})
        open_book.clear()

    out = pd.DataFrame(rows)
    executed = out[out["executed"] == True] if not out.empty else out  # noqa: E712
    wins = executed[executed["pnl_usd"] > 0]["pnl_usd"] if len(executed) else pd.Series(dtype=float)
    losses = executed[executed["pnl_usd"] < 0]["pnl_usd"] if len(executed) else pd.Series(dtype=float)
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(-losses.sum()) if len(losses) else 0.0
    pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
    n_exec = int(len(executed))
    n_dec = int(len(wins) + len(losses))

    summary = {
        "arm": f"A{fraction*100:g}_M{max_open}",
        "fraction": float(fraction),
        "max_open": int(max_open),
        "trades_signal": int(len(df)),
        "trades_executed": n_exec,
        "skipped_max_open": int(skipped_max_open),
        "exposure_limited": int(exposure_limited),
        "fully_available": int(fully_available),
        "final_equity_usd": float(equity),
        "net_pnl_usd": float(equity - starting_equity),
        "return_pct": 100.0 * (equity / starting_equity - 1.0),
        "max_dd": float(max_dd),
        "max_dd_pct": 100.0 * float(max_dd),
        "max_simultaneous": int(max_simultaneous),
        "max_exposure": float(max_exposure),
        "win_rate_pct": (100.0 * len(wins) / n_dec) if n_dec else None,
        "profit_factor": None if pf == float("inf") else float(pf),
        "avg_pnl_usd": float(executed["pnl_usd"].mean()) if n_exec else None,
        "sum_win_usd": gp,
        "sum_loss_usd": -gl,
    }
    return out, summary


def _plot_alloc_curve(df: pd.DataFrame, phase: str, path: Path) -> None:
    sub = df[(df["phase"] == phase) & (df["sweep"] == "alloc_at_M8")].sort_values("fraction")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(sub["fraction"] * 100, sub["return_pct"], marker="o", label="Return %")
    ax2 = ax.twinx()
    ax2.plot(sub["fraction"] * 100, sub["max_dd_pct"], marker="s", color="#d62728", label="Max DD %")
    ax.set_xlabel("Allocation per trade (%)")
    ax.set_ylabel("Return %")
    ax2.set_ylabel("Max DD %")
    ax.set_title(f"Stage4 alloc sweep @ max_open=8 ({phase})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_heatmap(df: pd.DataFrame, phase: str, value: str, path: Path, title: str) -> None:
    sub = df[(df["phase"] == phase) & (df["sweep"] == "grid")].copy()
    if sub.empty:
        return
    piv = sub.pivot(index="max_open", columns="fraction", values=value)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(piv.astype(float).fillna(0.0).values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c*100:g}%" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index.tolist())
    ax.set_xlabel("Allocation")
    ax.set_ylabel("max_open")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(stage1_dir: Path) -> Path:
    trades = load_candidate_trades(stage1_dir)
    meta1 = json.loads((stage1_dir / "run_meta.json").read_text(encoding="utf-8"))
    select_trades = trades[trades["phase"] == "select"].copy()
    holdout_trades = trades[trades["phase"] == "holdout"].copy()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / f"stage4_sizing_{ENTRY}_{EXIT}_90d_1s_{run_id}"
    plots = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    trade_tables: dict[str, pd.DataFrame] = {}

    def _run_pair(frac: float, mo: int, sweep: str) -> None:
        for phase, tdf in (("select", select_trades), ("holdout", holdout_trades), ("full", trades)):
            out, sm = simulate_portfolio(tdf, fraction=frac, max_open=mo)
            sm["phase"] = phase
            sm["sweep"] = sweep
            sm["candidate"] = f"{ENTRY}+{EXIT}"
            summaries.append(sm)
            trade_tables[f"{sm['arm']}_{phase}"] = out

    # 1) Alloc sweep at live max_open=8
    for frac in ALLOC_FRACS:
        _run_pair(frac, BASELINE_MAX_OPEN, "alloc_at_M8")

    # 2) Max-open sweep at live alloc=12.5%
    for mo in MAX_OPENS:
        _run_pair(BASELINE_FRAC, mo, "maxopen_at_A12_5")

    # 3) Grid
    for frac in ALLOC_FRACS:
        for mo in MAX_OPENS:
            _run_pair(frac, mo, "grid")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "sizing_summary_all.csv", index=False)

    # Rank alloc@M8 on select
    alloc_sel = summary_df[(summary_df["sweep"] == "alloc_at_M8") & (summary_df["phase"] == "select")].copy()
    alloc_sel = alloc_sel.sort_values(
        by=["return_pct", "max_dd"], ascending=[False, False]
    ).reset_index(drop=True)
    alloc_hold = summary_df[(summary_df["sweep"] == "alloc_at_M8") & (summary_df["phase"] == "holdout")]

    # Rank maxopen@12.5 on select
    mo_sel = summary_df[(summary_df["sweep"] == "maxopen_at_A12_5") & (summary_df["phase"] == "select")].copy()
    mo_sel = mo_sel.sort_values(by=["return_pct", "max_dd"], ascending=[False, False]).reset_index(drop=True)
    mo_hold = summary_df[(summary_df["sweep"] == "maxopen_at_A12_5") & (summary_df["phase"] == "holdout")]

    # Grid best on select with holdout confirmation filters
    grid_sel = summary_df[(summary_df["sweep"] == "grid") & (summary_df["phase"] == "select")].copy()
    grid_hold = summary_df[(summary_df["sweep"] == "grid") & (summary_df["phase"] == "holdout")].copy()
    grid_sel = grid_sel.sort_values(by=["return_pct", "max_dd"], ascending=[False, False]).reset_index(drop=True)

    def _hold_row(arm: str, hold_df: pd.DataFrame) -> dict[str, Any]:
        sub = hold_df[hold_df["arm"] == arm]
        return sub.iloc[0].to_dict() if not sub.empty else {}

    # Baseline metrics
    base_sel = _hold_row(f"A{BASELINE_FRAC*100:g}_M{BASELINE_MAX_OPEN}", alloc_sel)
    # alloc_sel is select-ranked; get baseline from summary
    base_sel = summary_df[
        (summary_df["sweep"] == "alloc_at_M8")
        & (summary_df["phase"] == "select")
        & (summary_df["fraction"] == BASELINE_FRAC)
        & (summary_df["max_open"] == BASELINE_MAX_OPEN)
    ]
    base_hold = summary_df[
        (summary_df["sweep"] == "alloc_at_M8")
        & (summary_df["phase"] == "holdout")
        & (summary_df["fraction"] == BASELINE_FRAC)
        & (summary_df["max_open"] == BASELINE_MAX_OPEN)
    ]
    baseline = {
        "arm": f"A{BASELINE_FRAC*100:g}_M{BASELINE_MAX_OPEN}",
        "fraction": BASELINE_FRAC,
        "max_open": BASELINE_MAX_OPEN,
        "select_return_pct": float(base_sel.iloc[0]["return_pct"]) if not base_sel.empty else None,
        "select_max_dd_pct": float(base_sel.iloc[0]["max_dd_pct"]) if not base_sel.empty else None,
        "holdout_return_pct": float(base_hold.iloc[0]["return_pct"]) if not base_hold.empty else None,
        "holdout_max_dd_pct": float(base_hold.iloc[0]["max_dd_pct"]) if not base_hold.empty else None,
    }

    # Pick grid winner: best select return among arms with holdout return > 0
    # and holdout DD not worse than 1.5x baseline holdout DD (if baseline DD < 0)
    candidates = []
    for _, row in grid_sel.iterrows():
        h = _hold_row(str(row["arm"]), grid_hold)
        if not h:
            continue
        if float(h.get("return_pct") or -1e9) <= 0:
            continue
        base_hdd = baseline.get("holdout_max_dd_pct")
        hdd = float(h.get("max_dd_pct") or 0)
        if base_hdd is not None and base_hdd < 0 and hdd < 1.5 * base_hdd:
            continue  # much worse DD (more negative)
        candidates.append({**row.to_dict(), "holdout_return_pct": h.get("return_pct"), "holdout_max_dd_pct": h.get("max_dd_pct")})

    if candidates:
        winner = candidates[0]  # already sorted by select return
        keep_new = not (
            abs(float(winner["fraction"]) - BASELINE_FRAC) < 1e-12
            and int(winner["max_open"]) == BASELINE_MAX_OPEN
        )
    else:
        winner = {
            "arm": baseline["arm"],
            "fraction": BASELINE_FRAC,
            "max_open": BASELINE_MAX_OPEN,
            "return_pct": baseline["select_return_pct"],
            "max_dd_pct": baseline["select_max_dd_pct"],
            "holdout_return_pct": baseline["holdout_return_pct"],
            "holdout_max_dd_pct": baseline["holdout_max_dd_pct"],
        }
        keep_new = False

    decision = {
        "stage": 4,
        "frozen_entry": ENTRY,
        "frozen_exit": EXIT,
        "baseline": baseline,
        "winner_on_select": {
            "arm": winner.get("arm"),
            "allocation_per_trade": winner.get("fraction"),
            "max_open": winner.get("max_open"),
            "select_return_pct": winner.get("return_pct"),
            "select_max_dd_pct": winner.get("max_dd_pct"),
            "holdout_return_pct": winner.get("holdout_return_pct"),
            "holdout_max_dd_pct": winner.get("holdout_max_dd_pct"),
        },
        "change_from_baseline": keep_new,
        "live_candidate": {
            "entry": ENTRY,
            "exit": EXIT,
            "selector": None,
            "allocation_per_trade": winner.get("fraction"),
            "max_simultaneous_trades": winner.get("max_open"),
        },
        "rule": "Pick sizing on 60d select; require holdout return > 0 and DD not >1.5× baseline holdout DD. Prefer baseline if no candidate passes.",
        "notes": [
            "E2+T7 stream from Stage 1; max concurrent in raw stream was 8",
            "pnl_usd = pnl_pct * actual_notional; exposure capped at 100% equity",
        ],
    }

    meta = {
        "source_stage1": str(stage1_dir),
        "eval_start": meta1.get("eval_start"),
        "eval_end": meta1.get("eval_end"),
        "n_trades_full": int(len(trades)),
        "n_trades_select": int(len(select_trades)),
        "n_trades_holdout": int(len(holdout_trades)),
        "decision": decision,
    }

    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    alloc_sel.to_csv(out_dir / "alloc_ranking_select60d.csv", index=False)
    mo_sel.to_csv(out_dir / "maxopen_ranking_select60d.csv", index=False)
    grid_sel.to_csv(out_dir / "grid_ranking_select60d.csv", index=False)
    alloc_hold.to_csv(out_dir / "alloc_summary_holdout30d.csv", index=False)
    mo_hold.to_csv(out_dir / "maxopen_summary_holdout30d.csv", index=False)
    grid_hold.to_csv(out_dir / "grid_summary_holdout30d.csv", index=False)

    # Save baseline + winner trade details
    for phase, tdf in (("select", select_trades), ("holdout", holdout_trades)):
        _, sm = simulate_portfolio(
            tdf,
            fraction=float(winner.get("fraction") or BASELINE_FRAC),
            max_open=int(winner.get("max_open") or BASELINE_MAX_OPEN),
        )
        out, _ = simulate_portfolio(
            tdf,
            fraction=float(winner.get("fraction") or BASELINE_FRAC),
            max_open=int(winner.get("max_open") or BASELINE_MAX_OPEN),
        )
        out.to_csv(out_dir / f"winner_trades_{phase}.csv", index=False)

    _plot_alloc_curve(summary_df, "select", plots / "alloc_curve_select.png")
    _plot_alloc_curve(summary_df, "holdout", plots / "alloc_curve_holdout.png")
    _plot_heatmap(
        summary_df,
        "select",
        "return_pct",
        plots / "grid_return_select.png",
        "Stage4 grid Return % (select)",
    )
    _plot_heatmap(
        summary_df,
        "holdout",
        "return_pct",
        plots / "grid_return_holdout.png",
        "Stage4 grid Return % (holdout)",
    )
    _plot_heatmap(
        summary_df,
        "select",
        "max_dd_pct",
        plots / "grid_dd_select.png",
        "Stage4 grid Max DD % (select)",
    )

    # max_open bar at 12.5%
    mo_s = summary_df[(summary_df["sweep"] == "maxopen_at_A12_5") & (summary_df["phase"] == "select")].sort_values("max_open")
    mo_h = summary_df[(summary_df["sweep"] == "maxopen_at_A12_5") & (summary_df["phase"] == "holdout")].sort_values("max_open")
    if not mo_s.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(mo_s))
        ax.bar(x - 0.2, mo_s["return_pct"], width=0.4, label="select")
        ax.bar(x + 0.2, [_hold_row(a, mo_h).get("return_pct", 0) for a in mo_s["arm"]], width=0.4, label="holdout")
        ax.set_xticks(x)
        ax.set_xticklabels(mo_s["max_open"].astype(int))
        ax.set_xlabel("max_open")
        ax.set_ylabel("Return %")
        ax.set_title("Stage4 max_open sweep @ 12.5% alloc")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots / "maxopen_return_select_vs_holdout.png", dpi=130)
        plt.close(fig)

    report = [
        f"# Stage 4 — Sizing / capacity ({ENTRY}+{EXIT})",
        "",
        f"- Source: `{stage1_dir}`",
        f"- Baseline live config: **{BASELINE_FRAC*100:g}%** alloc × **max_open={BASELINE_MAX_OPEN}**",
        "",
        "## Decision",
        f"- Winner: `{winner.get('arm')}` → alloc **{float(winner.get('fraction') or 0)*100:g}%**, max_open **{winner.get('max_open')}**",
        f"- Change from baseline: `{keep_new}`",
        f"- Select return: `{winner.get('return_pct')}` | DD: `{winner.get('max_dd_pct')}`",
        f"- Holdout return: `{winner.get('holdout_return_pct')}` | DD: `{winner.get('holdout_max_dd_pct')}`",
        "",
        "## Live candidate (final)",
        "```json",
        json.dumps(decision["live_candidate"], indent=2),
        "```",
        "",
        "## Alloc sweep @ max_open=8 (select)",
        "```",
        alloc_sel[
            ["arm", "fraction", "trades_executed", "skipped_max_open", "return_pct", "max_dd_pct", "profit_factor"]
        ].to_string(index=False),
        "```",
        "",
        "## Max-open sweep @ 12.5% (select)",
        "```",
        mo_sel[
            ["arm", "max_open", "trades_executed", "skipped_max_open", "return_pct", "max_dd_pct", "max_exposure"]
        ].to_string(index=False),
        "```",
        "",
        "## Notes",
        "- Raw E2+T7 stream never exceeded 8 concurrent trades, so raising max_open above 8 has little effect unless allocation changes entry density.",
        "- Prefer baseline unless select+holdout clearly improve.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(str(x) for x in report), encoding="utf-8")

    logger.info("Stage4 done → %s | winner=%s change=%s", out_dir, winner.get("arm"), keep_new)
    print(alloc_sel[["arm", "return_pct", "max_dd_pct", "trades_executed"]].head(10).to_string(index=False))
    print(f"\nWinner: {winner.get('arm')} change_from_baseline={keep_new}")
    print(f"Live candidate: {decision['live_candidate']}")
    print(f"Results: {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1)
    args = ap.parse_args()
    run(args.stage1_dir.resolve())


if __name__ == "__main__":
    main()
