"""Part 4D — Overlap/capacity audit + consolidated candidate comparison.

Uses existing Part 4C opportunity stream / accounting. Does NOT re-run the
full 1y Selector E backtest. Paper/live untouched.

Policy A: partial allocation (existing Part 4C semantics)
Policy B: full-size-or-skip when remaining exposure < requested
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from btcc.sim.selector_E_adaptive_sizing import (
    BASE_NOTIONAL_USD,
    MAX_OPEN,
    STARTING_CAPITAL_USD,
    _utc,
    load_selector_E_stream,
)
from btcc.sim.selector_E_position_size_ladder_part4c import ARM_ORDER as LADDER_ARMS
from btcc.sim.selector_E_position_size_ladder_part4c import FRACTION_BY_ARM

logger = logging.getLogger(__name__)

AUDIT_FRACTIONS = (0.10, 0.125, 0.15, 0.175, 0.20, 0.25)
AUDIT_ARM_NAMES = {
    0.10: "FIXED_10",
    0.125: "FIXED_12_5",
    0.15: "FIXED_15",
    0.175: "FIXED_17_5",
    0.20: "FIXED_20",
    0.25: "FIXED_25",
}

# Serious candidates for consolidated table (exclude confounded Part 4A as primary)
CONSOLIDATED_FIXED = ["FIXED_10", "FIXED_12_5", "FIXED_15", "FIXED_17_5", "FIXED_20", "FIXED_25"]
CONSOLIDATED_CAUSAL = [
    "7D_SQRT_CAUSAL",
    "7D_MEDIAN_SQRT_CAUSAL",
    "7D_INVERSE_CAUSAL",
    "7_30_INVERSE_CAUSAL",
]
PART4A_DIAGNOSTIC = ["7D_INVERSE", "7D_MEDIAN_SQRT", "7D_SQRT"]


def concurrency_profile(trades: pd.DataFrame) -> dict[str, Any]:
    """Distribution of simultaneous open E positions on the frozen stream."""
    ex = trades.copy()
    ex["entry_ts"] = pd.to_datetime(ex["entry_ts"], utc=True)
    ex["exit_ts"] = pd.to_datetime(ex["exit_ts"], utc=True)
    ex = ex.sort_values("entry_ts").reset_index(drop=True)

    events: list[tuple[pd.Timestamp, int]] = []
    for _, r in ex.iterrows():
        events.append((_utc(r["entry_ts"]), 1))
        events.append((_utc(r["exit_ts"]), -1))
    events.sort(key=lambda x: (x[0], x[1]))  # exits (-1) before entries (+1) at same ts? 
    # For capacity at entry, process exits first when timestamps equal so a close frees capacity.
    events.sort(key=lambda x: (x[0], 0 if x[1] < 0 else 1))

    c = 0
    mx = 0
    path_counts: dict[int, int] = {}
    for _, d in events:
        c += d
        mx = max(mx, c)
        path_counts[c] = path_counts.get(c, 0) + 1

    at_entry = []
    for _, r in ex.iterrows():
        n_before = int(((ex["entry_ts"] < r["entry_ts"]) & (ex["exit_ts"] > r["entry_ts"])).sum())
        at_entry.append(n_before + 1)
    s = pd.Series(at_entry)
    dist = s.value_counts().sort_index()
    return {
        "n_trades": int(len(ex)),
        "max_concurrent": int(mx),
        "mean_concurrent_at_entry": float(s.mean()),
        "median_concurrent_at_entry": float(s.median()),
        "concurrency_at_entry_counts": {int(k): int(v) for k, v in dist.items()},
        "pct_entries_with_concurrency_ge": {
            str(k): float((s >= k).mean() * 100.0) for k in range(1, int(s.max()) + 1)
        },
        "theoretical_max_full_slots": {f"{int(f*100) if f!=0.125 else '12.5'}%": int(1.0 // f) for f in AUDIT_FRACTIONS},
    }


def simulate_policy(
    trades: pd.DataFrame,
    *,
    fraction: float,
    policy: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """policy: 'partial' (A) or 'full_or_skip' (B)."""
    if policy not in ("partial", "full_or_skip"):
        raise ValueError(policy)

    rows: list[dict[str, Any]] = []
    equity = STARTING_CAPITAL_USD
    reserved = 0.0
    open_book: dict[str, dict[str, Any]] = {}

    events: list[tuple[pd.Timestamp, int, int, str]] = []
    for i, r in trades.iterrows():
        events.append((_utc(r["entry_ts"]), 0, int(i), "entry"))
        events.append((_utc(r["exit_ts"]), 1, int(i), "exit"))
    events.sort(key=lambda x: (x[0], x[1], x[2]))

    peak = equity
    max_dd = 0.0
    n_full = n_partial = n_skip_exposure = n_skip_max_open = 0
    constrained_pnl_if_taken = 0.0  # diagnostic
    max_exposure = 0.0
    max_sim = 0

    for ts, _, i, kind in events:
        r = trades.loc[i]
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

        if len(open_book) >= MAX_OPEN:
            n_skip_max_open += 1
            rows.append(
                {
                    "opportunity_id": oid,
                    "fraction": fraction,
                    "policy": policy,
                    "entry_ts": r["entry_ts"],
                    "exit_ts": r["exit_ts"],
                    "executed": False,
                    "status": "SKIP_MAX_OPEN",
                    "requested_allocation": fraction,
                    "actual_allocation": 0.0,
                    "actual_notional": 0.0,
                    "pnl_usd": 0.0,
                    "ref_pnl_usd": float(r["pnl_usd_equiv"]),
                    "equity_before": equity,
                    "n_open_before": len(open_book),
                }
            )
            continue

        req = fraction * STARTING_CAPITAL_USD
        available = max(0.0, equity - reserved)
        n_open_before = len(open_book)

        if available + 1e-12 >= req:
            actual = req
            status = "FULL"
            n_full += 1
        else:
            if policy == "partial":
                actual = available
                if actual <= 1e-12:
                    status = "SKIP_NO_EXPOSURE"
                    n_skip_exposure += 1
                    actual = 0.0
                else:
                    status = "PARTIAL"
                    n_partial += 1
                    # counterfactual full-size pnl for capacity diagnostic
                    constrained_pnl_if_taken += float(r["pnl_usd_equiv"]) * (req / BASE_NOTIONAL_USD)
            else:
                # full_or_skip
                status = "SKIP_NO_EXPOSURE"
                n_skip_exposure += 1
                actual = 0.0
                constrained_pnl_if_taken += float(r["pnl_usd_equiv"]) * (req / BASE_NOTIONAL_USD)

        if actual <= 0:
            rows.append(
                {
                    "opportunity_id": oid,
                    "fraction": fraction,
                    "policy": policy,
                    "entry_ts": r["entry_ts"],
                    "exit_ts": r["exit_ts"],
                    "executed": False,
                    "status": status,
                    "requested_allocation": fraction,
                    "actual_allocation": 0.0,
                    "requested_notional": req,
                    "actual_notional": 0.0,
                    "pnl_usd": 0.0,
                    "ref_pnl_usd": float(r["pnl_usd_equiv"]),
                    "equity_before": equity,
                    "available_before": available,
                    "n_open_before": n_open_before,
                }
            )
            continue

        scale = actual / BASE_NOTIONAL_USD
        pnl = float(r["pnl_usd_equiv"]) * scale
        reserved += actual
        max_sim = max(max_sim, len(open_book) + 1)
        max_exposure = max(max_exposure, reserved / equity if equity else 0.0)

        open_book[oid] = {
            "opportunity_id": oid,
            "fraction": fraction,
            "policy": policy,
            "entry_ts": r["entry_ts"],
            "exit_ts": r["exit_ts"],
            "executed": True,
            "status": status,
            "requested_allocation": fraction,
            "actual_allocation": actual / STARTING_CAPITAL_USD,
            "requested_notional": req,
            "actual_notional": actual,
            "pnl_usd": pnl,
            "pnl_pct": float(r["pnl_pct"]),
            "ref_pnl_usd": float(r["pnl_usd_equiv"]),
            "scale": scale,
            "equity_before": equity,
            "available_before": available,
            "reserved_before": reserved - actual,
            "n_open_before": n_open_before,
            "n_open_after": len(open_book) + 1,
            "exposure_after_entry": reserved / equity if equity else 0.0,
        }

    if open_book:
        raise RuntimeError(f"trades still open under {policy} @{fraction}")

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("exit_ts").reset_index(drop=True)
    executed = out[out["executed"] == True] if not out.empty else out  # noqa: E712
    skipped = out[out["executed"] == False] if not out.empty else out  # noqa: E712

    # PnL contribution by status among rows that were executed
    pnl_full = float(executed.loc[executed["status"] == "FULL", "pnl_usd"].sum()) if len(executed) else 0.0
    pnl_partial = float(executed.loc[executed["status"] == "PARTIAL", "pnl_usd"].sum()) if len(executed) else 0.0
    total_pnl = float(equity - STARTING_CAPITAL_USD)

    # Share of reference stream PnL that sits on constrained entries (entry-time capacity issue)
    ref_total = float(trades["pnl_usd_equiv"].sum())
    constrained_oids = set(out.loc[out["status"].isin(["PARTIAL", "SKIP_NO_EXPOSURE"]), "opportunity_id"].astype(str))
    ref_on_constrained = float(
        trades.loc[trades["opportunity_id"].astype(str).isin(constrained_oids), "pnl_usd_equiv"].sum()
    )

    summary = {
        "arm": AUDIT_ARM_NAMES.get(fraction, f"FIXED_{fraction}"),
        "fraction": fraction,
        "policy": policy,
        "final_equity_usd": float(equity),
        "total_return_pct": 100.0 * (equity / STARTING_CAPITAL_USD - 1.0),
        "max_drawdown_pct": 100.0 * float(max_dd),
        "n_opportunities": int(len(trades)),
        "n_executed": int(len(executed)),
        "n_full": int(n_full),
        "n_partial": int(n_partial),
        "n_skip_exposure": int(n_skip_exposure),
        "n_skip_max_open": int(n_skip_max_open),
        "pct_full": 100.0 * n_full / len(trades) if len(trades) else 0.0,
        "pct_partial": 100.0 * n_partial / len(trades) if len(trades) else 0.0,
        "pct_skip_exposure": 100.0 * n_skip_exposure / len(trades) if len(trades) else 0.0,
        "pct_skip_max_open": 100.0 * n_skip_max_open / len(trades) if len(trades) else 0.0,
        "pnl_from_full_usd": pnl_full,
        "pnl_from_partial_usd": pnl_partial,
        "pct_pnl_from_partial": 100.0 * pnl_partial / total_pnl if total_pnl else 0.0,
        "ref_pnl_on_constrained_entries_usd": ref_on_constrained,
        "pct_ref_pnl_on_constrained_entries": 100.0 * ref_on_constrained / ref_total if ref_total else 0.0,
        "mean_actual_allocation": float(executed["actual_allocation"].mean()) if len(executed) else 0.0,
        "max_exposure": float(max_exposure),
        "max_simultaneous": int(max_sim),
        "return_over_abs_dd": (
            float(100.0 * (equity / STARTING_CAPITAL_USD - 1.0) / abs(100.0 * max_dd)) if max_dd < 0 else float("inf")
        ),
        "win_rate_pct": 100.0 * float((executed["pnl_usd"] > 0).mean()) if len(executed) else 0.0,
        "profit_factor": _pf(executed["pnl_usd"]) if len(executed) else 0.0,
    }
    return out, summary


def _pf(pnl: pd.Series) -> float:
    pos = float(pnl[pnl > 0].sum())
    neg = float(-pnl[pnl < 0].sum())
    if neg <= 0:
        return float("inf") if pos > 0 else 0.0
    return pos / neg


def load_existing_summaries(
    part4c_dir: Path,
    part4b_dir: Path,
    part4a_dir: Path | None,
) -> dict[str, pd.DataFrame]:
    out = {
        "part4c": pd.read_csv(part4c_dir / "summary.csv"),
        "part4b": pd.read_csv(part4b_dir / "adaptive_sizing_summary.csv"),
    }
    if part4a_dir and (part4a_dir / "adaptive_sizing_summary.csv").exists():
        out["part4a"] = pd.read_csv(part4a_dir / "adaptive_sizing_summary.csv")
    return out


def build_consolidated_table(
    existing: dict[str, pd.DataFrame],
    policy_a: pd.DataFrame,
    policy_b: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add(source, arm, label, row, *, confounded=False):
        rows.append(
            {
                "strategy": label,
                "arm_key": arm,
                "source": source,
                "confounded_part4a": confounded,
                "return_pct": float(row.get("total_return_pct", np.nan)),
                "final_equity_usd": float(row.get("final_equity_usd", np.nan)),
                "max_drawdown_pct": float(row.get("max_drawdown_pct", np.nan)),
                "profit_factor": float(row.get("profit_factor", np.nan)),
                "n_trades": int(row.get("n_executed", row.get("n_trades", 0))),
                "avg_size": float(
                    row.get("mean_actual_allocation", row.get("avg_allocation", row.get("avg_actual_allocation", np.nan)))
                ),
                "pct_constrained": float(
                    row.get(
                        "pct_exposure_limited",
                        row.get("pct_partial", 0.0) + row.get("pct_skip_exposure", 0.0),
                    )
                ),
                "return_over_abs_dd": float(row.get("return_over_abs_dd", np.nan))
                if np.isfinite(row.get("return_over_abs_dd", np.nan))
                else (
                    float(row["total_return_pct"] / abs(row["max_drawdown_pct"]))
                    if row.get("max_drawdown_pct", 0) not in (0, None) and float(row.get("max_drawdown_pct", 0)) < 0
                    else np.nan
                ),
            }
        )

    p4c = existing["part4c"].set_index("arm")
    for arm in CONSOLIDATED_FIXED:
        if arm in p4c.index:
            add("part4c_policy_A", arm, arm.replace("FIXED_", "Fixed "), p4c.loc[arm])

    p4b = existing["part4b"].set_index("arm")
    labels = {
        "7D_SQRT_CAUSAL": "7D SQRT causal",
        "7D_MEDIAN_SQRT_CAUSAL": "7D Median SQRT causal",
        "7D_INVERSE_CAUSAL": "7D Inverse causal",
        "7_30_INVERSE_CAUSAL": "7/30 Inverse causal",
    }
    for arm in CONSOLIDATED_CAUSAL:
        if arm in p4b.index:
            r = p4b.loc[arm]
            # constrained approx: pct above floor isn't constraint; use exposure limited if present
            add("part4b", arm, labels[arm], r)

    # Policy B rows for high sizes
    pb = policy_b.set_index("arm")
    for arm in CONSOLIDATED_FIXED:
        if arm in pb.index:
            add("part4d_policy_B", arm, f"{arm.replace('FIXED_', 'Fixed ')} (full-or-skip)", pb.loc[arm])

    if "part4a" in existing:
        p4a = existing["part4a"].set_index("arm")
        for arm in PART4A_DIAGNOSTIC:
            if arm in p4a.index:
                add("part4a_diagnostic", arm, f"{arm} [confounded]", p4a.loc[arm], confounded=True)

    return pd.DataFrame(rows)


def make_results_dir(root: Path | None = None) -> Path:
    root = Path(root or "results")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = root / f"selector_E_capacity_audit_part4d_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "plots").mkdir(exist_ok=True)
    return out


def run_part4d(
    *,
    ref_dir: Path,
    part4c_dir: Path,
    part4b_dir: Path,
    part4a_dir: Path | None,
) -> dict[str, Any]:
    _, trades, ref_metrics = load_selector_E_stream(ref_dir)
    # Prefer Part 4C FIXED_10 trade list identity
    t10 = pd.read_csv(part4c_dir / "trades_FIXED_10.csv")
    t10 = t10[t10["executed"] == True]  # noqa: E712
    # Align to reference trade columns needed
    base = trades.copy()

    conc = concurrency_profile(base)

    policy_a_rows = []
    policy_b_rows = []
    trade_tables: dict[str, pd.DataFrame] = {}

    for frac in AUDIT_FRACTIONS:
        ta, sa = simulate_policy(base, fraction=frac, policy="partial")
        tb, sb = simulate_policy(base, fraction=frac, policy="full_or_skip")
        policy_a_rows.append(sa)
        policy_b_rows.append(sb)
        arm = AUDIT_ARM_NAMES[frac]
        trade_tables[f"{arm}__partial"] = ta
        trade_tables[f"{arm}__full_or_skip"] = tb

    policy_a = pd.DataFrame(policy_a_rows)
    policy_b = pd.DataFrame(policy_b_rows)
    compare_ab = policy_a.merge(policy_b, on=["arm", "fraction"], suffixes=("_A", "_B"))

    existing = load_existing_summaries(part4c_dir, part4b_dir, part4a_dir)
    consolidated = build_consolidated_table(existing, policy_a, policy_b)

    # Cross-check Policy A vs Part 4C summary equity
    p4c = existing["part4c"].set_index("arm")
    match_rows = []
    for frac in AUDIT_FRACTIONS:
        arm = AUDIT_ARM_NAMES[frac]
        a = policy_a[policy_a["arm"] == arm].iloc[0]
        if arm in p4c.index:
            match_rows.append(
                {
                    "arm": arm,
                    "part4d_policyA_equity": float(a["final_equity_usd"]),
                    "part4c_equity": float(p4c.loc[arm, "final_equity_usd"]),
                    "abs_diff": abs(float(a["final_equity_usd"]) - float(p4c.loc[arm, "final_equity_usd"])),
                    "match": abs(float(a["final_equity_usd"]) - float(p4c.loc[arm, "final_equity_usd"])) < 1e-4,
                }
            )
    policy_a_vs_4c = pd.DataFrame(match_rows)

    return {
        "ref_dir": str(ref_dir),
        "part4c_dir": str(part4c_dir),
        "part4b_dir": str(part4b_dir),
        "part4a_dir": str(part4a_dir) if part4a_dir else None,
        "concurrency": conc,
        "policy_a": policy_a,
        "policy_b": policy_b,
        "compare_ab": compare_ab,
        "consolidated": consolidated,
        "policy_a_vs_4c": policy_a_vs_4c,
        "trade_tables": trade_tables,
        "ref_metrics": ref_metrics,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
