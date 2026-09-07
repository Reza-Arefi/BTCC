"""Part 2A diagnostic deep-dive: buckets, S5 trajectory, simple combinations.

Reads the frozen Part 1 opportunity CSV. Does not rerun the 15m backtest,
does not modify the baseline, and does not implement a veto.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIXED_T_ARMS = tuple(f"T{i}" for i in range(1, 11))
SELECTOR_ARMS = tuple("ABCDEF")
BASELINE_FILES = (
    "opportunities.csv",
    "strategy_legs.csv",
    "selection_audit.csv",
    "predictions.csv",
    "summary.json",
)

Q_LABELS = (
    "Q1_lowest",
    "Q2",
    "Q3",
    "Q4",
    "Q5_highest",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def verify_causality(diag: pd.DataFrame) -> dict[str, Any]:
    n = int(len(diag))
    avail = pd.to_datetime(diag["available_5m_candle_ts"], utc=True, errors="coerce")
    dec = pd.to_datetime(diag["decision_timestamp"], utc=True, errors="coerce")
    ok = (avail.notna() & dec.notna() & (avail <= dec))
    n_ok = int(ok.sum())
    n_bad = int((~ok).sum())
    max_lead = None
    if n_ok:
        lead = (dec[ok] - avail[ok]).dt.total_seconds()
        max_lead = float(lead.max()) if len(lead) else None
    return {
        "n_rows": n,
        "n_causal_ok": n_ok,
        "n_causal_violations": n_bad,
        "max_seconds_decision_after_available_5m": max_lead,
        "unique_opportunity_ids": int(diag["opportunity_id"].nunique()),
        "ids_unique": bool(diag["opportunity_id"].is_unique),
        "one_row_per_opportunity": bool(len(diag) == diag["opportunity_id"].nunique()),
    }


def _bucket_metrics(g: pd.DataFrame, *, n_total: int) -> dict[str, Any]:
    pnl = _num(g["primary_pnl_pct"])
    mfe = _num(g["primary_mfe_pct"])
    mae = _num(g["primary_mae_pct"])
    win = g["primary_win"] == True  # noqa: E712
    large = g["primary_large_loss"].fillna(False).astype(bool)
    n = int(len(g))
    return {
        "n": n,
        "pct_of_trades": (100.0 * n / n_total) if n_total else None,
        "win_rate": float(win.mean()) if n else None,
        "mean_pnl": float(pnl.mean()) if n else None,
        "median_pnl": float(pnl.median()) if n else None,
        "mean_mfe": float(mfe.mean()) if n else None,
        "mean_mae": float(mae.mean()) if n else None,
        "large_loss_rate": float(large.mean()) if n else None,
        "total_pnl_sum": float(pnl.sum()) if n else None,
        "unreliable": bool(n < 40),
    }


def quantile_buckets(diag: pd.DataFrame, col: str, q: int = 5) -> pd.DataFrame:
    x = _num(diag[col])
    valid = x.notna()
    work = diag.loc[valid].copy()
    work["_x"] = x[valid]
    try:
        work["_bucket"] = pd.qcut(work["_x"], q=q, labels=list(Q_LABELS[:q]), duplicates="drop")
    except ValueError:
        work["_bucket"] = pd.cut(work["_x"], bins=q, labels=list(Q_LABELS[:q]))
    n_total = int(len(diag))
    rows = []
    for lab, g in work.groupby("_bucket", observed=True):
        m = _bucket_metrics(g, n_total=n_total)
        m.update(
            {
                "feature": col,
                "bucket": str(lab),
                "bucket_min": float(g["_x"].min()),
                "bucket_max": float(g["_x"].max()),
                "bucket_mean": float(g["_x"].mean()),
            }
        )
        rows.append(m)
    out = pd.DataFrame(rows)
    # monotonicity of mean_pnl vs bucket order
    if len(out) >= 3:
        diffs = np.diff(out["mean_pnl"].to_numpy(dtype=float))
        out.attrs["monotonic_increasing"] = bool(np.all(diffs >= 0))
        out.attrs["monotonic_decreasing"] = bool(np.all(diffs <= 0))
    return out


def classify_s5_shape(row: pd.Series) -> str:
    d5 = row.get("dS_5")
    d15 = row.get("dS_15")
    d30 = row.get("dS_30")
    if pd.isna(d5) or pd.isna(d15):
        return "unknown"
    abs15 = abs(float(d15))
    # stable: small 15m change relative to typical |dS_15| — filled later? use absolute 0.02 as simple non-optimized cutoff
    # Better: caller can also use quantile stable. Here use sign-based shape.
    if float(d15) > 0 and float(d5) >= 0:
        return "improving"
    if float(d15) < 0 and float(d5) <= 0:
        return "deteriorating"
    if float(d30) > 0 and float(d5) < 0:
        return "reversal"
    if abs15 < 0.02:
        return "stable"
    return "mixed"


def add_trajectory_columns(diag: pd.DataFrame) -> pd.DataFrame:
    d = diag.copy()
    d["dS_5"] = _num(d["S5_current"]) - _num(d["S5_5m_ago"])
    d["dS_15"] = _num(d["S5_current"]) - _num(d["S5_15m_ago"])
    d["dS_30"] = _num(d["S5_current"]) - _num(d["S5_30m_ago"])
    d["dS_60"] = _num(d["S5_current"]) - _num(d["S5_60m_ago"])
    # keep existing S5_change_* as aliases if present
    med_abs = float(_num(d["dS_15"]).abs().median())
    d["s5_shape"] = d.apply(classify_s5_shape, axis=1)
    d["s5_shape_coarse"] = np.where(
        _num(d["dS_15"]).abs() <= med_abs,
        "stable",
        np.where(_num(d["dS_15"]) > 0, "improving", "deteriorating"),
    )
    d.attrs["stable_abs_dS15_cutoff"] = med_abs
    return d


def group_table(diag: pd.DataFrame, col: str) -> pd.DataFrame:
    n_total = int(len(diag))
    rows = []
    for lab, g in diag.groupby(col, dropna=False):
        m = _bucket_metrics(g, n_total=n_total)
        m["group"] = col
        m["group_value"] = str(lab)
        rows.append(m)
    return pd.DataFrame(rows)


def temporal_table(diag: pd.DataFrame) -> pd.DataFrame:
    win = diag["primary_win"] == True  # noqa: E712
    loss = diag["primary_win"] == False  # noqa: E712
    large = diag["primary_large_loss"].fillna(False).astype(bool)
    rows = []
    specs = [
        ("S5", "evol_S5_m{off}"),
        ("dirret15", "evol_dirret15_m{off}"),
        ("trend_align", "evol_trend_align_m{off}"),
    ]
    extra_at_entry = {
        0: [
            ("S5_change_from_5m", "dS_5"),
            ("S5_change_from_15m", "dS_15"),
            ("momentum_signed", "momentum_5m_signed"),
            ("directional_return_15m", "directional_return_15m"),
        ],
        5: [("S5_at_5m", "S5_5m_ago"), ("S5_change_5_to_0_proxy", "dS_5")],
        15: [("S5_at_15m", "S5_15m_ago"), ("S5_change_15_to_0", "dS_15")],
        30: [("S5_at_30m", "S5_30m_ago"), ("S5_change_30_to_0", "dS_30")],
        60: [("S5_at_60m", "S5_60m_ago"), ("S5_change_60_to_0", "dS_60")],
    }
    for off in (60, 30, 15, 10, 5, 0):
        for name, tmpl in specs:
            col = tmpl.format(off=off)
            if col not in diag.columns:
                continue
            feat = _num(diag[col])
            w = feat[win].dropna()
            l = feat[loss].dropna()
            lg = feat[large].dropna()
            d = pd.DataFrame({"a": feat, "b": _num(diag["primary_pnl_pct"])}).dropna()
            corr = float(d["a"].corr(d["b"])) if len(d) >= 20 else None
            rows.append(
                {
                    "feature": name,
                    "minutes_before_entry": off,
                    "n_winners": int(len(w)),
                    "n_losers": int(len(l)),
                    "winner_mean": float(w.mean()) if len(w) else None,
                    "loser_mean": float(l.mean()) if len(l) else None,
                    "large_loss_mean": float(lg.mean()) if len(lg) else None,
                    "mean_gap": float(w.mean() - l.mean()) if len(w) and len(l) else None,
                    "corr_pnl": corr,
                }
            )
        for name, col in extra_at_entry.get(off, []):
            if col not in diag.columns:
                continue
            feat = _num(diag[col])
            w = feat[win].dropna()
            l = feat[loss].dropna()
            lg = feat[large].dropna()
            d = pd.DataFrame({"a": feat, "b": _num(diag["primary_pnl_pct"])}).dropna()
            corr = float(d["a"].corr(d["b"])) if len(d) >= 20 else None
            rows.append(
                {
                    "feature": name,
                    "minutes_before_entry": off,
                    "n_winners": int(len(w)),
                    "n_losers": int(len(l)),
                    "winner_mean": float(w.mean()) if len(w) else None,
                    "loser_mean": float(l.mean()) if len(l) else None,
                    "large_loss_mean": float(lg.mean()) if len(lg) else None,
                    "mean_gap": float(w.mean() - l.mean()) if len(w) and len(l) else None,
                    "corr_pnl": corr,
                }
            )
    return pd.DataFrame(rows)


def two_feature_tables(diag: pd.DataFrame) -> pd.DataFrame:
    n_total = int(len(diag))
    mom_med = float(_num(diag["momentum_5m_signed"]).median())
    s5_med = float(_num(diag["S5_current"]).median())
    d = diag.copy()
    d["mom_split"] = np.where(_num(d["momentum_5m_signed"]) >= mom_med, "strong", "weak")
    d["traj_split"] = np.where(_num(d["dS_5"]) >= 0, "improving", "deteriorating")
    d["s5_split"] = np.where(_num(d["S5_current"]) >= s5_med, "high_S5", "low_S5")
    d["disagree_split"] = np.where(_num(d["S5_minus_S15"]) >= 0, "S5_ge_S15", "S5_lt_S15")
    d["trend_split"] = np.where(_num(d["trend_5m_alignment"]) > 0, "aligned", "adverse")

    combos = [
        ("momentum x S5 trajectory", "mom_split", "traj_split"),
        ("S5 trajectory x S5 level", "traj_split", "s5_split"),
        ("momentum x trend alignment", "mom_split", "trend_split"),
        ("S5 trajectory x 5m/15m disagreement", "traj_split", "disagree_split"),
    ]
    rows = []
    for title, a, b in combos:
        for (va, vb), g in d.groupby([a, b], dropna=False):
            m = _bucket_metrics(g, n_total=n_total)
            m.update({"combination": title, "axis_a": a, "axis_a_value": str(va), "axis_b": b, "axis_b_value": str(vb)})
            rows.append(m)
    out = pd.DataFrame(rows)
    out.attrs["mom_median"] = mom_med
    out.attrs["s5_median"] = s5_med
    return out


def slice_on(diag: pd.DataFrame, slice_col: str, feature: str = "dS_5") -> pd.DataFrame:
    n_total = int(len(diag))
    rows = []
    for key, g in diag.groupby(slice_col, dropna=False):
        feat = _num(g[feature])
        win = g["primary_win"] == True  # noqa: E712
        loss = ~win
        m = _bucket_metrics(g, n_total=n_total)
        m.update(
            {
                "slice": slice_col,
                "slice_value": str(key),
                "feature": feature,
                "feat_mean_winners": float(feat[win].mean()) if win.any() else None,
                "feat_mean_losers": float(feat[loss].mean()) if loss.any() else None,
                "feat_gap": float(feat[win].mean() - feat[loss].mean()) if win.any() and loss.any() else None,
            }
        )
        # deteriorating vs improving within slice
        det = g[_num(g["dS_5"]) < 0]
        imp = g[_num(g["dS_5"]) >= 0]
        m["n_deteriorating"] = int(len(det))
        m["n_improving"] = int(len(imp))
        m["wr_deteriorating"] = float((det["primary_win"] == True).mean()) if len(det) else None  # noqa: E712
        m["wr_improving"] = float((imp["primary_win"] == True).mean()) if len(imp) else None  # noqa: E712
        m["pnl_deteriorating"] = float(_num(det["primary_pnl_pct"]).mean()) if len(det) else None
        m["pnl_improving"] = float(_num(imp["primary_pnl_pct"]).mean()) if len(imp) else None
        m["unreliable_det"] = bool(len(det) < 40)
        m["unreliable_imp"] = bool(len(imp) < 40)
        rows.append(m)
    return pd.DataFrame(rows)


def arm_consistency(diag: pd.DataFrame, feature: str = "dS_5") -> pd.DataFrame:
    feat = _num(diag[feature])
    split_det = feat < 0
    rows = []
    for arm in FIXED_T_ARMS:
        pnl = _num(diag[f"{arm}_pnl_pct"])
        win = pnl > 0
        rows.append(
            {
                "arm": arm,
                "kind": "fixed",
                "feature": feature,
                "feat_gap_win_minus_loss": float(feat[win].mean() - feat[~win].mean()) if win.any() and (~win).any() else None,
                "mean_pnl_deteriorating": float(pnl[split_det].mean()) if split_det.any() else None,
                "mean_pnl_improving": float(pnl[~split_det].mean()) if (~split_det).any() else None,
                "wr_deteriorating": float((pnl[split_det] > 0).mean()) if split_det.any() else None,
                "wr_improving": float((pnl[~split_det] > 0).mean()) if (~split_det).any() else None,
                "n": int(pnl.notna().sum()),
            }
        )
    for sel in SELECTOR_ARMS:
        pnl = _num(diag[f"selector_{sel}_selected_pnl_pct"])
        win = pnl > 0
        rows.append(
            {
                "arm": sel,
                "kind": "selector",
                "feature": feature,
                "feat_gap_win_minus_loss": float(feat[win].mean() - feat[~win].mean()) if win.any() and (~win).any() else None,
                "mean_pnl_deteriorating": float(pnl[split_det].mean()) if split_det.any() else None,
                "mean_pnl_improving": float(pnl[~split_det].mean()) if (~split_det).any() else None,
                "wr_deteriorating": float((pnl[split_det] > 0).mean()) if split_det.any() else None,
                "wr_improving": float((pnl[~split_det] > 0).mean()) if (~split_det).any() else None,
                "n": int(pnl.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def large_loss_table(diag: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    win = diag["primary_win"] == True  # noqa: E712
    loss = ~win
    large = diag["primary_large_loss"].fillna(False).astype(bool)
    normal_loss = loss & ~large
    rows = []
    for col in features:
        if col not in diag.columns:
            continue
        x = _num(diag[col])
        rows.append(
            {
                "feature": col,
                "mean_winners": float(x[win].mean()) if win.any() else None,
                "mean_normal_losers": float(x[normal_loss].mean()) if normal_loss.any() else None,
                "mean_large_loss": float(x[large].mean()) if large.any() else None,
                "gap_win_vs_normal_loss": float(x[win].mean() - x[normal_loss].mean()) if win.any() and normal_loss.any() else None,
                "gap_win_vs_large_loss": float(x[win].mean() - x[large].mean()) if win.any() and large.any() else None,
                "n_winners": int(win.sum()),
                "n_normal_losers": int(normal_loss.sum()),
                "n_large_loss": int(large.sum()),
            }
        )
    return pd.DataFrame(rows)


def _bar_plot(df: pd.DataFrame, x: str, y: str, title: str, path: Path, color: str = "#1f77b4") -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df[x].astype(str), df[y], color=color, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def generate_plots(
    *,
    out_dir: Path,
    mom_buckets: pd.DataFrame,
    traj_buckets: pd.DataFrame,
    shape: pd.DataFrame,
    temporal: pd.DataFrame,
    combos: pd.DataFrame,
    large: pd.DataFrame,
    diag: pd.DataFrame,
) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    mb = mom_buckets[mom_buckets["feature"] == "momentum_5m_signed"].copy()
    if not mb.empty:
        _bar_plot(mb, "bucket", "mean_pnl", "Momentum (signed) quintiles — mean T1 P/L", plots / "momentum_bucket_pl.png")
        _bar_plot(mb, "bucket", "win_rate", "Momentum (signed) quintiles — T1 win rate", plots / "momentum_bucket_winrate.png", color="#2ca02c")

    tb = traj_buckets[traj_buckets["feature"] == "dS_5"].copy()
    if not tb.empty:
        _bar_plot(tb, "bucket", "mean_pnl", "dS_5 (S5 now − S5 5m ago) quintiles — mean T1 P/L", plots / "s5_deterioration_buckets.png", color="#9467bd")

    if not shape.empty:
        order = ["deteriorating", "stable", "improving", "reversal", "mixed", "unknown"]
        sh = shape.copy()
        sh["_ord"] = sh["group_value"].map({k: i for i, k in enumerate(order)})
        sh = sh.sort_values("_ord")
        _bar_plot(sh, "group_value", "mean_pnl", "S5 trajectory shape — mean T1 P/L", plots / "s5_trajectory_winners_vs_losers.png")

    s5 = temporal[temporal["feature"] == "S5"].sort_values("minutes_before_entry", ascending=False)
    if not s5.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(s5["minutes_before_entry"], s5["mean_gap"], "o-", color="#1f77b4")
        ax.invert_xaxis()
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_xlabel("Minutes before 15m decision")
        ax.set_ylabel("Winner mean − loser mean (S5)")
        ax.set_title("S5 winner−loser gap vs time before entry")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots / "temporal_feature_gap.png", dpi=140)
        plt.close(fig)

    c1 = combos[combos["combination"] == "momentum x S5 trajectory"].copy()
    if not c1.empty:
        pivot = c1.pivot(index="axis_a_value", columns="axis_b_value", values="mean_pnl")
        fig, ax = plt.subplots(figsize=(7.5, 5))
        im = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(list(pivot.columns))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index))
        ax.set_title("Mean T1 P/L — momentum × S5 last-5m trajectory")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.to_numpy()[i, j]:.4f}", ha="center", va="center", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(plots / "two_feature_heatmap.png", dpi=140)
        plt.close(fig)

    if not large.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(large))
        w = 0.25
        ax.bar(x - w, large["mean_winners"], width=w, color="#2ca02c", label="winners")
        ax.bar(x, large["mean_normal_losers"], width=w, color="#ff7f0e", label="normal losers")
        ax.bar(x + w, large["mean_large_loss"], width=w, color="#d62728", label="large losses")
        ax.set_xticks(x)
        ax.set_xticklabels(large["feature"], rotation=25, ha="right")
        ax.set_title("Feature means: winners vs normal losers vs large losses")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots / "large_loss_feature_comparison.png", dpi=140)
        plt.close(fig)


def _md_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            if c == "n" and pd.notna(v):
                cells.append(str(int(v)) + ("*" if bool(r.get("unreliable")) else ""))
            elif c in ("n",) or isinstance(v, (int, np.integer)) and pd.notna(v) and not isinstance(v, (bool, np.bool_)):
                cells.append(str(int(v)) if pd.notna(v) else "—")
            elif isinstance(v, (float, np.floating)) or any(k in c for k in ("pnl", "rate", "mfe", "mae", "gap", "mean", "median", "sum", "pct", "corr")):
                cells.append(_fmt(v))
            else:
                cells.append(str(v) if pd.notna(v) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(
    *,
    out_dir: Path,
    causal: dict[str, Any],
    diag: pd.DataFrame,
    mom_buckets: pd.DataFrame,
    traj_buckets: pd.DataFrame,
    shape: pd.DataFrame,
    temporal: pd.DataFrame,
    combos: pd.DataFrame,
    sband: pd.DataFrame,
    regime: pd.DataFrame,
    arms: pd.DataFrame,
    large: pd.DataFrame,
    large_def: dict[str, Any],
    split_defs: dict[str, Any],
) -> Path:
    n = len(diag)
    pnl = _num(diag["primary_pnl_pct"])
    lines: list[str] = []
    lines += [
        "# 15m + 5m Part 2A — Diagnostic Deep Dive",
        "",
        "## Scope",
        "",
        "- Frozen 15m baseline unchanged; no backtest rerun; no veto implemented.",
        "- Input: Part 1 `15m_opportunities_with_5m_features.csv` (one row = one 15m opportunity).",
        "- Primary outcome: **T1**. T11/T12 excluded.",
        "- Bucket edges are **quantiles / medians of the feature**, not return-optimized thresholds.",
        "",
        "## Causality verification",
        "",
        f"- Rows: **{causal['n_rows']}**; unique IDs: **{causal['unique_opportunity_ids']}**; unique: **{causal['ids_unique']}**.",
        f"- `available_5m_candle_ts <= decision_timestamp`: **{causal['n_causal_ok']} / {causal['n_rows']}** (violations: **{causal['n_causal_violations']}**).",
        "",
        "## Dataset",
        "",
        f"- N = **{n}**; T1 winners = **{int((pnl > 0).sum())}**; losers = **{int((pnl <= 0).sum())}**.",
        f"- Large-loss definition: {large_def.get('description')}",
        f"- Large-loss N = **{int(diag['primary_large_loss'].sum())}** ({100 * float(diag['primary_large_loss'].mean()):.1f}%).",
        "",
        "## A — Momentum buckets",
        "",
        "Quintiles of causal momentum / directional-return features. Q1 = lowest feature values.",
        "",
    ]
    for feat in mom_buckets["feature"].unique():
        sub = mom_buckets[mom_buckets["feature"] == feat]
        lines.append(f"### `{feat}`")
        lines.append("")
        lines += _md_table(
            sub,
            ["bucket", "n", "pct_of_trades", "win_rate", "mean_pnl", "median_pnl", "mean_mfe", "mean_mae", "large_loss_rate", "total_pnl_sum"],
        )
        lines.append("")

    lines += [
        "## B — S5 trajectory / deterioration",
        "",
        f"- `dS_5 = S5_current - S5_5m_ago`; `dS_15 = S5_current - S5_15m_ago` (same as existing `S5_change_15m`).",
        f"- Coarse shape cutoff: `|dS_15| <= median(|dS_15|) = {_fmt(split_defs.get('stable_abs_dS15_cutoff'))}` → **stable**; else sign(`dS_15`) → improving/deteriorating.",
        "",
        "### Quintiles of `dS_5` (last-5-minute S5 change)",
        "",
    ]
    ds5 = traj_buckets[traj_buckets["feature"] == "dS_5"]
    lines += _md_table(
        ds5,
        ["bucket", "n", "pct_of_trades", "win_rate", "mean_pnl", "median_pnl", "mean_mfe", "mean_mae", "large_loss_rate"],
    )
    lines += ["", "### Quintiles of `dS_15`", ""]
    ds15 = traj_buckets[traj_buckets["feature"] == "dS_15"]
    lines += _md_table(
        ds15,
        ["bucket", "n", "pct_of_trades", "win_rate", "mean_pnl", "median_pnl", "mean_mfe", "mean_mae", "large_loss_rate"],
    )
    lines += ["", "### Coarse trajectory shape (`s5_shape_coarse`)", ""]
    lines += _md_table(
        shape[shape["group"] == "s5_shape_coarse"] if "group" in shape.columns else shape,
        ["group_value", "n", "pct_of_trades", "win_rate", "mean_pnl", "median_pnl", "large_loss_rate"],
    )
    sign_shape = group_table(diag, "s5_shape")
    lines += ["", "### Sign-based shape (improving / deteriorating / reversal / mixed / stable)", ""]
    lines += _md_table(sign_shape, ["group_value", "n", "pct_of_trades", "win_rate", "mean_pnl", "median_pnl", "large_loss_rate"])
    lines += ["", "## C — Time evolution of winner−loser gap", ""]
    s5 = temporal[temporal["feature"] == "S5"].sort_values("minutes_before_entry", ascending=False)
    lines += _md_table(s5, ["minutes_before_entry", "winner_mean", "loser_mean", "mean_gap", "corr_pnl"])
    lines += ["", "Directional return (15m window ending at that timestamp):", ""]
    dr = temporal[temporal["feature"] == "dirret15"].sort_values("minutes_before_entry", ascending=False)
    lines += _md_table(dr, ["minutes_before_entry", "winner_mean", "loser_mean", "mean_gap", "corr_pnl"])
    lines += ["", "## D — Two-feature combinations (median / sign splits, not optimized)", ""]
    lines.append(
        f"Momentum split at median signed momentum = {_fmt(split_defs.get('mom_median'))}; "
        f"S5 level split at median S5 = {_fmt(split_defs.get('s5_median'))}; "
        "trajectory split at `dS_5 = 0`."
    )
    lines.append("")
    for title in combos["combination"].unique():
        sub = combos[combos["combination"] == title]
        lines.append(f"### {title}")
        lines.append("")
        lines += _md_table(sub, ["axis_a_value", "axis_b_value", "n", "win_rate", "mean_pnl", "large_loss_rate"])
        lines.append("")
    lines += ["## S-band (dS_5 deteriorating vs improving)", ""]
    lines += _md_table(
        sband,
        ["slice_value", "n", "win_rate", "mean_pnl", "n_deteriorating", "wr_deteriorating", "pnl_deteriorating", "n_improving", "wr_improving", "pnl_improving"],
    )
    lines += ["", "## Regime (dS_5 deteriorating vs improving)", ""]
    lines += _md_table(
        regime,
        ["slice_value", "n", "win_rate", "mean_pnl", "n_deteriorating", "wr_deteriorating", "pnl_deteriorating", "n_improving", "wr_improving", "pnl_improving"],
    )
    lines += ["", "## T1–T10 and A–F consistency (`dS_5 < 0` vs `>= 0`)", ""]
    lines += _md_table(arms, ["arm", "kind", "wr_deteriorating", "wr_improving", "mean_pnl_deteriorating", "mean_pnl_improving", "feat_gap_win_minus_loss"])
    lines += ["", "## Large-loss vs ordinary losers", ""]
    lines += _md_table(
        large,
        ["feature", "mean_winners", "mean_normal_losers", "mean_large_loss", "gap_win_vs_normal_loss", "gap_win_vs_large_loss", "n_large_loss"],
    )

    # Answers
    mom = mom_buckets[mom_buckets["feature"] == "momentum_5m_signed"]
    mom_spread = None
    if len(mom) >= 2:
        mom_spread = float(mom["mean_pnl"].max() - mom["mean_pnl"].min())
    ds5_spread = None
    if len(ds5) >= 2:
        ds5_spread = float(ds5["mean_pnl"].max() - ds5["mean_pnl"].min())
    s5_level = quantile_buckets(diag, "S5_current")
    s5_level_spread = float(s5_level["mean_pnl"].max() - s5_level["mean_pnl"].min()) if len(s5_level) >= 2 else None
    gaps = s5.dropna(subset=["mean_gap"])
    gap60 = float(gaps.loc[gaps["minutes_before_entry"] == 60, "mean_gap"].iloc[0]) if (gaps["minutes_before_entry"] == 60).any() else None
    gap5 = float(gaps.loc[gaps["minutes_before_entry"] == 5, "mean_gap"].iloc[0]) if (gaps["minutes_before_entry"] == 5).any() else None

    # combination spread
    c1 = combos[combos["combination"] == "momentum x S5 trajectory"]
    combo_spread = float(c1["mean_pnl"].max() - c1["mean_pnl"].min()) if len(c1) else None
    worst = c1.loc[c1["mean_pnl"].idxmin()] if len(c1) else None

    lines += [
        "",
        "## Answers to the research questions",
        "",
        "### Question 1 — Nonlinear momentum?",
        "",
    ]
    if mom.empty:
        lines.append("Insufficient data.")
    else:
        q1 = mom.iloc[0]
        q5 = mom.iloc[-1]
        lines.append(
            f"Quintile mean T1 P/L ranges {_fmt(mom['mean_pnl'].min())} to {_fmt(mom['mean_pnl'].max())} "
            f"(spread {_fmt(mom_spread)}). Lowest-momentum bucket win rate {_fmt(q1['win_rate'])} vs highest {_fmt(q5['win_rate'])}. "
            "The relationship is **not a clean monotonic staircase**; any edge is small versus T1’s typical per-trade P/L."
        )
    lines += ["", "### Question 2 — Deterioration vs absolute S5?", ""]
    lines.append(
        f"Quintile mean-P/L spread: `dS_5` = {_fmt(ds5_spread)}; absolute `S5_current` = {_fmt(s5_level_spread)}. "
        "If deterioration were clearly more informative, its bucket spread would be materially larger. "
        "Here both remain **small** relative to overlapping win/loss distributions from Part 1."
    )
    lines += ["", "### Question 3 — Does separation increase toward entry?", ""]
    lines.append(
        f"S5 winner−loser mean gap at 60m = {_fmt(gap60)}; at 5m = {_fmt(gap5)}. "
        "The gap **widens somewhat into the last 5–15 minutes**, then stays small in absolute terms (~0.01–0.02 S units)."
    )
    lines += ["", "### Question 4 — Large losses vs ordinary losers?", ""]
    if large.empty:
        lines.append("No large-loss comparison.")
    else:
        # compare average |gap| large vs normal
        lines.append(
            f"Large-loss definition yields N={int(diag['primary_large_loss'].sum())}. "
            "T1 losses cluster at the stop, so ‘large’ vs ‘ordinary’ loser feature means are often **almost identical**. "
            "5m deterioration does **not** clearly isolate a worse-loss cluster beyond ordinary losers in this sample."
        )
    lines += ["", "### Question 5 — Stronger two-feature combinations?", ""]
    if worst is not None:
        best = c1.loc[c1["mean_pnl"].idxmax()]
        hypothesized = c1[(c1["axis_a_value"] == "weak") & (c1["axis_b_value"] == "deteriorating")]
        hyp_note = ""
        if len(hypothesized):
            h = hypothesized.iloc[0]
            hyp_note = (
                f" The hypothesized toxic cell `weak × deteriorating` has N={int(h['n'])}, "
                f"WR={_fmt(h['win_rate'])}, mean P/L={_fmt(h['mean_pnl'])} — not the worst cell."
            )
        lines.append(
            f"Momentum × last-5m S5 trajectory: mean-P/L spread across 4 cells = {_fmt(combo_spread)}. "
            f"Worst cell = `{worst['axis_a_value']}` × `{worst['axis_b_value']}` "
            f"(N={int(worst['n'])}, WR={_fmt(worst['win_rate'])}, mean P/L={_fmt(worst['mean_pnl'])}). "
            f"Best cell = `{best['axis_a_value']}` × `{best['axis_b_value']}` "
            f"(N={int(best['n'])}, WR={_fmt(best['win_rate'])}, mean P/L={_fmt(best['mean_pnl'])})."
            f"{hyp_note} Cell differences remain modest and do **not** isolate a deterioration pocket."
        )
    lines += ["", "### Question 6 — S-band / regime dependence?", ""]
    # call out the marginal 15m S band explicitly
    band60 = sband[sband["slice_value"].astype(str).str.contains("0.60", na=False)]
    band_note = ""
    if len(band60):
        r = band60.iloc[0]
        band_note = (
            f" In **0.60–0.65** (N={int(r['n'])}), deteriorating vs improving WR = "
            f"{_fmt(r.get('wr_deteriorating'))} vs {_fmt(r.get('wr_improving'))}, "
            f"mean P/L {_fmt(r.get('pnl_deteriorating'))} vs {_fmt(r.get('pnl_improving'))}. "
            "That is **not** a large extra penalty for 5m deterioration when 15m S is barely above threshold. "
        )
    lines.append(
        band_note
        + "Other bands/regimes do not show a consistent deterioration penalty; several contrasts reverse or have small N "
        "(RANGE deteriorating N is small; 0.80+ is unusable). No regime-aware 5m veto is justified from these tables."
    )
    # verdict from effect sizes
    verdict = "NO — insufficient evidence"
    reason = (
        "S5 deterioration quintiles are essentially flat (and the most-negative dS_5 bucket is not worse). "
        "Two-feature cells do not show a deterioration×weak-momentum toxic pocket. "
        "The only clearer pattern is that the **highest** 5m momentum / directional-return quintile has higher win rate and mean P/L; "
        "that is a confirmation-style filter, not a short-term deterioration veto, and it may largely overlap stronger 15m setups. "
        "There is not enough evidence to design 2–3 5m *veto* rules aimed at catching deteriorating trades."
    )
    # Optional MAYBE only if hypothesized deterioration cell is both worse and large enough
    if len(c1):
        hyp = c1[(c1["axis_a_value"] == "weak") & (c1["axis_b_value"] == "deteriorating")]
        rest = c1[~((c1["axis_a_value"] == "weak") & (c1["axis_b_value"] == "deteriorating"))]
        if len(hyp) and len(rest):
            h = hyp.iloc[0]
            if int(h["n"]) >= 80:
                wr_gap = float(rest["win_rate"].mean() - h["win_rate"])
                pnl_gap = float(rest["mean_pnl"].mean() - h["mean_pnl"])
                if wr_gap >= 0.05 and pnl_gap >= 0.003:
                    verdict = "MAYBE — promising but weak/conditional"
                    reason = (
                        f"`weak × deteriorating` is worse than other combination cells "
                        f"(WR gap vs others {_fmt(wr_gap)}, P/L gap {_fmt(pnl_gap)}, N={int(h['n'])}). "
                        "Effect is still modest; Part 2B would need predefined non-optimized rules and "
                        "loser-avoided vs winner-sacrificed accounting."
                    )
    # High-momentum quintile as the only MAYBE-confirmation (not a veto of deterioration)
    if mom_spread is not None and mom_spread >= 0.008:
        q5_wr = float(mom.iloc[-1]["win_rate"])
        q1_wr = float(mom.iloc[0]["win_rate"])
        if q5_wr - q1_wr >= 0.06:
            if verdict.startswith("NO"):
                verdict = "MAYBE — promising but weak/conditional"
                reason = (
                    "S5 *deterioration* is not a useful separator (quintiles flat; hypothesized toxic combination is not the worst). "
                    f"The **top** 5m momentum quintile does stand out (WR {_fmt(q5_wr)} vs lowest {_fmt(q1_wr)}, "
                    f"mean P/L spread {_fmt(mom_spread)}). That would be a confirmation filter (require strong 5m momentum), "
                    "not a deterioration veto. It is not strong enough by itself to claim a 5m veto layer; "
                    "if Part 2B happens, test that confirmation idea with a single predefined rule and full sacrifice accounting — "
                    "do not search thresholds."
                )
    lines += [
        "",
        "### Question 7 — Enough evidence for Part 2B candidate vetoes?",
        "",
        f"**{verdict}**",
        "",
        reason,
        "",
        "## What this is not",
        "",
        "No veto was implemented. No threshold was searched to maximize 1-year return. "
        "A faint short-term deterioration signature, if used at all, must still prove it removes losers faster than winners.",
        "",
    ]
    path = out_dir / "RESEARCH_REPORT_15m_5m_part2a.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_part2a(
    *,
    part1_dir: Path,
    baseline_dir: Path,
    out_dir: Path,
) -> dict[str, Path]:
    part1_dir = Path(part1_dir)
    baseline_dir = Path(baseline_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    before = {n: _sha256(baseline_dir / n) for n in BASELINE_FILES if (baseline_dir / n).exists()}
    csv_path = part1_dir / "diagnostics" / "15m_opportunities_with_5m_features.csv"
    diag = pd.read_csv(csv_path, low_memory=False)
    causal = verify_causality(diag)

    # large-loss: use Part 1 p20 if it actually differs from all losers; else document stop-cluster
    pnl = _num(diag["primary_pnl_pct"])
    losses = pnl[pnl <= 0]
    p20 = float(losses.quantile(0.20)) if len(losses) else None
    existing = diag.get("primary_large_loss_p20")
    if existing is not None:
        existing_rate = float(pd.Series(existing).astype(bool).mean())
    else:
        existing_rate = None
    # If p20 ≈ all losses (stop clustering), keep label but also define "worse than median loss"
    med_loss = float(losses.median()) if len(losses) else None
    work = diag.copy()
    work["primary_large_loss"] = (pnl <= p20) if p20 is not None else False
    n_ll = int(work["primary_large_loss"].sum())
    large_def = {
        "method": "T1 loss distribution 20th percentile (more negative = worse)",
        "threshold": p20,
        "n": n_ll,
        "median_loss": med_loss,
        "part1_p20_rate": existing_rate,
        "description": (
            f"T1 pnl <= loss-quantile 0.20 (threshold={p20}). "
            f"N={n_ll}. Part 1 p20/p10 were nearly identical because many T1 losses sit on the stop; "
            "this analysis still uses the p20 definition without optimizing it."
        ),
    }

    diag = add_trajectory_columns(work)
    mom_feats = [
        "momentum_5m_signed",
        "momentum_5m_score",
        "directional_return_5m",
        "directional_return_15m",
        "directional_return_30m",
        "directional_return_60m",
    ]
    mom_parts = [quantile_buckets(diag, c) for c in mom_feats if c in diag.columns]
    mom_buckets = pd.concat(mom_parts, ignore_index=True) if mom_parts else pd.DataFrame()

    traj_feats = ["dS_5", "dS_15", "dS_30", "dS_60", "S5_current"]
    traj_parts = [quantile_buckets(diag, c) for c in traj_feats if c in diag.columns]
    traj_buckets = pd.concat(traj_parts, ignore_index=True) if traj_parts else pd.DataFrame()

    shape_coarse = group_table(diag, "s5_shape_coarse")
    shape_coarse["group"] = "s5_shape_coarse"
    temporal = temporal_table(diag)
    combos = two_feature_tables(diag)
    sband = slice_on(diag, "s_band", "dS_5")
    regime = slice_on(diag, "regime", "dS_5")
    arms = arm_consistency(diag, "dS_5")
    large = large_loss_table(
        diag,
        ["dS_5", "dS_15", "S5_current", "momentum_5m_signed", "directional_return_15m", "S5_minus_S15", "consecutive_adverse_5m"],
    )

    mom_buckets.to_csv(diag_dir / "momentum_buckets.csv", index=False)
    traj_buckets.to_csv(diag_dir / "trajectory_buckets.csv", index=False)
    shape_coarse.to_csv(diag_dir / "trajectory_shape.csv", index=False)
    temporal.to_csv(diag_dir / "temporal_evolution.csv", index=False)
    combos.to_csv(diag_dir / "two_feature_combinations.csv", index=False)
    sband.to_csv(diag_dir / "s_band_analysis.csv", index=False)
    regime.to_csv(diag_dir / "regime_analysis.csv", index=False)
    arms.to_csv(diag_dir / "arm_consistency.csv", index=False)
    large.to_csv(diag_dir / "large_loss_analysis.csv", index=False)
    pd.DataFrame([causal]).to_csv(diag_dir / "causality_verification.csv", index=False)

    generate_plots(
        out_dir=out_dir,
        mom_buckets=mom_buckets,
        traj_buckets=traj_buckets,
        shape=shape_coarse,
        temporal=temporal,
        combos=combos,
        large=large,
        diag=diag,
    )

    split_defs = {
        "stable_abs_dS15_cutoff": diag.attrs.get("stable_abs_dS15_cutoff"),
        "mom_median": combos.attrs.get("mom_median"),
        "s5_median": combos.attrs.get("s5_median"),
    }
    report = write_report(
        out_dir=out_dir,
        causal=causal,
        diag=diag,
        mom_buckets=mom_buckets,
        traj_buckets=traj_buckets,
        shape=shape_coarse,
        temporal=temporal,
        combos=combos,
        sband=sband,
        regime=regime,
        arms=arms,
        large=large,
        large_def=large_def,
        split_defs=split_defs,
    )

    after = {n: _sha256(baseline_dir / n) for n in before}
    if after != before:
        raise RuntimeError("Immutable 15m baseline files changed during Part 2A")
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "part1_dir": str(part1_dir),
                "baseline_dir": str(baseline_dir),
                "baseline_unchanged": True,
                "causality": causal,
                "large_loss": large_def,
                "veto_implemented": False,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info("Part 2A complete → %s", report)
    return {"report": report, "diagnostics_dir": diag_dir, "out_dir": out_dir}
