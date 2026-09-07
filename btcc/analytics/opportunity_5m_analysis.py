"""Analyze 5m diagnostic features vs immutable 15m opportunity outcomes.

Diagnostic only — no veto optimization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcc.analytics.opportunity_5m_diagnostics import (
    EVOLUTION_OFFSETS_MIN,
    FEATURE_OUTCOME_CSV,
    FEATURE_SUMMARY_CSV,
    FIXED_T_ARMS,
    REPORT_NAME,
    SELECTOR_ARMS,
)

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "S5_current",
    "S5_5m_ago",
    "S5_15m_ago",
    "S5_30m_ago",
    "S5_60m_ago",
    "S5_change_15m",
    "S5_change_30m",
    "S5_change_60m",
    "directional_return_5m",
    "directional_return_15m",
    "directional_return_30m",
    "directional_return_60m",
    "momentum_5m_score",
    "momentum_5m_signed",
    "natr_5m_current",
    "volatility_5m_recent",
    "natr_5m_change_ratio_30m",
    "trend_5m_alignment",
    "consecutive_favorable_5m",
    "consecutive_adverse_5m",
    "frac_favorable_5m_60m",
    "frac_adverse_5m_60m",
    "S5_persist_frac_ge060_15m",
    "S5_persist_frac_ge060_30m",
    "S5_persist_frac_ge060_60m",
    "S5_minus_S15",
    "disagreement_momentum",
    "disagreement_trend",
    "disagreement_s5_falling_15m",
]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _group_stats(feature: pd.Series, mask: pd.Series) -> dict[str, Any]:
    x = _num(feature)[mask].dropna()
    if x.empty:
        return {"n": 0}
    q = x.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std()) if len(x) > 1 else 0.0,
        "q10": float(q.get(0.1, np.nan)),
        "q25": float(q.get(0.25, np.nan)),
        "q50": float(q.get(0.5, np.nan)),
        "q75": float(q.get(0.75, np.nan)),
        "q90": float(q.get(0.9, np.nan)),
    }


def analyze_features(diag: pd.DataFrame) -> pd.DataFrame:
    win = diag["primary_win"] == True  # noqa: E712
    loss = diag["primary_win"] == False  # noqa: E712
    large = diag.get("primary_large_loss_p20", pd.Series(False, index=diag.index)).fillna(False).astype(bool)
    normal_loss = loss & ~large
    pnl = _num(diag["primary_pnl_pct"])
    mfe = _num(diag["primary_mfe_pct"])
    mae = _num(diag["primary_mae_pct"])

    rows = []
    for col in FEATURE_COLS:
        if col not in diag.columns:
            continue
        feat = _num(diag[col])
        # correlation with outcomes (information strength, not causation claim)
        def _corr(a, b):
            d = pd.DataFrame({"a": a, "b": b}).dropna()
            if len(d) < 10:
                return None
            return float(d["a"].corr(d["b"]))

        row = {
            "feature": col,
            "n": int(feat.notna().sum()),
            "mean_all": float(feat.mean()) if feat.notna().any() else None,
            "mean_winners": _group_stats(feat, win).get("mean"),
            "mean_losers": _group_stats(feat, loss).get("mean"),
            "median_winners": _group_stats(feat, win).get("median"),
            "median_losers": _group_stats(feat, loss).get("median"),
            "mean_large_loss": _group_stats(feat, large).get("mean"),
            "mean_normal_loss": _group_stats(feat, normal_loss).get("mean"),
            "winner_loser_mean_gap": None,
            "corr_pnl": _corr(feat, pnl),
            "corr_mfe": _corr(feat, mfe),
            "corr_mae": _corr(feat, mae),
        }
        if row["mean_winners"] is not None and row["mean_losers"] is not None:
            row["winner_loser_mean_gap"] = row["mean_winners"] - row["mean_losers"]
        # absolute gap for ranking
        row["abs_winner_loser_gap"] = None if row["winner_loser_mean_gap"] is None else abs(row["winner_loser_mean_gap"])
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("abs_winner_loser_gap", ascending=False, na_position="last")
    return out


def analyze_by_slice(diag: pd.DataFrame, slice_col: str, feature: str = "S5_change_15m") -> pd.DataFrame:
    rows = []
    if slice_col not in diag.columns or feature not in diag.columns:
        return pd.DataFrame()
    for key, g in diag.groupby(slice_col):
        win = g["primary_win"] == True  # noqa: E712
        loss = g["primary_win"] == False  # noqa: E712
        feat = _num(g[feature])
        rows.append(
            {
                "slice": slice_col,
                "slice_value": key,
                "feature": feature,
                "n": int(len(g)),
                "win_rate": float(win.mean()) if len(g) else None,
                "avg_pnl": float(_num(g["primary_pnl_pct"]).mean()),
                "feat_mean_winners": float(feat[win].mean()) if win.any() else None,
                "feat_mean_losers": float(feat[loss].mean()) if loss.any() else None,
                "feat_gap": (
                    float(feat[win].mean() - feat[loss].mean()) if win.any() and loss.any() else None
                ),
            }
        )
    return pd.DataFrame(rows)


def analyze_evolution(diag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    win = diag["primary_win"] == True  # noqa: E712
    loss = diag["primary_win"] == False  # noqa: E712
    large = diag.get("primary_large_loss_p20", pd.Series(False, index=diag.index)).fillna(False).astype(bool)
    for off in EVOLUTION_OFFSETS_MIN:
        col = f"evol_S5_m{off}"
        if col not in diag.columns:
            continue
        feat = _num(diag[col])
        rows.append(
            {
                "minutes_before_entry": off,
                "feature": "S5",
                "mean_winners": float(feat[win].mean()) if win.any() else None,
                "mean_losers": float(feat[loss].mean()) if loss.any() else None,
                "mean_large_loss": float(feat[large].mean()) if large.any() else None,
                "gap_win_minus_loss": (
                    float(feat[win].mean() - feat[loss].mean()) if win.any() and loss.any() else None
                ),
            }
        )
        col2 = f"evol_dirret15_m{off}"
        if col2 in diag.columns:
            feat2 = _num(diag[col2])
            rows.append(
                {
                    "minutes_before_entry": off,
                    "feature": "dirret15",
                    "mean_winners": float(feat2[win].mean()) if win.any() else None,
                    "mean_losers": float(feat2[loss].mean()) if loss.any() else None,
                    "mean_large_loss": float(feat2[large].mean()) if large.any() else None,
                    "gap_win_minus_loss": (
                        float(feat2[win].mean() - feat2[loss].mean()) if win.any() and loss.any() else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def analyze_by_t_and_selectors(diag: pd.DataFrame, feature: str = "S5_change_15m") -> pd.DataFrame:
    rows = []
    feat = _num(diag.get(feature, pd.Series(dtype=float)))
    for arm in FIXED_T_ARMS:
        pnl_c = f"{arm}_pnl_pct"
        if pnl_c not in diag.columns:
            continue
        pnl = _num(diag[pnl_c])
        win = pnl > 0
        loss = pnl <= 0
        rows.append(
            {
                "arm": arm,
                "kind": "fixed",
                "feature": feature,
                "n": int(pnl.notna().sum()),
                "feat_mean_winners": float(feat[win].mean()) if win.any() else None,
                "feat_mean_losers": float(feat[loss].mean()) if loss.any() else None,
                "feat_gap": float(feat[win].mean() - feat[loss].mean()) if win.any() and loss.any() else None,
                "avg_pnl": float(pnl.mean()),
            }
        )
    for sel in SELECTOR_ARMS:
        pnl_c = f"selector_{sel}_selected_pnl_pct"
        if pnl_c not in diag.columns:
            continue
        pnl = _num(diag[pnl_c])
        win = pnl > 0
        loss = pnl <= 0
        rows.append(
            {
                "arm": sel,
                "kind": "selector",
                "feature": feature,
                "n": int(pnl.notna().sum()),
                "feat_mean_winners": float(feat[win].mean()) if win.any() else None,
                "feat_mean_losers": float(feat[loss].mean()) if loss.any() else None,
                "feat_gap": float(feat[win].mean() - feat[loss].mean()) if win.any() and loss.any() else None,
                "avg_pnl": float(pnl.mean()),
            }
        )
    return pd.DataFrame(rows)


def _hist_win_loss(ax, diag: pd.DataFrame, col: str, title: str) -> None:
    win = _num(diag.loc[diag["primary_win"] == True, col])  # noqa: E712
    loss = _num(diag.loc[diag["primary_win"] == False, col])  # noqa: E712
    win = win.dropna()
    loss = loss.dropna()
    if win.empty and loss.empty:
        ax.set_title(title + " (no data)")
        return
    bins = 30
    ax.hist(win, bins=bins, alpha=0.55, color="#2ca02c", label=f"winners n={len(win)}")
    ax.hist(loss, bins=bins, alpha=0.55, color="#d62728", label=f"losers n={len(loss)}")
    if len(win):
        ax.axvline(win.mean(), color="#145a32", ls="--", lw=1.5)
    if len(loss):
        ax.axvline(loss.mean(), color="#7b241c", ls="--", lw=1.5)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)


def generate_diagnostic_plots(diag: pd.DataFrame, out_dir: Path, evolution: pd.DataFrame) -> None:
    plots = Path(out_dir) / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("S5_current", "5m S: winners vs losers"),
        ("S5_change_15m", "5m S deterioration (15m): winners vs losers"),
        ("S5_change_30m", "5m S deterioration (30m): winners vs losers"),
        ("directional_return_15m", "5m directional return 15m: winners vs losers"),
        ("momentum_5m_signed", "5m momentum signed: winners vs losers"),
        ("natr_5m_current", "5m NATR: winners vs losers"),
        ("S5_minus_S15", "5m/15m S disagreement: winners vs losers"),
        ("consecutive_adverse_5m", "Consecutive adverse 5m candles: winners vs losers"),
        ("S5_persist_frac_ge060_30m", "5m S persistence (>=0.60 over 30m)"),
    ]
    for col, title in pairs:
        if col not in diag.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        _hist_win_loss(ax, diag, col, title)
        fig.tight_layout()
        safe = col.replace("/", "_")
        fig.savefig(plots / f"winloss_{safe}.png", dpi=140)
        plt.close(fig)

    # feature vs pnl / mae / mfe scatter for top candidates
    for col in ("S5_change_15m", "directional_return_15m", "S5_minus_S15", "consecutive_adverse_5m"):
        if col not in diag.columns:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        x = _num(diag[col])
        for ax, ycol, ylab in zip(
            axes,
            ("primary_pnl_pct", "primary_mae_pct", "primary_mfe_pct"),
            ("realized P/L", "MAE", "MFE"),
        ):
            y = _num(diag[ycol])
            d = pd.DataFrame({"x": x, "y": y}).dropna()
            ax.scatter(d["x"], d["y"], s=8, alpha=0.25, color="#1f77b4")
            ax.set_xlabel(col)
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.25)
            ax.set_title(f"{col} vs {ylab}")
        fig.tight_layout()
        fig.savefig(plots / f"scatter_{col}.png", dpi=140)
        plt.close(fig)

    # evolution plot
    if not evolution.empty:
        for feat_name in ("S5", "dirret15"):
            sub = evolution[evolution["feature"] == feat_name].sort_values("minutes_before_entry", ascending=False)
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(sub["minutes_before_entry"], sub["mean_winners"], "o-", color="#2ca02c", label="winners")
            ax.plot(sub["minutes_before_entry"], sub["mean_losers"], "o-", color="#d62728", label="losers")
            ax.plot(sub["minutes_before_entry"], sub["mean_large_loss"], "s--", color="#9467bd", label="large losses")
            ax.invert_xaxis()
            ax.set_xlabel("Minutes before 15m decision")
            ax.set_ylabel(feat_name)
            ax.set_title(f"Feature evolution before entry — {feat_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plots / f"evolution_{feat_name}.png", dpi=140)
            plt.close(fig)


def write_research_report(
    *,
    out_dir: Path,
    diag: pd.DataFrame,
    feature_outcome: pd.DataFrame,
    evolution: pd.DataFrame,
    by_sband: pd.DataFrame,
    by_regime: pd.DataFrame,
    by_arm: pd.DataFrame,
) -> Path:
    out_dir = Path(out_dir)
    diag_dir = out_dir / "diagnostics"
    pnl = _num(diag["primary_pnl_pct"])
    lines = []
    lines.append("# 15m + 5m Diagnostic Research Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Immutable 15m baseline opportunities only (no strategy change, no veto implemented).")
    lines.append("- One row per 15m opportunity; T11/T12 excluded from labels.")
    lines.append("- All 5m features use candles with `timestamp <=` 15m decision timestamp.")
    lines.append("- Primary outcome arm for win/loss analysis: **T1**.")
    lines.append("")
    lines.append("## Dataset summary")
    lines.append("")
    lines.append(f"- Opportunities: **{len(diag)}**")
    lines.append(f"- Winners (T1): **{int((pnl > 0).sum())}**")
    lines.append(f"- Losers (T1): **{int((pnl <= 0).sum())}**")
    lines.append(f"- Avg T1 P/L: **{100 * float(pnl.mean()):.3f}%**")
    lines.append(f"- Avg T1 MFE: **{100 * float(_num(diag['primary_mfe_pct']).mean()):.3f}%**")
    lines.append(f"- Avg T1 MAE: **{100 * float(_num(diag['primary_mae_pct']).mean()):.3f}%**")
    if "five_m_data_available" in diag.columns:
        lines.append(f"- Rows with 5m data: **{int(diag['five_m_data_available'].fillna(False).sum())}**")
    lines.append("")
    lines.append("## Q1–Q3: Does 5m contain useful information? Strongest features?")
    lines.append("")
    top = feature_outcome.head(12) if not feature_outcome.empty else pd.DataFrame()
    if top.empty:
        lines.append("Insufficient feature coverage to rank.")
    else:
        lines.append("Top features by |mean(winners) − mean(losers)| on T1 outcomes:")
        lines.append("")
        lines.append("| feature | gap(w-l) | mean win | mean loss | mean large-loss | corr(pnl) | corr(mae) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, r in top.iterrows():
            lines.append(
                f"| `{r['feature']}` | {_fmt(r.get('winner_loser_mean_gap'))} | {_fmt(r.get('mean_winners'))} | "
                f"{_fmt(r.get('mean_losers'))} | {_fmt(r.get('mean_large_loss'))} | {_fmt(r.get('corr_pnl'))} | {_fmt(r.get('corr_mae'))} |"
            )
    lines.append("")
    lines.append("## Q4: 5m/15m disagreement")
    lines.append("")
    for col in ("S5_minus_S15", "disagreement_momentum", "disagreement_trend", "disagreement_s5_falling_15m"):
        sub = feature_outcome[feature_outcome["feature"] == col] if not feature_outcome.empty else pd.DataFrame()
        if sub.empty:
            continue
        r = sub.iloc[0]
        lines.append(
            f"- `{col}`: winner−loser gap={_fmt(r.get('winner_loser_mean_gap'))}, "
            f"corr_pnl={_fmt(r.get('corr_pnl'))}, corr_mae={_fmt(r.get('corr_mae'))}"
        )
    lines.append("")
    lines.append("## Q5: Dependence on 15m S band")
    lines.append("")
    if by_sband.empty:
        lines.append("No s_band slice results.")
    else:
        lines.append("| S band | n | win rate | feat gap (win−loss) | avg pnl |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in by_sband.iterrows():
            lines.append(
                f"| {r['slice_value']} | {int(r['n'])} | {_fmt(r.get('win_rate'))} | {_fmt(r.get('feat_gap'))} | {_fmt(r.get('avg_pnl'))} |"
            )
    lines.append("")
    lines.append("## Q6: Dependence on regime")
    lines.append("")
    if by_regime.empty:
        lines.append("No regime slice results.")
    else:
        lines.append("| regime | n | win rate | feat gap | avg pnl |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in by_regime.iterrows():
            lines.append(
                f"| {r['slice_value']} | {int(r['n'])} | {_fmt(r.get('win_rate'))} | {_fmt(r.get('feat_gap'))} | {_fmt(r.get('avg_pnl'))} |"
            )
    lines.append("")
    lines.append("## Q7–Q8: Dependence on T1–T10 and A–F")
    lines.append("")
    if by_arm.empty:
        lines.append("No arm-level results.")
    else:
        lines.append("| arm | kind | feat gap | avg pnl |")
        lines.append("|---|---|---:|---:|")
        for _, r in by_arm.iterrows():
            lines.append(f"| {r['arm']} | {r['kind']} | {_fmt(r.get('feat_gap'))} | {_fmt(r.get('avg_pnl'))} |")
    lines.append("")
    lines.append("## Q9: When does useful information appear?")
    lines.append("")
    if evolution.empty:
        lines.append("No evolution series.")
    else:
        s5 = evolution[evolution["feature"] == "S5"].sort_values("minutes_before_entry", ascending=False)
        lines.append("S5 mean by minutes before entry:")
        lines.append("")
        lines.append("| minutes before | winners | losers | large loss | gap |")
        lines.append("|---:|---:|---:|---:|---:|")
        for _, r in s5.iterrows():
            lines.append(
                f"| {int(r['minutes_before_entry'])} | {_fmt(r.get('mean_winners'))} | {_fmt(r.get('mean_losers'))} | "
                f"{_fmt(r.get('mean_large_loss'))} | {_fmt(r.get('gap_win_minus_loss'))} |"
            )
        gaps = s5.dropna(subset=["gap_win_minus_loss"])
        if not gaps.empty:
            best = gaps.loc[gaps["gap_win_minus_loss"].abs().idxmax()]
            lines.append("")
            lines.append(
                f"Largest |gap| for S5 occurs at **{int(best['minutes_before_entry'])} minutes** before entry "
                f"(gap={_fmt(best['gap_win_minus_loss'])})."
            )
    lines.append("")
    lines.append("## Q10: Most promising candidates for a *future* simple veto")
    lines.append("")
    lines.append("These are diagnostic candidates only — **not implemented**:")
    lines.append("")
    if not top.empty:
        for f in top["feature"].head(5).tolist():
            lines.append(f"- `{f}`")
    else:
        lines.append("- (none yet)")
    lines.append("")
    lines.append("## Research conclusion status")
    lines.append("")
    lines.append("Inspect the ranked gaps and evolution plots before designing any veto.")
    lines.append("Possible outcomes remain: strong / conditional / no useful evidence.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{diag_dir / '15m_opportunities_with_5m_features.csv'}`")
    lines.append(f"- `{diag_dir / FEATURE_SUMMARY_CSV}`")
    lines.append(f"- `{diag_dir / FEATURE_OUTCOME_CSV}`")
    lines.append(f"- `{out_dir / 'plots'}`")
    path = out_dir / REPORT_NAME
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def run_full_analysis(out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    diag_dir = out_dir / "diagnostics"
    csv_path = diag_dir / "15m_opportunities_with_5m_features.csv"
    diag = pd.read_csv(csv_path, low_memory=False)

    feature_outcome = analyze_features(diag)
    feature_outcome.to_csv(diag_dir / FEATURE_OUTCOME_CSV, index=False)

    # feature summary (overall quantiles)
    sum_rows = []
    for col in FEATURE_COLS:
        if col not in diag.columns:
            continue
        x = _num(diag[col]).dropna()
        if x.empty:
            continue
        sum_rows.append(
            {
                "feature": col,
                "n": int(len(x)),
                "mean": float(x.mean()),
                "median": float(x.median()),
                "q10": float(x.quantile(0.1)),
                "q25": float(x.quantile(0.25)),
                "q75": float(x.quantile(0.75)),
                "q90": float(x.quantile(0.9)),
            }
        )
    feature_summary = pd.DataFrame(sum_rows)
    feature_summary.to_csv(diag_dir / FEATURE_SUMMARY_CSV, index=False)

    evolution = analyze_evolution(diag)
    evolution.to_csv(diag_dir / "feature_evolution_before_entry.csv", index=False)
    by_sband = analyze_by_slice(diag, "s_band", "S5_change_15m")
    by_sband.to_csv(diag_dir / "analysis_by_s_band.csv", index=False)
    by_regime = analyze_by_slice(diag, "regime", "S5_change_15m")
    by_regime.to_csv(diag_dir / "analysis_by_regime.csv", index=False)
    by_arm = analyze_by_t_and_selectors(diag, "S5_change_15m")
    by_arm.to_csv(diag_dir / "analysis_by_arm.csv", index=False)

    generate_diagnostic_plots(diag, out_dir, evolution)
    report = write_research_report(
        out_dir=out_dir,
        diag=diag,
        feature_outcome=feature_outcome,
        evolution=evolution,
        by_sband=by_sband,
        by_regime=by_regime,
        by_arm=by_arm,
    )
    logger.info("Analysis complete → %s", report)
    return {
        "feature_outcome": diag_dir / FEATURE_OUTCOME_CSV,
        "feature_summary": diag_dir / FEATURE_SUMMARY_CSV,
        "report": report,
    }
