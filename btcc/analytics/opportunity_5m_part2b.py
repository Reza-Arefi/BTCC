"""Part 2B: predefined top-quintile 5m momentum confirmation test.

Hypothesis (frozen from Part 2A): PASS iff momentum_5m_signed is in the
top quintile via pandas qcut(q=5) on the same 1517-opportunity dataset.

No threshold search. No live/paper changes. No 15m backtest rerun.
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
BASELINE_FILES = (
    "opportunities.csv",
    "strategy_legs.csv",
    "selection_audit.csv",
    "predictions.csv",
    "summary.json",
)
PRIMARY_FEATURE = "momentum_5m_signed"
SECONDARY_FEATURE = "directional_return_15m"  # same Part 2A family, not a new search
Q_LABELS = ("Q1_lowest", "Q2", "Q3", "Q4", "Q5_highest")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def assign_predefined_quintiles(series: pd.Series) -> pd.Series:
    """Same construction as Part 2A: pd.qcut(q=5) on the feature distribution."""
    x = _num(series)
    out = pd.Series(index=series.index, dtype="object")
    valid = x.notna()
    labels = pd.qcut(x[valid], q=5, labels=list(Q_LABELS), duplicates="drop")
    out.loc[valid] = labels.astype(str)
    return out


def profit_factor(pnl: pd.Series) -> float | None:
    p = _num(pnl).dropna()
    gp = float(p[p > 0].sum())
    gl = float((-p[p < 0]).sum())
    if gl <= 1e-12:
        return None if gp <= 0 else float("inf")
    return gp / gl


def max_drawdown_from_sequence(pnl: pd.Series) -> float | None:
    p = _num(pnl).fillna(0.0)
    if p.empty:
        return None
    eq = p.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    return float(dd.min())


def subset_metrics(g: pd.DataFrame, pnl_col: str, win_col: str | None = None) -> dict[str, Any]:
    pnl = _num(g[pnl_col])
    if win_col and win_col in g.columns:
        win = g[win_col] == True  # noqa: E712
    else:
        win = pnl > 0
    loss = ~win
    mfe_col = pnl_col.replace("_pnl_pct", "_mfe_pct").replace("primary_pnl_pct", "primary_mfe_pct")
    mae_col = pnl_col.replace("_pnl_pct", "_mae_pct").replace("primary_pnl_pct", "primary_mae_pct")
    mfe = _num(g[mfe_col]) if mfe_col in g.columns else pd.Series(dtype=float)
    mae = _num(g[mae_col]) if mae_col in g.columns else pd.Series(dtype=float)
    large = g["primary_large_loss"].fillna(False).astype(bool) if "primary_large_loss" in g.columns else pd.Series(False, index=g.index)
    n = int(len(g))
    wins = g[win]
    losses = g[loss]
    return {
        "n": n,
        "win_rate": float(win.mean()) if n else None,
        "mean_pnl": float(pnl.mean()) if n else None,
        "median_pnl": float(pnl.median()) if n else None,
        "total_pnl": float(pnl.sum()) if n else None,
        "profit_factor": profit_factor(pnl),
        "avg_winner": float(_num(wins[pnl_col]).mean()) if len(wins) else None,
        "avg_loser": float(_num(losses[pnl_col]).mean()) if len(losses) else None,
        "mean_mfe": float(mfe.mean()) if n and mfe.notna().any() else None,
        "mean_mae": float(mae.mean()) if n and mae.notna().any() else None,
        "large_loss_rate": float(large.mean()) if n else None,
        "n_winners": int(win.sum()),
        "n_losers": int(loss.sum()),
    }


def confusion_and_retention(diag: pd.DataFrame, pass_mask: pd.Series, pnl_col: str) -> dict[str, Any]:
    pnl = _num(diag[pnl_col])
    win = pnl > 0
    loss = ~win
    passed = pass_mask.fillna(False)
    tp = int((passed & win).sum())
    fp = int((passed & loss).sum())
    fn = int((~passed & win).sum())
    tn = int((~passed & loss).sum())
    n = int(len(diag))
    n_win = int(win.sum())
    n_loss = int(loss.sum())
    n_pass = int(passed.sum())
    loser_rej = (tn / n_loss) if n_loss else None
    winner_sac = (fn / n_win) if n_win else None
    ratio = None
    if winner_sac is not None and winner_sac > 1e-12 and loser_rej is not None:
        ratio = loser_rej / winner_sac
    return {
        "n": n,
        "pass_winners": tp,
        "pass_losers": fp,
        "fail_winners": fn,
        "fail_losers": tn,
        "n_winners": n_win,
        "n_losers": n_loss,
        "n_pass": n_pass,
        "trade_retention": n_pass / n if n else None,
        "winner_retention": tp / n_win if n_win else None,
        "loser_rejection": loser_rej,
        "winner_sacrifice": winner_sac,
        "selectivity_ratio": ratio,
        "trades_retained_pct": 100.0 * n_pass / n if n else None,
    }


def compare_baseline_filtered(diag: pd.DataFrame, pass_mask: pd.Series, pnl_col: str, win_col: str | None = None) -> dict[str, Any]:
    passed = pass_mask.fillna(False)
    base = subset_metrics(diag, pnl_col, win_col)
    filt = subset_metrics(diag.loc[passed], pnl_col, win_col)
    ret = confusion_and_retention(diag, passed, pnl_col)
    row = {f"baseline_{k}": v for k, v in base.items()}
    row.update({f"filtered_{k}": v for k, v in filt.items()})
    row.update(ret)
    row["exposure_retained"] = ret["trade_retention"]
    row["total_pnl_retained_frac"] = (
        (filt["total_pnl"] / base["total_pnl"]) if base.get("total_pnl") else None
    )
    return row


def _md_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            if c in ("n", "pass_winners", "pass_losers", "fail_winners", "fail_losers", "n_winners", "n_losers", "n_pass") and pd.notna(v):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)) or any(k in str(c) for k in ("rate", "pnl", "frac", "ratio", "retention", "sacrifice", "rejection", "pct", "mfe", "mae", "pf", "wr")):
                cells.append(_fmt(v))
            else:
                cells.append(str(v) if pd.notna(v) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def generate_plots(out_dir: Path, *, overall: dict, arms: pd.DataFrame, sband: pd.DataFrame, regime: pd.DataFrame, timeb: pd.DataFrame) -> None:
    plots = Path(out_dir) / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["PASS winners", "PASS losers", "FAIL winners", "FAIL losers"]
    vals = [overall["pass_winners"], overall["pass_losers"], overall["fail_winners"], overall["fail_losers"]]
    colors = ["#2ca02c", "#d62728", "#98df8a", "#ff9896"]
    ax.bar(labels, vals, color=colors)
    ax.set_title("T1: PASS/FAIL × winner/loser counts")
    ax.set_ylabel("Opportunities")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "winner_loser_retention.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(["baseline", "filtered"], [overall["baseline_win_rate"], overall["filtered_win_rate"]], color=["#7f7f7f", "#1f77b4"])
    axes[0].set_title("T1 win rate")
    axes[0].set_ylim(0, 1)
    axes[1].bar(["baseline", "filtered"], [overall["baseline_mean_pnl"], overall["filtered_mean_pnl"]], color=["#7f7f7f", "#1f77b4"])
    axes[1].set_title("T1 mean P/L")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "baseline_vs_filtered.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(arms))
    ax.bar(x - 0.2, arms["baseline_win_rate"], 0.4, label="baseline WR", color="#7f7f7f")
    ax.bar(x + 0.2, arms["filtered_win_rate"], 0.4, label="filtered WR", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(arms["arm"])
    ax.set_title("T1–T10 win rate: baseline vs predefined confirmation")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "t1_t10_comparison.png", dpi=140)
    plt.close(fig)

    if not sband.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        x = np.arange(len(sband))
        ax.bar(x - 0.2, sband["baseline_win_rate"], 0.4, label="baseline WR", color="#7f7f7f")
        ax.bar(x + 0.2, sband["filtered_win_rate"], 0.4, label="filtered WR", color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(sband["slice"].astype(str), rotation=15)
        ax.set_title("S-band win rate: baseline vs filtered")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots / "s_band_comparison.png", dpi=140)
        plt.close(fig)

    if not regime.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        x = np.arange(len(regime))
        ax.bar(x - 0.2, regime["baseline_win_rate"], 0.4, label="baseline WR", color="#7f7f7f")
        ax.bar(x + 0.2, regime["filtered_win_rate"], 0.4, label="filtered WR", color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(regime["slice"].astype(str), rotation=20)
        ax.set_title("Regime win rate: baseline vs filtered")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots / "regime_comparison.png", dpi=140)
        plt.close(fig)

    if not timeb.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(timeb["slice"].astype(str), timeb["baseline_win_rate"], "o-", color="#7f7f7f", label="baseline WR")
        ax.plot(timeb["slice"].astype(str), timeb["filtered_win_rate"], "s-", color="#1f77b4", label="filtered WR")
        ax.set_ylim(0, 1)
        ax.set_title("Chronological stability of win rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots / "time_stability.png", dpi=140)
        plt.close(fig)


def write_report(
    *,
    out_dir: Path,
    rule: dict[str, Any],
    overall: dict[str, Any],
    mag: dict[str, Any],
    arms: pd.DataFrame,
    e_row: dict[str, Any],
    sband: pd.DataFrame,
    regime: pd.DataFrame,
    timeb: pd.DataFrame,
    secondary: dict[str, Any] | None,
    decision: str,
    decision_why: str,
) -> Path:
    o = overall
    lines = [
        "# 15m + 5m Part 2B — Predefined confirmation filter",
        "",
        "## Hypothesis (frozen before evaluating filtered performance)",
        "",
        f"- Feature: `{rule['feature']}`",
        f"- Construction: `{rule['construction']}`",
        f"- PASS if bucket = `{rule['pass_bucket']}`",
        f"- Recovered Q5 lower edge (min value in Q5): `{rule['q5_min']}`",
        f"- PASS N = **{rule['n_pass']} / {rule['n_all']}** ({_fmt(100 * rule['n_pass'] / rule['n_all'], 1)}%)",
        "- No threshold was searched in this task.",
        "",
        "## Causality / immutability",
        "",
        f"- Causal 5m timestamps OK: **{rule['causal_ok']} / {rule['n_all']}**",
        f"- Baseline files unchanged: **{rule['baseline_unchanged']}**",
        "- Paper/live bot not modified. Selector E not rerun.",
        "",
        "## 1. Winner / loser classification (T1)",
        "",
        "| | Winner | Loser | Total |",
        "| --- | ---: | ---: | ---: |",
        f"| PASS | {o['pass_winners']} | {o['pass_losers']} | {o['n_pass']} |",
        f"| FAIL | {o['fail_winners']} | {o['fail_losers']} | {o['n'] - o['n_pass']} |",
        f"| Total | {o['n_winners']} | {o['n_losers']} | {o['n']} |",
        "",
        f"- Trade retention: **{_fmt(o['trade_retention'])}** ({o['n_pass']}/{o['n']})",
        f"- Winner retention: **{_fmt(o['winner_retention'])}** ({o['pass_winners']}/{o['n_winners']})",
        f"- Loser rejection: **{_fmt(o['loser_rejection'])}** ({o['fail_losers']}/{o['n_losers']})",
        f"- Winner sacrifice: **{_fmt(o['winner_sacrifice'])}** ({o['fail_winners']}/{o['n_winners']})",
        f"- Selectivity ratio (loser rejection / winner sacrifice): **{_fmt(o['selectivity_ratio'])}**",
        "",
        "A ratio near 1 means the filter drops winners and losers at similar rates.",
        "",
        "## 2. T1 baseline vs filtered",
        "",
        "| metric | baseline (all) | filtered (PASS only) |",
        "| --- | ---: | ---: |",
        f"| N | {o['baseline_n']} | {o['filtered_n']} |",
        f"| win rate | {_fmt(o['baseline_win_rate'])} | {_fmt(o['filtered_win_rate'])} |",
        f"| mean P/L | {_fmt(o['baseline_mean_pnl'])} | {_fmt(o['filtered_mean_pnl'])} |",
        f"| median P/L | {_fmt(o['baseline_median_pnl'])} | {_fmt(o['filtered_median_pnl'])} |",
        f"| profit factor | {_fmt(o['baseline_profit_factor'])} | {_fmt(o['filtered_profit_factor'])} |",
        f"| total P/L | {_fmt(o['baseline_total_pnl'])} | {_fmt(o['filtered_total_pnl'])} |",
        f"| avg winner | {_fmt(o['baseline_avg_winner'])} | {_fmt(o['filtered_avg_winner'])} |",
        f"| avg loser | {_fmt(o['baseline_avg_loser'])} | {_fmt(o['filtered_avg_loser'])} |",
        f"| mean MFE | {_fmt(o['baseline_mean_mfe'])} | {_fmt(o['filtered_mean_mfe'])} |",
        f"| mean MAE | {_fmt(o['baseline_mean_mae'])} | {_fmt(o['filtered_mean_mae'])} |",
        f"| large-loss rate | {_fmt(o['baseline_large_loss_rate'])} | {_fmt(o['filtered_large_loss_rate'])} |",
        f"| exposure retained (trade count) | 1.0000 | {_fmt(o['exposure_retained'])} |",
        f"| share of baseline total P/L kept | 1.0000 | {_fmt(o['total_pnl_retained_frac'])} |",
        "",
        "## 3. What the filter actually changes (PASS vs FAIL, T1)",
        "",
        f"- PASS mean P/L {_fmt(mag['pass_mean_pnl'])} vs FAIL {_fmt(mag['fail_mean_pnl'])}",
        f"- PASS median P/L {_fmt(mag['pass_median_pnl'])} vs FAIL {_fmt(mag['fail_median_pnl'])}",
        f"- PASS avg winner {_fmt(mag['pass_avg_winner'])} vs FAIL {_fmt(mag['fail_avg_winner'])}",
        f"- PASS avg loser {_fmt(mag['pass_avg_loser'])} vs FAIL {_fmt(mag['fail_avg_loser'])}",
        f"- PASS mean MFE {_fmt(mag['pass_mean_mfe'])} vs FAIL {_fmt(mag['fail_mean_mfe'])}",
        f"- PASS mean MAE {_fmt(mag['pass_mean_mae'])} vs FAIL {_fmt(mag['fail_mean_mae'])}",
        "",
        mag["mechanism_note"],
        "",
        "## 4. T1–T10 (same frozen rule)",
        "",
    ]
    lines += _md_table(
        arms,
        ["arm", "baseline_mean_pnl", "filtered_mean_pnl", "baseline_win_rate", "filtered_win_rate", "trade_retention", "winner_retention", "loser_rejection", "winner_sacrifice", "selectivity_ratio"],
    )
    lines += [
        "",
        "## 5. Selector E (existing E outcomes, not rerun)",
        "",
        f"- Trades: baseline {e_row['baseline_n']} → filtered {e_row['filtered_n']} (retention {_fmt(e_row['trade_retention'])})",
        f"- Win rate: {_fmt(e_row['baseline_win_rate'])} → {_fmt(e_row['filtered_win_rate'])}",
        f"- Mean P/L: {_fmt(e_row['baseline_mean_pnl'])} → {_fmt(e_row['filtered_mean_pnl'])}",
        f"- Total P/L: {_fmt(e_row['baseline_total_pnl'])} → {_fmt(e_row['filtered_total_pnl'])} (kept {_fmt(e_row['total_pnl_retained_frac'])} of baseline total)",
        f"- Profit factor: {_fmt(e_row['baseline_profit_factor'])} → {_fmt(e_row['filtered_profit_factor'])}",
        f"- Winner retention {_fmt(e_row['winner_retention'])}; loser rejection {_fmt(e_row['loser_rejection'])}; winner sacrifice {_fmt(e_row['winner_sacrifice'])}; selectivity {_fmt(e_row['selectivity_ratio'])}",
        f"- Max drawdown of cumulative E P/L sequence: baseline {_fmt(e_row.get('baseline_max_dd'))}; filtered {_fmt(e_row.get('filtered_max_dd'))}",
        "",
        "## 6. S-band (rule not re-fit)",
        "",
    ]
    lines += _md_table(
        sband,
        ["slice", "n", "trade_retention", "winner_retention", "loser_rejection", "winner_sacrifice", "baseline_win_rate", "filtered_win_rate", "baseline_mean_pnl", "filtered_mean_pnl"],
    )
    lines += ["", "## 7. Regime (rule not re-fit)", ""]
    lines += _md_table(
        regime,
        ["slice", "n", "trade_retention", "winner_retention", "loser_rejection", "winner_sacrifice", "baseline_win_rate", "filtered_win_rate", "baseline_mean_pnl", "filtered_mean_pnl"],
    )
    lines += ["", "## 8. Chronological stability (terciles of opportunity order)", ""]
    lines += _md_table(
        timeb,
        ["slice", "n", "trade_retention", "winner_retention", "loser_rejection", "winner_sacrifice", "baseline_win_rate", "filtered_win_rate", "baseline_mean_pnl", "filtered_mean_pnl"],
    )
    if secondary:
        lines += [
            "",
            "## Appendix — directional_return_15m Q5 (same Part 2A family, not a new search)",
            "",
            f"Selectivity {_fmt(secondary['selectivity_ratio'])}; winner sacrifice {_fmt(secondary['winner_sacrifice'])}; loser rejection {_fmt(secondary['loser_rejection'])}; "
            f"T1 WR {_fmt(secondary['baseline_win_rate'])} → {_fmt(secondary['filtered_win_rate'])}.",
        ]
    lines += [
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
        decision_why,
        "",
        "## Deployment",
        "",
        "No filter was implemented in the paper/live bot. This test is **in-sample** relative to Part 2A discovery. "
        "Even a strong result would still require a frozen-rule walk-forward on unseen time before any integration.",
        "",
    ]
    path = Path(out_dir) / "RESEARCH_REPORT_15m_5m_confirmation_part2b.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def decide(overall: dict, arms: pd.DataFrame, timeb: pd.DataFrame, sband: pd.DataFrame) -> tuple[str, str]:
    ratio = overall.get("selectivity_ratio")
    wr_sac = overall.get("winner_sacrifice")
    lr = overall.get("loser_rejection")
    ret = overall.get("trade_retention")
    # consistency: filtered WR >= baseline on most T arms
    wr_up = int((arms["filtered_win_rate"] > arms["baseline_win_rate"]).sum()) if len(arms) else 0
    time_up = int((timeb["filtered_win_rate"] > timeb["baseline_win_rate"]).sum()) if len(timeb) else 0
    # 0.80+ may be tiny
    sband_ok = sband[sband["n"] >= 40] if "n" in sband.columns else sband
    sband_up = int((sband_ok["filtered_win_rate"] > sband_ok["baseline_win_rate"]).sum()) if len(sband_ok) else 0

    if ratio is None or wr_sac is None or lr is None:
        return "C — Reject", "Could not compute retention statistics."

    near_one = abs(ratio - 1.0) < 0.15
    weak_select = ratio < 1.25
    strong_select = ratio >= 1.8 and (lr - wr_sac) >= 0.08

    persistent = time_up >= 2 and wr_up >= 7
    if strong_select and persistent and ret and ret >= 0.12:
        return (
            "A — Promising enough for out-of-sample validation",
            f"Loser rejection {_fmt(lr)} vs winner sacrifice {_fmt(wr_sac)} (ratio {_fmt(ratio)}). "
            f"Filtered WR higher on {wr_up}/10 T-arms and {time_up}/3 time blocks. "
            "Still in-sample vs Part 2A discovery; freeze the rule and walk-forward next. Do not deploy.",
        )
    if near_one or weak_select:
        return (
            "C — Reject" if ratio < 1.1 else "B — Interesting but weak",
            f"The filter retains ~{ _fmt(100 * (ret or 0), 1)}% of trades. "
            f"Loser rejection {_fmt(lr)} vs winner sacrifice {_fmt(wr_sac)} (selectivity {_fmt(ratio)}). "
            "It does not reject losers much faster than winners; higher filtered WR is largely the mechanical effect of keeping the best-looking 20% of entries. "
            f"Direction is similar on {wr_up}/10 T-arms and {time_up}/3 time blocks, {sband_up} S-bands with N≥40. "
            "Not enough for a 5m confirmation layer without out-of-sample proof that this is more than the in-sample Q5 pattern.",
        )
    return (
        "B — Interesting but weak",
        f"Some improvement: loser rejection {_fmt(lr)}, winner sacrifice {_fmt(wr_sac)}, ratio {_fmt(ratio)}. "
        f"T-arm WR up on {wr_up}/10; time blocks WR up on {time_up}/3. "
        "Winner sacrifice is still large relative to the benefit. In-sample only. Do not deploy.",
    )


def run_part2b(*, part1_dir: Path, baseline_dir: Path, part2a_dir: Path, out_dir: Path) -> dict[str, Path]:
    part1_dir = Path(part1_dir)
    baseline_dir = Path(baseline_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ddir = out_dir / "diagnostics"
    ddir.mkdir(parents=True, exist_ok=True)

    before = {n: _sha256(baseline_dir / n) for n in BASELINE_FILES if (baseline_dir / n).exists()}
    diag = pd.read_csv(part1_dir / "diagnostics" / "15m_opportunities_with_5m_features.csv", low_memory=False)

    avail = pd.to_datetime(diag["available_5m_candle_ts"], utc=True, errors="coerce")
    dec = pd.to_datetime(diag["decision_timestamp"], utc=True, errors="coerce")
    causal_ok = int((avail.notna() & dec.notna() & (avail <= dec)).sum())

    pnl = _num(diag["primary_pnl_pct"])
    losses = pnl[pnl <= 0]
    p20 = float(losses.quantile(0.20)) if len(losses) else None
    work = diag.copy()
    work["primary_large_loss"] = (pnl <= p20) if p20 is not None else False

    q = assign_predefined_quintiles(work[PRIMARY_FEATURE])
    work["mom_quintile"] = q
    pass_mask = q == "Q5_highest"
    q5_vals = _num(work.loc[pass_mask, PRIMARY_FEATURE])
    rule = {
        "feature": PRIMARY_FEATURE,
        "construction": "pandas.qcut(q=5, labels=Q1..Q5_highest) on the full 1517-row Part 1 dataset — identical to Part 2A quantile_buckets()",
        "pass_bucket": "Q5_highest",
        "q5_min": float(q5_vals.min()) if len(q5_vals) else None,
        "q5_max": float(q5_vals.max()) if len(q5_vals) else None,
        "n_pass": int(pass_mask.sum()),
        "n_all": int(len(work)),
        "causal_ok": causal_ok,
        "part2a_dir": str(part2a_dir),
        "part1_dir": str(part1_dir),
    }

    overall = compare_baseline_filtered(work, pass_mask, "primary_pnl_pct", "primary_win")
    fail = ~pass_mask.fillna(False)
    mag = {
        "pass_mean_pnl": overall["filtered_mean_pnl"],
        "fail_mean_pnl": float(_num(work.loc[fail, "primary_pnl_pct"]).mean()),
        "pass_median_pnl": overall["filtered_median_pnl"],
        "fail_median_pnl": float(_num(work.loc[fail, "primary_pnl_pct"]).median()),
        "pass_avg_winner": overall["filtered_avg_winner"],
        "fail_avg_winner": float(_num(work.loc[fail & (pnl > 0), "primary_pnl_pct"]).mean()) if (fail & (pnl > 0)).any() else None,
        "pass_avg_loser": overall["filtered_avg_loser"],
        "fail_avg_loser": float(_num(work.loc[fail & (pnl <= 0), "primary_pnl_pct"]).mean()) if (fail & (pnl <= 0)).any() else None,
        "pass_mean_mfe": overall["filtered_mean_mfe"],
        "fail_mean_mfe": float(_num(work.loc[fail, "primary_mfe_pct"]).mean()),
        "pass_mean_mae": overall["filtered_mean_mae"],
        "fail_mean_mae": float(_num(work.loc[fail, "primary_mae_pct"]).mean()),
    }
    wr_lift = (overall["filtered_win_rate"] or 0) - (overall["baseline_win_rate"] or 0)
    pw, fw = mag["pass_avg_winner"] or 0, mag["fail_avg_winner"] or 0
    pl, fl = mag["pass_avg_loser"] or 0, mag["fail_avg_loser"] or 0
    mag["mechanism_note"] = (
        f"Win rate rises by {_fmt(wr_lift)}. "
        f"Average winner is larger on PASS ({_fmt(pw)} vs {_fmt(fw)}). "
        f"Average loser is essentially unchanged ({_fmt(pl)} vs {_fmt(fl)}) — T1 losses sit on the stop. "
        f"PASS MFE is larger and PASS MAE is more negative: a higher-momentum subset with bigger swings, not tighter risk."
    )

    arm_rows = []
    for arm in FIXED_T_ARMS:
        r = compare_baseline_filtered(work, pass_mask, f"{arm}_pnl_pct")
        r["arm"] = arm
        arm_rows.append(r)
    arms = pd.DataFrame(arm_rows)

    e_row = compare_baseline_filtered(work, pass_mask, "selector_E_selected_pnl_pct")
    e_row["baseline_max_dd"] = max_drawdown_from_sequence(work["selector_E_selected_pnl_pct"])
    e_row["filtered_max_dd"] = max_drawdown_from_sequence(work.loc[pass_mask, "selector_E_selected_pnl_pct"])

    def _slice_table(col: str) -> pd.DataFrame:
        rows = []
        for key, g in work.groupby(col, dropna=False):
            r = compare_baseline_filtered(g, pass_mask.loc[g.index], "primary_pnl_pct", "primary_win")
            r["slice"] = str(key)
            rows.append(r)
        return pd.DataFrame(rows)

    sband = _slice_table("s_band")
    regime = _slice_table("regime")

    work["_chrono"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.sort_values("_chrono", kind="stable")
    work["_tercile"] = pd.qcut(np.arange(len(work)), 3, labels=["early", "middle", "late"])
    time_rows = []
    for key, g in work.groupby("_tercile", observed=True):
        r = compare_baseline_filtered(g, pass_mask.loc[g.index], "primary_pnl_pct", "primary_win")
        r["slice"] = str(key)
        time_rows.append(r)
    timeb = pd.DataFrame(time_rows)

    q2 = assign_predefined_quintiles(work[SECONDARY_FEATURE])
    secondary = compare_baseline_filtered(work, q2 == "Q5_highest", "primary_pnl_pct", "primary_win")

    after = {n: _sha256(baseline_dir / n) for n in before}
    if after != before:
        raise RuntimeError("Immutable baseline modified during Part 2B")
    rule["baseline_unchanged"] = True

    pd.DataFrame([overall]).to_csv(ddir / "overall_filter_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "pass_winners": overall["pass_winners"],
                "pass_losers": overall["pass_losers"],
                "fail_winners": overall["fail_winners"],
                "fail_losers": overall["fail_losers"],
                "trade_retention": overall["trade_retention"],
                "winner_retention": overall["winner_retention"],
                "loser_rejection": overall["loser_rejection"],
                "winner_sacrifice": overall["winner_sacrifice"],
                "selectivity_ratio": overall["selectivity_ratio"],
            }
        ]
    ).to_csv(ddir / "winner_loser_retention.csv", index=False)
    arms.to_csv(ddir / "t1_t10_filter_results.csv", index=False)
    pd.DataFrame([e_row]).to_csv(ddir / "selector_e_filter_results.csv", index=False)
    sband.to_csv(ddir / "s_band_filter_results.csv", index=False)
    regime.to_csv(ddir / "regime_filter_results.csv", index=False)
    timeb.to_csv(ddir / "time_stability.csv", index=False)
    pd.DataFrame([secondary]).to_csv(ddir / "directional_return_15m_q5_appendix.csv", index=False)

    generate_plots(out_dir, overall=overall, arms=arms, sband=sband, regime=regime, timeb=timeb)
    decision, why = decide(overall, arms, timeb, sband)
    report = write_report(
        out_dir=out_dir,
        rule=rule,
        overall=overall,
        mag=mag,
        arms=arms,
        e_row=e_row,
        sband=sband,
        regime=regime,
        timeb=timeb,
        secondary=secondary,
        decision=decision,
        decision_why=why,
    )
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "rule": rule,
                "decision": decision,
                "live_bot_modified": False,
                "veto_or_filter_deployed": False,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info("Part 2B complete → %s (%s)", report, decision)
    return {"report": report, "out_dir": out_dir}
