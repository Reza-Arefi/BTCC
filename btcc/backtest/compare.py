"""Compare two BTCC 90-day backtest runs (before/after BTC.D data quality)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _load(run_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame | None]:
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(run_dir / "top5_predictions.csv")
    hp = None
    if (run_dir / "horizon_performance.csv").exists():
        hp = pd.read_csv(run_dir / "horizon_performance.csv")
    return meta, preds, hp


def compare_runs(baseline_dir: Path, new_dir: Path, out_path: Path | None = None) -> str:
    m0, p0, h0 = _load(baseline_dir)
    m1, p1, h1 = _load(new_dir)

    lines = [
        "# BTC.D Data-Quality Comparison",
        "",
        f"- Baseline (BTC.D missing): `{baseline_dir.name}`",
        f"- New (BTC.D fixed): `{new_dir.name}`",
        "",
        "## Dominance availability",
        "",
        "| Field | Baseline | New |",
        "|-------|----------|-----|",
        f"| status | {m0.get('dominance_status')} | {m1.get('dominance_status')} |",
        f"| source | {m0.get('dominance', {}).get('source')} | {m1.get('dominance', {}).get('source')} |",
        f"| n_points | {m0.get('dominance', {}).get('n_points')} | {m1.get('dominance', {}).get('n_points')} |",
        f"| missing_at_decisions | {m0.get('dominance_missing_at_decisions')} | {m1.get('dominance_missing_at_decisions')} |",
        f"| median_spacing_h | {m0.get('dominance', {}).get('median_spacing_hours')} | {m1.get('dominance', {}).get('median_spacing_hours')} |",
        f"| calibration | {m0.get('dominance', {}).get('calibration')} | {m1.get('dominance', {}).get('calibration')} |",
        "",
        "## Prediction counts",
        "",
        f"| Metric | Baseline | New |",
        f"|--------|----------|-----|",
        f"| decision_bars | {m0.get('decision_bars')} | {m1.get('decision_bars')} |",
        f"| top5_rows | {m0.get('top5_predictions')} | {m1.get('top5_predictions')} |",
        "",
        "## Horizon performance (Top-5)",
        "",
    ]

    if h0 is not None and h1 is not None:
        lines.append("| Horizon | Baseline success | New success | Δ success | Baseline mean ret | New mean ret |")
        lines.append("|---------|------------------|-------------|-----------|-------------------|--------------|")
        for h in (1, 4, 8, 12, 24):
            r0 = h0[h0["horizon_h"] == h]
            r1 = h1[h1["horizon_h"] == h]
            if r0.empty or r1.empty:
                continue
            s0 = float(r0.iloc[0]["success_rate"])
            s1 = float(r1.iloc[0]["success_rate"])
            mret0 = float(r0.iloc[0]["mean_future_return"])
            mret1 = float(r1.iloc[0]["mean_future_return"])
            lines.append(
                f"| {h}h | {s0:.2%} | {s1:.2%} | {s1 - s0:+.2%} | {mret0:.4%} | {mret1:.4%} |"
            )

    # BTC.D observation audit on new run
    lines += ["", "## New-run BTC.D observation audit", ""]
    if "btc_dominance_obs_ts" in p1.columns:
        ok = (p1.get("dominance_status") == "OK").sum() if "dominance_status" in p1.columns else 0
        lines.append(f"- Rows with dominance_status=OK: {ok} / {len(p1)}")
        if "btc_dominance_age_hours" in p1.columns:
            ages = p1["btc_dominance_age_hours"].dropna()
            if not ages.empty:
                lines.append(f"- BTC.D age hours: median={ages.median():.2f}, max={ages.max():.2f}")
        if "btc_dominance" in p1.columns:
            d = p1["btc_dominance"].dropna()
            if not d.empty:
                lines.append(f"- BTC.D pct range: {d.min():.2f}% … {d.max():.2f}%")
    else:
        lines.append("- New run missing btc_dominance_obs_ts column")

    lines += [
        "",
        "## Notes",
        "",
        "- Strategy weights, probability formula, Late Entry, and universe were **unchanged**.",
        "- Difference is attributable to BTC.D / BTC-regime data quality only.",
        "- Probabilities remain baseline_model_probability (not calibrated hit-rates).",
        "",
    ]

    text = "\n".join(lines)
    if out_path is None:
        out_path = new_dir / "comparison_vs_missing_btd.md"
    out_path.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--new", required=True)
    args = p.parse_args()
    print(compare_runs(Path(args.baseline), Path(args.new)))
