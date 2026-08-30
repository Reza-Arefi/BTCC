"""A/B/C weight-arm comparison: Static vs Equal vs Adaptive.

Fairness: shared candles, universe, BTC.D, factor scores, threshold, execution.
Only the weighting methodology differs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from btcc.sim.backtest import run_adaptive_sim_backtest
from btcc.sim.config import load_sim_config
from btcc.sim.score import FACTOR_KEYS, equal_factor_weights, static_factor_weights

logger = logging.getLogger(__name__)

ARM_STATIC = "static"
ARM_EQUAL = "equal"
ARM_ADAPTIVE = "adaptive"
ARM_ORDER = (ARM_STATIC, ARM_EQUAL, ARM_ADAPTIVE)
ARM_LABELS = {
    ARM_STATIC: "Static (config factors.weights)",
    ARM_EQUAL: "Equal (1/N)",
    ARM_ADAPTIVE: "Adaptive (90d init + daily rolling 90d)",
}


def run_abc_comparison(
    *,
    days: int = 365,
    force_download: bool = False,
    long_threshold: float = 0.60,
    out_root: Path | None = None,
    signal_cfg: dict[str, Any] | None = None,
    sim_cfg: dict[str, Any] | None = None,
    init_days: int | None = None,
    roll_days: int | None = None,
) -> Path:
    """Run arms A/B/C with identical shared inputs; write comparison report.

    Does NOT force identical trade opportunities — signals may differ by design.
    """
    from btcc.backtest.config import load_backtest_config
    from btcc.config import load_config

    sim = dict(sim_cfg or load_sim_config())
    sim["long_threshold"] = float(long_threshold)
    wf = dict(sim.get("walk_forward") or {})
    if init_days is not None:
        wf["init_days"] = int(init_days)
    if roll_days is not None:
        wf["daily_rolling_window_days"] = int(roll_days)
    sim["walk_forward"] = wf

    cfg = signal_cfg or load_config()
    # Ensure backtest-compatible merge fields exist
    try:
        bt = load_backtest_config()
        for k in ("backtest", "backtest_data", "backtest_output", "_root"):
            if k in bt and k not in cfg:
                cfg[k] = bt[k]
        if "_root" not in cfg:
            cfg["_root"] = bt.get("_root")
    except Exception:
        pass

    root = Path(cfg.get("_root") or Path(__file__).resolve().parents[2])
    out_root = Path(out_root) if out_root else root / "results"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    thr_tag = f"{float(long_threshold):.2f}".replace(".", "p")
    cmp_dir = out_root / f"abc_compare_{days}d_thr{thr_tag}_{run_id}"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    static_w = static_factor_weights(cfg)
    equal_w = equal_factor_weights()
    (cmp_dir / "arm_weights.json").write_text(
        json.dumps(
            {
                "static": {
                    "source": "configs/signal_config.yaml → factors.weights",
                    "reason": "Pre-Adaptive V2 configured indicator weights; fixed for entire run",
                    "weights": static_w,
                },
                "equal": {
                    "source": "1/N over FACTOR_KEYS",
                    "n": len(FACTOR_KEYS),
                    "rule_missing_indicator": (
                        "Factor still listed at 1/N; signed score treated as 0 (neutral) "
                        "in combined_score — same N for all bars"
                    ),
                    "weights": equal_w,
                },
                "adaptive": {
                    "source": "90d init then daily rolling 90d (matured outcomes only)",
                    "seed_pre_init": static_w,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Shared factor cache: first arm fills, later arms reuse → identical factor inputs
    factor_cache: dict[tuple[str, str], dict[str, float]] = {}
    arm_dirs: dict[str, Path] = {}
    arm_summaries: dict[str, Any] = {}

    first = True
    for arm in ARM_ORDER:
        logger.info("=== ABC arm %s (days=%d thr=%.2f) ===", arm, days, long_threshold)
        out = run_adaptive_sim_backtest(
            signal_cfg=cfg,
            days=days,
            sim_cfg=sim,
            force_download=force_download and first,
            long_threshold=float(long_threshold),
            out_root=cmp_dir,
            weight_mode=arm,
            factor_cache=factor_cache,
            run_tag=arm,
        )
        first = False
        arm_dirs[arm] = out
        arm_summaries[arm] = json.loads((out / "summary.json").read_text(encoding="utf-8"))

    comparison = _build_comparison(arm_summaries, arm_dirs, static_w, equal_w, sim)
    (cmp_dir / "abc_comparison.json").write_text(
        json.dumps(comparison, indent=2, default=str), encoding="utf-8"
    )
    (cmp_dir / "abc_comparison.md").write_text(
        _comparison_md(comparison, days, long_threshold), encoding="utf-8"
    )
    # Flat strategy grid CSV
    rows = []
    for arm in ARM_ORDER:
        for sk, st in (arm_summaries[arm].get("strategies") or {}).items():
            rows.append({
                "arm": arm,
                "strategy": sk,
                "opportunities": arm_summaries[arm].get("n_trades_opened"),
                "n_signals": arm_summaries[arm].get("n_signals"),
                "n_trades": st.get("n_trades"),
                "winning_trades": st.get("winning_trades"),
                "losing_trades": st.get("losing_trades"),
                "win_rate": st.get("win_rate"),
                "average_return_pct": st.get("average_return_pct"),
                "net_btc_pnl": st.get("sum_pnl_btc"),
                "cumulative_btc": st.get("cumulative_btc"),
                "usd_equiv_pnl": st.get("sum_pnl_usd_equiv"),
                "max_drawdown_btc": st.get("max_drawdown_btc"),
                "profit_factor": st.get("profit_factor"),
                "avg_holding_hours": st.get("mean_holding_hours"),
                "max_simultaneous": arm_summaries[arm].get("max_simultaneous_opportunities"),
                "max_open_rejects": arm_summaries[arm].get("n_rejected_max_open_trades"),
            })
    pd.DataFrame(rows).to_csv(cmp_dir / "abc_strategy_grid.csv", index=False)
    logger.info("ABC comparison complete → %s", cmp_dir)

    # Shared analytics / plots (Telegram DISABLED for backtest)
    try:
        from btcc.analytics.pipeline import build_abc_analytics

        analytics_root = build_abc_analytics(cmp_dir, telegram_enabled=False)
        logger.info("ABC analytics → %s", analytics_root)
    except Exception as e:
        logger.exception("ABC analytics failed: %s", e)

    return cmp_dir


def _build_comparison(
    summaries: dict[str, Any],
    arm_dirs: dict[str, Path],
    static_w: dict[str, float],
    equal_w: dict[str, float],
    sim: dict[str, Any],
) -> dict[str, Any]:
    grid: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        grid[arm] = summaries[arm].get("strategies") or {}

    def _btc(arm: str, sk: str) -> float | None:
        st = (grid.get(arm) or {}).get(sk) or {}
        v = st.get("sum_pnl_btc")
        return float(v) if v is not None else None

    strategies = sorted({sk for g in grid.values() for sk in g.keys()})
    q: dict[str, Any] = {}
    for sk in strategies:
        a, b, c = _btc(ARM_STATIC, sk), _btc(ARM_EQUAL, sk), _btc(ARM_ADAPTIVE, sk)
        q[sk] = {
            "Q1_adaptive_vs_static": None if a is None or c is None else (c > a),
            "Q2_adaptive_vs_equal": None if b is None or c is None else (c > b),
            "Q3_static_vs_equal": None if a is None or b is None else (a > b),
            "static_sum_btc": a,
            "equal_sum_btc": b,
            "adaptive_sum_btc": c,
        }

    # Prediction accuracy across arms
    acc = {
        arm: summaries[arm].get("prediction_accuracy_sign")
        or summaries[arm].get("signal_hit_rate_4h")
        for arm in ARM_ORDER
    }
    q_pred = {
        "accuracy_by_arm": acc,
        "Q4_adaptive_improves_prediction": (
            None
            if acc[ARM_ADAPTIVE] is None or acc[ARM_STATIC] is None
            else acc[ARM_ADAPTIVE] > acc[ARM_STATIC]
        ),
    }

    # Weight stability for adaptive
    adaptive_dir = arm_dirs[ARM_ADAPTIVE]
    wh_path = adaptive_dir / "weight_history.csv"
    weight_stability: dict[str, Any] = {"n_daily_updates": summaries[ARM_ADAPTIVE].get("n_daily_updates")}
    if wh_path.exists():
        wh = pd.read_csv(wh_path)
        if not wh.empty and "new_weight" in wh.columns:
            piv = wh.pivot_table(index="update_id", columns="indicator", values="new_weight", aggfunc="last")
            if len(piv) >= 2:
                diffs = piv.diff().abs().dropna(how="all")
                weight_stability["mean_abs_weight_change_per_update"] = float(diffs.mean().mean())
                weight_stability["max_abs_weight_change"] = float(diffs.max().max())

    return {
        "threshold": float(sim.get("long_threshold", 0.60)),
        "fairness": {
            "shared_candles": True,
            "shared_universe": True,
            "shared_btc_d": True,
            "shared_factor_scores_via_cache": True,
            "shared_threshold": 0.60,
            "shared_execution": True,
            "only_difference": "weighting_methodology",
            "opportunities_may_differ": True,
        },
        "fixed_weights": {"static": static_w, "equal": equal_w},
        "arms": {
            arm: {
                "label": ARM_LABELS[arm],
                "out_dir": str(arm_dirs[arm]),
                "n_signals": summaries[arm].get("n_signals"),
                "n_opportunities": summaries[arm].get("n_trades_opened"),
                "n_predictions": summaries[arm].get("n_predictions"),
                "n_weight_updates": summaries[arm].get("n_weight_updates"),
                "n_daily_updates": summaries[arm].get("n_daily_updates"),
                "weight_mode": summaries[arm].get("weight_mode"),
                "prediction_accuracy_sign": summaries[arm].get("prediction_accuracy_sign"),
                "signal_hit_rate_4h": summaries[arm].get("signal_hit_rate_4h"),
                "strategies": summaries[arm].get("strategies"),
            }
            for arm in ARM_ORDER
        },
        "strategy_grid": grid,
        "questions": {
            "by_strategy_btc_pnl": q,
            "prediction": q_pred,
            "Q6_answer_differs_by_strategy": len({
                (q[sk]["Q1_adaptive_vs_static"], q[sk]["Q2_adaptive_vs_equal"])
                for sk in q
            }) > 1 if q else None,
            "Q8_weight_stability": weight_stability,
            "Q9_n_daily_updates": summaries[ARM_ADAPTIVE].get("n_daily_updates"),
            "Q10_early_vs_late": "requires monthly slice analysis on full-year run",
            "Q7_monthly_consistency": "requires monthly slice analysis on full-year run",
        },
    }


def _comparison_md(cmp: dict[str, Any], days: int, thr: float) -> str:
    lines = [
        "# A/B/C Weight Arm Comparison",
        "",
        f"- Days: {days} | Threshold: {thr} (fixed for all arms)",
        "- Arms: **Static** | **Equal** | **Adaptive**",
        "- Only intentional difference: weighting methodology",
        "",
        "## Strategy grid (net BTC PnL)",
        "",
        "| Arm | " + " | ".join(sorted({
            sk for arm in ARM_ORDER for sk in (cmp["arms"][arm].get("strategies") or {})
        })) + " |",
        "|-----|" + "|".join(["-----"] * max(1, len((cmp["arms"][ARM_STATIC].get("strategies") or {})))) + "|",
    ]
    strategies = sorted({
        sk for arm in ARM_ORDER for sk in (cmp["arms"][arm].get("strategies") or {})
    })
    if strategies:
        lines[6] = "| Arm | " + " | ".join(strategies) + " |"
        lines[7] = "|-----|" + "|".join(["-----"] * len(strategies)) + "|"
        for arm in ARM_ORDER:
            cells = []
            for sk in strategies:
                st = (cmp["arms"][arm].get("strategies") or {}).get(sk) or {}
                cells.append(f"{st.get('sum_pnl_btc')}")
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "## Signals vs opportunities", ""]
    for arm in ARM_ORDER:
        a = cmp["arms"][arm]
        lines.append(
            f"- **{arm}**: signals={a.get('n_signals')} opportunities={a.get('n_opportunities')} "
            f"updates={a.get('n_weight_updates')}"
        )
    lines += ["", "## Research questions (preliminary)", ""]
    for sk, qq in (cmp.get("questions") or {}).get("by_strategy_btc_pnl", {}).items():
        lines.append(
            f"- **{sk}**: Adaptive>Static={qq.get('Q1_adaptive_vs_static')} "
            f"Adaptive>Equal={qq.get('Q2_adaptive_vs_equal')} "
            f"Static>Equal={qq.get('Q3_static_vs_equal')}"
        )
    lines += [
        "",
        f"- Q4 prediction: {cmp['questions']['prediction']}",
        f"- Q9 daily updates: {cmp['questions']['Q9_n_daily_updates']}",
        f"- Q7/Q10: {cmp['questions']['Q7_monthly_consistency']}",
        "",
    ]
    return "\n".join(lines)
