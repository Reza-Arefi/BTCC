"""Emit organized analytics plots into stable subdirectories.

Layout (under plots/ or plots/<arm>/):
  primary/      — capital, P/L%, drawdown, win rate, S1–S5 comparison
  adaptive/     — weights, learning (Adaptive only)
  prediction/   — score vs return, correlations, buckets
  trades/       — wins/losses, expectancy, profit factor, recovered
  risk/         — concurrent opportunities, max-open
  contextual/   — BTC.D (analysis only)
  comparisons/  — Normal vs Late, Static/Equal/Adaptive overlays
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from btcc.analytics import metrics as M
from btcc.analytics import plots as P
from btcc.analytics.capital import (
    btc_equity_daily,
    capital_daily_series,
    capital_drawdown_series,
    daily_btc_price_map,
)

PLOT_SUBDIRS = (
    "primary",
    "adaptive",
    "prediction",
    "trades",
    "risk",
    "contextual",
    "comparisons",
)

STRATEGY_KEYS = M.STRATEGY_KEYS


def ensure_plot_layout(plots_root: Path) -> dict[str, Path]:
    out = {"root": plots_root}
    plots_root.mkdir(parents=True, exist_ok=True)
    for name in PLOT_SUBDIRS:
        p = plots_root / name
        p.mkdir(parents=True, exist_ok=True)
        out[name] = p
    return out


def emit_arm_plots(
    *,
    arm: str,
    plots_root: Path,
    pred: pd.DataFrame,
    opp: pd.DataFrame,
    legs: pd.DataFrame,
    wh_ts: pd.DataFrame,
    cum: pd.DataFrame,
    wr: pd.DataFrame,
    corr: pd.DataFrame,
    buckets: pd.DataFrame,
    svr: pd.DataFrame,
    wl: pd.DataFrame,
    dd: pd.DataFrame,
    sim: pd.DataFrame,
    rej: pd.DataFrame,
    btcd: pd.DataFrame,
    learn: pd.DataFrame | None = None,
    wvu: pd.DataFrame | None = None,
    init_days: int | None = 90,
    max_open: int = 10,
    starting_capital_usd: float = 1000.0,
    max_day: int | None = None,
    lite: bool = False,
) -> dict[str, Any]:
    """Write arm plots into organized subdirs.

    lite=True → primary risk/dashboard plots only (for daily checkpoints).
    lite=False → full research suite (end of arm / ABC).
    """
    dirs = ensure_plot_layout(plots_root)
    written: list[str] = []

    # --- Capital / compounding (PRIMARY) ---
    cap = capital_daily_series(
        legs, starting_capital_usd=starting_capital_usd, max_day=max_day,
    )
    dd_cap = capital_drawdown_series(cap, starting_capital_usd=starting_capital_usd)
    btc_eq = btc_equity_daily(
        legs,
        max_day=max_day,
        starting_capital_usd=starting_capital_usd,
        btc_price_by_day=daily_btc_price_map(pred),
    )
    exp = M.expectancy_summary(legs)
    funnel = M.opportunity_funnel(pred, opp, legs=legs)
    rej_clean = M.rejection_without_btcd(pred)
    btcd_days = M.btcd_over_days(pred)

    for pol in sorted(cap["entry_policy"].dropna().unique()) if not cap.empty else []:
        sub = cap[cap["entry_policy"] == pol]
        p = P.plot_s1_s5_portfolio(
            sub, dirs["primary"],
            title=f"Cumulative Return (%) from $1,000 Start — {arm} / {pol}",
            filename=f"portfolio_1000_{arm}_{pol}.png",
            starting_capital_usd=starting_capital_usd,
            init_days=init_days,
        )
        if p:
            written.append(str(p))
        p = P.plot_s1_s5_cumulative_pl(
            sub, dirs["primary"],
            title=f"Cumulative P/L (%) — {arm} / {pol}",
            filename=f"cumulative_pl_pct_{arm}_{pol}.png",
            init_days=init_days,
        )
        if p:
            written.append(str(p))
        dd_sub = dd_cap[dd_cap["entry_policy"] == pol] if not dd_cap.empty else dd_cap
        p = P.plot_s1_s5_drawdown_pct(
            dd_sub, dirs["primary"],
            title=f"Portfolio Drawdown (%) — {arm} / {pol}",
            filename=f"drawdown_pct_{arm}_{pol}.png",
            init_days=init_days,
        )
        if p:
            written.append(str(p))
        # Trading P/L % uses the same $1,000-normalized return series (per strategy).
        p = P.plot_s1_s5_btc_pnl(
            sub, dirs["primary"],
            title=f"Cumulative P/L (%) from Trading — {arm} / {pol}",
            filename=f"btc_pnl_from_trading_{arm}_{pol}.png",
            init_days=init_days,
            starting_capital_usd=starting_capital_usd,
        )
        if p:
            written.append(str(p))
        be = btc_eq[btc_eq["entry_policy"] == pol] if not btc_eq.empty else btc_eq
        p = P.plot_s1_s5_btc_total_equiv(
            be, dirs["primary"],
            title=f"BTC-Equivalent Cumulative Return (%) — {arm} / {pol}",
            filename=f"btc_total_equiv_{arm}_{pol}.png",
            init_days=init_days,
        )
        if p:
            written.append(str(p))

    p = P.plot_rolling_win_rate({arm: wr}, dirs["primary"], init_days=init_days)
    written.extend(str(x) for x in (p or []))
    p = P.plot_cumulative_win_rate({arm: wr}, dirs["primary"], init_days=init_days)
    written.extend(str(x) for x in (p or []))
    p = P.plot_simultaneous(sim, dirs["risk"], max_open=max_open, init_days=init_days)
    if p:
        written.append(str(p))
    if arm == "adaptive":
        p = P.plot_adaptive_weights(wh_ts, dirs["adaptive"], init_days=init_days)
        if p:
            written.append(str(p))
    p = P.plot_btcd_over_days(btcd_days, dirs["contextual"], init_days=init_days)
    if p:
        written.append(str(p))
    p = P.plot_funnel(funnel, dirs["risk"], arm=arm)
    if p:
        written.append(str(p))
    p = P.plot_win_loss_counts(wl if not wl.empty else M.win_loss_summary(legs), dirs["trades"], arm=arm)
    if p:
        written.append(str(p))
    if not exp.empty:
        p = P.plot_expectancy_bars(exp, dirs["trades"], arm=arm)
        if p:
            written.append(str(p))

    if lite:
        return {
            "arm": arm,
            "lite": True,
            "n_plots": len(written),
            "plot_dirs": {k: str(v) for k, v in dirs.items() if k != "root"},
            "capital_daily_rows": int(len(cap)),
            "paths": written[:50],
        }

    # --- Full research suite ---
    exp_day = M.expectancy_by_day(legs)
    if not exp.empty:
        p = P.plot_profit_factor_bars(exp, dirs["trades"], arm=arm)
        if p:
            written.append(str(p))
    if not exp_day.empty:
        for pol in sorted(exp_day["entry_policy"].unique()):
            p = P.plot_expectancy_over_days(
                exp_day[exp_day["entry_policy"] == pol],
                dirs["trades"],
                filename=f"expectancy_over_days_{arm}_{pol}.png",
                init_days=init_days,
            )
            if p:
                written.append(str(p))

    p = P.plot_win_loss_bars({arm: wl}, dirs["trades"])
    if p:
        written.append(str(p))
    p = P.plot_entry_policy_btc(legs, dirs["comparisons"])
    written.extend(str(x) for x in (p or []))
    p = P.plot_entry_policy_win_rate(legs, dirs["comparisons"])
    if p:
        written.append(str(p))
    p = P.plot_normal_vs_late_summary(legs, dirs["comparisons"], arm=arm)
    if p:
        written.append(str(p))
    p = P.plot_recovered_trades(legs, dirs["trades"])
    written.extend(str(x) for x in (p or []))

    if arm == "adaptive":
        if wvu is not None and not wvu.empty:
            p = P.plot_weight_vs_usefulness(wvu, dirs["adaptive"])
            if p:
                written.append(str(p))
        if learn is not None and not learn.empty:
            p = P.plot_learning_progress(learn, dirs["adaptive"])
            if p:
                written.append(str(p))

    p = P.plot_indicator_correlations(corr, dirs["prediction"])
    if p:
        written.append(str(p))
    p = P.plot_score_vs_return(svr, dirs["prediction"])
    written.extend(str(x) for x in (p or []))
    p = P.plot_accuracy_buckets(buckets, dirs["prediction"])
    if p:
        written.append(str(p))
    p = P.plot_entry_classification_scores(pred, dirs["prediction"])
    if p:
        written.append(str(p))
    p = P.plot_entry_future_returns(pred, dirs["prediction"])
    if p:
        written.append(str(p))

    p = P.plot_rejections(rej_clean if not rej_clean.empty else rej, dirs["risk"])
    if p:
        written.append(str(p))
    p = P.plot_drawdown({arm: dd}, dirs["risk"], init_days=init_days)
    written.extend(str(x) for x in (p or []))
    p = P.plot_btc_accumulation_by_strategy({arm: cum}, dirs["primary"], init_days=init_days)
    written.extend(str(x) for x in (p or []))
    p = P.plot_btcd_regimes(btcd, dirs["contextual"])
    if p:
        written.append(str(p))

    return {
        "arm": arm,
        "lite": False,
        "n_plots": len(written),
        "plot_dirs": {k: str(v) for k, v in dirs.items() if k != "root"},
        "capital_daily_rows": int(len(cap)),
        "paths": written[:50],
    }


def emit_abc_comparison_plots(
    *,
    plots_root: Path,
    cap_by_arm_policy: dict[str, pd.DataFrame],
    cum_by_arm: dict[str, pd.DataFrame],
    wr_by_arm: dict[str, pd.DataFrame],
    dd_by_arm: dict[str, pd.DataFrame],
    wl_by_arm: dict[str, pd.DataFrame],
    init_days: int | None = 90,
    starting_capital_usd: float = 1000.0,
) -> None:
    dirs = ensure_plot_layout(plots_root)
    # Cross-arm compounded overlays per strategy (labels like static/NORMAL_FILTERED)
    if cap_by_arm_policy:
        for sk in STRATEGY_KEYS:
            P.plot_compounded_capital(
                cap_by_arm_policy, dirs["comparisons"], strategy_key=sk,
                starting_capital_usd=starting_capital_usd, init_days=init_days,
            )
            P.plot_cumulative_pl_pct(
                cap_by_arm_policy, dirs["comparisons"], strategy_key=sk,
                starting_capital_usd=starting_capital_usd, init_days=init_days,
            )
        # Per-arm S1–S5 already in arm folders; also Normal vs Late for Adaptive if present
        for arm in ("static", "equal", "adaptive"):
            for pol in ("NORMAL_FILTERED", "LATE_ENTRY_ALLOWED"):
                key = f"{arm}/{pol}"
                if key not in cap_by_arm_policy:
                    continue
                P.plot_s1_s5_portfolio(
                    cap_by_arm_policy[key], dirs["primary"],
                    title=f"Cumulative Return (%) from $1,000 — {arm} / {pol}",
                    filename=f"portfolio_1000_{arm}_{pol}.png",
                    starting_capital_usd=starting_capital_usd,
                    init_days=init_days,
                )
    P.plot_btc_accumulation_by_strategy(cum_by_arm, dirs["comparisons"], init_days=init_days)
    P.plot_rolling_win_rate(wr_by_arm, dirs["comparisons"], init_days=init_days)
    P.plot_drawdown(dd_by_arm, dirs["comparisons"], init_days=init_days)
    P.plot_win_loss_bars(wl_by_arm, dirs["comparisons"])
    for sk in STRATEGY_KEYS:
        adv = M.adaptive_advantage_series(cum_by_arm, sk)
        if not adv.empty:
            P.plot_adaptive_advantage(adv, dirs["comparisons"], sk)
