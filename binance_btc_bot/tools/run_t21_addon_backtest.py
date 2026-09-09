"""Addon exit-only backtest: T21 on the same entries as the T1–T20 experiment.

T21 (user candidate):
  Hard SL: -1.00%
  Activation: +0.75%
  Trail: 0.75% until peak >= +1.50%, then 0.25%

Does not touch live bot / LIVE config.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CANDLE_DIR = REPO / "data" / "backtest_candles_binance"
PRIOR = REPO / "results" / "t_strategy_21d_1m_20260909_001724"
STARTING_BTC = 0.00783355
ALLOC_FRAC = 0.125

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("t21_addon")


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def t21_spec():
    from btcc.sim.exits import StrategySpec

    return StrategySpec(
        key="trail_21",
        name="T21",
        stop_loss_pct=0.01,
        take_profit_pct=None,
        trail_activation_pct=0.0075,
        trail_distance_pct=0.0075,
        adaptive_mode="two_stage",
        adaptive_cfg={
            "trail_initial": 0.0075,
            "trail_at_1_5pct": 0.0025,
            "trail_at_2pct": 0.0025,  # stay tight after +1.5%
        },
    )


def load_exit_panels(bases: list[str]) -> dict[str, Any]:
    from btcc.data.candles import candle_path, load_candles
    from btcc.series.relative import build_alt_btc

    btc = load_candles(candle_path(CANDLE_DIR, "BTCUSDT", "1m"))
    if btc is None or btc.empty:
        raise RuntimeError("Missing BTCUSDT 1m candles")
    btc = btc.copy()
    btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True)
    coins: dict[str, Any] = {}
    for base in bases:
        alt = load_candles(candle_path(CANDLE_DIR, f"{base}USDT", "1m"))
        if alt is None or alt.empty:
            continue
        alt = alt.copy()
        alt["timestamp"] = pd.to_datetime(alt["timestamp"], utc=True)
        rel = build_alt_btc(alt, btc)
        if rel is None or rel.empty:
            continue
        rel = rel.copy()
        rel["timestamp"] = pd.to_datetime(rel["timestamp"], utc=True)
        coins[base] = rel
    return {"btc_close": btc.set_index("timestamp")["close"], "coins": coins}


def metrics(legs: pd.DataFrame, label: str) -> dict[str, Any]:
    pnl = pd.to_numeric(legs["pnl_btc"], errors="coerce").fillna(0.0)
    pnl_pct = pd.to_numeric(legs["pnl_pct"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 1e-12]
    losses = pnl[pnl < -1e-12]
    n = len(pnl)
    n_dec = len(wins) + len(losses)
    reasons = legs["exit_reason"].astype(str)
    trail_n = int(reasons.str.contains("TRAILING_STOP", na=False).sum())
    same_n = int(reasons.str.contains("STOP_LOSS_TRAIL_ACTIVATION_SAME_CANDLE", na=False).sum())
    sl_n = int(reasons.str.contains("STOP_LOSS", na=False).sum()) - same_n
    ordered = legs.sort_values("exit_ts")
    eq = STARTING_BTC + pd.to_numeric(ordered["pnl_btc"], errors="coerce").fillna(0.0).cumsum()
    peak = eq.cummax()
    dd = float(((eq - peak) / peak.replace(0, np.nan)).min()) if len(eq) else 0.0
    final = float(eq.iloc[-1]) if len(eq) else STARTING_BTC
    pf = None
    if losses.sum() < 0 and abs(losses.sum()) > 0:
        pf = float(wins.sum() / abs(losses.sum()))
    hold = pd.to_numeric(legs.get("holding_hours"), errors="coerce")
    return {
        "Strategy": label,
        "Trades": n,
        "Wins": int(len(wins)),
        "Losses": int(len(losses)),
        "Win %": round(100.0 * len(wins) / n_dec, 2) if n_dec else None,
        "Expectancy": float(pnl_pct.mean()) if n else None,
        "Avg Win %": (
            float(pd.to_numeric(legs.loc[wins.index, "pnl_pct"], errors="coerce").mean() * 100)
            if len(wins)
            else None
        ),
        "Avg Loss %": (
            float(pd.to_numeric(legs.loc[losses.index, "pnl_pct"], errors="coerce").mean() * 100)
            if len(losses)
            else None
        ),
        "Profit Factor": pf,
        "Net BTC": float(pnl.sum()),
        "Return %": 100.0 * (final / STARTING_BTC - 1.0),
        "Max DD": dd,
        "Final Equity": final,
        "Avg Hold h": float(hold.mean()) if hold.notna().any() else None,
        "Trail exits": trail_n,
        "SL exits": sl_n,
        "Same-candle SL": same_n,
    }


def run() -> Path:
    from btcc.sim.accounting import CostModel
    from btcc.sim.exits import leg_to_record, open_opportunity_legs, process_bars_until_closed

    params = json.loads((PRIOR / "strategy_parameters.json").read_text(encoding="utf-8"))
    entries = pd.read_csv(PRIOR / "entry_diagnostics.csv")
    prior_cmp = pd.read_csv(PRIOR / "trade_level_comparison.csv")
    entries["entry_ts"] = pd.to_datetime(entries["entry_ts"], utc=True)
    bases = sorted(entries["base"].dropna().unique().tolist())
    panels = load_exit_panels(bases)
    btc_close = panels["btc_close"]
    costs = CostModel(0.0, 0.0)
    spec = t21_spec()

    out_dir = REPO / "results" / "t21_addon_21d_1m"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    forced = 0
    for i, r in entries.iterrows():
        base = str(r["base"])
        oid = str(r["opportunity_id"])
        entry_ts = _utc(r["entry_ts"])
        entry_mid = float(r["entry_price"])
        rel = panels["coins"].get(base)
        if rel is None:
            logger.warning("No 1m panel for %s — skip %s", base, oid)
            continue
        hist = btc_close[btc_close.index <= entry_ts]
        btc_px = float(hist.iloc[-1]) if len(hist) else float(btc_close.iloc[0])
        notional = float(STARTING_BTC * ALLOC_FRAC * btc_px)
        legs = open_opportunity_legs(
            alt_btc_entry_mid=entry_mid,
            btc_usdt=btc_px,
            notional_usd=notional,
            costs=costs,
            specs=[spec],
            entry_ts=entry_ts,
        )
        bars = rel[rel["timestamp"] > entry_ts].copy()
        if bars.empty:
            logger.warning("No bars after entry for %s", oid)
            continue
        process_bars_until_closed(
            legs,
            bars,
            btc_usdt_series=btc_close,
            default_btc_usdt=btc_px,
            costs=costs,
            same_candle_conflict="assume_sl_first",
        )
        leg = legs[0]
        if not leg.closed:
            last = bars.iloc[-1]
            from btcc.sim.exits import _close_leg

            ts = _utc(last["timestamp"])
            try:
                px = float(btc_close.loc[ts]) if ts in btc_close.index else float(btc_close.iloc[-1])
            except Exception:
                px = float(btc_close.iloc[-1])
            _close_leg(
                leg,
                exit_mid=float(last["close"]),
                btc_usdt=px,
                costs=costs,
                exit_ts=ts,
                reason="END_OF_DATA",
            )
            forced += 1
        rec = leg_to_record(leg, oid)
        rec["arm_key"] = "T21"
        rec["base"] = base
        rec["symbol"] = r["symbol"]
        rec["signal_ts"] = r["signal_ts"]
        rows.append(rec)
        if (i + 1) % 50 == 0:
            logger.info("Processed %d/%d", i + 1, len(entries))

    legs_df = pd.DataFrame(rows)
    logger.info("T21 closed trades=%d forced_eod=%d", len(legs_df), forced)

    m21 = metrics(legs_df, "T21")
    # Prior benchmarks from same entry set
    bench_rows = []
    for lab, pnl_col, reason_col in (
        ("T1", "T1_pnl_pct", "T1_exit_reason"),
        ("T7", "T7_pnl_pct", "T7_exit_reason"),
        ("T11", "T11_pnl_pct", "T11_exit_reason"),
        ("T13", "T13_pnl_pct", "T13_exit_reason"),
    ):
        # reconstruct approx BTC pnl from pct * entry notional / via merge
        merged = prior_cmp.merge(
            legs_df[["opportunity_id", "entry_btc_spent", "pnl_btc"]].rename(
                columns={"opportunity_id": "trade_id", "pnl_btc": "T21_pnl_btc"}
            ),
            left_on="trade_id",
            right_on="trade_id",
            how="inner",
        )
        # Use T21 entry_btc_spent * pnl_pct as notional proxy for prior arms
        spent = pd.to_numeric(merged["entry_btc_spent"], errors="coerce").fillna(0.0)
        pct = pd.to_numeric(merged[pnl_col], errors="coerce").fillna(0.0)
        fake = pd.DataFrame(
            {
                "pnl_btc": spent * pct,
                "pnl_pct": pct,
                "exit_reason": merged[reason_col],
                "exit_ts": merged.get(f"{lab}_exit_ts", pd.Series([None] * len(merged))),
                "holding_hours": np.nan,
            }
        )
        # holding hours unknown for prior in this reconstruct — leave nan
        if f"{lab}_exit_ts" in merged.columns:
            fake["exit_ts"] = pd.to_datetime(merged[f"{lab}_exit_ts"], utc=True, errors="coerce")
        else:
            fake["exit_ts"] = pd.RangeIndex(len(fake))  # stable order fallback
        bench_rows.append(metrics(fake, lab))

    summary = pd.DataFrame([m21] + bench_rows)
    # Prefer order T21, T1, T7, T11, T13
    summary = summary.set_index("Strategy").loc[["T21", "T1", "T7", "T11", "T13"]].reset_index()

    # Trade-level vs T1/T7
    cmp = prior_cmp.merge(
        legs_df[
            [
                "opportunity_id",
                "exit_ts",
                "exit_fill_price",
                "exit_reason",
                "pnl_pct",
                "pnl_btc",
                "mfe_pct",
                "mae_pct",
                "holding_hours",
            ]
        ].rename(
            columns={
                "opportunity_id": "trade_id",
                "exit_ts": "T21_exit_ts",
                "exit_fill_price": "T21_exit_price",
                "exit_reason": "T21_exit_reason",
                "pnl_pct": "T21_pnl_pct",
                "pnl_btc": "T21_pnl_btc",
                "holding_hours": "T21_holding_hours",
            }
        ),
        on="trade_id",
        how="inner",
    )
    cmp["T21_minus_T1_pct"] = cmp["T21_pnl_pct"] - cmp["T1_pnl_pct"]
    cmp["T21_minus_T7_pct"] = cmp["T21_pnl_pct"] - cmp["T7_pnl_pct"]

    meta = {
        "strategy": "T21",
        "description": "SL -1%, act +0.75%, trail 0.75% until peak +1.5% then trail 0.25%",
        "prior_experiment": str(PRIOR),
        "eval_start_utc": params.get("eval_start_utc"),
        "eval_end_utc": params.get("eval_end_utc"),
        "unique_entries": int(len(entries)),
        "t21_trades": int(len(legs_df)),
        "forced_end_of_data": forced,
        "fees": 0.0,
        "slippage": 0.0,
        "same_entries_as_prior": True,
        "live_untouched": True,
    }

    legs_df.to_csv(out_dir / "t21_legs.csv", index=False)
    summary.to_csv(out_dir / "t21_vs_benchmarks.csv", index=False)
    cmp.to_csv(out_dir / "t21_trade_level_vs_prior.csv", index=False)
    (out_dir / "t21_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = [
        "# T21 Addon Backtest (same 21d entries / 1m exits)",
        "",
        "## Geometry",
        "- Hard SL: **-1.00%**",
        "- Activation: **+0.75%**",
        "- Trail: **0.75%** until peak ≥ **+1.50%**, then **0.25%**",
        "- Transitions permanent / never loosen",
        "",
        f"Window: `{params.get('eval_start_utc')}` → `{params.get('eval_end_utc')}`",
        f"Shared entries: **{len(entries)}** (from prior exit-only experiment)",
        "",
        "## Results vs benchmarks",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        f"- T21 beats T1 on {(cmp['T21_minus_T1_pct'] > 0).mean()*100:.1f}% of trades "
        f"(mean Δ = {100*cmp['T21_minus_T1_pct'].mean():.3f}%)",
        f"- T21 beats T7 on {(cmp['T21_minus_T7_pct'] > 0).mean()*100:.1f}% of trades "
        f"(mean Δ = {100*cmp['T21_minus_T7_pct'].mean():.3f}%)",
        "",
        "Live bot was not modified.",
    ]
    (out_dir / "t21_summary.md").write_text("\n".join(report), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"\nWrote {out_dir}")
    return out_dir


if __name__ == "__main__":
    run()
