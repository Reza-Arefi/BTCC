"""Replay the 2026-10-12 live pack (T30 + W2 cohorts) through fixed-trailing and fixed SL/TP exits.

Every exit keeps the live entry (time + fill price). Paths are native ALTBTC 1m klines
(Binance REST, cached under data/live_replay_1m_native/). SL is assumed first when SL and
target/activation touch in the same 1m bar. Fees 0.1%/side on every arm, including LIVE.
Positions still open after --max-hold-days or at data end are marked to the last close;
the open ICPBTC trade and the manually cancelled DOTBTC trade are included the same way.

Families:
  fixed trail   T1–T10, T16, T17, T29, T30, W1, W2, W3, TRN_1..5 (0.5% trail), TRW_1,2,3,5 (2% trail)
  fixed SL/TP   SL{2,3,4,5}_TP{1..5}  (no trailing)
  LIVE          what the bot actually did (W2 trail until 2026-10-06, then fixed TP +4%)

Writes: results/live_exit_compare_2026-10-12/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from btcc.sim.accounting import CostModel  # noqa: E402
from btcc.sim.exits import StrategySpec, open_opportunity_legs, process_bars_until_closed  # noqa: E402

PACK = REPO / "live_reports" / "live_trades_t30_w2_2026-10-12" / "live_trades_t30_w2_2026-10-12"
CACHE = REPO / "data" / "live_replay_1m_native"
OUT = REPO / "results" / "live_exit_compare_2026-10-12"
KLINES_URL = "https://api.binance.com/api/v3/klines"
DEFAULT_BTC_USDT = 100_000.0
COSTS = CostModel(fee_rate_per_side=0.0, slippage_rate_per_side=0.0)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_exit_compare")

# name: (stop_loss, trail_activation, trail_distance)
FIXED_TRAIL: dict[str, tuple[float, float, float]] = {
    "T1": (0.0075, 0.0075, 0.0025),
    "T2": (0.01, 0.01, 0.0025),
    "T3": (0.0075, 0.0075, 0.005),
    "T4": (0.01, 0.01, 0.005),
    "T5": (0.015, 0.01, 0.005),
    "T6": (0.015, 0.015, 0.005),
    "T7": (0.02, 0.02, 0.005),
    "T8": (0.0075, 0.0075, 0.0075),
    "T9": (0.015, 0.015, 0.0025),
    "T10": (0.02, 0.015, 0.005),
    "T16": (0.0075, 0.01, 0.005),
    "T17": (0.0075, 0.0125, 0.0075),
    "T29": (0.04, 0.01, 0.005),
    "T30": (0.03, 0.01, 0.0025),
    "W1": (0.03, 0.03, 0.015),
    "W2": (0.05, 0.04, 0.02),
    "W3": (0.04, 0.025, 0.02),
    **{f"TRN_{lv}": (0.05, lv / 100, 0.005) for lv in (1, 2, 3, 4, 5)},
    **{f"TRW_{lv}": (0.05, lv / 100, 0.02) for lv in (1, 2, 3, 5)},
}
SL_TP_GRID = [(sl, tp) for sl in (2, 3, 4, 5) for tp in (1, 2, 3, 4, 5)]


def build_specs() -> list[StrategySpec]:
    specs = [
        StrategySpec(key=k, name=k, stop_loss_pct=sl, trail_activation_pct=act, trail_distance_pct=dist)
        for k, (sl, act, dist) in FIXED_TRAIL.items()
    ]
    specs += [
        StrategySpec(key=f"SL{sl}_TP{tp}", name=f"SL{sl}_TP{tp}", stop_loss_pct=sl / 100, take_profit_pct=tp / 100)
        for sl, tp in SL_TP_GRID
    ]
    return specs


def family(key: str) -> str:
    if key == "LIVE":
        return "live"
    return "fixed SL/TP" if key.startswith("SL") else "fixed trail"


def load_trades() -> pd.DataFrame:
    closed = pd.read_csv(PACK / "trades_closed.csv")
    still_open = [json.loads(p.read_text())["summary"] for p in PACK.glob("still_open/*/trade.json")]
    d = pd.concat([closed, pd.DataFrame(still_open)], ignore_index=True)
    live_exit: dict[str, str] = {}
    for p in list(PACK.glob("trades/*/trade.json")) + list(PACK.glob("still_open/*/trade.json")):
        t = json.loads(p.read_text())
        cfg = t.get("strategy_config") or {}
        s = t["summary"]
        if s["strategy"] == "T30":
            live_exit[s["trade_id"]] = "T30"
        else:
            live_exit[s["trade_id"]] = "SL5_TP4" if cfg.get("fixed_take_profit") else "W2"
    d["live_exit"] = d["trade_id"].map(live_exit)
    d["entry_time"] = pd.to_datetime(d["entry_time"], utc=True)
    d["exit_time"] = pd.to_datetime(d["exit_time"], utc=True)
    d["cohort"] = d["strategy"]
    return d.sort_values("entry_time").reset_index(drop=True)


def fetch_klines(session: requests.Session, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows: list[list[Any]] = []
    t = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while t < end_ms:
        for attempt in range(5):
            try:
                r = session.get(
                    KLINES_URL,
                    params={"symbol": symbol, "interval": "1m", "startTime": t, "endTime": end_ms, "limit": 1000},
                    timeout=20,
                )
                r.raise_for_status()
                break
            except requests.RequestException as exc:
                logger.warning("%s retry %d: %s", symbol, attempt + 1, exc)
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"klines failed for {symbol}")
        k = r.json()
        if not k:
            break
        rows.extend(k)
        t = int(k[-1][0]) + 60_000
    df = pd.DataFrame([row[:6] for row in rows], columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def load_1m(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = CACHE / f"{symbol}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if df["timestamp"].min() <= start and df["timestamp"].max() >= end - pd.Timedelta(minutes=2):
            return df
    with requests.Session() as session:
        df = fetch_klines(session, symbol, start, end)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("%s 1m %s -> %s (%d bars)", symbol, df["timestamp"].min(), df["timestamp"].max(), len(df))
    return df


def max_consecutive_losses(net: pd.Series) -> int:
    best = run = 0
    for x in net:
        run = run + 1 if x < 0 else 0
        best = max(best, run)
    return best


def paired_bootstrap_p_better(diff: np.ndarray, n: int = 5000, seed: int = 7) -> float:
    if len(diff) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    return float((means > 0).mean())


def replay_trade(t: pd.Series, m: pd.DataFrame, specs: list[StrategySpec], max_hold: pd.Timedelta, fee_rt: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entry_ts = pd.Timestamp(t["entry_time"])
    entry_px = float(t["entry_price"])
    first = entry_ts.floor("min") + pd.Timedelta(minutes=1)
    bars = m[(m["timestamp"] >= first) & (m["timestamp"] < entry_ts + max_hold)].reset_index(drop=True)
    entry_bar = m[m["timestamp"] == entry_ts.floor("min")]
    audit = {
        "trade_id": t["trade_id"],
        "symbol": t["symbol"],
        "entry_px": entry_px,
        "entry_bar_low": float(entry_bar["low"].iloc[0]) if not entry_bar.empty else np.nan,
        "entry_bar_high": float(entry_bar["high"].iloc[0]) if not entry_bar.empty else np.nan,
        "forward_bars": len(bars),
        "max_gap_min": float(bars["timestamp"].diff().dt.total_seconds().div(60).max()) if len(bars) > 1 else np.nan,
    }
    legs = open_opportunity_legs(
        alt_btc_entry_mid=entry_px, btc_usdt=DEFAULT_BTC_USDT, notional_usd=1000.0, costs=COSTS, specs=specs, entry_ts=entry_ts
    )
    for leg in legs:
        leg.position["entry_fill_price"] = entry_px
        leg.position["entry_alt_btc_mid"] = entry_px
        leg.initial_sl = entry_px * (1.0 - leg.spec.stop_loss_pct)
        leg.tp = entry_px * (1.0 + leg.spec.take_profit_pct) if leg.spec.take_profit_pct is not None else None
    process_bars_until_closed(legs, bars, btc_usdt_series=None, default_btc_usdt=DEFAULT_BTC_USDT, costs=COSTS)

    last_px = float(bars["close"].iloc[-1])
    last_ts = pd.Timestamp(bars["timestamp"].iloc[-1])
    base = {
        "trade_id": t["trade_id"],
        "cohort": t["cohort"],
        "symbol": t["symbol"],
        "entry_utc": entry_ts,
        "entry_px": entry_px,
        "btc_value": float(t["btc_value"]),
        "alloc": float(t["actual_allocation_pct"]),
        "live_exit": t["live_exit"],
    }
    rows: list[dict[str, Any]] = []
    for leg in legs:
        if leg.closed:
            exit_px = float((leg.exit_result or {})["exit_alt_btc_mid"])
            exit_ts = pd.Timestamp(leg.exit_ts)
            reason = str(leg.exit_reason)
        else:
            exit_px, exit_ts, reason = last_px, last_ts, "OPEN_MTM"
        gross = exit_px / entry_px - 1.0
        rows.append(
            {
                **base,
                "strategy": leg.spec.key,
                "exit_utc": exit_ts,
                "exit_px": exit_px,
                "exit_reason": reason,
                "gross_pct": gross,
                "net_pct": gross - fee_rt,
                "mfe_pct": float(leg.mfe_pct),
                "mae_pct": float(leg.mae_pct),
                "hold_h": (exit_ts - entry_ts).total_seconds() / 3600.0,
            }
        )

    live_closed = pd.notna(t["exit_price"]) and t["outcome"] != "MANUAL"
    live_px = float(t["exit_price"]) if live_closed else last_px
    live_ts = pd.Timestamp(t["exit_time"]) if live_closed else last_ts
    gross = live_px / entry_px - 1.0
    rows.append(
        {
            **base,
            "strategy": "LIVE",
            "exit_utc": live_ts,
            "exit_px": live_px,
            "exit_reason": str(t["outcome"]) if live_closed else "OPEN_MTM",
            "gross_pct": gross,
            "net_pct": gross - fee_rt,
            "mfe_pct": np.nan,
            "mae_pct": np.nan,
            "hold_h": (live_ts - entry_ts).total_seconds() / 3600.0,
        }
    )
    return rows, audit


def summarize(detail: pd.DataFrame, cohort: str) -> pd.DataFrame:
    g = detail[detail["cohort"] == cohort].sort_values("entry_utc")
    live = g[g["strategy"] == "LIVE"].set_index("trade_id")["net_pct"]
    rows = []
    for name, h in g.groupby("strategy", sort=False):
        net = h["net_pct"]
        gains, losses = net[net > 0].sum(), -net[net < 0].sum()
        diff = (h.set_index("trade_id")["net_pct"] - live).dropna().to_numpy()
        reasons = h["exit_reason"].astype(str)
        rows.append(
            {
                "strategy": name,
                "family": family(name),
                "trades": len(h),
                "wins": int((net > 0).sum()),
                "losses": int((net < 0).sum()),
                "win_rate": float((net > 0).mean()),
                "equity_return_pct": float((net * h["alloc"]).sum() * 100),
                "net_btc": float((net * h["btc_value"]).sum()),
                "net_sum_pct": float(net.sum() * 100),
                "mean_pct": float(net.mean() * 100),
                "median_pct": float(net.median() * 100),
                "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
                "worst_pct": float(net.min() * 100),
                "max_consec_losses": max_consecutive_losses(net),
                "stop_loss_exits": int(reasons.str.startswith("STOP_LOSS").sum()),
                "open_mtm": int((reasons == "OPEN_MTM").sum()),
                "avg_hold_h": float(h["hold_h"].mean()),
                "vs_live_sum_pct": float(diff.sum() * 100),
                "p_better_than_live": paired_bootstrap_p_better(diff) if name != "LIVE" else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("equity_return_pct", ascending=False).reset_index(drop=True)


def validate(detail: pd.DataFrame) -> pd.DataFrame:
    live = detail[detail["strategy"] == "LIVE"].set_index("trade_id")
    sim = detail.set_index(["trade_id", "strategy"])
    rows = []
    for tid, r in live.iterrows():
        if r["exit_reason"] == "OPEN_MTM":
            continue
        s = sim.loc[(tid, r["live_exit"])]
        rows.append(
            {
                "trade_id": tid,
                "cohort": r["cohort"],
                "symbol": r["symbol"],
                "live_exit": r["live_exit"],
                "live_gross_pct": r["gross_pct"] * 100,
                "sim_gross_pct": s["gross_pct"] * 100,
                "abs_diff_pp": abs(r["gross_pct"] - s["gross_pct"]) * 100,
                "same_sign": bool(np.sign(r["gross_pct"]) == np.sign(s["gross_pct"])),
                "sim_reason": s["exit_reason"],
                "live_hold_h": r["hold_h"],
                "sim_hold_h": s["hold_h"],
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hold-days", type=float, default=7.0)
    ap.add_argument("--fee-per-side", type=float, default=0.001)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    fee_rt = 2.0 * args.fee_per_side
    max_hold = pd.Timedelta(days=args.max_hold_days)
    now = pd.Timestamp(datetime.now(timezone.utc)).floor("min")
    trades = load_trades()
    specs = build_specs()
    logger.info("%d trades (%s), %d exit specs", len(trades), trades["cohort"].value_counts().to_dict(), len(specs))

    windows = trades.groupby("symbol")["entry_time"].agg(["min", "max"])
    windows["start"] = windows["min"].dt.floor("min") - pd.Timedelta(minutes=10)
    windows["end"] = (windows["max"] + max_hold).clip(upper=now)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {sym: ex.submit(load_1m, sym, w["start"], w["end"]) for sym, w in windows.iterrows()}
        paths = {sym: f.result() for sym, f in futs.items()}

    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for _, t in trades.iterrows():
        r, a = replay_trade(t, paths[t["symbol"]], specs, max_hold, fee_rt)
        rows.extend(r)
        audits.append(a)
    detail = pd.DataFrame(rows)
    audit = pd.DataFrame(audits)
    audit["entry_in_bar_range"] = (audit["entry_px"] >= audit["entry_bar_low"] * 0.999) & (
        audit["entry_px"] <= audit["entry_bar_high"] * 1.001
    )

    OUT.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT / "per_trade_by_strategy.csv", index=False)
    audit.to_csv(OUT / "entry_audit.csv", index=False)
    val = validate(detail)
    val.to_csv(OUT / "live_vs_sim_validation.csv", index=False)
    summaries = {c: summarize(detail, c) for c in ("W2", "T30")}
    for c, s in summaries.items():
        s.to_csv(OUT / f"summary_{c}.csv", index=False)
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_end_utc": str(now),
        "max_hold_days": args.max_hold_days,
        "fee_per_side": args.fee_per_side,
        "n_trades": trades["cohort"].value_counts().to_dict(),
        "n_specs": len(specs),
        "entries_outside_bar_range": int((~audit["entry_in_bar_range"]).sum()),
        "validation_same_sign_rate": val.groupby("live_exit")["same_sign"].mean().to_dict(),
        "validation_median_abs_diff_pp": val.groupby("live_exit")["abs_diff_pp"].median().to_dict(),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    logger.info("meta %s", json.dumps(meta, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
