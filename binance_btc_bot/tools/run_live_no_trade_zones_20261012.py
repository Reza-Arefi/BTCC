"""Search for no-trade conditions on the 2026-10-12 live pack (T30 + W2 entries).

Each live entry gets a feature row (entry snapshot, BTC / alt-market context from Binance 1h
klines, time of day, open positions, recent outcomes). Outcomes come from
results/live_exit_compare_2026-10-12/per_trade_by_strategy.csv (run the exit compare first).

Every candidate rule is "skip the entry when <feature> >= / <= threshold" (or a time bucket).
A rule is ROBUST when skipping improves the summed net return in BOTH cohorts (T30, W2) under
BOTH reference exits (TRN_4, W2). A within-cohort outcome permutation test estimates how many
robust rules appear by chance.

Writes: results/live_no_trade_zones_2026-10-12/
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml

REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "live_reports" / "live_trades_t30_w2_2026-10-12" / "live_trades_t30_w2_2026-10-12"
EXITS = REPO / "results" / "live_exit_compare_2026-10-12" / "per_trade_by_strategy.csv"
CACHE = REPO / "data" / "live_replay_1h_native"
OUT = REPO / "results" / "live_no_trade_zones_2026-10-12"
KLINES_URL = "https://api.binance.com/api/v3/klines"
REF_EXITS = ("TRN_4", "W2")
COHORTS = ("T30", "W2")
PEGGED = {"PAXGBTC", "WBTCBTC", "XAUTBTC"}
N_PERM = 300

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("no_trade_zones")

SNAPSHOT_FIELDS = {
    "S": ("S",),
    "S_previous": ("S_previous",),
    "mom_ret_15m": ("factors", "momentum", "return_15m"),
    "mom_ret_1h": ("factors", "momentum", "return_1h"),
    "mom_ret_4h": ("factors", "momentum", "return_4h"),
    "mom_ret_24h": ("factors", "momentum", "return_24h"),
    "mom_accel_4h": ("factors", "momentum", "accel_4h"),
    "adx": ("factors", "trend", "adx"),
    "rsi14": ("factors", "rsi", "rsi14"),
    "rsi_slope": ("factors", "rsi", "rsi_slope"),
    "rvol": ("factors", "volume", "rvol"),
    "natr": ("factors", "volatility", "natr"),
    "percent_b": ("factors", "volatility", "percent_b"),
    "bb_bandwidth": ("factors", "volatility", "bandwidth"),
    "dist_to_resistance": ("factors", "structure", "dist_to_resistance"),
    "dist_to_support": ("factors", "structure", "dist_to_support"),
    "btc_ret_1h": ("factors", "btc_regime", "btc_return_1h"),
    "btc_ret_4h": ("factors", "btc_regime", "btc_return_4h"),
    "btc_ret_24h": ("factors", "btc_regime", "btc_return_24h"),
    "btc_natr": ("factors", "btc_regime", "btc_natr"),
    "btc_regime_score": ("factors", "btc_regime", "score"),
    "late_entry_score": ("late_entry", "late_entry_score"),
    "score_momentum": ("factor_scores", "momentum"),
    "score_trend": ("factor_scores", "trend"),
    "score_volume": ("factor_scores", "volume"),
    "score_volatility": ("factor_scores", "volatility"),
    "score_structure": ("factor_scores", "structure"),
}
BOOL_FIELDS = {
    "btc_ema_bull": ("factors", "btc_regime", "btc_ema_bull"),
    "rsi_overheated": ("factors", "rsi", "overheated"),
    "breakout": ("factors", "structure", "breakout"),
    "above_upper_band": ("factors", "volatility", "above_upper_band"),
    "volume_divergence": ("factors", "volume", "volume_divergence"),
}


def _get(d: dict[str, Any], path: tuple[str, ...]) -> Any:
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def load_snapshots() -> pd.DataFrame:
    rows = []
    for p in list(PACK.glob("trades/*/entry_snapshot.json")) + list(PACK.glob("still_open/*/entry_snapshot.json")):
        s = json.loads(p.read_text())
        row: dict[str, Any] = {"trade_id": p.parent.name}
        for name, path in SNAPSHOT_FIELDS.items():
            v = _get(s, path)
            row[name] = float(v) if v is not None else np.nan
        for name, path in BOOL_FIELDS.items():
            v = _get(s, path)
            row[name] = float(bool(v)) if v is not None else np.nan
        rows.append(row)
    d = pd.DataFrame(rows)
    d["S_jump"] = d["S"] - d["S_previous"]
    return d


def fetch_1h(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = CACHE / f"{symbol}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if df["timestamp"].min() <= start and df["timestamp"].max() >= end - pd.Timedelta(hours=2):
            return df
    rows: list[list[Any]] = []
    t, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    with requests.Session() as s:
        while t < end_ms:
            r = s.get(KLINES_URL, params={"symbol": symbol, "interval": "1h", "startTime": t, "endTime": end_ms, "limit": 1000}, timeout=20)
            r.raise_for_status()
            k = r.json()
            if not k:
                break
            rows.extend(k)
            t = int(k[-1][0]) + 3_600_000
    df = pd.DataFrame([x[:6] for x in rows], columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def market_features(entries: pd.Series) -> pd.DataFrame:
    """Context from the last fully closed 1h bar before each entry."""
    start = entries.min().floor("h") - pd.Timedelta(days=8)
    end = entries.max().ceil("h") + pd.Timedelta(hours=1)
    cfg = yaml.safe_load((REPO / "binance_btc_bot" / "config" / "binance_bot.yaml").read_text(encoding="utf-8"))
    universe = [s for s in cfg["universe"]["btc_pairs"] if s not in PEGGED]

    b = fetch_1h("BTCUSDT", start, end).set_index("timestamp")
    btc, btc_hi, btc_lo = b["close"], b["high"], b["low"]
    alts = pd.DataFrame({s: fetch_1h(s, start, end).set_index("timestamp")["close"] for s in universe}).sort_index()

    m = pd.DataFrame(index=btc.index)
    m["btc_ret_72h"] = btc.pct_change(72)
    m["btc_ret_7d"] = btc.pct_change(168)
    m["btc_dist_ema50_1h"] = btc / btc.ewm(span=50, adjust=False).mean() - 1
    m["btc_range_24h"] = (btc_hi.rolling(24).max() - btc_lo.rolling(24).min()) / btc
    for h in (4, 24, 72):
        r = alts.pct_change(h)
        m[f"alt_breadth_{h}h"] = (r > 0).sum(axis=1) / r.notna().sum(axis=1)
        m[f"alt_index_ret_{h}h"] = r.median(axis=1)
    m.index = m.index + pd.Timedelta(hours=1)  # bar close time: usable only after the bar closes
    idx = m.index.searchsorted(entries.to_numpy(), side="right") - 1
    return m.iloc[idx].reset_index(drop=True)


def position_features(t: pd.DataFrame) -> pd.DataFrame:
    """Open positions, clustering and recent live outcomes known at entry time."""
    rows = []
    for _, r in t.iterrows():
        e = r["entry_utc"]
        before_closed = t[(t["live_exit_utc"] <= e)].sort_values("live_exit_utc")
        same = t[(t["symbol"] == r["symbol"]) & (t["entry_utc"] < e)].sort_values("entry_utc")
        same_closed = same[same["live_exit_utc"] <= e]
        sl_before = before_closed[before_closed["live_net"] < -0.03]
        consec = 0
        for x in before_closed["live_net"].to_numpy()[::-1]:
            if x < 0:
                consec += 1
            else:
                break
        rows.append(
            {
                "open_positions": int(((t["entry_utc"] < e) & (t["live_exit_utc"] > e)).sum()),
                "entries_prev_4h": int(((t["entry_utc"] < e) & (t["entry_utc"] >= e - pd.Timedelta(hours=4))).sum()),
                "hours_since_same_coin": (e - same["entry_utc"].iloc[-1]).total_seconds() / 3600 if len(same) else np.nan,
                "same_coin_prev_loss": float(same_closed["live_net"].iloc[-1] < 0) if len(same_closed) else np.nan,
                "prev_trade_loss": float(before_closed["live_net"].iloc[-1] < 0) if len(before_closed) else np.nan,
                "consec_losses_before": consec,
                "hours_since_last_sl": (e - sl_before["live_exit_utc"].iloc[-1]).total_seconds() / 3600 if len(sl_before) else np.nan,
                "losses_prev_24h": int(((before_closed["live_exit_utc"] >= e - pd.Timedelta(hours=24)) & (before_closed["live_net"] < 0)).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_table() -> pd.DataFrame:
    d = pd.read_csv(EXITS, parse_dates=["entry_utc", "exit_utc"])
    t = d[d["strategy"] == "LIVE"][["trade_id", "cohort", "symbol", "entry_utc", "alloc", "exit_utc", "net_pct"]]
    t = t.rename(columns={"exit_utc": "live_exit_utc", "net_pct": "live_net"})
    for ex in REF_EXITS:
        t = t.merge(d[d["strategy"] == ex][["trade_id", "net_pct"]].rename(columns={"net_pct": f"net_{ex}"}), on="trade_id")
    t = t.sort_values("entry_utc").reset_index(drop=True)
    t = t.merge(load_snapshots(), on="trade_id", how="left")
    t = pd.concat([t, market_features(t["entry_utc"]), position_features(t)], axis=1)
    t["hour_utc"] = t["entry_utc"].dt.hour
    t["weekday"] = t["entry_utc"].dt.day_name().str[:3]
    t["weekend"] = (t["entry_utc"].dt.weekday >= 5).astype(float)
    return t


def candidate_rules(t: pd.DataFrame, numeric: list[str]) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    masks: list[np.ndarray] = []
    n = len(t)
    for f in numeric:
        x = t[f].to_numpy(dtype=float)
        ok = ~np.isnan(x)
        if ok.sum() < 0.7 * n:
            continue
        uniq = np.unique(x[ok])
        thresholds = uniq if len(uniq) <= 12 else np.unique(np.nanquantile(x, np.linspace(0.05, 0.95, 19)))
        for thr in thresholds:
            for op, m in ((">=", ok & (x >= thr)), ("<=", ok & (x <= thr))):
                if 0.05 * n <= m.sum() <= 0.4 * n:
                    names.append(f"{f} {op} {thr:.4g}")
                    masks.append(m)
    for day in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"):
        names.append(f"weekday == {day}")
        masks.append((t["weekday"] == day).to_numpy())
    for h0 in range(0, 24, 2):
        for width in (4, 6, 8):
            hrs = [(h0 + k) % 24 for k in range(width)]
            names.append(f"hour_utc in {h0:02d}-{(h0 + width) % 24:02d}")
            masks.append(t["hour_utc"].isin(hrs).to_numpy())
    keep = [i for i, m in enumerate(masks) if 0.05 * n <= m.sum() <= 0.4 * n]
    return [names[i] for i in keep], np.array([masks[i] for i in keep])


def evaluate(masks: np.ndarray, t: pd.DataFrame, nets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per cohort x exit: change in mean net return per cohort trade if the masked trades are skipped."""
    out: dict[str, np.ndarray] = {}
    for c in COHORTS:
        cm = (t["cohort"] == c).to_numpy()
        n_c = cm.sum()
        for ex, net in nets.items():
            out[f"{c}_{ex}"] = -(masks[:, cm] @ net[cm]) / n_c
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t = build_table()
    t.to_csv(OUT / "trade_features.csv", index=False)
    skip = {"trade_id", "cohort", "symbol", "entry_utc", "alloc", "live_exit_utc", "live_net", "weekday", "hour_utc"}
    numeric = [c for c in t.columns if c not in skip and not c.startswith("net_")]
    names, masks = candidate_rules(t, numeric)
    nets = {ex: t[f"net_{ex}"].to_numpy() for ex in REF_EXITS}
    ev = evaluate(masks.astype(float), t, nets)
    score = np.min(np.vstack(list(ev.values())), axis=0)
    robust = score > 0
    logger.info("%d candidate rules, %d robust", len(names), robust.sum())

    rng = np.random.default_rng(11)
    perm_robust, perm_best = [], []
    cohort_idx = {c: np.flatnonzero((t["cohort"] == c).to_numpy()) for c in COHORTS}
    for _ in range(N_PERM):
        order = np.arange(len(t))
        for idx in cohort_idx.values():
            order[idx] = rng.permutation(idx)
        pev = evaluate(masks.astype(float), t, {ex: v[order] for ex, v in nets.items()})
        ps = np.min(np.vstack(list(pev.values())), axis=0)
        perm_robust.append(int((ps > 0).sum()))
        perm_best.append(float(ps.max()))
    perm_robust_arr, perm_best_arr = np.array(perm_robust), np.array(perm_best)

    rows = []
    for i, name in enumerate(names):
        m = masks[i]
        row: dict[str, Any] = {"rule": name, "feature": name.split(" ")[0], "score": score[i] * 100, "robust": bool(robust[i])}
        for c in COHORTS:
            cm = (t["cohort"] == c).to_numpy() & m
            row[f"{c}_skipped"] = int(cm.sum())
            for ex in REF_EXITS:
                net = t.loc[cm, f"net_{ex}"]
                row[f"{c}_{ex}_losses_skipped"] = int((net < 0).sum())
                row[f"{c}_{ex}_wins_skipped"] = int((net > 0).sum())
                row[f"{c}_{ex}_delta_sum_pct"] = float(-net.sum() * 100)
                row[f"{c}_{ex}_delta_account_pct"] = float(-(net * t.loc[cm, "alloc"]).sum() * 100)
            row[f"{c}_live_delta_sum_pct"] = float(-t.loc[cm, "live_net"].sum() * 100)
        row["p_value_best_vs_perm"] = float((perm_best_arr >= score[i]).mean())
        rows.append(row)
    rules = pd.DataFrame(rows).sort_values("score", ascending=False)
    rules.to_csv(OUT / "rules_all.csv", index=False)
    best_per_feature = rules[rules["robust"]].drop_duplicates("feature")
    best_per_feature.to_csv(OUT / "rules_robust_best_per_feature.csv", index=False)
    meta = {
        "n_trades": t["cohort"].value_counts().to_dict(),
        "n_candidate_rules": len(names),
        "n_robust_rules": int(robust.sum()),
        "perm_robust_rules_mean": float(perm_robust_arr.mean()),
        "perm_robust_rules_p95": float(np.percentile(perm_robust_arr, 95)),
        "p_robust_count_vs_perm": float((perm_robust_arr >= robust.sum()).mean()),
        "perm_best_score_p95_pct": float(np.percentile(perm_best_arr, 95) * 100),
        "real_best_score_pct": float(score.max() * 100),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("meta %s", json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
