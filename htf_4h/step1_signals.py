"""HTF4H step 1 — build the 4h S panel and look at the signals it produces.

1a  history check: how many 4h bars of history does S need before it stops changing?
    (live 15m uses 1000 bars; 1000 x 4h = 167 days is longer than our data)
1b  score all 38 coins on 4h bars for both variants (literal / scaled), chosen history
1c  signal census: NEW_CROSS counts at 0.55-0.75, per month, per coin, S distribution,
    compared with the 15m signals on the same days

    python -m htf_4h.step1_signals --check            # 1a only
    python -m htf_4h.step1_signals --history 200      # 1b + 1c

Writes: results/htf_4h/step1/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from htf_4h.common import RESULTS, THRESHOLD, VARIANTS, bases, load_coin, new_cross_signals, score_bar, score_coin_4h, weights

OUT = RESULTS / "step1"
CHECK_COINS = ("ETH", "SOL", "LINK", "WLD", "ICP", "TAO")
CHECK_HISTORIES = (100, 150, 200, 300)
CHECK_LAST_BARS = 150
THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75)
S15_PANEL = RESULTS.parent / "candidate_trn4_wknd_local_7m" / "s_panel.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("htf4h.step1")


# ------------------------------------------------------------- 1a history ---


def _check_coin(base: str) -> list[dict]:
    d = load_coin(base)
    w = weights()
    n = len(d["rel4"])
    rows = []
    for variant in VARIANTS:
        for i in range(n - CHECK_LAST_BARS, n):
            full = score_bar(d, i, i + 1, variant, w)["S"]
            for h in CHECK_HISTORIES:
                rows.append({"base": base, "variant": variant, "i": i, "history": h, "full_bars": i + 1,
                             "S_full": full, "S_h": score_bar(d, i, h, variant, w)["S"]})
    return rows


def history_check(workers: int) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_check_coin, b) for b in CHECK_COINS]):
            rows += fut.result()
    df = pd.DataFrame(rows)
    df["abs_dS"] = (df["S_h"] - df["S_full"]).abs()
    df["zone_agree"] = (df["S_h"] >= THRESHOLD) == (df["S_full"] >= THRESHOLD)
    out = df.groupby(["variant", "history"]).agg(
        bars=("abs_dS", "size"), mean_abs_dS=("abs_dS", "mean"), p95_abs_dS=("abs_dS", lambda x: x.quantile(0.95)),
        max_abs_dS=("abs_dS", "max"), zone_agreement=("zone_agree", "mean"),
    ).reset_index()
    out.to_csv(OUT / "history_check.csv", index=False)
    return out


# --------------------------------------------------------------- 1b panel ---


def build_panel(history: int, workers: int) -> pd.DataFrame:
    path = OUT / f"s_panel_4h_h{history}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    parts, t0 = [], time.time()
    tasks = [(b, v) for b in bases() for v in VARIANTS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(score_coin_4h, b, v, history): (b, v) for b, v in tasks}
        for n, fut in enumerate(as_completed(futs), 1):
            b, v = futs[fut]
            df = fut.result()
            if not df.empty:
                parts.append(df)
            log.info("scored %s/%s (%d/%d) bars=%d %.0fs", b, v, n, len(tasks), len(df), time.time() - t0)
    s = pd.concat(parts, ignore_index=True)
    s.to_parquet(path, index=False)
    return s


# -------------------------------------------------------------- 1c census ---


def census(s: pd.DataFrame) -> dict:
    start = s["close_ts"].min()
    end = s["close_ts"].max()
    days = (end - start).total_seconds() / 86400
    thr_rows, month_rows = [], []
    sigs = {}
    for thr in THRESHOLDS:
        sg = new_cross_signals(s, thr)
        for v, g in sg.groupby("variant"):
            thr_rows.append({"variant": v, "threshold": thr, "signals": len(g), "per_day": len(g) / days,
                             "coins": g["base"].nunique(),
                             "share_bars_in_zone": float((s.loc[s["variant"] == v, "S"] >= thr).mean())})
        if thr == THRESHOLD:
            sigs = {v: g for v, g in sg.groupby("variant")}
    for v, g in sigs.items():
        cnt = g.groupby(g["entry_utc"].dt.strftime("%Y-%m")).size()
        for m, n in cnt.items():
            month_rows.append({"variant": v, "month": m, "signals": int(n)})

    s15_n = None
    if S15_PANEL.exists():
        p = pd.read_parquet(S15_PANEL, columns=["base", "timestamp", "S"])
        p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True)
        p = p[(p["timestamp"] >= start) & (p["timestamp"] <= end)].sort_values(["base", "timestamp"])
        z = p["S"] >= THRESHOLD
        s15_n = int((z & ~z.groupby(p["base"]).shift(1, fill_value=False)).sum())
        for m, n in p.loc[z & ~z.groupby(p["base"]).shift(1, fill_value=False), "timestamp"].dt.strftime("%Y-%m").value_counts().items():
            month_rows.append({"variant": "15m live", "month": m, "signals": int(n)})

    thr_df = pd.DataFrame(thr_rows)
    month_df = pd.DataFrame(month_rows).pivot(index="month", columns="variant", values="signals").fillna(0).astype(int)
    coin_df = pd.concat([g.groupby("base").size().rename(v) for v, g in sigs.items()], axis=1).fillna(0).astype(int)
    thr_df.to_csv(OUT / "signals_by_threshold.csv", index=False)
    month_df.to_csv(OUT / "signals_by_month.csv")
    coin_df.to_csv(OUT / "signals_by_coin.csv")
    for v, g in sigs.items():
        g.to_csv(OUT / f"signals_{v}_S65.csv", index=False)
    return {"start": start, "end": end, "days": days, "thr": thr_df, "month": month_df, "coin": coin_df,
            "sigs": sigs, "s15_signals_same_span": s15_n}


def plots(s: pd.DataFrame, c: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    ax = axes[0][0]
    for v, col in zip(VARIANTS, ("#1f77b4", "#d62728")):
        ax.hist(s.loc[s["variant"] == v, "S"], bins=np.linspace(-1, 1, 81), alpha=0.5, color=col, label=v)
    ax.axvline(THRESHOLD, color="black", ls="--", lw=1)
    ax.set_title("Distribution of 4h S (dashed = 0.65)")
    ax.legend()

    ax = axes[0][1]
    t = c["thr"].pivot(index="threshold", columns="variant", values="per_day")
    t.plot(kind="bar", ax=ax, color=["#1f77b4", "#d62728"])
    if c["s15_signals_same_span"]:
        ax.axhline(c["s15_signals_same_span"] / c["days"], color="black", ls=":", label="15m live S>=0.65")
    ax.set_title("NEW_CROSS signals per day (all 38 coins)")
    ax.legend()

    ax = axes[1][0]
    c["month"].plot(kind="bar", ax=ax)
    ax.set_title("S>=0.65 signals per month")

    ax = axes[1][1]
    coin = c["coin"].sort_values(c["coin"].columns[0], ascending=False)
    coin.plot(kind="bar", ax=ax, width=0.8, color=["#1f77b4", "#d62728"])
    ax.set_title("S>=0.65 signals per coin")
    ax.tick_params(axis="x", labelsize=7)
    for a in axes.ravel():
        a.grid(alpha=0.3, axis="y")
    fig.suptitle(f"HTF4H step 1 — 4h S panel {c['start']:%Y-%m-%d} -> {c['end']:%Y-%m-%d}")
    fig.tight_layout()
    fig.savefig(OUT / "step1_signals.png", dpi=130)
    plt.close(fig)


def example_chart(s: pd.DataFrame, c: dict, base: str = "ETH") -> None:
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
    g = s[(s["base"] == base) & (s["variant"] == "literal")]
    axes[0].plot(g["close_ts"], g["close"], color="black", lw=1)
    for ax, v, col in zip(axes[1:], VARIANTS, ("#1f77b4", "#d62728")):
        gv = s[(s["base"] == base) & (s["variant"] == v)]
        ax.plot(gv["close_ts"], gv["S"], color=col, lw=0.9)
        ax.axhline(THRESHOLD, color="black", ls="--", lw=0.8)
        ax.set_ylabel(f"S {v}")
        sg = c["sigs"][v]
        sg = sg[sg["base"] == base]
        axes[0].scatter(sg["entry_utc"], sg["entry_px"], marker="^", color=col, s=40, zorder=3, label=f"{v} signal")
        ax.grid(alpha=0.3)
    axes[0].set_title(f"{base}/BTC 4h close with S>=0.65 NEW_CROSS signals")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"example_{base}.png", dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--history", type=int, default=200)
    ap.add_argument("--workers", type=int, default=18)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    if args.check:
        print(history_check(args.workers).round(4).to_string(index=False))
        return 0
    s = build_panel(args.history, args.workers)
    for col in ("timestamp", "close_ts"):
        s[col] = pd.to_datetime(s[col], utc=True)
    c = census(s)
    plots(s, c)
    example_chart(s, c, "ETH")
    meta = {"history_bars": args.history, "window": [str(c["start"]), str(c["end"])], "days": c["days"],
            "rows": len(s), "coins": int(s["base"].nunique()), "s15_signals_same_span": c["s15_signals_same_span"]}
    (OUT / "config.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(json.dumps(meta, indent=1, default=str))
    print(c["thr"].round(3).to_string(index=False))
    print(c["month"].to_string())
    print(c["coin"].describe().round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
