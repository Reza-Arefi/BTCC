"""Part 4A — Selector E adaptive capital allocation by opportunity frequency.

Historical research only. Replays the immutable Selector E trade stream and
varies ONLY position size. Does not modify paper/live bots or Selector E logic.

Accounting convention (matches frozen MEXC 1y reference):
  - starting_capital_usd = 1000
  - baseline notional = 100 (= 10% of starting capital)
  - reference used compound_notional=false
  - FIXED_10 replays stored pnl_usd_equiv exactly
  - adaptive arms scale that PnL by (actual_notional / 100)
  - requested notional = requested_allocation * starting_capital_usd
  - exposure / available capital use *current* equity (no leverage)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REF_DIR_DEFAULT = Path("results/selector_compare_365d_thr0p60_20260901_132655")
STARTING_CAPITAL_USD = 1000.0
BASE_ALLOCATION = 0.10
BASE_NOTIONAL_USD = STARTING_CAPITAL_USD * BASE_ALLOCATION  # 100
MAX_ALLOCATION = 0.25
MAX_OPEN = 10
ZERO_FREQ_ALLOCATION = MAX_ALLOCATION  # no divide-by-zero; request cap

ARM_ORDER = (
    "FIXED_10",
    "3D_INVERSE",
    "7D_INVERSE",
    "14D_INVERSE",
    "30D_INVERSE",
    "7D_SQRT",
    "7_30_SQRT",
    "7D_MEDIAN_SQRT",
    "DEADBAND",
)

FREQ_BUCKETS = (
    ("0-4", 0.0, 4.0),
    ("4-6", 4.0, 6.0),
    ("6-8", 6.0, 8.0),
    ("8-10", 8.0, 10.0),
    ("10+", 10.0, float("inf")),
)


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _clamp_alloc(x: float) -> float:
    if not np.isfinite(x) or x <= 0:
        return ZERO_FREQ_ALLOCATION
    return float(min(MAX_ALLOCATION, max(BASE_ALLOCATION, x)))


def _inv_freq(n: float) -> float:
    if n is None or not np.isfinite(n) or n <= 0:
        return ZERO_FREQ_ALLOCATION
    return _clamp_alloc(1.0 / float(n))


def _sqrt_alloc(n: float) -> float:
    if n is None or not np.isfinite(n) or n <= 0:
        return ZERO_FREQ_ALLOCATION
    return _clamp_alloc(BASE_ALLOCATION * float(np.sqrt(10.0 / float(n))))


@dataclass(frozen=True)
class ArmSpec:
    key: str
    warmup_days: int
    allocator: Callable[[dict[str, float]], float]


def _build_arm_specs() -> dict[str, ArmSpec]:
    return {
        "FIXED_10": ArmSpec("FIXED_10", 0, lambda _m: BASE_ALLOCATION),
        "3D_INVERSE": ArmSpec("3D_INVERSE", 3, lambda m: _inv_freq(m["n3"])),
        "7D_INVERSE": ArmSpec("7D_INVERSE", 7, lambda m: _inv_freq(m["n7"])),
        "14D_INVERSE": ArmSpec("14D_INVERSE", 14, lambda m: _inv_freq(m["n14"])),
        "30D_INVERSE": ArmSpec("30D_INVERSE", 30, lambda m: _inv_freq(m["n30"])),
        "7D_SQRT": ArmSpec("7D_SQRT", 7, lambda m: _sqrt_alloc(m["n7"])),
        "7_30_SQRT": ArmSpec(
            "7_30_SQRT",
            30,
            lambda m: _sqrt_alloc(0.7 * m["n7"] + 0.3 * m["n30"]),
        ),
        "7D_MEDIAN_SQRT": ArmSpec("7D_MEDIAN_SQRT", 7, lambda m: _sqrt_alloc(m["n7_median"])),
        "DEADBAND": ArmSpec(
            "DEADBAND",
            30,
            lambda m: (
                BASE_ALLOCATION
                if (0.7 * m["n7"] + 0.3 * m["n30"]) >= 8.0
                else _sqrt_alloc(0.7 * m["n7"] + 0.3 * m["n30"])
            ),
        ),
    }


def load_selector_E_stream(ref_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load immutable E opportunities + selected legs. Does not modify ref_dir."""
    ref_dir = Path(ref_dir)
    opp = pd.read_csv(ref_dir / "opportunities.csv", low_memory=False)
    legs = pd.read_csv(ref_dir / "strategy_legs.csv", low_memory=False)
    e = legs[(legs["arm_key"] == "E") & (~legs["is_counterfactual"].astype(bool))].copy()
    if e.empty:
        raise RuntimeError(f"No Selector E legs in {ref_dir}")
    e["entry_ts"] = pd.to_datetime(e["entry_ts"], utc=True)
    e["exit_ts"] = pd.to_datetime(e["exit_ts"], utc=True)
    e["pnl_pct"] = pd.to_numeric(e["pnl_pct"], errors="coerce").fillna(0.0)
    e["pnl_usd_equiv"] = pd.to_numeric(e["pnl_usd_equiv"], errors="coerce").fillna(0.0)
    e["pnl_btc"] = pd.to_numeric(e.get("pnl_btc"), errors="coerce").fillna(0.0)
    e["notional_usd"] = pd.to_numeric(e.get("notional_usd"), errors="coerce").fillna(BASE_NOTIONAL_USD)

    opp = opp.copy()
    opp["opened_ts"] = pd.to_datetime(opp["opened_ts"], utc=True)
    opp["signal_timestamp"] = pd.to_datetime(opp["signal_timestamp"], utc=True)
    opp["entry_fill_ts"] = pd.to_datetime(opp["entry_fill_ts"], utc=True)
    opp["day_number"] = pd.to_numeric(opp.get("day_number"), errors="coerce")

    # Join opportunity metadata onto E legs
    meta_cols = [
        "opportunity_id",
        "opened_ts",
        "signal_timestamp",
        "day_number",
        "symbol",
        "base",
        "S",
        "entry_btc_usdt",
        "regime",
    ]
    meta_cols = [c for c in meta_cols if c in opp.columns]
    merged = e.merge(opp[meta_cols], on="opportunity_id", how="left", suffixes=("", "_opp"))
    if "day_number" not in merged.columns and "day_number_opp" in merged.columns:
        merged["day_number"] = merged["day_number_opp"]
    if "entry_btc_usdt" not in merged.columns and "entry_btc_usdt_opp" in merged.columns:
        merged["entry_btc_usdt"] = merged["entry_btc_usdt_opp"]

    merged = merged.sort_values(["entry_ts", "opportunity_id"]).reset_index(drop=True)
    ref_metrics = {}
    metrics_path = ref_dir / "analytics" / "selector_metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text())
        ref_metrics = (payload.get("metrics") or {}).get("E") or {}
    return opp, merged, ref_metrics


def build_daily_opportunity_counts(opportunities: pd.DataFrame) -> pd.Series:
    """Eligible 15m opportunity count per UTC calendar day (original stream)."""
    ts = pd.to_datetime(opportunities["opened_ts"], utc=True)
    day = ts.dt.floor("D")
    counts = day.value_counts().sort_index()
    counts.index = pd.DatetimeIndex(counts.index).tz_convert("UTC")
    # Dense daily index from first to last opportunity day (zeros for quiet days)
    full = pd.date_range(counts.index.min(), counts.index.max(), freq="D", tz="UTC")
    return counts.reindex(full, fill_value=0).astype(float)


def frequency_features_for_day(daily_counts: pd.Series, trade_day: pd.Timestamp) -> dict[str, float]:
    """Lookback features using ONLY completed days strictly before trade_day.

    N7 = mean(count[D-1], ..., count[D-7]). Day D itself is never included.
    """
    d = _utc(trade_day).floor("D")
    idx = daily_counts.index
    # Prior completed days present in the series (and calendar-complete before d)
    prior = daily_counts[idx < d]

    def window_mean(n: int) -> float:
        # Require n complete calendar days immediately before D (D-n .. D-1).
        end = d - pd.Timedelta(days=1)
        start = d - pd.Timedelta(days=n)
        w = daily_counts[(idx >= start) & (idx <= end)]
        if len(w) < n:
            return float("nan")
        return float(w.mean())

    def window_median(n: int) -> float:
        end = d - pd.Timedelta(days=1)
        start = d - pd.Timedelta(days=n)
        w = daily_counts[(idx >= start) & (idx <= end)]
        if len(w) < n:
            return float("nan")
        return float(w.median())

    n3 = window_mean(3)
    n7 = window_mean(7)
    n14 = window_mean(14)
    n30 = window_mean(30)
    n7_med = window_median(7)
    nblend = (
        float("nan")
        if (not np.isfinite(n7) or not np.isfinite(n30))
        else (0.7 * n7 + 0.3 * n30)
    )
    return {
        "trade_day": d,
        "n_prior_days": float(len(prior)),
        "n3": n3,
        "n7": n7,
        "n14": n14,
        "n30": n30,
        "n7_median": n7_med,
        "nblend": nblend,
        "day_count_same_day_excluded": float(daily_counts.get(d, 0.0)),
    }


def verify_no_lookahead(daily_counts: pd.Series, trades: pd.DataFrame) -> dict[str, Any]:
    """Explicit check: sizing features never include day-D opportunity counts."""
    violations = 0
    samples = []
    for _, r in trades.head(50).iterrows():
        d = _utc(r["entry_ts"]).floor("D")
        feats = frequency_features_for_day(daily_counts, d)
        # Reconstruct windows and ensure d not in index
        for n, key in ((3, "n3"), (7, "n7"), (14, "n14"), (30, "n30")):
            end = d - pd.Timedelta(days=1)
            start = d - pd.Timedelta(days=n)
            window_idx = daily_counts.index[(daily_counts.index >= start) & (daily_counts.index <= end)]
            if d in window_idx:
                violations += 1
        samples.append(
            {
                "entry_ts": str(r["entry_ts"]),
                "trade_day": str(d),
                "same_day_count_excluded": feats["day_count_same_day_excluded"],
                "n7": feats["n7"],
            }
        )
    return {"violations": violations, "ok": violations == 0, "samples": samples}


def verify_fixed10_vs_reference(
    trades: pd.DataFrame,
    ref_metrics: dict[str, Any],
    *,
    tol_equity: float = 1e-6,
) -> dict[str, Any]:
    """FIXED_10 must reproduce Selector E reference equity / trade stats."""
    n = int(len(trades))
    total_pnl = float(trades["pnl_usd_equiv"].sum())
    final_eq = STARTING_CAPITAL_USD + total_pnl
    wins = int((trades["pnl_usd_equiv"] > 0).sum())
    losses = int((trades["pnl_usd_equiv"] < 0).sum())
    win_rate = 100.0 * wins / n if n else 0.0
    avg_ret = 100.0 * float(trades["pnl_pct"].mean()) if n else 0.0
    pos = float(trades.loc[trades["pnl_usd_equiv"] > 0, "pnl_usd_equiv"].sum())
    neg = float(-trades.loc[trades["pnl_usd_equiv"] < 0, "pnl_usd_equiv"].sum())
    pf = (pos / neg) if neg > 0 else float("inf")

    ref_final = float(ref_metrics.get("final_equity_usd", float("nan")))
    ref_ret = float(ref_metrics.get("cumulative_return_pct", float("nan")))
    ref_n = int(ref_metrics.get("n_trades", -1))
    ref_wr = float(ref_metrics.get("win_rate_pct", float("nan")))
    ref_pf = float(ref_metrics.get("profit_factor", float("nan")))
    ref_avg = float(ref_metrics.get("avg_trade_return_pct", float("nan")))
    ref_pnl = float(ref_metrics.get("total_pnl_usd", float("nan")))

    checks = {
        "n_trades": abs(n - ref_n) == 0,
        "final_equity_usd": abs(final_eq - ref_final) <= tol_equity,
        "total_pnl_usd": abs(total_pnl - ref_pnl) <= tol_equity,
        "cumulative_return_pct": abs((100.0 * total_pnl / STARTING_CAPITAL_USD) - ref_ret) <= 1e-6,
        "win_rate_pct": abs(win_rate - ref_wr) <= 1e-6,
        "profit_factor": abs(pf - ref_pf) <= 1e-8,
        "avg_trade_return_pct": abs(avg_ret - ref_avg) <= 1e-8,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "fixed10": {
            "n_trades": n,
            "final_equity_usd": final_eq,
            "total_pnl_usd": total_pnl,
            "cumulative_return_pct": 100.0 * total_pnl / STARTING_CAPITAL_USD,
            "win_rate_pct": win_rate,
            "profit_factor": pf,
            "avg_trade_return_pct": avg_ret,
            "winning_trades": wins,
            "losing_trades": losses,
        },
        "reference": {
            "n_trades": ref_n,
            "final_equity_usd": ref_final,
            "total_pnl_usd": ref_pnl,
            "cumulative_return_pct": ref_ret,
            "win_rate_pct": ref_wr,
            "profit_factor": ref_pf,
            "avg_trade_return_pct": ref_avg,
        },
    }


def simulate_arm(
    trades: pd.DataFrame,
    daily_counts: pd.Series,
    arm: ArmSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Event-driven replay of E trades with arm-specific sizing."""
    rows: list[dict[str, Any]] = []
    equity = STARTING_CAPITAL_USD
    reserved = 0.0
    open_book: dict[str, dict[str, Any]] = {}

    # Precompute features per trade calendar day
    trade_days = {_utc(ts).floor("D") for ts in trades["entry_ts"]}
    feat_cache = {d: frequency_features_for_day(daily_counts, d) for d in trade_days}

    events: list[tuple[pd.Timestamp, int, int, str]] = []
    for i, r in trades.iterrows():
        events.append((_utc(r["entry_ts"]), 0, int(i), "entry"))
        events.append((_utc(r["exit_ts"]), 1, int(i), "exit"))
    events.sort(key=lambda x: (x[0], x[1], x[2]))

    peak = equity
    max_dd = 0.0
    max_exposure = 0.0
    exposure_limited = 0
    cap_25_requested = 0
    cap_25_actual = 0
    max_req_alloc = 0.0
    max_act_alloc = 0.0
    uncapped_req_peak = 0.0  # before 25% cap, for inverse/sqrt raw

    equity_curve: list[dict[str, Any]] = []

    for ts, _, i, kind in events:
        r = trades.loc[i]
        oid = str(r["opportunity_id"])
        if kind == "exit":
            pos = open_book.pop(oid, None)
            if pos is None:
                continue
            reserved = max(0.0, reserved - float(pos["actual_notional"]))
            pnl = float(pos["pnl_usd"])
            equity += pnl
            peak = max(peak, equity)
            dd = (equity - peak) / peak if peak else 0.0
            max_dd = min(max_dd, dd)
            rows.append({**pos, "equity_after": equity, "drawdown_pct": 100.0 * dd})
            equity_curve.append({"ts": ts, "equity_usd": equity, "event": "exit", "opportunity_id": oid})
            continue

        # entry
        if len(open_book) >= MAX_OPEN:
            # Preserve max-open rule; should not occur on this reference stream
            d = _utc(r["entry_ts"]).floor("D")
            feats = feat_cache[d]
            rows.append(
                {
                    "opportunity_id": oid,
                    "arm": arm.key,
                    "entry_ts": r["entry_ts"],
                    "exit_ts": r["exit_ts"],
                    "trade_day": d,
                    "executed": False,
                    "skip_reason": "MAX_OPEN",
                    "requested_allocation": BASE_ALLOCATION,
                    "actual_allocation": 0.0,
                    "requested_notional": 0.0,
                    "actual_notional": 0.0,
                    "pnl_usd": 0.0,
                    "pnl_pct": float(r["pnl_pct"]),
                    "ref_pnl_usd": float(r["pnl_usd_equiv"]),
                    "exposure_limited": False,
                    "cap_25_hit": False,
                    "equity_before": equity,
                    "n3": feats["n3"],
                    "n7": feats["n7"],
                    "n14": feats["n14"],
                    "n30": feats["n30"],
                    "n7_median": feats["n7_median"],
                    "nblend": feats["nblend"],
                    "warmup": False,
                }
            )
            continue

        d = _utc(r["entry_ts"]).floor("D")
        feats = feat_cache[d]
        warmup = feats["n_prior_days"] < arm.warmup_days
        if warmup or arm.key == "FIXED_10":
            req_alloc = BASE_ALLOCATION
            raw_uncapped = BASE_ALLOCATION
        else:
            # Raw uncapped (for reporting) before clamp where applicable
            raw_uncapped = _raw_uncapped(arm.key, feats)
            req_alloc = float(arm.allocator(feats))

        if raw_uncapped > MAX_ALLOCATION + 1e-15:
            cap_25_requested += 1
        max_req_alloc = max(max_req_alloc, req_alloc)
        uncapped_req_peak = max(uncapped_req_peak, raw_uncapped)

        req_notional = req_alloc * STARTING_CAPITAL_USD
        available = max(0.0, equity - reserved)
        actual_notional = min(req_notional, available)
        exp_lim = actual_notional + 1e-9 < req_notional
        if exp_lim:
            exposure_limited += 1
        act_alloc = (actual_notional / STARTING_CAPITAL_USD) if STARTING_CAPITAL_USD else 0.0
        # Also express vs current equity for diagnostics
        act_alloc_of_equity = (actual_notional / equity) if equity > 0 else 0.0
        max_act_alloc = max(max_act_alloc, act_alloc)
        if abs(req_alloc - MAX_ALLOCATION) < 1e-12:
            cap_25_actual += 1

        scale = actual_notional / BASE_NOTIONAL_USD if BASE_NOTIONAL_USD else 0.0
        pnl_usd = float(r["pnl_usd_equiv"]) * scale
        pnl_btc = float(r["pnl_btc"]) * scale

        reserved += actual_notional
        max_exposure = max(max_exposure, reserved / equity if equity else 0.0)

        open_book[oid] = {
            "opportunity_id": oid,
            "arm": arm.key,
            "entry_ts": r["entry_ts"],
            "exit_ts": r["exit_ts"],
            "trade_day": d,
            "day_number": r.get("day_number"),
            "symbol": r.get("symbol"),
            "executed": True,
            "skip_reason": "",
            "requested_allocation": req_alloc,
            "actual_allocation": act_alloc,
            "actual_allocation_of_equity": act_alloc_of_equity,
            "requested_notional": req_notional,
            "actual_notional": actual_notional,
            "pnl_usd": pnl_usd,
            "pnl_btc": pnl_btc,
            "pnl_pct": float(r["pnl_pct"]),
            "ref_pnl_usd": float(r["pnl_usd_equiv"]),
            "scale": scale,
            "exposure_limited": exp_lim,
            "cap_25_hit": abs(req_alloc - MAX_ALLOCATION) < 1e-12,
            "raw_uncapped_allocation": raw_uncapped,
            "equity_before": equity,
            "reserved_before": reserved - actual_notional,
            "available_before": available,
            "n3": feats["n3"],
            "n7": feats["n7"],
            "n14": feats["n14"],
            "n30": feats["n30"],
            "n7_median": feats["n7_median"],
            "nblend": feats["nblend"],
            "warmup": warmup,
            "same_day_opp_count_excluded": feats["day_count_same_day_excluded"],
        }
        equity_curve.append(
            {
                "ts": ts,
                "equity_usd": equity,
                "event": "entry",
                "opportunity_id": oid,
                "reserved": reserved,
                "exposure": reserved / equity if equity else 0.0,
            }
        )

    if open_book:
        raise RuntimeError(f"Arm {arm.key}: {len(open_book)} trades still open at end")

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("exit_ts").reset_index(drop=True)

    executed = out[out.get("executed", True) == True] if not out.empty else out  # noqa: E712
    summary = {
        "arm": arm.key,
        "n_opportunities": int(len(trades)),
        "n_executed": int(len(executed)),
        "final_equity_usd": float(equity),
        "total_return_pct": 100.0 * (equity / STARTING_CAPITAL_USD - 1.0),
        "total_pnl_usd": float(equity - STARTING_CAPITAL_USD),
        "max_drawdown_pct": 100.0 * float(max_dd),
        "win_rate_pct": 100.0 * float((executed["pnl_usd"] > 0).mean()) if len(executed) else 0.0,
        "profit_factor": _profit_factor(executed["pnl_usd"]) if len(executed) else 0.0,
        "avg_allocation": float(executed["requested_allocation"].mean()) if len(executed) else BASE_ALLOCATION,
        "median_allocation": float(executed["requested_allocation"].median()) if len(executed) else BASE_ALLOCATION,
        "avg_actual_allocation": float(executed["actual_allocation"].mean()) if len(executed) else BASE_ALLOCATION,
        "median_actual_allocation": float(executed["actual_allocation"].median()) if len(executed) else BASE_ALLOCATION,
        "max_requested_allocation": float(max_req_alloc),
        "max_actual_allocation": float(max_act_alloc),
        "max_uncapped_requested_allocation": float(uncapped_req_peak),
        "max_exposure": float(max_exposure),
        "avg_exposure_at_entry": (
            float(
                (
                    (executed["reserved_before"] + executed["actual_notional"])
                    / executed["equity_before"].replace(0, np.nan)
                ).mean()
            )
            if len(executed)
            else 0.0
        ),
        "n_cap_25_requested_gt": int(cap_25_requested),
        "n_cap_25_binding": int(cap_25_actual),
        "pct_cap_25_binding": 100.0 * cap_25_actual / len(executed) if len(executed) else 0.0,
        "n_exposure_limited": int(exposure_limited),
        "pct_exposure_limited": 100.0 * exposure_limited / len(executed) if len(executed) else 0.0,
        "n_warmup_trades": int(executed["warmup"].sum()) if len(executed) and "warmup" in executed else 0,
    }
    return out, summary


def _raw_uncapped(arm_key: str, feats: dict[str, float]) -> float:
    """Requested allocation before the 25% cap (for binding diagnostics)."""
    def safe_inv(n):
        if n is None or not np.isfinite(n) or n <= 0:
            return float("inf")
        return 1.0 / float(n)

    def safe_sqrt(n):
        if n is None or not np.isfinite(n) or n <= 0:
            return float("inf")
        return BASE_ALLOCATION * float(np.sqrt(10.0 / float(n)))

    if arm_key == "FIXED_10":
        return BASE_ALLOCATION
    if arm_key == "3D_INVERSE":
        return max(BASE_ALLOCATION, safe_inv(feats["n3"]))
    if arm_key == "7D_INVERSE":
        return max(BASE_ALLOCATION, safe_inv(feats["n7"]))
    if arm_key == "14D_INVERSE":
        return max(BASE_ALLOCATION, safe_inv(feats["n14"]))
    if arm_key == "30D_INVERSE":
        return max(BASE_ALLOCATION, safe_inv(feats["n30"]))
    if arm_key == "7D_SQRT":
        return max(BASE_ALLOCATION, safe_sqrt(feats["n7"]))
    if arm_key == "7_30_SQRT":
        blend = 0.7 * feats["n7"] + 0.3 * feats["n30"]
        return max(BASE_ALLOCATION, safe_sqrt(blend))
    if arm_key == "7D_MEDIAN_SQRT":
        return max(BASE_ALLOCATION, safe_sqrt(feats["n7_median"]))
    if arm_key == "DEADBAND":
        blend = 0.7 * feats["n7"] + 0.3 * feats["n30"]
        if np.isfinite(blend) and blend >= 8.0:
            return BASE_ALLOCATION
        return max(BASE_ALLOCATION, safe_sqrt(blend))
    return BASE_ALLOCATION


def _profit_factor(pnl: pd.Series) -> float:
    pos = float(pnl[pnl > 0].sum())
    neg = float(-pnl[pnl < 0].sum())
    if neg <= 0:
        return float("inf") if pos > 0 else 0.0
    return pos / neg


def run_all_arms(
    ref_dir: Path,
) -> dict[str, Any]:
    opp, trades, ref_metrics = load_selector_E_stream(ref_dir)
    daily_counts = build_daily_opportunity_counts(opp)
    lookahead = verify_no_lookahead(daily_counts, trades)
    baseline = verify_fixed10_vs_reference(trades, ref_metrics)

    arm_specs = _build_arm_specs()
    trade_tables: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, Any]] = []

    # Always simulate FIXED_10 first (must match baseline)
    fixed_trades, fixed_sum = simulate_arm(trades, daily_counts, arm_specs["FIXED_10"])
    trade_tables["FIXED_10"] = fixed_trades
    summaries.append(fixed_sum)

    # Gate: FIXED_10 equity must match reference
    fixed_match = abs(fixed_sum["final_equity_usd"] - baseline["fixed10"]["final_equity_usd"]) <= 1e-6
    baseline["fixed10_sim_matches_replay"] = fixed_match
    baseline["fixed10_sim_final_equity_usd"] = fixed_sum["final_equity_usd"]

    if not baseline["passed"] or not fixed_match:
        return {
            "stopped": True,
            "reason": "FIXED_10 does not reproduce Selector E reference",
            "baseline_check": baseline,
            "lookahead_check": lookahead,
            "daily_counts": daily_counts,
            "trades": trades,
            "opportunities": opp,
            "ref_metrics": ref_metrics,
            "trade_tables": trade_tables,
            "summaries": summaries,
        }

    for key in ARM_ORDER:
        if key == "FIXED_10":
            continue
        tdf, sm = simulate_arm(trades, daily_counts, arm_specs[key])
        trade_tables[key] = tdf
        summaries.append(sm)

    return {
        "stopped": False,
        "baseline_check": baseline,
        "lookahead_check": lookahead,
        "daily_counts": daily_counts,
        "trades": trades,
        "opportunities": opp,
        "ref_metrics": ref_metrics,
        "trade_tables": trade_tables,
        "summaries": summaries,
        "ref_dir": str(ref_dir),
        "starting_capital_usd": STARTING_CAPITAL_USD,
        "base_allocation": BASE_ALLOCATION,
        "max_allocation": MAX_ALLOCATION,
        "zero_frequency_policy": (
            "If lookback mean/median frequency is 0 (or non-finite), requested "
            f"allocation is set to the {MAX_ALLOCATION:.0%} cap. No division by zero."
        ),
        "sizing_convention": (
            "requested_notional_usd = requested_allocation * starting_capital_usd "
            f"({STARTING_CAPITAL_USD:.0f}). FIXED_10 uses allocation=10% → ${BASE_NOTIONAL_USD:.0f} "
            "notional, reproducing the frozen reference (compound_notional=false). "
            "Adaptive arms only change this allocation. Exposure uses current equity; "
            "no leverage (open notional sum ≤ equity). PnL scaled from reference "
            "pnl_usd_equiv by actual_notional/100."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def make_results_dir(root: Path | None = None) -> Path:
    root = Path(root or "results")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = root / f"selector_E_adaptive_sizing_part4a_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "plots").mkdir(exist_ok=True)
    return out
