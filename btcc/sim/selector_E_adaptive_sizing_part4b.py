"""Part 4B — Causal adaptive capital allocation for Selector E.

Corrects Part 4A's fixed normal_frequency=10 assumption.

  N*_D = expanding mean of daily eligible opportunity counts over ALL
         completed calendar days strictly before D (including zeros).
  N_recent = N7 / N7_median / Nblend (also lagged; day D excluded).
  R_D = N*_D / N_recent
  allocation = 10% * f(R_D), floored at 10%, capped at 25%.

Historical research only. Paper/live bot untouched.
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

from btcc.sim.selector_E_adaptive_sizing import (
    BASE_ALLOCATION,
    BASE_NOTIONAL_USD,
    MAX_ALLOCATION,
    MAX_OPEN,
    STARTING_CAPITAL_USD,
    ZERO_FREQ_ALLOCATION,
    _utc,
    build_daily_opportunity_counts,
    load_selector_E_stream,
    verify_fixed10_vs_reference,
)

logger = logging.getLogger(__name__)

# Minimum completed days before using expanding N* (causal warmup; no backfill).
NSTAR_MIN_DAYS = 7

ARM_ORDER = (
    "FIXED_10",
    "7D_INVERSE_CAUSAL",
    "7D_SQRT_CAUSAL",
    "7D_MEDIAN_SQRT_CAUSAL",
    "7_30_SQRT_CAUSAL",
    "7_30_INVERSE_CAUSAL",
    "DEADBAND_CAUSAL",
)

SCARCITY_BUCKETS = (
    ("R<0.75", 0.0, 0.75),
    ("0.75-1.00", 0.75, 1.00),
    ("1.00-1.25", 1.00, 1.25),
    ("1.25-1.50", 1.25, 1.50),
    ("R>1.50", 1.50, float("inf")),
)

PART4A_ARM_MAP = {
    "7D_INVERSE_CAUSAL": "7D_INVERSE",
    "7D_SQRT_CAUSAL": "7D_SQRT",
    "7D_MEDIAN_SQRT_CAUSAL": "7D_MEDIAN_SQRT",
    "7_30_SQRT_CAUSAL": "7_30_SQRT",
    "DEADBAND_CAUSAL": "DEADBAND",
}


def _clamp_alloc(x: float) -> float:
    if not np.isfinite(x) or x <= 0:
        return ZERO_FREQ_ALLOCATION
    return float(min(MAX_ALLOCATION, max(BASE_ALLOCATION, x)))


def _safe_ratio(nstar: float, n_recent: float) -> float:
    """R = N*/N_recent. Zero/non-finite recent → treat as infinite scarcity (cap)."""
    if nstar is None or not np.isfinite(nstar) or nstar < 0:
        return float("nan")
    if n_recent is None or not np.isfinite(n_recent) or n_recent <= 0:
        return float("inf")
    return float(nstar) / float(n_recent)


def _alloc_from_r(r: float, *, mode: str) -> tuple[float, float]:
    """Return (requested_allocation, raw_uncapped_allocation)."""
    if not np.isfinite(r):
        # inf scarcity or nan → request cap / floor respectively
        if r == float("inf"):
            return MAX_ALLOCATION, float("inf")
        return BASE_ALLOCATION, BASE_ALLOCATION
    if mode == "inverse":
        raw = BASE_ALLOCATION * r
    elif mode == "sqrt":
        raw = BASE_ALLOCATION * float(np.sqrt(max(r, 0.0)))
    elif mode == "deadband":
        if r <= 1.0:
            return BASE_ALLOCATION, BASE_ALLOCATION
        raw = BASE_ALLOCATION * float(np.sqrt(r))
    else:
        raise ValueError(mode)
    return _clamp_alloc(raw), float(max(BASE_ALLOCATION, raw))


@dataclass(frozen=True)
class ArmSpec:
    key: str
    warmup_days: int  # min completed prior days required
    needs_n30: bool
    mode: str  # inverse | sqrt | deadband | fixed
    recent_key: str  # n7 | n7_median | nblend | none


def _build_arm_specs() -> dict[str, ArmSpec]:
    return {
        "FIXED_10": ArmSpec("FIXED_10", 0, False, "fixed", "none"),
        "7D_INVERSE_CAUSAL": ArmSpec("7D_INVERSE_CAUSAL", 7, False, "inverse", "n7"),
        "7D_SQRT_CAUSAL": ArmSpec("7D_SQRT_CAUSAL", 7, False, "sqrt", "n7"),
        "7D_MEDIAN_SQRT_CAUSAL": ArmSpec("7D_MEDIAN_SQRT_CAUSAL", 7, False, "sqrt", "n7_median"),
        "7_30_SQRT_CAUSAL": ArmSpec("7_30_SQRT_CAUSAL", 30, True, "sqrt", "nblend"),
        "7_30_INVERSE_CAUSAL": ArmSpec("7_30_INVERSE_CAUSAL", 30, True, "inverse", "nblend"),
        "DEADBAND_CAUSAL": ArmSpec("DEADBAND_CAUSAL", 30, True, "deadband", "nblend"),
    }


def causal_frequency_features(daily_counts: pd.Series, trade_day: pd.Timestamp) -> dict[str, Any]:
    """Lagged features for day D: N*, N7, N30, N7_median, blends — day D excluded.

    N*_D = mean(count[d] for all completed calendar days d < D in the series),
    including zero-opportunity days.
    """
    d = _utc(trade_day).floor("D")
    idx = daily_counts.index
    prior = daily_counts[idx < d]

    def window_stats(n: int) -> tuple[float, float]:
        end = d - pd.Timedelta(days=1)
        start = d - pd.Timedelta(days=n)
        w = daily_counts[(idx >= start) & (idx <= end)]
        if len(w) < n:
            return float("nan"), float("nan")
        return float(w.mean()), float(w.median())

    n_prior = int(len(prior))
    nstar = float(prior.mean()) if n_prior >= NSTAR_MIN_DAYS else float("nan")
    n7, n7_med = window_stats(7)
    n30, _ = window_stats(30)
    nblend = (
        float("nan")
        if (not np.isfinite(n7) or not np.isfinite(n30))
        else (0.7 * n7 + 0.3 * n30)
    )

    r7 = _safe_ratio(nstar, n7)
    r7_med = _safe_ratio(nstar, n7_med)
    rblend = _safe_ratio(nstar, nblend)

    return {
        "trade_day": d,
        "n_prior_days": n_prior,
        "nstar": nstar,
        "n7": n7,
        "n7_median": n7_med,
        "n30": n30,
        "nblend": nblend,
        "r_n7": r7,
        "r_n7_median": r7_med,
        "r_nblend": rblend,
        "same_day_count_excluded": float(daily_counts.get(d, 0.0)),
        "nstar_includes_day_d": False,  # explicit audit flag
    }


def verify_nstar_excludes_day_d(daily_counts: pd.Series, trades: pd.DataFrame) -> dict[str, Any]:
    violations = 0
    samples = []
    for _, r in trades.head(80).iterrows():
        d = _utc(r["entry_ts"]).floor("D")
        feats = causal_frequency_features(daily_counts, d)
        prior = daily_counts[daily_counts.index < d]
        if d in prior.index:
            violations += 1
        # Recompute N* manually
        if len(prior) >= NSTAR_MIN_DAYS:
            manual = float(prior.mean())
            if abs(manual - feats["nstar"]) > 1e-12:
                violations += 1
        samples.append(
            {
                "entry_ts": str(r["entry_ts"]),
                "trade_day": str(d),
                "nstar": feats["nstar"],
                "n7": feats["n7"],
                "r_n7": feats["r_n7"],
                "same_day_excluded": feats["same_day_count_excluded"],
                "n_prior_days": feats["n_prior_days"],
            }
        )
    # Also verify zeros are included: compare mean of all prior days vs mean of prior days with count>0
    d_mid = daily_counts.index[len(daily_counts) // 2]
    prior = daily_counts[daily_counts.index < d_mid]
    zeros_ok = True
    if len(prior):
        with_zeros = float(prior.mean())
        nonzero_only = float(prior[prior > 0].mean()) if (prior > 0).any() else float("nan")
        zeros_ok = (prior == 0).any() and (with_zeros < nonzero_only - 1e-12 or not np.isfinite(nonzero_only))
    return {
        "violations": violations,
        "ok": violations == 0,
        "zeros_included_in_nstar": bool(zeros_ok),
        "nstar_definition": (
            f"N*_D = mean(daily_eligible_opportunity_count) over ALL completed calendar "
            f"days strictly before D present in the opportunity-span series, including "
            f"zero-opportunity days. Warmup: require >= {NSTAR_MIN_DAYS} prior days. "
            "Day D is never included."
        ),
        "samples": samples,
    }


def _request_allocation(arm: ArmSpec, feats: dict[str, Any]) -> tuple[float, float, float, bool]:
    """Return req_alloc, raw_uncapped, scarcity_ratio_used, warmup."""
    if arm.mode == "fixed":
        return BASE_ALLOCATION, BASE_ALLOCATION, float("nan"), False

    warmup = feats["n_prior_days"] < max(arm.warmup_days, NSTAR_MIN_DAYS)
    if arm.needs_n30 and not np.isfinite(feats.get("n30", float("nan"))):
        warmup = True
    if arm.recent_key == "n7" and not np.isfinite(feats.get("n7", float("nan"))):
        warmup = True
    if arm.recent_key == "n7_median" and not np.isfinite(feats.get("n7_median", float("nan"))):
        warmup = True
    if arm.recent_key == "nblend" and not np.isfinite(feats.get("nblend", float("nan"))):
        warmup = True
    if not np.isfinite(feats.get("nstar", float("nan"))):
        warmup = True

    if warmup:
        return BASE_ALLOCATION, BASE_ALLOCATION, float("nan"), True

    if arm.recent_key == "n7":
        r = feats["r_n7"]
    elif arm.recent_key == "n7_median":
        r = feats["r_n7_median"]
    else:
        r = feats["r_nblend"]

    req, raw = _alloc_from_r(r, mode=arm.mode)
    return req, raw, float(r) if np.isfinite(r) or r == float("inf") else float("nan"), False


def simulate_arm(
    trades: pd.DataFrame,
    daily_counts: pd.Series,
    arm: ArmSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    equity = STARTING_CAPITAL_USD
    reserved = 0.0
    open_book: dict[str, dict[str, Any]] = {}

    trade_days = {_utc(ts).floor("D") for ts in trades["entry_ts"]}
    feat_cache = {d: causal_frequency_features(daily_counts, d) for d in trade_days}

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
    uncapped_peak = 0.0

    for ts, _, i, kind in events:
        r = trades.loc[i]
        oid = str(r["opportunity_id"])
        if kind == "exit":
            pos = open_book.pop(oid, None)
            if pos is None:
                continue
            reserved = max(0.0, reserved - float(pos["actual_notional"]))
            equity += float(pos["pnl_usd"])
            peak = max(peak, equity)
            dd = (equity - peak) / peak if peak else 0.0
            max_dd = min(max_dd, dd)
            rows.append({**pos, "equity_after": equity, "drawdown_pct": 100.0 * dd})
            continue

        if len(open_book) >= MAX_OPEN:
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
                    "warmup": True,
                    **{k: feats[k] for k in ("nstar", "n7", "n7_median", "n30", "nblend", "r_n7", "r_nblend")},
                    "scarcity_ratio": float("nan"),
                }
            )
            continue

        d = _utc(r["entry_ts"]).floor("D")
        feats = feat_cache[d]
        req_alloc, raw_uncapped, scarcity_r, warmup = _request_allocation(arm, feats)

        if raw_uncapped > MAX_ALLOCATION + 1e-15:
            cap_25_requested += 1
        max_req_alloc = max(max_req_alloc, req_alloc)
        if np.isfinite(raw_uncapped):
            uncapped_peak = max(uncapped_peak, raw_uncapped)

        req_notional = req_alloc * STARTING_CAPITAL_USD
        available = max(0.0, equity - reserved)
        actual_notional = min(req_notional, available)
        exp_lim = actual_notional + 1e-9 < req_notional
        if exp_lim:
            exposure_limited += 1
        act_alloc = actual_notional / STARTING_CAPITAL_USD if STARTING_CAPITAL_USD else 0.0
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
            "actual_allocation_of_equity": (actual_notional / equity) if equity > 0 else 0.0,
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
            "nstar": feats["nstar"],
            "n7": feats["n7"],
            "n7_median": feats["n7_median"],
            "n30": feats["n30"],
            "nblend": feats["nblend"],
            "r_n7": feats["r_n7"],
            "r_n7_median": feats["r_n7_median"],
            "r_nblend": feats["r_nblend"],
            "scarcity_ratio": scarcity_r,
            "warmup": warmup,
            "same_day_opp_count_excluded": feats["same_day_count_excluded"],
            "n_prior_days": feats["n_prior_days"],
            "nstar_includes_day_d": False,
        }

    if open_book:
        raise RuntimeError(f"Arm {arm.key}: {len(open_book)} trades still open")

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("exit_ts").reset_index(drop=True)
    executed = out[out["executed"] == True] if not out.empty else out  # noqa: E712

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
        "max_requested_allocation": float(max_req_alloc),
        "max_actual_allocation": float(max_act_alloc),
        "max_uncapped_requested_allocation": float(uncapped_peak) if uncapped_peak else BASE_ALLOCATION,
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
        "n_warmup_trades": int(executed["warmup"].sum()) if len(executed) else 0,
        "pct_at_10": 100.0 * float((executed["requested_allocation"] <= BASE_ALLOCATION + 1e-12).mean())
        if len(executed)
        else 100.0,
        "pct_above_10": 100.0 * float((executed["requested_allocation"] > BASE_ALLOCATION + 1e-12).mean())
        if len(executed)
        else 0.0,
        "avg_scarcity_ratio": float(pd.to_numeric(executed.get("scarcity_ratio"), errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).mean())
        if len(executed)
        else float("nan"),
        "avg_nstar": float(pd.to_numeric(executed.get("nstar"), errors="coerce").mean())
        if len(executed)
        else float("nan"),
    }
    return out, summary


def _profit_factor(pnl: pd.Series) -> float:
    pos = float(pnl[pnl > 0].sum())
    neg = float(-pnl[pnl < 0].sum())
    if neg <= 0:
        return float("inf") if pos > 0 else 0.0
    return pos / neg


def build_causal_frequency_history(daily_counts: pd.Series) -> pd.DataFrame:
    """Per-day audit table of lagged N*, recent freqs, and scarcity ratios."""
    rows = []
    for d in daily_counts.index:
        feats = causal_frequency_features(daily_counts, d)
        # Allocations for each non-fixed arm at this day (for audit)
        specs = _build_arm_specs()
        row = {
            "date": d,
            "opp_count_day": float(daily_counts.loc[d]),
            "historical_normal_Nstar": feats["nstar"],
            "recent_N7": feats["n7"],
            "recent_N30": feats["n30"],
            "recent_N7_median": feats["n7_median"],
            "nblend": feats["nblend"],
            "scarcity_ratio_N7": feats["r_n7"],
            "scarcity_ratio_N7_median": feats["r_n7_median"],
            "scarcity_ratio_Nblend": feats["r_nblend"],
            "n_prior_days": feats["n_prior_days"],
            "nstar_includes_day_d": False,
            "same_day_excluded_from_features": True,
        }
        for key, arm in specs.items():
            if key == "FIXED_10":
                row[f"req_alloc_{key}"] = BASE_ALLOCATION
                continue
            req, _, _, warm = _request_allocation(arm, feats)
            row[f"req_alloc_{key}"] = req
            row[f"warmup_{key}"] = warm
        rows.append(row)
    return pd.DataFrame(rows)


def run_all_arms(ref_dir: Path) -> dict[str, Any]:
    opp, trades, ref_metrics = load_selector_E_stream(ref_dir)
    daily_counts = build_daily_opportunity_counts(opp)
    causality = verify_nstar_excludes_day_d(daily_counts, trades)
    baseline = verify_fixed10_vs_reference(trades, ref_metrics)

    arm_specs = _build_arm_specs()
    trade_tables: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, Any]] = []

    fixed_trades, fixed_sum = simulate_arm(trades, daily_counts, arm_specs["FIXED_10"])
    trade_tables["FIXED_10"] = fixed_trades
    summaries.append(fixed_sum)
    fixed_match = abs(fixed_sum["final_equity_usd"] - baseline["fixed10"]["final_equity_usd"]) <= 1e-6
    baseline["fixed10_sim_matches_replay"] = fixed_match
    baseline["fixed10_sim_final_equity_usd"] = fixed_sum["final_equity_usd"]

    if not baseline["passed"] or not fixed_match:
        return {
            "stopped": True,
            "reason": "FIXED_10 does not reproduce Selector E reference",
            "baseline_check": baseline,
            "causality_check": causality,
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

    hist = build_causal_frequency_history(daily_counts)

    return {
        "stopped": False,
        "baseline_check": baseline,
        "causality_check": causality,
        "daily_counts": daily_counts,
        "trades": trades,
        "opportunities": opp,
        "ref_metrics": ref_metrics,
        "trade_tables": trade_tables,
        "summaries": summaries,
        "causal_frequency_history": hist,
        "ref_dir": str(ref_dir),
        "nstar_min_days": NSTAR_MIN_DAYS,
        "nstar_definition": causality["nstar_definition"],
        "zero_frequency_policy": (
            "If N_recent is 0 (or non-finite), scarcity ratio is treated as infinite and "
            f"requested allocation is set to the {MAX_ALLOCATION:.0%} cap. No division by zero."
        ),
        "sizing_convention": (
            "requested_notional_usd = requested_allocation * starting_capital_usd "
            f"({STARTING_CAPITAL_USD:.0f}). FIXED_10 = 10% → ${BASE_NOTIONAL_USD:.0f} reproduces "
            "frozen reference (compound_notional=false). Adaptive: allocation = 10% * f(R) "
            "with R = N*_D / N_recent, floor 10%, cap 25%. Exposure uses current equity; no leverage."
        ),
        "warmup_policy": (
            f"Before {NSTAR_MIN_DAYS} completed prior days, or before the arm's required "
            "lookback (7 for N7 arms, 30 for N30/blend arms) is available, allocation = 10%. "
            "No historical backfill."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def make_results_dir(root: Path | None = None) -> Path:
    root = Path(root or "results")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = root / f"selector_E_adaptive_sizing_part4b_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "plots").mkdir(exist_ok=True)
    return out
