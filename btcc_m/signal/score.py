"""Signed S-score + NEW CROSS entry generation on absolute BTC."""

from __future__ import annotations

from typing import Any

import pandas as pd

from btcc.sim.score import combined_score, extract_factor_scores, normalize_weights
from btcc.sim.state_machine import CrossingStateMachine
from btcc_m.signal.factors import BASELINE_WEIGHTS, compute_btc_factors


def frozen_baseline_weights(raw: dict[str, float] | None = None) -> dict[str, float]:
    w = dict(BASELINE_WEIGHTS)
    if raw:
        for k, v in raw.items():
            if k in w:
                w[k] = float(v)
    w["btc_regime"] = 0.0
    return normalize_weights(w)


def score_at_bar(
    hist: pd.DataFrame,
    *,
    interval: str,
    weights: dict[str, float],
    momentum_profile: str = "e2",
) -> dict[str, Any]:
    factors = compute_btc_factors(hist, interval=interval, momentum_profile=momentum_profile)
    scores = extract_factor_scores(factors)
    return combined_score(scores, weights)


def generate_entries(
    signal_df: pd.DataFrame,
    *,
    interval: str,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    long_threshold: float = 0.65,
    weights: dict[str, float] | None = None,
    momentum_profile: str = "e2",
    warmup_bars: int = 1000,
    pair_key: str = "BTCUSDT",
) -> pd.DataFrame:
    """Walk closed signal bars; emit NEW CROSS entries filled at next bar open."""
    w = frozen_baseline_weights(weights)
    sm = CrossingStateMachine(long_threshold=float(long_threshold), max_open=1, one_per_pair=True)
    rows: list[dict[str, Any]] = []

    eval_start = pd.Timestamp(eval_start)
    if eval_start.tzinfo is None:
        eval_start = eval_start.tz_localize("UTC")
    eval_end = pd.Timestamp(eval_end)
    if eval_end.tzinfo is None:
        eval_end = eval_end.tz_localize("UTC")

    # Trailing window: e2 needs ~24h of bars; keep headroom for structure/ADX.
    max_lookback = max(int(warmup_bars), 1500)
    n = len(signal_df)
    for i in range(n - 1):
        if i + 1 < min(warmup_bars, 100):
            continue
        bar = signal_df.iloc[i]
        ts = pd.Timestamp(bar["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if ts > eval_end:
            break

        start_i = max(0, i + 1 - max_lookback)
        hist = signal_df.iloc[start_i : i + 1]
        scored = score_at_bar(hist, interval=interval, weights=w, momentum_profile=momentum_profile)
        S = float(scored["S"])
        decision = sm.evaluate(pair_key, S)

        # Warm FSM before eval_start so first in-window bar is not a false NEW CROSS.
        if ts < eval_start:
            continue
        if not decision["trade_suggested"]:
            continue

        nxt = signal_df.iloc[i + 1]
        entry_ts = pd.Timestamp(nxt["timestamp"])
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        if entry_ts > eval_end:
            continue

        opp_id = f"{pair_key}_{interval}_{entry_ts.strftime('%Y%m%d%H%M%S')}"
        # Entry list = all NEW CROSSes; engine enforces one-at-a-time capital.
        rows.append(
            {
                "opportunity_id": opp_id,
                "symbol": pair_key,
                "signal_ts": ts,
                "entry_ts": entry_ts,
                "entry_mid": float(nxt["open"]),
                "S": S,
                "interval": interval,
                **{f"contrib_{k}": float(v) for k, v in scored["contributions"].items()},
            }
        )

    return pd.DataFrame(rows)
