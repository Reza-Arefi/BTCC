"""Phase V — Ichimoku-only entry timing (no new indicators).

Triggers defer entry after a filtered base signal. Fixed exploratory windows
(not grid-searched): E1 waits exactly 1 bar; E2/E3 search up to MAX_WAIT bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

# Fixed research windows — not optimized.
MAX_WAIT_PULLBACK = 12
MAX_WAIT_CONTINUATION = 8

Family = Literal["primary", "secondary"]
TriggerId = Literal["E0", "E1", "E2", "E3"]


@dataclass
class TimingOutcome:
    base_idx: int
    entry_idx: int | None
    delay_bars: int | None
    status: str  # entered | missed_timeout | missed_confirm_fail | no_fill
    reason: str
    immediate_entry_px: float
    delayed_entry_px: float | None
    entry_penalty: float | None  # delayed / immediate - 1


def _next_open(df: pd.DataFrame, i: int) -> float | None:
    if i + 1 >= len(df):
        return None
    px = float(df.iloc[i + 1]["open"])
    return px if np.isfinite(px) else None


def _valid_row(df: pd.DataFrame, i: int) -> bool:
    row = df.iloc[i]
    return bool(
        pd.notna(row.get("tenkan"))
        and pd.notna(row.get("kijun"))
        and pd.notna(row.get("kumo_top"))
        and pd.notna(row.get("kumo_bot"))
    )


def resolve_immediate(df: pd.DataFrame, base_idxs: list[int]) -> list[TimingOutcome]:
    out: list[TimingOutcome] = []
    for i in base_idxs:
        imm = _next_open(df, i)
        if imm is None:
            out.append(
                TimingOutcome(i, None, None, "no_fill", "signal_at_last_bar", float("nan"), None, None)
            )
            continue
        out.append(
            TimingOutcome(
                base_idx=i,
                entry_idx=i,
                delay_bars=0,
                status="entered",
                reason="immediate",
                immediate_entry_px=imm,
                delayed_entry_px=imm,
                entry_penalty=0.0,
            )
        )
    return out


def resolve_one_bar_confirm_primary(df: pd.DataFrame, base_idxs: list[int]) -> list[TimingOutcome]:
    """E1 — next closed bar still holds breakout structure (above kumo + chikou + not over-extended)."""
    out: list[TimingOutcome] = []
    n = len(df)
    for i in base_idxs:
        imm = _next_open(df, i)
        if imm is None or not np.isfinite(imm):
            out.append(TimingOutcome(i, None, None, "no_fill", "no_immediate_ref", float("nan"), None, None))
            continue
        j = i + 1
        if j >= n or not _valid_row(df, j):
            out.append(
                TimingOutcome(i, None, None, "missed_timeout", "no_next_bar", imm, None, None)
            )
            continue
        row = df.iloc[j]
        ok = (
            float(row["close"]) > float(row["kumo_top"])
            and bool(row.get("chikou_above"))
            and pd.notna(row.get("price_kijun_atr"))
            and float(row["price_kijun_atr"]) <= 2.0
        )
        if not ok:
            out.append(
                TimingOutcome(
                    i, None, None, "missed_confirm_fail", "one_bar_structure_broke", imm, None, None
                )
            )
            continue
        delayed = _next_open(df, j)
        if delayed is None:
            out.append(TimingOutcome(i, None, 1, "no_fill", "confirm_at_last_bar", imm, None, None))
            continue
        out.append(
            TimingOutcome(
                i, j, 1, "entered", "one_bar_confirm", imm, delayed, delayed / imm - 1.0
            )
        )
    return out


def resolve_pullback_reclaim_primary(
    df: pd.DataFrame, base_idxs: list[int], max_wait: int = MAX_WAIT_PULLBACK
) -> list[TimingOutcome]:
    """E2 — after breakout, first pullback that reclaims above Kumo top with bullish close."""
    out: list[TimingOutcome] = []
    n = len(df)
    c = df["close"].astype(float).values
    o = df["open"].astype(float).values
    low = df["low"].astype(float).values
    top = df["kumo_top"].astype(float).values
    bot = df["kumo_bot"].astype(float).values
    for i in base_idxs:
        imm = _next_open(df, i)
        if imm is None or not np.isfinite(imm):
            out.append(TimingOutcome(i, None, None, "no_fill", "no_immediate_ref", float("nan"), None, None))
            continue
        found = False
        for j in range(i + 1, min(n, i + 1 + max_wait)):
            if not _valid_row(df, j):
                continue
            if not (np.isfinite(c[j]) and np.isfinite(top[j]) and np.isfinite(bot[j])):
                continue
            pulled = low[j] < c[i]
            reclaim = c[j] > top[j] and c[j] > o[j]
            held = c[j] >= bot[j]
            if pulled and reclaim and held:
                delayed = _next_open(df, j)
                if delayed is None:
                    out.append(
                        TimingOutcome(i, None, j - i, "no_fill", "confirm_at_last_bar", imm, None, None)
                    )
                else:
                    out.append(
                        TimingOutcome(
                            i,
                            j,
                            j - i,
                            "entered",
                            "pullback_reclaim",
                            imm,
                            delayed,
                            delayed / imm - 1.0,
                        )
                    )
                found = True
                break
        if not found:
            out.append(
                TimingOutcome(
                    i, None, None, "missed_timeout", "no_pullback_reclaim_in_window", imm, None, None
                )
            )
    return out


def resolve_kumo_kijun_retest_primary(
    df: pd.DataFrame, base_idxs: list[int], max_wait: int = MAX_WAIT_PULLBACK
) -> list[TimingOutcome]:
    """E3 — touch Kumo top or Kijun, then reclaim above both with bullish close."""
    out: list[TimingOutcome] = []
    n = len(df)
    c = df["close"].astype(float).values
    o = df["open"].astype(float).values
    low = df["low"].astype(float).values
    top = df["kumo_top"].astype(float).values
    kijun = df["kijun"].astype(float).values
    for i in base_idxs:
        imm = _next_open(df, i)
        if imm is None or not np.isfinite(imm):
            out.append(TimingOutcome(i, None, None, "no_fill", "no_immediate_ref", float("nan"), None, None))
            continue
        found = False
        for j in range(i + 1, min(n, i + 1 + max_wait)):
            if not _valid_row(df, j):
                continue
            if not all(np.isfinite(x) for x in (c[j], o[j], low[j], top[j], kijun[j])):
                continue
            touch = low[j] <= top[j] or low[j] <= kijun[j]
            reclaim = c[j] > top[j] and c[j] > kijun[j] and c[j] > o[j]
            if touch and reclaim:
                delayed = _next_open(df, j)
                if delayed is None:
                    out.append(
                        TimingOutcome(i, None, j - i, "no_fill", "confirm_at_last_bar", imm, None, None)
                    )
                else:
                    out.append(
                        TimingOutcome(
                            i,
                            j,
                            j - i,
                            "entered",
                            "kumo_kijun_retest_reclaim",
                            imm,
                            delayed,
                            delayed / imm - 1.0,
                        )
                    )
                found = True
                break
        if not found:
            out.append(
                TimingOutcome(
                    i, None, None, "missed_timeout", "no_retest_reclaim_in_window", imm, None, None
                )
            )
    return out


def resolve_one_bar_confirm_secondary(df: pd.DataFrame, base_idxs: list[int]) -> list[TimingOutcome]:
    """E1 — next bar still TK bullish above cloud with positive Kijun slope."""
    out: list[TimingOutcome] = []
    n = len(df)
    for i in base_idxs:
        imm = _next_open(df, i)
        if imm is None or not np.isfinite(imm):
            out.append(TimingOutcome(i, None, None, "no_fill", "no_immediate_ref", float("nan"), None, None))
            continue
        j = i + 1
        if j >= n or not _valid_row(df, j):
            out.append(TimingOutcome(i, None, None, "missed_timeout", "no_next_bar", imm, None, None))
            continue
        row = df.iloc[j]
        ok = (
            float(row["tenkan"]) > float(row["kijun"])
            and str(row.get("cloud_pos")) == "above"
            and float(row.get("kijun_slope") or 0) > 0
        )
        if not ok:
            out.append(
                TimingOutcome(
                    i, None, None, "missed_confirm_fail", "one_bar_tk_or_cloud_failed", imm, None, None
                )
            )
            continue
        delayed = _next_open(df, j)
        if delayed is None:
            out.append(TimingOutcome(i, None, 1, "no_fill", "confirm_at_last_bar", imm, None, None))
            continue
        out.append(
            TimingOutcome(i, j, 1, "entered", "one_bar_confirm", imm, delayed, delayed / imm - 1.0)
        )
    return out


def resolve_continuation_secondary(
    df: pd.DataFrame, base_idxs: list[int], max_wait: int = MAX_WAIT_CONTINUATION
) -> list[TimingOutcome]:
    """E2 — first later bar with higher close vs signal, TK still bullish above cloud."""
    out: list[TimingOutcome] = []
    n = len(df)
    c = df["close"].astype(float).values
    tenkan = df["tenkan"].astype(float).values
    kijun = df["kijun"].astype(float).values
    slope = df["kijun_slope"].astype(float).values
    cloud = df["cloud_pos"].astype(str).values
    for i in base_idxs:
        imm = _next_open(df, i)
        if imm is None or not np.isfinite(imm):
            out.append(TimingOutcome(i, None, None, "no_fill", "no_immediate_ref", float("nan"), None, None))
            continue
        found = False
        for j in range(i + 1, min(n, i + 1 + max_wait)):
            if not _valid_row(df, j):
                continue
            if not all(np.isfinite(x) for x in (c[j], tenkan[j], kijun[j], slope[j])):
                continue
            ok = (
                c[j] > c[i]
                and tenkan[j] > kijun[j]
                and cloud[j] == "above"
                and slope[j] > 0
            )
            if ok:
                delayed = _next_open(df, j)
                if delayed is None:
                    out.append(
                        TimingOutcome(i, None, j - i, "no_fill", "confirm_at_last_bar", imm, None, None)
                    )
                else:
                    out.append(
                        TimingOutcome(
                            i,
                            j,
                            j - i,
                            "entered",
                            "continuation_higher_close",
                            imm,
                            delayed,
                            delayed / imm - 1.0,
                        )
                    )
                found = True
                break
        if not found:
            out.append(
                TimingOutcome(
                    i, None, None, "missed_timeout", "no_continuation_in_window", imm, None, None
                )
            )
    return out


def resolve_kijun_hold_secondary(
    df: pd.DataFrame, base_idxs: list[int], max_wait: int = MAX_WAIT_PULLBACK
) -> list[TimingOutcome]:
    """E3 — pullback that touches Kijun and holds (close back above), TK still bullish above cloud."""
    out: list[TimingOutcome] = []
    n = len(df)
    c = df["close"].astype(float).values
    o = df["open"].astype(float).values
    low = df["low"].astype(float).values
    tenkan = df["tenkan"].astype(float).values
    kijun = df["kijun"].astype(float).values
    slope = df["kijun_slope"].astype(float).values
    cloud = df["cloud_pos"].astype(str).values
    for i in base_idxs:
        imm = _next_open(df, i)
        if imm is None or not np.isfinite(imm):
            out.append(TimingOutcome(i, None, None, "no_fill", "no_immediate_ref", float("nan"), None, None))
            continue
        found = False
        for j in range(i + 1, min(n, i + 1 + max_wait)):
            if not _valid_row(df, j):
                continue
            if not all(np.isfinite(x) for x in (c[j], o[j], low[j], tenkan[j], kijun[j], slope[j])):
                continue
            touch = low[j] <= kijun[j]
            hold = c[j] > kijun[j]
            struct = tenkan[j] > kijun[j] and cloud[j] == "above" and slope[j] > 0
            if touch and hold and struct:
                delayed = _next_open(df, j)
                if delayed is None:
                    out.append(
                        TimingOutcome(i, None, j - i, "no_fill", "confirm_at_last_bar", imm, None, None)
                    )
                else:
                    out.append(
                        TimingOutcome(
                            i,
                            j,
                            j - i,
                            "entered",
                            "kijun_hold",
                            imm,
                            delayed,
                            delayed / imm - 1.0,
                        )
                    )
                found = True
                break
        if not found:
            out.append(
                TimingOutcome(
                    i, None, None, "missed_timeout", "no_kijun_hold_in_window", imm, None, None
                )
            )
    return out


PRIMARY_TRIGGERS: dict[str, Callable[[pd.DataFrame, list[int]], list[TimingOutcome]]] = {
    "E0": resolve_immediate,
    "E1": resolve_one_bar_confirm_primary,
    "E2": resolve_pullback_reclaim_primary,
    "E3": resolve_kumo_kijun_retest_primary,
}

SECONDARY_TRIGGERS: dict[str, Callable[[pd.DataFrame, list[int]], list[TimingOutcome]]] = {
    "E0": resolve_immediate,
    "E1": resolve_one_bar_confirm_secondary,
    "E2": resolve_continuation_secondary,
    "E3": resolve_kijun_hold_secondary,
}

TRIGGER_LABELS = {
    "primary": {
        "E0": "immediate_breakout",
        "E1": "one_bar_confirm",
        "E2": "first_pullback_reclaim",
        "E3": "kumo_kijun_retest_reclaim",
    },
    "secondary": {
        "E0": "immediate_cross",
        "E1": "one_bar_confirm",
        "E2": "continuation_confirm",
        "E3": "kijun_hold_confirm",
    },
}


def base_signal_indices(
    panel: pd.DataFrame,
    fire: pd.Series,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> list[int]:
    def _utc(ts) -> pd.Timestamp:
        t = pd.Timestamp(ts)
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

    eval_start, eval_end = _utc(eval_start), _utc(eval_end)
    idxs: list[int] = []
    for i in range(len(panel)):
        if not bool(fire.iloc[i]):
            continue
        ts = _utc(panel.iloc[i]["timestamp"])
        if ts < eval_start or ts > eval_end:
            continue
        idxs.append(i)
    return idxs


def outcomes_to_fire(panel: pd.DataFrame, outcomes: list[TimingOutcome]) -> pd.Series:
    """Boolean fire on confirmation (or immediate) bars; one True per unique entry bar."""
    fire = pd.Series(False, index=panel.index)
    seen: set[int] = set()
    for oc in outcomes:
        if oc.status != "entered" or oc.entry_idx is None:
            continue
        if oc.entry_idx in seen:
            continue
        seen.add(oc.entry_idx)
        fire.iloc[oc.entry_idx] = True
    return fire


def summarize_timing(
    panel: pd.DataFrame,
    outcomes: list[TimingOutcome],
    *,
    horizon: int = 6,
) -> dict[str, Any]:
    """Aggregate delay / penalty / continuation / rejection quality vs immediate path."""
    col = f"fwd_{horizon}"
    entered = [o for o in outcomes if o.status == "entered" and o.entry_idx is not None]
    missed = [o for o in outcomes if o.status.startswith("missed") or o.status == "no_fill"]
    n_base = len(outcomes)
    n_entered = len(entered)
    n_missed = len(missed)

    delays = [o.delay_bars for o in entered if o.delay_bars is not None]
    penalties = [o.entry_penalty for o in entered if o.entry_penalty is not None and np.isfinite(o.entry_penalty)]

    # Forward from entry (confirmation) bar
    fwd_entry = []
    for o in entered:
        v = panel.iloc[o.entry_idx][col]  # type: ignore[index]
        if pd.notna(v):
            fwd_entry.append(float(v))

    # Counterfactual immediate forward from base bar (all base signals)
    fwd_base_all = []
    for o in outcomes:
        v = panel.iloc[o.base_idx][col]
        if pd.notna(v):
            fwd_base_all.append(float(v))

    # Missed signal forward (was skipping good or bad?)
    fwd_missed = []
    for o in missed:
        v = panel.iloc[o.base_idx][col]
        if pd.notna(v):
            fwd_missed.append(float(v))

    good_skip = int(sum(1 for x in fwd_missed if x <= 0))
    bad_skip = int(sum(1 for x in fwd_missed if x > 0))

    reason_counts: dict[str, int] = {}
    for o in outcomes:
        reason_counts[o.reason] = reason_counts.get(o.reason, 0) + 1

    return {
        "n_base": n_base,
        "n_entered": n_entered,
        "n_missed": n_missed,
        "pct_missed": (n_missed / n_base) if n_base else float("nan"),
        "mean_delay_bars": float(np.mean(delays)) if delays else float("nan"),
        "median_delay_bars": float(np.median(delays)) if delays else float("nan"),
        "mean_entry_penalty": float(np.mean(penalties)) if penalties else float("nan"),
        "median_entry_penalty": float(np.median(penalties)) if penalties else float("nan"),
        "mean_fwd_entry": float(np.mean(fwd_entry)) if fwd_entry else float("nan"),
        "median_fwd_entry": float(np.median(fwd_entry)) if fwd_entry else float("nan"),
        "p_continuation_entry": float(np.mean([x > 0 for x in fwd_entry])) if fwd_entry else float("nan"),
        "mean_fwd_base_all": float(np.mean(fwd_base_all)) if fwd_base_all else float("nan"),
        "mean_fwd_missed": float(np.mean(fwd_missed)) if fwd_missed else float("nan"),
        "good_skips": good_skip,
        "bad_skips": bad_skip,
        "skip_precision": (good_skip / n_missed) if n_missed else float("nan"),
        "reason_counts": reason_counts,
    }
