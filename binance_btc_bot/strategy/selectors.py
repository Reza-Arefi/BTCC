"""Selector definitions for research compatibility.

LIVE_SELECTOR = None. These functions are intentionally isolated from live
execution so A–F can be activated later after Binance backtest validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


SELECTOR_KINDS = {
    "A": "ewma_7d",
    "B": "multi_horizon_ewma",
    "C": "regime_conditional",
    "D": "recent_plus_regime",
    "E": "rank_ewma",
    "F": "downside_aware",
}


@dataclass(frozen=True)
class SelectorSpec:
    key: str
    kind: str
    params: dict[str, Any]


def load_selector_specs(cfg: Mapping[str, Any] | None) -> dict[str, SelectorSpec]:
    raw = (cfg or {}).get("selectors") or {}
    out: dict[str, SelectorSpec] = {}
    for key, kind in SELECTOR_KINDS.items():
        params = dict(raw.get(key) or {})
        params.setdefault("kind", kind)
        out[key] = SelectorSpec(key=key, kind=str(params.get("kind") or kind), params=params)
    return out


def live_selector_disabled() -> None:
    """Live path must not call selectors. Explicit sentinel."""
    return None


def select_strategy_for_research(
    selector_key: str,
    *,
    candidate_scores: Mapping[str, float],
    default: str = "T1",
) -> str:
    """Research helper: pick argmax strategy key from provided scores.

    Not used by the live engine.
    """
    key = str(selector_key).upper()
    if key not in SELECTOR_KINDS:
        raise KeyError(f"unknown selector {selector_key}")
    if not candidate_scores:
        return default
    return max(candidate_scores.items(), key=lambda kv: float(kv[1]))[0]
