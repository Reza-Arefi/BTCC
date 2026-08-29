"""Rank coins by 4h probability; attach late-entry classification."""

from __future__ import annotations

from typing import Any


def rank_signals(rows: list[dict[str, Any]], primary_horizon: int = 4) -> list[dict[str, Any]]:
    key = f"p_{primary_horizon}h"
    ranked = sorted(rows, key=lambda r: r.get(key, 0.0), reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return ranked
