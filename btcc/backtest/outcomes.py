"""Future ALT/BTC return and outperformance labels."""

from __future__ import annotations

from typing import Any

import pandas as pd

from btcc.series.relative import horizon_bars


def future_outcomes(
    rel_close: pd.Series,
    decision_idx: int,
    interval: str = "15m",
    horizons: list[int] | None = None,
) -> dict[str, Any]:
    """Compute forward relative returns from decision_idx (no lookahead at decision time).

    R_h = close[t+h] / close[t] - 1
    outperformed = R_h > 0
    """
    horizons = horizons or [1, 4, 8, 12, 24]
    out: dict[str, Any] = {}
    px0 = float(rel_close.iloc[decision_idx])
    if px0 == 0:
        for h in horizons:
            out[f"future_return_{h}h"] = None
            out[f"outperformed_{h}h"] = None
        return out

    for h in horizons:
        bars = horizon_bars(h, interval)
        j = decision_idx + bars
        if j >= len(rel_close):
            out[f"future_return_{h}h"] = None
            out[f"outperformed_{h}h"] = None
            continue
        px1 = float(rel_close.iloc[j])
        ret = px1 / px0 - 1.0
        out[f"future_return_{h}h"] = ret
        out[f"outperformed_{h}h"] = int(ret > 0)

    return out
