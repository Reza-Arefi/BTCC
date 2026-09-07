"""Independent safety monitor for Selector E-v1 live engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class SafetyState(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    HALT = "HALT"


@dataclass
class SafetyEvaluation:
    state: SafetyState
    reasons: list[str]
    metrics: dict[str, Any]


class SelectorSafetyMonitor:
    """Configurable safety layer — independent from selector E logic."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.thresholds = dict(cfg.get("thresholds") or {})
        self.halt_blocks_new_entries = bool(cfg.get("halt_blocks_new_entries", True))
        self._state = SafetyState(str(cfg.get("initial_state", "NORMAL")))

    @property
    def state(self) -> SafetyState:
        return self._state

    def allow_new_entries(self) -> bool:
        if not self.enabled:
            return True
        if self._state == SafetyState.HALT and self.halt_blocks_new_entries:
            return False
        return True

    def evaluate(
        self,
        e_legs: pd.DataFrame,
        *,
        regret_mean_pct: float | None = None,
        data_health_ok: bool = True,
        execution_ok: bool = True,
    ) -> SafetyEvaluation:
        reasons: list[str] = []
        metrics: dict[str, Any] = {}
        if not self.enabled:
            return SafetyEvaluation(SafetyState.NORMAL, [], {})

        if not data_health_ok:
            reasons.append("DATA_HEALTH_FAILURE")
        if not execution_ok:
            reasons.append("EXECUTION_FAILURE")

        if e_legs is not None and not e_legs.empty:
            pnl = pd.to_numeric(e_legs.get("pnl_pct"), errors="coerce").fillna(0.0)
            usd = pd.to_numeric(e_legs.get("pnl_usd_equiv"), errors="coerce").fillna(0.0)
            metrics["n_trades"] = int(len(e_legs))
            # consecutive losses (most recent first)
            streak = 0
            for v in reversed(pnl.tolist()):
                if v < 0:
                    streak += 1
                else:
                    break
            metrics["consecutive_losses"] = streak
            eq = 1000.0 + usd.cumsum()
            peak = eq.cummax()
            dd_pct = float(100 * (eq / peak - 1.0).min()) if len(eq) else 0.0
            metrics["max_dd_pct"] = round(dd_pct, 2)
            gp = float(usd[usd > 0].sum())
            gl = float(abs(usd[usd < 0].sum()))
            pf = gp / gl if gl > 1e-9 else float("inf")
            metrics["profit_factor"] = round(pf, 2) if math.isfinite(pf) else None

            th = self.thresholds
            if streak >= int(th.get("halt_consecutive_losses", 10)):
                reasons.append(f"CONSECUTIVE_LOSSES_HALT>={streak}")
            elif streak >= int(th.get("warning_consecutive_losses", 5)):
                reasons.append(f"CONSECUTIVE_LOSSES_WARNING>={streak}")

            if dd_pct <= -float(th.get("halt_rolling_dd_pct", 6.0)):
                reasons.append(f"DRAWDOWN_HALT<={dd_pct:.2f}%")
            elif dd_pct <= -float(th.get("warning_rolling_dd_pct", 3.0)):
                reasons.append(f"DRAWDOWN_WARNING<={dd_pct:.2f}%")

            if pf < float(th.get("halt_pf_below", 0.8)):
                reasons.append(f"PF_HALT<{pf:.2f}")
            elif pf < float(th.get("warning_pf_below", 1.0)):
                reasons.append(f"PF_WARNING<{pf:.2f}")

        if regret_mean_pct is not None:
            metrics["mean_regret_pct"] = regret_mean_pct
            if regret_mean_pct > float(self.thresholds.get("warning_mean_regret_pct", 0.5)):
                reasons.append(f"REGRET_WARNING>{regret_mean_pct:.3f}%")

        halt_reasons = [r for r in reasons if "HALT" in r or "DATA_HEALTH" in r or "EXECUTION" in r]
        warn_reasons = [r for r in reasons if r not in halt_reasons]

        if halt_reasons:
            new_state = SafetyState.HALT
        elif warn_reasons:
            new_state = SafetyState.WARNING
        else:
            new_state = SafetyState.NORMAL

        self._state = new_state
        return SafetyEvaluation(new_state, reasons, metrics)
