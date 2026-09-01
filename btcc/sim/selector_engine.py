"""Counterfactual history and dynamic selectors A–F (chronological, no lookahead)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from btcc.sim.selector_config import (
    DEFAULT_STRATEGY_KEY,
    FIXED_ARM_LABELS,
    FIXED_STRATEGY_KEYS,
    SELECTOR_ARM_LABELS,
    SELECTOR_IDS,
    _label_for_key,
)

STRATEGY_INDEX = {k: i for i, k in enumerate(FIXED_STRATEGY_KEYS)}


@dataclass
class ClosedCounterfactual:
    opportunity_id: str
    strategy_key: str
    exit_ts: pd.Timestamp
    pnl_pct: float
    regime: str | None


@dataclass
class CounterfactualHistory:
    """Past T1–T12 counterfactual outcomes only (never selector legs)."""

    trades: list[ClosedCounterfactual] = field(default_factory=list)

    def record(self, *, opportunity_id: str, strategy_key: str, exit_ts, pnl_pct: float, regime: str | None) -> None:
        self.trades.append(
            ClosedCounterfactual(
                opportunity_id=str(opportunity_id),
                strategy_key=str(strategy_key),
                exit_ts=pd.Timestamp(exit_ts).tz_convert("UTC") if pd.Timestamp(exit_ts).tzinfo else pd.Timestamp(exit_ts).tz_localize("UTC"),
                pnl_pct=float(pnl_pct),
                regime=regime,
            )
        )

    def prior(
        self,
        strategy_key: str,
        asof: pd.Timestamp,
        *,
        regime: str | None = None,
        lookback_days: float | None = None,
    ) -> list[ClosedCounterfactual]:
        asof = pd.Timestamp(asof)
        if asof.tzinfo is None:
            asof = asof.tz_localize("UTC")
        else:
            asof = asof.tz_convert("UTC")
        out = [
            t for t in self.trades
            if t.strategy_key == strategy_key and t.exit_ts < asof and (regime is None or t.regime == regime)
        ]
        if lookback_days is not None and lookback_days > 0:
            cutoff = asof - pd.Timedelta(days=float(lookback_days))
            out = [t for t in out if t.exit_ts >= cutoff]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "trades": [
                {
                    "opportunity_id": t.opportunity_id,
                    "strategy_key": t.strategy_key,
                    "exit_ts": str(t.exit_ts),
                    "pnl_pct": t.pnl_pct,
                    "regime": t.regime,
                }
                for t in self.trades
            ]
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CounterfactualHistory:
        h = cls()
        for row in raw.get("trades") or []:
            h.record(
                opportunity_id=row["opportunity_id"],
                strategy_key=row["strategy_key"],
                exit_ts=row["exit_ts"],
                pnl_pct=float(row["pnl_pct"]),
                regime=row.get("regime"),
            )
        return h


def _ewma(values: list[tuple[pd.Timestamp, float]], asof: pd.Timestamp, half_life_seconds: float) -> float:
    if not values or half_life_seconds <= 0:
        return 0.0
    num = 0.0
    den = 0.0
    for ts, val in values:
        if ts >= asof:
            continue
        age = max(0.0, (asof - ts).total_seconds())
        w = math.pow(0.5, age / half_life_seconds)
        num += w * val
        den += w
    return num / den if den > 1e-12 else 0.0


def _strategy_returns(
    history: CounterfactualHistory,
    strategy_key: str,
    asof: pd.Timestamp,
    *,
    lookback_days: float | None = None,
) -> list[tuple[pd.Timestamp, float]]:
    return [(t.exit_ts, t.pnl_pct) for t in history.prior(strategy_key, asof, lookback_days=lookback_days)]


def _scores_ewma(
    history: CounterfactualHistory,
    asof: pd.Timestamp,
    half_life_days: float,
) -> dict[str, float]:
    hl = half_life_days * 86400.0
    return {k: _ewma(_strategy_returns(history, k, asof), asof, hl) for k in FIXED_STRATEGY_KEYS}


def _scores_multi_horizon(history: CounterfactualHistory, asof: pd.Timestamp, cfg: dict[str, Any]) -> dict[str, float]:
    w = cfg.get("weights") or {}
    w4 = float(w.get("h4", 0.15))
    w1 = float(w.get("d1", 0.30))
    w7 = float(w.get("d7", 0.55))
    hl4 = float(cfg.get("half_life_hours_4h", 4)) * 3600.0
    hl1 = float(cfg.get("half_life_days_1d", 1)) * 86400.0
    hl7 = float(cfg.get("half_life_days_7d", 7)) * 86400.0
    out: dict[str, float] = {}
    for k in FIXED_STRATEGY_KEYS:
        rets = _strategy_returns(history, k, asof)
        q = (
            w4 * _ewma(rets, asof, hl4)
            + w1 * _ewma(rets, asof, hl1)
            + w7 * _ewma(rets, asof, hl7)
        )
        out[k] = q
    return out


def _scores_regime(
    history: CounterfactualHistory,
    asof: pd.Timestamp,
    regime: str | None,
    *,
    half_life_days: float,
    min_obs: int,
    fallback: dict[str, float],
) -> dict[str, float]:
    hl = half_life_days * 86400.0
    out: dict[str, float] = {}
    for k in FIXED_STRATEGY_KEYS:
        prior = history.prior(k, asof, regime=regime)
        if len(prior) >= min_obs:
            rets = [(t.exit_ts, t.pnl_pct) for t in prior]
            out[k] = _ewma(rets, asof, hl)
        else:
            # Smooth fallback toward general recent performance
            blend = len(prior) / max(min_obs, 1)
            reg_score = _ewma([(t.exit_ts, t.pnl_pct) for t in prior], asof, hl) if prior else fallback.get(k, 0.0)
            out[k] = blend * reg_score + (1.0 - blend) * fallback.get(k, 0.0)
    return out


def _scores_rank_ewma(
    history: CounterfactualHistory,
    asof: pd.Timestamp,
    half_life_days: float,
    *,
    lookback_days: float | None = None,
    strategy_keys: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Lower score is better (EWMA of rank). Invert for argmax selection."""
    keys = strategy_keys or FIXED_STRATEGY_KEYS
    hl = half_life_days * 86400.0
    asof = pd.Timestamp(asof)
    if asof.tzinfo is None:
        asof = asof.tz_localize("UTC")
    else:
        asof = asof.tz_convert("UTC")
    cutoff = asof - pd.Timedelta(days=float(lookback_days)) if lookback_days else None
    events: dict[pd.Timestamp, dict[str, float]] = {}
    for t in history.trades:
        if t.exit_ts >= asof:
            continue
        if cutoff is not None and t.exit_ts < cutoff:
            continue
        events.setdefault(t.exit_ts, {})[t.strategy_key] = t.pnl_pct
    rank_samples: dict[str, list[tuple[pd.Timestamp, float]]] = {k: [] for k in keys}
    for ts in sorted(events.keys()):
        rets = {k: events[ts].get(k) for k in keys if k in events[ts]}
        if not rets:
            continue
        ordered = sorted(rets.items(), key=lambda x: x[1], reverse=True)
        rank_map = {k: i + 1 for i, (k, _) in enumerate(ordered)}
        for k in rank_map:
            if k in rank_samples:
                rank_samples[k].append((ts, float(rank_map[k])))
    return {k: -_ewma(rank_samples[k], asof, hl) for k in keys}


def _scores_downside_aware(
    history: CounterfactualHistory,
    asof: pd.Timestamp,
    *,
    half_life_days: float,
    downside_lambda: float,
) -> dict[str, float]:
    hl = half_life_days * 86400.0
    out: dict[str, float] = {}
    for k in FIXED_STRATEGY_KEYS:
        rets = _strategy_returns(history, k, asof)
        r = _ewma(rets, asof, hl)
        neg = [(ts, abs(v)) for ts, v in rets if v < 0]
        d = _ewma(neg, asof, hl) if neg else 0.0
        out[k] = r - float(downside_lambda) * d
    return out


def _pick_best(scores: dict[str, float]) -> tuple[str, float, str, float]:
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_k, best_v = ordered[0]
    second_k, second_v = ordered[1] if len(ordered) > 1 else (best_k, best_v)
    return best_k, best_v, second_k, second_v


@dataclass
class SelectorState:
    selector_id: str
    arm_label: str
    kind: str
    cfg: dict[str, Any]
    current_strategy_key: str = DEFAULT_STRATEGY_KEY
    last_switch_ts: pd.Timestamp | None = None
    n_switches: int = 0
    time_on_strategy_seconds: dict[str, float] = field(default_factory=dict)
    min_duration_hours: float = 6.0
    switch_margin: float = 0.0005
    lookback_days: float | None = None  # available history window (distinct from EWMA half-life)

    def compute_scores(
        self,
        history: CounterfactualHistory,
        asof: pd.Timestamp,
        regime: str | None,
        strategy_keys: tuple[str, ...] | None = None,
    ) -> dict[str, float]:
        se = self.cfg
        lb = self.lookback_days
        keys = strategy_keys
        if self.kind == "ewma_7d":
            return _scores_ewma(history, asof, float(se.get("half_life_days", 7)))
        if self.kind == "multi_horizon_ewma":
            return _scores_multi_horizon(history, asof, se)
        if self.kind == "regime_conditional":
            fb = _scores_ewma(history, asof, float(se.get("half_life_days", 7)))
            return _scores_regime(
                history, asof, regime,
                half_life_days=float(se.get("half_life_days", 7)),
                min_obs=int(se.get("min_regime_observations", 5)),
                fallback=fb,
            )
        if self.kind == "recent_plus_regime":
            recent = _scores_ewma(history, asof, float(se.get("half_life_days", 7)))
            regime_s = _scores_regime(
                history, asof, regime,
                half_life_days=float(se.get("half_life_days", 7)),
                min_obs=int(se.get("min_regime_observations", 5)),
                fallback=recent,
            )
            wr = float(se.get("recent_weight", 0.65))
            wg = float(se.get("regime_weight", 0.35))
            return {k: wr * recent[k] + wg * regime_s[k] for k in FIXED_STRATEGY_KEYS}
        if self.kind == "rank_ewma":
            return _scores_rank_ewma(
                history, asof, float(se.get("half_life_days", 7)),
                lookback_days=lb, strategy_keys=keys,
            )
        if self.kind == "downside_aware":
            return _scores_downside_aware(
                history, asof,
                half_life_days=float(se.get("half_life_days", 7)),
                downside_lambda=float(se.get("downside_lambda", 0.5)),
            )
        return _scores_ewma(history, asof, 7.0)

    def select(
        self,
        history: CounterfactualHistory,
        asof: pd.Timestamp,
        regime: str | None,
        strategy_keys: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        scores = self.compute_scores(history, asof, regime, strategy_keys=strategy_keys)
        if strategy_keys:
            scores = {k: scores.get(k, 0.0) for k in strategy_keys}
        candidate_k, candidate_v, second_k, second_v = _pick_best(scores)
        current_k = self.current_strategy_key
        current_v = scores.get(current_k, 0.0)
        switched = False
        if self.last_switch_ts is None:
            chosen = candidate_k
            switched = True
        else:
            asof_ts = pd.Timestamp(asof)
            if asof_ts.tzinfo is None:
                asof_ts = asof_ts.tz_localize("UTC")
            min_elapsed = (asof_ts - self.last_switch_ts).total_seconds() / 3600.0
            if min_elapsed < self.min_duration_hours:
                chosen = current_k
            elif candidate_k != current_k and candidate_v > current_v + self.switch_margin:
                chosen = candidate_k
                switched = True
            else:
                chosen = current_k
        if switched and (self.last_switch_ts is None or chosen != self.current_strategy_key):
            if self.last_switch_ts is not None:
                asof_ts = pd.Timestamp(asof)
                if asof_ts.tzinfo is None:
                    asof_ts = asof_ts.tz_localize("UTC")
                dt = (asof_ts - self.last_switch_ts).total_seconds()
                self.time_on_strategy_seconds[self.current_strategy_key] = (
                    self.time_on_strategy_seconds.get(self.current_strategy_key, 0.0) + dt
                )
                self.n_switches += 1
            self.current_strategy_key = chosen
            asof_ts = pd.Timestamp(asof)
            self.last_switch_ts = asof_ts.tz_localize("UTC") if asof_ts.tzinfo is None else asof_ts.tz_convert("UTC")
        ranks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rank_map = {k: i + 1 for i, (k, _) in enumerate(ranks)}
        return {
            "selector_id": self.selector_id,
            "arm_label": self.arm_label,
            "selected_strategy_key": chosen,
            "selected_arm_label": _label_for_key(chosen),
            "scores": {k: float(v) for k, v in scores.items()},
            "selected_score": float(scores[chosen]),
            "second_best_strategy_key": second_k,
            "second_best_score": float(second_v),
            "strategy_rank": int(rank_map.get(chosen, 0)),
            "switched": switched and chosen != current_k,
            "regime": regime,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector_id": self.selector_id,
            "arm_label": self.arm_label,
            "kind": self.kind,
            "cfg": self.cfg,
            "current_strategy_key": self.current_strategy_key,
            "last_switch_ts": str(self.last_switch_ts) if self.last_switch_ts is not None else None,
            "n_switches": self.n_switches,
            "time_on_strategy_seconds": dict(self.time_on_strategy_seconds),
            "min_duration_hours": self.min_duration_hours,
            "switch_margin": self.switch_margin,
            "lookback_days": self.lookback_days,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SelectorState:
        lst = raw.get("last_switch_ts")
        last_ts = None
        if lst:
            ts = pd.Timestamp(lst)
            last_ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        st = cls(
            selector_id=str(raw["selector_id"]),
            arm_label=str(raw["arm_label"]),
            kind=str(raw["kind"]),
            cfg=dict(raw.get("cfg") or {}),
            current_strategy_key=str(raw.get("current_strategy_key", DEFAULT_STRATEGY_KEY)),
            last_switch_ts=last_ts,
            n_switches=int(raw.get("n_switches", 0)),
            time_on_strategy_seconds=dict(raw.get("time_on_strategy_seconds") or {}),
            min_duration_hours=float(raw.get("min_duration_hours", 6)),
            switch_margin=float(raw.get("switch_margin", 0.0005)),
            lookback_days=float(raw["lookback_days"]) if raw.get("lookback_days") is not None else None,
        )
        return st


def build_selector_memory_group(lookbacks: dict[str, int], *, switching: dict[str, Any] | None = None) -> dict[str, SelectorState]:
    """Build E-3..E-90 rank_ewma selectors differing only by lookback_days."""
    sw = switching or {}
    min_h = float(sw.get("minimum_selection_duration_hours", 6))
    margin = float(sw.get("switch_margin", 0.0005))
    cfg = {"kind": "rank_ewma", "half_life_days": 7}
    out: dict[str, SelectorState] = {}
    for label, days in lookbacks.items():
        sid = f"selector_{label.lower().replace('-', '_')}"
        out[sid] = SelectorState(
            selector_id=sid,
            arm_label=label,
            kind="rank_ewma",
            cfg=dict(cfg),
            min_duration_hours=min_h,
            switch_margin=margin,
            lookback_days=float(days),
        )
    return out


def build_selector_group(sim: dict[str, Any]) -> dict[str, SelectorState]:
    se = sim.get("selector_experiment") or {}
    sel_cfg = se.get("selectors") or {}
    sw = se.get("switching") or {}
    min_h = float(sw.get("minimum_selection_duration_hours", 6))
    margin = float(sw.get("switch_margin", 0.0005))
    kind_map = {
        "A": "ewma_7d",
        "B": "multi_horizon_ewma",
        "C": "regime_conditional",
        "D": "recent_plus_regime",
        "E": "rank_ewma",
        "F": "downside_aware",
    }
    out: dict[str, SelectorState] = {}
    for label, sid in zip(SELECTOR_ARM_LABELS, SELECTOR_IDS):
        cfg = dict(sel_cfg.get(label) or {})
        cfg.setdefault("kind", kind_map[label])
        out[sid] = SelectorState(
            selector_id=sid,
            arm_label=label,
            kind=str(cfg.get("kind", kind_map[label])),
            cfg=cfg,
            min_duration_hours=min_h,
            switch_margin=margin,
        )
    return out


def build_selector_e(sl: dict[str, Any]) -> SelectorState:
    """Build single Selector E instance for live engine (baseline E-v1)."""
    sw = sl.get("switching") or {}
    cfg = dict(sl.get("selector") or {})
    cfg.setdefault("kind", "rank_ewma")
    return SelectorState(
        selector_id=str(sl.get("selector_id", "selector_e")),
        arm_label=str(sl.get("selector_arm_label", "E")),
        kind=str(cfg.get("kind", "rank_ewma")),
        cfg=cfg,
        min_duration_hours=float(sw.get("minimum_selection_duration_hours", 6)),
        switch_margin=float(sw.get("switch_margin", 0.0005)),
    )


def oracle_best_counterfactual(counterfactual_pnls: dict[str, float]) -> tuple[str, float]:
    if not counterfactual_pnls:
        return DEFAULT_STRATEGY_KEY, 0.0
    best_k = max(counterfactual_pnls, key=counterfactual_pnls.get)
    return best_k, float(counterfactual_pnls[best_k])
