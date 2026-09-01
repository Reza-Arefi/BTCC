"""Selector E-v1 live paper engine — E selector + T1–T10 counterfactuals per opportunity."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from btcc.monitoring.selector_safety import SelectorSafetyMonitor, SafetyState
from btcc.monitoring.selector_windows import selection_concentration_report
from btcc.sim.accounting import CostModel
from btcc.sim.exits import (
    StrategySpec,
    leg_to_record,
    open_opportunity_legs,
    process_bars_until_closed,
    rebuild_leg_from_snapshot,
    specs_from_config,
)
from btcc.sim.health import evaluate_health
from btcc.sim.regime import classify_regime
from btcc.sim.score import FACTOR_KEYS, combined_score, extract_factor_scores, static_factor_weights
from btcc.sim.selector_engine import CounterfactualHistory, SelectorState, build_selector_e, oracle_best_counterfactual
from btcc.sim.selector_live_config import LIVE_ARM_LABELS, LIVE_TRAIL_KEYS, version_manifest
from btcc.sim.selector_live_store import SelectorLiveStore
from btcc.sim.state_machine import CrossingStateMachine
from btcc.sim.trail_entry import evaluate_trail_entry
from btcc.telegram.selector_trades import SelectorTradeNotifier

logger = logging.getLogger(__name__)


def _label_for_key(key: str) -> str:
    if key.startswith("trail_"):
        return f"T{key.split('_')[1]}"
    return key


def _s_band(s: float, bands: list[dict[str, Any]]) -> str:
    for b in bands:
        lo = float(b["lo"])
        hi = b.get("hi")
        if hi is None and s >= lo:
            return str(b["label"])
        if hi is not None and lo <= s < float(hi):
            return str(b["label"])
    return "unknown"


class SelectorLiveEngine:
    """Live/paper engine: Selector E chooses T1–T10; counterfactuals always recorded."""

    def __init__(self, signal_cfg: dict[str, Any], sim_cfg: dict[str, Any], tg_send=None):
        self.signal_cfg = signal_cfg
        self.sim = sim_cfg
        self.enabled = bool(sim_cfg.get("enabled", False))
        self.store = SelectorLiveStore(sim_cfg)
        self.sm = self.store.load_state_machine(sim_cfg)
        self.sm.long_threshold = float(sim_cfg.get("long_threshold", 0.60))
        self.sm.upper_threshold = sim_cfg.get("upper_threshold")
        self.sm.max_open = int(sim_cfg.get("max_open_opportunities", 10))
        self.costs = CostModel(
            fee_rate_per_side=float(sim_cfg.get("fee_rate_per_side", 0.001)),
            slippage_rate_per_side=float(sim_cfg.get("slippage_rate_per_side", 0.0005)),
        )
        self.specs = specs_from_config(sim_cfg)
        self.spec_by_key = {s.key: s for s in self.specs}
        self.cf_specs = [self.spec_by_key[k] for k in LIVE_TRAIL_KEYS if k in self.spec_by_key]
        self.weights = static_factor_weights({**signal_cfg, "sim": sim_cfg})
        sl_raw = sim_cfg.get("selector_live_raw") or {}
        self.selector: SelectorState = build_selector_e(sl_raw)
        self.cf_history = CounterfactualHistory()
        self.history_recorded: set[tuple[str, str]] = set()
        rt = self.store.load_runtime_state()
        sid = str(sl_raw.get("selector_id", "selector_e"))
        if rt:
            self.cf_history = CounterfactualHistory.from_dict(rt.get("counterfactual_history") or {})
            if sid in (rt.get("selectors") or {}):
                self.selector = SelectorState.from_dict(rt["selectors"][sid])
        self.regime_rules = (sl_raw.get("regime") or {}).get("rules") or {}
        self.s_bands = list(sl_raw.get("s_bands") or [])
        self.open_opps: list[dict[str, Any]] = []
        self._restore_open_book()
        self.trade_notifier = SelectorTradeNotifier(tg_send or (lambda _t: False), enabled=True)
        self.safety = SelectorSafetyMonitor(sim_cfg.get("safety") or {})
        self._prev_safety_state = self.safety.state
        self.version = version_manifest(sim_cfg)
        tg_cfg = sim_cfg.get("telegram_live") or {}
        self._notify_open = bool(tg_cfg.get("send_trade_open", True))
        self._notify_close = bool(tg_cfg.get("send_trade_close", True))

    def _restore_open_book(self) -> None:
        all_specs = self.specs
        for opp in self.store.open_opportunities():
            leg_items = []
            for snap in opp.get("legs_snapshot") or []:
                leg = rebuild_leg_from_snapshot(snap, all_specs)
                if leg is None:
                    continue
                arm = snap.get("arm_key") or _label_for_key(leg.spec.key)
                leg_items.append({
                    "leg": leg,
                    "arm_key": arm,
                    "is_counterfactual": bool(snap.get("is_counterfactual")),
                    "selector_id": snap.get("selector_id"),
                })
            opp = dict(opp)
            opp["leg_items"] = leg_items
            opp.pop("legs_snapshot", None)
            self.open_opps.append(opp)
            if opp.get("status") in ("OPEN", "PENDING_ENTRY") and opp.get("symbol") and opp.get("opportunity_id"):
                self.sm.register_open(opp["symbol"], opp["opportunity_id"])

    def notify_startup(self) -> None:
        logger.info(
            "Selector E-v1 live engine started | version=%s | threshold=%.2f | cf=%s",
            self.version.get("selector_version"),
            self.sm.long_threshold,
            LIVE_TRAIL_KEYS,
        )

    def _persist(self) -> None:
        serializable = []
        for opp in self.open_opps:
            row = {k: v for k, v in opp.items() if k != "leg_items"}
            snaps = []
            for item in opp.get("leg_items") or []:
                rec = leg_to_record(item["leg"], opp["opportunity_id"])
                rec["arm_key"] = item["arm_key"]
                rec["is_counterfactual"] = item.get("is_counterfactual")
                rec["selector_id"] = item.get("selector_id")
                snaps.append(rec)
            row["legs_snapshot"] = snaps
            serializable.append(row)
        self.store.set_open_opportunities(serializable)
        self.store.save_state_machine(self.sm, extra={"version": self.version})
        self.store.save_runtime_state({
            "counterfactual_history": self.cf_history.to_dict(),
            "selectors": {self.selector.selector_id: self.selector.to_dict()},
            "version": self.version,
        })

    def _advance_open(self, rel_panels: dict[str, pd.DataFrame], btc_df: pd.DataFrame) -> list[dict]:
        closed_rows: list[dict] = []
        btc = btc_df.copy()
        btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True)
        btc_close = btc.set_index("timestamp")["close"]
        still: list[dict] = []

        for opp in self.open_opps:
            sym = opp.get("symbol")
            rel = rel_panels.get(sym)
            if rel is None or rel.empty:
                still.append(opp)
                continue
            rel = rel.copy()
            rel["timestamp"] = pd.to_datetime(rel["timestamp"], utc=True)
            last = pd.Timestamp(opp.get("last_processed_ts") or opp.get("entry_fill_ts"))
            if last.tzinfo is None:
                last = last.tz_localize("UTC")
            bars = rel[(rel["timestamp"] > last) & (rel["timestamp"] <= btc.iloc[-1]["timestamp"])]
            if bars.empty and opp.get("status") != "PENDING_ENTRY":
                still.append(opp)
                continue

            if opp.get("status") == "PENDING_ENTRY":
                self._fill_entry(opp, rel, float(btc_close.iloc[-1]))
                if opp.get("status") == "PENDING_ENTRY":
                    still.append(opp)
                    continue

            legs_only = [it["leg"] for it in opp.get("leg_items") or []]
            if bars.empty:
                still.append(opp)
                continue
            process_bars_until_closed(
                legs_only,
                bars,
                btc_usdt_series=btc_close,
                default_btc_usdt=float(btc_close.iloc[-1]),
                costs=self.costs,
                same_candle_conflict=str(self.sim.get("same_candle_conflict", "assume_sl_first")),
            )
            opp["last_processed_ts"] = str(bars.iloc[-1]["timestamp"])
            oid = opp["opportunity_id"]
            regime = opp.get("regime")
            for item in opp.get("leg_items") or []:
                leg = item["leg"]
                if not leg.closed:
                    continue
                sk = leg.spec.key
                key = (str(oid), sk)
                if item.get("is_counterfactual") and key not in self.history_recorded:
                    self.cf_history.record(
                        opportunity_id=oid,
                        strategy_key=sk,
                        exit_ts=leg.exit_ts,
                        pnl_pct=float((leg.exit_result or {}).get("pnl_pct") or 0.0),
                        regime=regime,
                    )
                    self.history_recorded.add(key)

            if all(it["leg"].closed for it in opp.get("leg_items") or []):
                cf_pnls: dict[str, float] = {}
                for item in opp.get("leg_items") or []:
                    leg = item["leg"]
                    rec = leg_to_record(leg, oid)
                    rec["arm_key"] = item["arm_key"]
                    rec["is_counterfactual"] = bool(item.get("is_counterfactual"))
                    rec["selector_id"] = item.get("selector_id")
                    rec["regime"] = regime
                    rec["s_band"] = opp.get("s_band")
                    rec["closed"] = True
                    rec.update({k: self.version.get(k) for k in ("bot_version", "selector_version") if k in self.version})
                    closed_rows.append(rec)
                    if item.get("is_counterfactual"):
                        cf_pnls[leg.spec.key] = float((leg.exit_result or {}).get("pnl_pct") or 0.0)
                best_k, best_v = oracle_best_counterfactual(cf_pnls)
                e_rec = next((r for r in closed_rows if r.get("opportunity_id") == oid and r.get("arm_key") == "E"), None)
                if e_rec and self._notify_close:
                    self.trade_notifier.trade_closed(
                        opp,
                        e_rec,
                        best_cf_arm=_label_for_key(best_k),
                        best_cf_pnl_pct=best_v,
                        regret_pct=best_v - float(e_rec.get("pnl_pct") or 0),
                        version=self.version,
                    )
                opp["status"] = "CLOSED"
                self.sm.register_close(oid, sym)
            else:
                still.append(opp)

        if closed_rows:
            self.store.append_strategy_legs(closed_rows)
        self.open_opps = still
        self._persist()
        return closed_rows

    def _fill_entry(self, opp: dict, rel: pd.DataFrame, btc_usdt: float) -> None:
        decision_ts = pd.Timestamp(opp["opened_ts"])
        if decision_ts.tzinfo is None:
            decision_ts = decision_ts.tz_localize("UTC")
        future = rel[rel["timestamp"] > decision_ts]
        if future.empty:
            return
        bar = future.iloc[0]
        entry_mid = float(bar["open"])
        entry_ts = bar["timestamp"]
        factors_raw = opp.get("factors_raw") or {}
        regime_info = classify_regime(factors_raw, rules=self.regime_rules)
        pick = self.selector.select(self.cf_history, entry_ts, regime_info["regime"], strategy_keys=LIVE_TRAIL_KEYS)
        sk = pick["selected_strategy_key"]
        cf_legs = open_opportunity_legs(
            alt_btc_entry_mid=entry_mid,
            btc_usdt=btc_usdt,
            notional_usd=float(opp.get("notional_usd", 100.0)),
            costs=self.costs,
            specs=self.cf_specs,
            entry_ts=entry_ts,
        )
        leg_items: list[dict] = []
        for arm_label, leg in zip(LIVE_ARM_LABELS, cf_legs):
            leg_items.append({"leg": leg, "arm_key": arm_label, "is_counterfactual": True, "selector_id": None})
        sel_spec = self.spec_by_key[sk]
        sel_legs = open_opportunity_legs(
            alt_btc_entry_mid=entry_mid,
            btc_usdt=btc_usdt,
            notional_usd=float(opp.get("notional_usd", 100.0)),
            costs=self.costs,
            specs=[sel_spec],
            entry_ts=entry_ts,
        )
        leg_items.append({"leg": sel_legs[0], "arm_key": "E", "is_counterfactual": False, "selector_id": self.selector.selector_id})
        opp.update({
            "status": "OPEN",
            "entry_fill_ts": str(entry_ts),
            "entry_alt_btc_mid": entry_mid,
            "entry_btc_usdt": btc_usdt,
            "leg_items": leg_items,
            "last_processed_ts": str(entry_ts),
            "regime": regime_info["regime"],
            "selected_arm_label": pick["selected_arm_label"],
            "selector_pick": pick,
        })
        audit = {
            "opportunity_id": opp["opportunity_id"],
            "entry_ts": str(entry_ts),
            "selector_id": self.selector.selector_id,
            "arm_label": "E",
            "regime": regime_info["regime"],
            "s_band": opp.get("s_band"),
            "S": opp.get("S"),
            "selected_strategy_key": sk,
            "selected_arm_label": pick["selected_arm_label"],
            "selected_score": pick["selected_score"],
            "second_best_strategy_key": pick["second_best_strategy_key"],
            "second_best_score": pick["second_best_score"],
            "strategy_rank": pick["strategy_rank"],
            "switched": pick["switched"],
            **{f"score_{k}": pick["scores"].get(k) for k in LIVE_TRAIL_KEYS},
            **self.version,
        }
        self.store.append_selection_audit([audit])
        if self._notify_open:
            self.trade_notifier.trade_opened(opp, {**pick, "arm_label": "E"}, version=self.version)

    def _run_safety(self) -> None:
        legs = self.store.read_legs()
        is_cf = legs.get("is_counterfactual", pd.Series()).astype(str).str.lower().isin(["true", "1"]) if not legs.empty else pd.Series(dtype=bool)
        e_legs = legs[(~is_cf) & (legs.get("arm_key") == "E")] if not legs.empty else pd.DataFrame()
        ev = self.safety.evaluate(e_legs, data_health_ok=True)
        if ev.state != self._prev_safety_state:
            alerts = self.sim.get("alerts") or {}
            if alerts.get("send_safety_state_change", True):
                self.trade_notifier.safety_state_change(
                    self._prev_safety_state.value, ev.state.value, ev.reasons
                )
            self._prev_safety_state = ev.state
        state = self.store.load_state()
        state["safety_state"] = ev.state.value
        state["safety_reasons"] = ev.reasons
        self.store.save_state(state)

    def process_ranked_cycle(
        self,
        ranked: list[dict[str, Any]],
        *,
        decision_ts: Any,
        rel_panels: dict[str, pd.DataFrame],
        btc_df: pd.DataFrame,
        dominance_pct: float | None,
        dominance_ts: datetime | None,
        dominance_source: str | None,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        self._advance_open(rel_panels, btc_df)
        self._run_safety()

        dts = decision_ts
        if hasattr(dts, "to_pydatetime"):
            dts = dts.to_pydatetime()
        health = evaluate_health(
            sim_cfg=self.sim,
            dominance_pct=dominance_pct,
            dominance_ts=dominance_ts,
            dominance_source=dominance_source,
            decision_candle_ts=dts if isinstance(dts, datetime) else None,
        )
        allow_new = health.allow_new_trades and self.safety.allow_new_entries()
        threshold = float(self.sim.get("long_threshold", 0.60))
        pred_rows: list[dict] = []

        for r in ranked:
            pair = r["symbol"]
            base = r.get("base")
            factor_scores = extract_factor_scores(r.get("factors") or {})
            scored = combined_score(factor_scores, self.weights)
            s_val = float(scored["S"])
            decision = self.sm.evaluate(pair, s_val)
            ep = evaluate_trail_entry(sm_decision=decision, health_allow_new_trades=allow_new)
            trade_opened = False
            rejection = ep["rejection_reason"]
            if not self.safety.allow_new_entries() and ep["trade_suggested"]:
                rejection = "SAFETY_HALT"
                ep = {"trade_suggested": False, "rejection_reason": rejection, "entry_classification": rejection}
            opportunity_id = None
            factors_raw = r.get("factors") or {}
            regime_info = classify_regime(factors_raw, rules=self.regime_rules)
            s_band = _s_band(s_val, self.s_bands)

            if ep["trade_suggested"]:
                opportunity_id = f"opp_{uuid.uuid4().hex[:12]}"
                opp = {
                    "opportunity_id": opportunity_id,
                    "opened_ts": str(decision_ts),
                    "signal_timestamp": str(decision_ts),
                    "symbol": pair,
                    "base": base,
                    "S": s_val,
                    "s_band": s_band,
                    "status": "PENDING_ENTRY",
                    "notional_usd": float(self.sim.get("notional_usd", 100.0)),
                    "factors_raw": factors_raw,
                    "regime": regime_info["regime"],
                }
                rel = rel_panels.get(pair)
                if rel is not None:
                    self._fill_entry(opp, rel.copy(), float(r.get("btc_price") or 0))
                self.open_opps.append(opp)
                self.sm.register_open(pair, opportunity_id)
                trade_opened = True
                rejection = None
                self.store.append_opportunities([{
                    **opp,
                    "long_threshold": threshold,
                    "entry_classification": ep["entry_classification"],
                    "weight_mode": "selector_live_E_v1",
                    **self.version,
                }])

            pred_rows.append({
                "timestamp": str(decision_ts),
                "symbol": pair,
                "base": base,
                "S": s_val,
                "s_band": s_band,
                "long_threshold": threshold,
                "signal_generated": bool(decision["signal_generated"]),
                "trade_opened": trade_opened,
                "rejection_reason": rejection,
                "opportunity_id": opportunity_id,
                "regime": regime_info["regime"],
                "health_allow_new_trades": allow_new,
                "safety_state": self.safety.state.value,
                **self.version,
            })

        if pred_rows:
            self.store.append_predictions(pred_rows)
        self._persist()

        try:
            from btcc.monitoring.selector_report import update_selector_live_analytics
            update_selector_live_analytics(self.sim)
        except Exception as e:
            logger.warning("Selector live analytics update failed: %s", e)

        return pred_rows
