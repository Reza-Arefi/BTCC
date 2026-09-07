"""Adaptive V2 live/sim engine — LONG ALT/BTC opportunities on top of existing SignalEngine.

Preserves:
- MEXC data pipeline
- existing factors / Late Entry overlay
- BTC.D feed
- 15m closed-bar timing
- legacy Top-5 Telegram ranking
- adaptive research archive (Champion path untouched)

Adds:
- S ∈ [-1,+1] from signed factors
- crossing state machine + max 10 / one-per-pair
- three virtual exit strategies with BTC PnL
- rolling 90d daily weight update @ 23:00 America/Sao_Paulo
- ops Telegram alerts
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from btcc.sim.accounting import CostModel
from btcc.sim.alerts import OpsAlerter
from btcc.sim.config import load_sim_config
from btcc.sim.daily_update import run_daily_weight_update, should_run_daily_update
from btcc.sim.exits import (
    leg_to_record,
    open_opportunity_legs,
    process_bar_on_leg,
    rebuild_leg_from_snapshot,
    specs_from_config,
)
from btcc.sim.health import evaluate_health
from btcc.sim.score import FACTOR_KEYS, combined_score, extract_factor_scores, normalize_weights
from btcc.sim.store import SimStore

logger = logging.getLogger(__name__)


class AdaptiveSimEngine:
    def __init__(self, signal_cfg: dict[str, Any], sim_cfg: dict[str, Any] | None = None, tg_send=None):
        self.signal_cfg = signal_cfg
        self.sim = sim_cfg or load_sim_config()
        self.sim["_fallback_factor_weights"] = dict(signal_cfg.get("factors", {}).get("weights") or {})
        self.enabled = bool(self.sim.get("enabled", True))
        self.store = SimStore(self.sim)
        self.sm = self.store.load_state_machine(self.sim)
        # Sync threshold/max from config (authoritative)
        self.sm.long_threshold = float(self.sim.get("long_threshold", 0.60))
        self.sm.max_open = int(self.sim.get("max_open_opportunities", 10))
        self.sm.one_per_pair = bool(self.sim.get("one_opportunity_per_pair", True))
        self.costs = CostModel(
            fee_rate_per_side=float(self.sim.get("fee_rate_per_side", 0.001)),
            slippage_rate_per_side=float(self.sim.get("slippage_rate_per_side", 0.0005)),
        )
        self.specs = specs_from_config(self.sim)
        self.weights = normalize_weights(
            self.store.get_current_weights(self.sim["_fallback_factor_weights"])
        )
        self.open_opps: list[dict[str, Any]] = []
        for opp in self.store.open_opportunities_state():
            # Restore live leg objects from snapshots after crash/restart
            snaps = opp.get("legs_snapshot") or []
            legs = []
            for snap in snaps:
                leg = rebuild_leg_from_snapshot(snap, self.specs)
                if leg is not None:
                    legs.append({"spec_key": leg.spec.key, "state": leg})
            opp = dict(opp)
            opp["legs"] = legs
            opp.pop("legs_snapshot", None)
            self.open_opps.append(opp)
            oid = opp.get("opportunity_id")
            pair = opp.get("symbol")
            if oid and pair and opp.get("status") in ("OPEN", "PENDING_ENTRY"):
                self.sm.register_open(pair, oid)
        self.alerter = OpsAlerter(
            send_fn=tg_send or (lambda _t: False),
            enabled=True,
            cfg=self.sim.get("alerts") or {},
        )
        self._activity_opens: list[dict[str, Any]] = []
        self._btcd_alerted = False
        # Weights calculated at 23:00 become effective on the NEXT cycle.
        # Persist pending_* in state.json so a reboot mid-deferral still activates next bar.
        state0 = self.store.load_state()
        pw = state0.get("pending_weights")
        self._pending_weights: dict[str, float] | None = (
            normalize_weights(pw) if isinstance(pw, dict) and pw else None
        )
        self._pending_weights_meta: dict[str, Any] | None = state0.get("pending_weights_meta")
        if self._pending_weights:
            logger.info(
                "Restored pending weights from state (will activate next cycle) | update_id=%s",
                (self._pending_weights_meta or {}).get("update_id"),
            )

    def weights_version(self) -> str:
        meta = self.store.load_state().get("weights_meta") or {}
        return str(meta.get("update_id") or "config_initial")

    def notify_startup(self) -> None:
        self.alerter.startup({
            "sim_enabled": self.enabled,
            "long_threshold": self.sm.long_threshold,
            "max_open": self.sm.max_open,
            "weights_version": self.weights_version(),
        })

    def notify_shutdown(self, reason: str = "user_stop") -> None:
        self.alerter.shutdown(reason)

    def _maybe_live_analytics(self, daily_result: dict[str, Any] | None = None) -> None:
        """Update live analytics directory; Telegram daily summary only (not every cycle)."""
        from zoneinfo import ZoneInfo

        alerts = self.sim.get("alerts") or {}
        wu = self.sim.get("weight_update") or {}
        tz = ZoneInfo(str(wu.get("timezone", "America/Sao_Paulo")))
        now_local = datetime.now(tz)
        local_date = now_local.date().isoformat()
        state = self.store.load_state()
        if state.get("last_analytics_local_date") == local_date and not (
            daily_result and daily_result.get("status") == "OK"
        ):
            return
        if (now_local.hour, now_local.minute) < (int(wu.get("hour", 23)), int(wu.get("minute", 0))):
            if not (daily_result and daily_result.get("status") in ("OK", "SKIPPED", "FAILED")):
                return

        from btcc.analytics.pipeline import update_live_analytics

        def _tg(text: str) -> bool:
            self.alerter.daily_summary(text)
            return True

        root = update_live_analytics(
            sim_storage=self.sim.get("storage"),
            max_open=int(self.sim.get("max_open_opportunities", 10)),
            send_daily_telegram=bool(alerts.get("send_daily_summary", True)),
            tg_send=_tg if alerts.get("send_daily_summary", True) else None,
        )
        state = self.store.load_state()
        state["last_analytics_local_date"] = local_date
        state["last_analytics_root"] = str(root)
        self.store.save_state(state)
        logger.info("Live analytics updated → %s", root)

    def maybe_daily_update(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if should_run_daily_update(self.sim, self.store):
            prev = dict(self.weights)
            result = run_daily_weight_update(self.sim, self.store)
            # run_daily_weight_update may have activated weights — defer to next cycle
            if result.get("status") == "OK" and result.get("weights"):
                self._pending_weights = normalize_weights(result["weights"])
                self._pending_weights_meta = {
                    "update_id": result.get("update_id"),
                    "calculated_at": datetime.now(timezone.utc).isoformat(),
                    "effective_from": "next_cycle",
                    "local_date": result.get("local_date"),
                }
                # Restore previous known-good as active until next cycle
                self.store.set_current_weights(prev, meta={"deferred": True, "active": "pre_update"})
                self.weights = normalize_weights(prev)
                state = self.store.load_state()
                state["pending_weights"] = self._pending_weights
                state["pending_weights_meta"] = self._pending_weights_meta
                self.store.save_state(state)
            return result
        return None

    def _activate_pending_weights(self) -> None:
        """Apply weights scheduled from a prior cycle's 23:00 update."""
        state = self.store.load_state()
        pending = self._pending_weights or state.get("pending_weights")
        meta = self._pending_weights_meta or state.get("pending_weights_meta")
        if pending:
            self.weights = normalize_weights(pending)
            self.store.set_current_weights(self.weights, meta={
                **(meta or {}),
                "effective_from": "activated",
            })
            state = self.store.load_state()
            state.pop("pending_weights", None)
            state.pop("pending_weights_meta", None)
            self.store.save_state(state)
            self._pending_weights = None
            self._pending_weights_meta = None

    def _pending_entry_fill(
        self,
        opp: dict[str, Any],
        rel: pd.DataFrame,
        btc_usdt: float,
    ) -> None:
        """Fill entry at next bar open after decision timestamp."""
        if opp.get("status") != "PENDING_ENTRY":
            return
        decision_ts = pd.Timestamp(opp["opened_ts"])
        if decision_ts.tzinfo is None:
            decision_ts = decision_ts.tz_localize("UTC")
        future = rel[pd.to_datetime(rel["timestamp"], utc=True) > decision_ts]
        if future.empty:
            return
        bar = future.iloc[0]
        entry_mid = float(bar["open"])
        entry_ts = bar["timestamp"]
        legs = open_opportunity_legs(
            alt_btc_entry_mid=entry_mid,
            btc_usdt=btc_usdt,
            notional_usd=float(self.sim.get("notional_usd", 100.0)),
            costs=self.costs,
            specs=self.specs,
            entry_ts=entry_ts,
        )
        opp["status"] = "OPEN"
        opp["entry_fill_ts"] = str(entry_ts)
        opp["entry_alt_btc_mid"] = entry_mid
        opp["entry_btc_usdt"] = btc_usdt
        opp["legs"] = [
            {
                "spec_key": leg.spec.key,
                "state": leg,
            }
            for leg in legs
        ]

    def _advance_open(self, rel_panels: dict[str, pd.DataFrame], btc_df: pd.DataFrame) -> None:
        """Advance pending fills and open legs with new bars."""
        btc = btc_df.copy()
        btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True)
        btc_indexed = btc.set_index("timestamp")["close"]
        still_open: list[dict[str, Any]] = []
        closed_leg_rows: list[dict[str, Any]] = []

        for opp in self.open_opps:
            sym = opp["symbol"]
            rel = rel_panels.get(sym)
            if rel is None or rel.empty:
                still_open.append(opp)
                continue
            rel = rel.copy()
            rel["timestamp"] = pd.to_datetime(rel["timestamp"], utc=True)
            last_btc = float(btc_indexed.iloc[-1]) if len(btc_indexed) else float(opp.get("entry_btc_usdt") or 0)

            if opp.get("status") == "PENDING_ENTRY":
                self._pending_entry_fill(opp, rel, last_btc)
                if opp.get("status") == "PENDING_ENTRY":
                    still_open.append(opp)
                    continue

            # Process new bars after last_processed
            last_ts = opp.get("last_processed_ts")
            if last_ts:
                last_ts = pd.Timestamp(last_ts)
                if last_ts.tzinfo is None:
                    last_ts = last_ts.tz_localize("UTC")
                bars = rel[rel["timestamp"] > last_ts]
            else:
                entry_ts = pd.Timestamp(opp["entry_fill_ts"])
                if entry_ts.tzinfo is None:
                    entry_ts = entry_ts.tz_localize("UTC")
                bars = rel[rel["timestamp"] > entry_ts]

            legs = [x["state"] for x in opp.get("legs") or []]
            for _, row in bars.iterrows():
                ts = row["timestamp"]
                btc_usdt = float(btc_indexed.loc[ts]) if ts in btc_indexed.index else last_btc
                bar = {
                    "timestamp": ts,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
                for leg in legs:
                    newly = process_bar_on_leg(
                        leg,
                        bar=bar,
                        btc_usdt=btc_usdt,
                        costs=self.costs,
                        same_candle_conflict=str(self.sim.get("same_candle_conflict", "assume_sl_first")),
                    )
                    if newly:
                        closed_leg_rows.append(leg_to_record(leg, opp["opportunity_id"]))
                opp["last_processed_ts"] = str(ts)
                if all(leg.closed for leg in legs):
                    break

            if legs and all(leg.closed for leg in legs):
                opp["status"] = "CLOSED"
                opp["closed_ts"] = str(legs[0].exit_ts)
                self.sm.register_close(opp["opportunity_id"], sym)
                # Persist final leg snapshots if not already
                for leg in legs:
                    if leg.closed:
                        rec = leg_to_record(leg, opp["opportunity_id"])
                        if rec not in closed_leg_rows:
                            closed_leg_rows.append(rec)
            else:
                still_open.append(opp)

        if closed_leg_rows:
            # Deduplicate by opportunity+strategy+exit_ts
            uniq = {}
            for r in closed_leg_rows:
                key = (r["opportunity_id"], r["strategy_key"], r.get("exit_ts"))
                uniq[key] = r
            self.store.append_strategy_legs(list(uniq.values()))

        self.open_opps = still_open
        # Persist serializable open state (without live leg objects — rebuild from JSON)
        self._persist_open_book()

    def _persist_open_book(self) -> None:
        serializable = []
        for opp in self.open_opps:
            row = {k: v for k, v in opp.items() if k != "legs"}
            legs_dump = []
            for item in opp.get("legs") or []:
                leg = item["state"]
                legs_dump.append(leg_to_record(leg, opp["opportunity_id"]))
            row["legs_snapshot"] = legs_dump
            serializable.append(row)
        self.store.set_open_opportunities_state(serializable)
        self.store.save_state_machine(
            self.sm,
            extra={"current_weights": self.weights, "weights_meta": self.store.load_state().get("weights_meta")},
        )

    def _check_activity_burst(self) -> None:
        alerts = self.sim.get("alerts") or {}
        window_m = int(alerts.get("activity_window_minutes", 60))
        thresh = int(alerts.get("activity_open_threshold", 3))
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=window_m)
        recent = []
        for o in self._activity_opens:
            try:
                ts = datetime.fromisoformat(str(o["ts"]).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    recent.append(o)
            except Exception:
                continue
        self._activity_opens = recent
        if len(recent) >= thresh:
            self.alerter.activity_burst(
                opens=recent,
                n_open=self.sm.n_open(),
                slots_remaining=self.sm.slots_remaining(),
            )
            # Prevent alert spam: clear window after alert
            self._activity_opens = []

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
        """Run Adaptive V2 logic after the existing ranking cycle.

        Returns list of prediction records written this cycle.
        """
        if not self.enabled:
            return []

        # Activate any weights scheduled from a previous 23:00 update FIRST
        self._activate_pending_weights()
        self.weights = normalize_weights(
            self.store.get_current_weights(self.sim["_fallback_factor_weights"])
        )

        # May schedule NEW weights for the next cycle — must not affect this cycle
        daily = self.maybe_daily_update()

        # Live analytics refresh + optional daily Telegram summary (never per-prediction)
        try:
            self._maybe_live_analytics(daily_result=daily)
        except Exception as e:
            logger.warning("Live analytics refresh failed: %s", e)

        # Advance open opportunities with latest bars first
        try:
            self._advance_open(rel_panels, btc_df)
        except Exception as e:
            logger.exception("Advance open opportunities failed: %s", e)
            self.alerter.runtime_error(f"advance_open: {e}")

        # Global health (candle / BTC.D)
        try:
            dts = decision_ts
            if hasattr(dts, "to_pydatetime"):
                dts = dts.to_pydatetime()
            if isinstance(dts, datetime) and dts.tzinfo is None:
                dts = dts.replace(tzinfo=timezone.utc)
        except Exception:
            dts = datetime.now(timezone.utc)

        health = evaluate_health(
            sim_cfg=self.sim,
            dominance_pct=dominance_pct,
            dominance_ts=dominance_ts,
            dominance_source=dominance_source,
            decision_candle_ts=dts if isinstance(dts, datetime) else None,
        )
        if not health.btc_d_available and not self._btcd_alerted:
            self.alerter.btcd_failure("; ".join(health.reasons))
            self._btcd_alerted = True
        if health.btc_d_available:
            self._btcd_alerted = False
        if not health.ok and any("STALE_CANDLES" in r for r in health.reasons):
            self.alerter.data_failure("; ".join(health.reasons))

        pred_rows: list[dict[str, Any]] = []
        opp_rows: list[dict[str, Any]] = []
        threshold = float(self.sim.get("long_threshold", 0.60))
        horizon = int(self.sim.get("primary_horizon_hours", 4))
        bot_version = str(self.sim.get("bot_version", "2.0.0"))

        for r in ranked:
            pair = r["symbol"]
            base = r.get("base")
            factor_scores = extract_factor_scores(r.get("factors") or {})
            # Per-symbol bar count health
            n_bars = int(r.get("n_relative_bars") or 0)
            local_health = evaluate_health(
                sim_cfg=self.sim,
                dominance_pct=dominance_pct,
                dominance_ts=dominance_ts,
                dominance_source=dominance_source,
                decision_candle_ts=dts if isinstance(dts, datetime) else None,
                n_relative_bars=n_bars,
                indicator_ok=all(
                    k in factor_scores and factor_scores[k] == factor_scores[k] for k in FACTOR_KEYS
                ),
            )

            scored = combined_score(factor_scores, self.weights)
            S = float(scored["S"])
            decision = self.sm.evaluate(pair, S)

            trade_opened = False
            rejection = decision["rejection_reason"]
            opportunity_id = None

            if decision["trade_suggested"]:
                if not local_health.allow_new_trades:
                    rejection = "DATA_HEALTH_BLOCK"
                    decision["trade_suggested"] = False
                else:
                    opportunity_id = f"opp_{uuid.uuid4().hex[:12]}"
                    wver = self.weights_version()
                    opp = {
                        "opportunity_id": opportunity_id,
                        "opened_ts": str(decision_ts),
                        "signal_timestamp": str(decision_ts),
                        "symbol": pair,
                        "base": base,
                        "S": S,
                        "threshold": threshold,
                        "status": "PENDING_ENTRY",
                        "entry_btc_usdt": float(r.get("btc_price") or 0),
                        "notional_usd": float(self.sim.get("notional_usd", 100.0)),
                        "weight_version_id": wver,
                        "weights_at_entry": dict(self.weights),
                    }
                    # Try immediate fill if next bar already present
                    rel = rel_panels.get(pair)
                    if rel is not None:
                        self._pending_entry_fill(opp, rel, float(r.get("btc_price") or 0))
                    self.open_opps.append(opp)
                    self.sm.register_open(pair, opportunity_id)
                    trade_opened = True
                    rejection = None
                    opp_rows.append({
                        "opportunity_id": opportunity_id,
                        "opened_ts": str(decision_ts),
                        "signal_timestamp": str(decision_ts),
                        "symbol": pair,
                        "base": base,
                        "S": S,
                        "threshold": threshold,
                        "entry_fill_ts": opp.get("entry_fill_ts"),
                        "entry_alt_btc_mid": opp.get("entry_alt_btc_mid"),
                        "entry_btc_usdt": opp.get("entry_btc_usdt"),
                        "notional_usd": opp.get("notional_usd"),
                        "status": opp.get("status"),
                        "closed_ts": None,
                        "n_legs_closed": 0,
                        "rejection_on_signal": None,
                        "weight_version_id": wver,
                        "weights_calculated_at": (self.store.load_state().get("weights_meta") or {}).get("calculated_at"),
                        "weights_effective_from": (self.store.load_state().get("weights_meta") or {}).get("effective_from"),
                        **{f"weight_{k}": self.weights.get(k) for k in FACTOR_KEYS},
                    })
                    self._activity_opens.append({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "symbol": pair,
                        "S": S,
                    })

            # Signals that did not open still recorded
            if decision["signal_generated"] and not trade_opened and rejection is None:
                rejection = decision.get("rejection_reason") or "NOT_OPENED"

            pred_id = f"pred_{uuid.uuid4().hex[:12]}"
            row = {
                "prediction_id": pred_id,
                "timestamp": str(decision_ts),
                "symbol": pair,
                "base": base,
                "construction": r.get("construction"),
                "alt_btc_price": r.get("alt_btc_price"),
                "btc_price": r.get("btc_price"),
                "btc_usdt": r.get("btc_price"),
                "S": S,
                "long_threshold": threshold,
                "signal_generated": bool(decision["signal_generated"]),
                "trade_opened": trade_opened,
                "rejection_reason": rejection if not trade_opened else None,
                "zone_state": decision["zone"],
                "crossed_into": decision["crossed_into"],
                "opportunity_id": opportunity_id,
                "horizon_hours": horizon,
                "model_weights_version": self.weights_version(),
                "signal_score_legacy": r.get("signal_score"),
                "p_4h_legacy": r.get("p_4h"),
                "late_entry_score": (r.get("late_entry") or {}).get("late_entry_score"),
                "late_entry_class": (r.get("late_entry") or {}).get("classification"),
                "btc_dominance": dominance_pct,
                "btc_d_status": local_health.btc_d_status,
                "btc_d_age_seconds": local_health.btc_d_age_seconds,
                "btc_d_available": local_health.btc_d_available,
                "health_ok": local_health.ok,
                "health_allow_new_trades": local_health.allow_new_trades,
                "health_reasons": ";".join(local_health.reasons),
                "bot_version": bot_version,
                "config_version": self.sim.get("version"),
                "outcome_ready": False,
            }
            for k in FACTOR_KEYS:
                row[f"signed_{k}"] = scored["signed"].get(k)
                row[f"weight_{k}"] = scored["weights"].get(k)
                row[f"factor_{k}"] = factor_scores.get(k)
            pred_rows.append(row)

        self.store.append_predictions(pred_rows)
        if opp_rows:
            self.store.append_opportunities(opp_rows)
        self._persist_open_book()
        self._check_activity_burst()

        # Backfill 4h outcomes when possible
        try:
            self.backfill_prediction_outcomes(rel_panels)
        except Exception as e:
            logger.warning("Outcome backfill failed: %s", e)

        return pred_rows

    def backfill_prediction_outcomes(self, rel_panels: dict[str, pd.DataFrame]) -> int:
        """Attach 4h ALT/BTC outcomes without overwriting original prediction fields."""
        from btcc.series.relative import horizon_bars

        df = self.store.load_predictions()
        if df.empty:
            return 0
        horizon = int(self.sim.get("primary_horizon_hours", 4))
        bars = horizon_bars(horizon, str(self.sim.get("candle_interval", "15m")))
        updated = 0
        if "outcome_ready" not in df.columns:
            df["outcome_ready"] = False

        for i, row in df.iterrows():
            if bool(row.get("outcome_ready")):
                continue
            sym = row.get("symbol")
            rel = rel_panels.get(sym)
            if rel is None or rel.empty:
                continue
            rel = rel.copy()
            rel["timestamp"] = pd.to_datetime(rel["timestamp"], utc=True)
            ts = pd.Timestamp(row["timestamp"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            # Find decision index
            idxs = rel.index[rel["timestamp"] == ts]
            if len(idxs) == 0:
                # nearest previous
                prev = rel[rel["timestamp"] <= ts]
                if prev.empty:
                    continue
                idx = prev.index[-1]
            else:
                idx = idxs[0]
            pos = rel.index.get_loc(idx)
            if isinstance(pos, slice):
                continue
            if pos + bars >= len(rel):
                continue
            px0 = float(rel.iloc[pos]["close"])
            px1 = float(rel.iloc[pos + bars]["close"])
            if px0 == 0:
                continue
            ret = px1 / px0 - 1.0
            S = float(row.get("S") or 0)
            direction_actual = 1 if ret > 0 else (-1 if ret < 0 else 0)
            # Prediction correct if sign(S) matches sign(return) when |S| meaningful;
            # for long-only research: S>=threshold and ret>0 counts as hit when signal_generated
            if bool(row.get("signal_generated")):
                prediction_correct = int(ret > 0)
            else:
                prediction_correct = int((S >= 0 and ret > 0) or (S < 0 and ret < 0) or ret == 0)
            df.at[i, "future_alt_btc_price"] = px1
            df.at[i, "future_return_4h"] = ret
            df.at[i, "direction_actual"] = direction_actual
            df.at[i, "prediction_correct"] = prediction_correct
            df.at[i, "prediction_score"] = float(S) * float(ret)  # alignment score
            df.at[i, "outcome_ts"] = str(rel.iloc[pos + bars]["timestamp"])
            df.at[i, "outcome_ready"] = True
            updated += 1

        if updated:
            from btcc.sim.store import _atomic_write_df
            _atomic_write_df(self.store.predictions_path, df)
        return updated
