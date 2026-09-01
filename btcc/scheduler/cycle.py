"""15-minute decision cycle — SIGNAL ONLY."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from btcc.adaptive.checkpoint import run_checkpoint
from btcc.adaptive.model import get_champion
from btcc.adaptive.store import AdaptivePredictionStore
from btcc.backtest.extract import flatten_indicators_and_factors
from btcc.data.candles import candle_path, load_candles
from btcc.data.dominance import DominanceFeed
from btcc.data.websocket import MexcPublicREST
from btcc.factors.combine import compute_all_factors
from btcc.late_entry.score import late_entry_score
from btcc.probability.score import classify_signal, horizons_probabilities, load_calibration
from btcc.ranking.ranker import rank_signals
from btcc.series.relative import build_alt_btc, native_btc_as_relative
from btcc.storage.predictions import PredictionStore
from btcc.telegram.notifier import TelegramNotifier
from btcc.universe import (
    build_universe_audit,
    format_universe_audit,
    resolve_btc_market,
    resolve_data_market,
    resolve_usdt_symbol,
)

try:
    from btcc.sim.engine import AdaptiveSimEngine
except Exception:  # pragma: no cover - sim package always present in tree
    AdaptiveSimEngine = None  # type: ignore

try:
    from btcc.sim.selector_live_config import load_selector_live_config
    from btcc.sim.selector_live_engine import SelectorLiveEngine
except Exception:  # pragma: no cover
    load_selector_live_config = None  # type: ignore
    SelectorLiveEngine = None  # type: ignore

from btcc.runtime_persist import RuntimeState

logger = logging.getLogger(__name__)


class SignalEngine:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.rest = MexcPublicREST(cfg["data"]["mexc_rest"])
        self.candle_dir = cfg["data"]["candle_dir"]
        self.interval = cfg["experiment"]["candle_interval"]
        self.lookback = int(cfg["experiment"]["lookback_bars"])
        self.panels: dict[str, pd.DataFrame] = {}
        self.btc: pd.DataFrame | None = None
        hist = cfg["data"].get("dominance_history_path")
        self.dominance = DominanceFeed(
            cfg["data"]["dominance_url"],
            cfg["data"]["dominance_source"],
            int(cfg["data"]["dominance_poll_seconds"]),
            history_path=hist,
        )
        self.store = PredictionStore(Path(cfg["data"]["prediction_dir"]) / "predictions.csv")
        self.calib = load_calibration(Path(cfg["data"]["prediction_dir"]) / "calibration.json")
        self.tg = TelegramNotifier(
            cfg["telegram"].get("bot_token", ""),
            cfg["telegram"].get("chat_id", ""),
            cfg["telegram"].get("enabled", True),
        )
        self.unavailable: set[str] = set()
        self.construction: dict[str, str] = {}
        self.native_btc_panels: dict[str, pd.DataFrame] = {}  # base -> native *BTC df
        self.last_universe_audit: list[dict[str, Any]] = []
        self.last_dominance_summary: dict[str, Any] = {}
        self.last_data_meta: dict[str, Any] = {}
        # Adaptive walk-forward (Champion/Challenger) — SIGNAL ONLY
        self.adaptive_cfg = dict(cfg.get("adaptive") or {})
        self.adaptive_enabled = bool(self.adaptive_cfg.get("enabled", False))
        self.models_dir = Path(self.adaptive_cfg.get("models_dir", "data/models"))
        self.reports_dir = Path(self.adaptive_cfg.get("reports_dir", "logs/adaptive"))
        pred_path = self.adaptive_cfg.get(
            "predictions_path",
            str(Path(cfg["data"]["prediction_dir"]) / "adaptive_predictions.csv"),
        )
        self.adaptive_store = AdaptivePredictionStore(pred_path) if self.adaptive_enabled else None
        self.champion = get_champion(self.models_dir) if self.adaptive_enabled else None
        self.champion_weights = (
            self.champion.normalized_weights() if self.champion is not None else None
        )
        self._checkpoint_state_path = self.models_dir / "last_checkpoint.json"
        if self.adaptive_enabled:
            logger.info(
                "Adaptive learning enabled | champion=%s | store=%s",
                self.champion.version if self.champion else "none (run adaptive-bootstrap)",
                pred_path,
            )
        # Adaptive V2 — LONG ALT/BTC virtual sim (separate from Champion archive)
        self.sim_engine = None
        self.selector_live_engine = None
        sim_cfg = cfg.get("sim") or {}
        rt_path = (sim_cfg.get("storage") or {}).get(
            "runtime_state_path", "data/sim/runtime_state.json"
        )
        root = Path(cfg.get("_root") or ".")
        self.runtime = RuntimeState(root / rt_path if not Path(rt_path).is_absolute() else Path(rt_path))

        selector_live_cfg = None
        if load_selector_live_config is not None:
            try:
                selector_live_cfg = load_selector_live_config()
            except Exception as e:
                logger.warning("Selector live config load failed: %s", e)

        if (
            SelectorLiveEngine is not None
            and selector_live_cfg is not None
            and bool(selector_live_cfg.get("enabled", False))
        ):
            self.selector_live_engine = SelectorLiveEngine(
                cfg,
                selector_live_cfg,
                tg_send=self.tg.send,
            )
            self.selector_live_engine.notify_startup()
            logger.info(
                "Selector E-v1 live enabled | version=%s | threshold=%.2f",
                selector_live_cfg.get("selector_version"),
                float(selector_live_cfg.get("long_threshold", 0.60)),
            )
        elif AdaptiveSimEngine is not None and sim_cfg.get("enabled", False):
            self.sim_engine = AdaptiveSimEngine(
                cfg,
                sim_cfg,
                tg_send=self.tg.send,
            )
            logger.info(
                "Adaptive V2 sim enabled | threshold=%.2f | max_open=%d | weights=%s",
                float(sim_cfg.get("long_threshold", 0.60)),
                int(sim_cfg.get("max_open_opportunities", 10)),
                self.sim_engine.weights_version(),
            )

    def bootstrap(self) -> None:
        btc_sym = self.cfg["universe"]["btc_symbol"]
        df_btc = self.rest.bootstrap_symbol(btc_sym, self.interval, self.lookback, self.candle_dir)
        if df_btc is None or df_btc.empty:
            logger.error("DATA_UNAVAILABLE: %s", btc_sym)
            self.unavailable.add(btc_sym)
        else:
            self.btc = df_btc
            logger.info("Loaded %s (%d bars)", btc_sym, len(df_btc))

        for base in self.cfg["universe"]["bases"]:
            meta = resolve_data_market(base, self.cfg)
            resolved = meta["resolved_market"]
            usdt = resolve_usdt_symbol(base, self.cfg)
            btc_mkt = resolve_btc_market(base, self.cfg)
            df = self.rest.bootstrap_symbol(
                resolved, self.interval, self.lookback, self.candle_dir
            )
            if df is not None and not df.empty:
                self.panels[resolved] = df
                self.construction[base] = meta["construction"]
                if meta["mode"] == "native_btc":
                    self.native_btc_panels[base] = df
                logger.info(
                    "Loaded %s logical=%s (%d bars) construction=%s",
                    resolved, base, len(df), self.construction[base],
                )
                continue
            self.unavailable.add(resolved)
            # Synthetic pairs only: one native fallback if USDT vanished
            if meta["mode"] == "synthetic_usdt":
                native = self.rest.bootstrap_symbol(
                    btc_mkt, self.interval, self.lookback, self.candle_dir
                )
                if native is not None and not native.empty:
                    self.native_btc_panels[base] = native
                    self.panels[btc_mkt] = native
                    self.construction[base] = f"native {btc_mkt} (no USDT pair)"
                    logger.info(
                        "Loaded native %s (%d bars) — USDT %s unavailable",
                        btc_mkt, len(native), usdt,
                    )
                    continue
                self.unavailable.add(btc_mkt)
            self.construction[base] = "DATA_UNAVAILABLE"
            logger.error("DATA_UNAVAILABLE: logical=%s resolved=%s", base, resolved)

        self.dominance.fetch(force=True)
        self.last_universe_audit = build_universe_audit(
            self.cfg, self.panels, self.unavailable, self.construction
        )
        logger.info("\n%s", format_universe_audit(self.last_universe_audit))

    def refresh_latest(self) -> None:
        """Pull latest completed candles via REST (WS updates optional)."""
        btc_sym = self.cfg["universe"]["btc_symbol"]
        symbols = {btc_sym} | set(self.panels.keys())
        for sym in symbols:
            try:
                fresh = self.rest.fetch_klines(sym, self.interval, limit=5)
                if fresh.empty:
                    continue
                path = candle_path(self.candle_dir, sym, self.interval)
                existing = load_candles(path)
                if existing is None:
                    existing = fresh
                else:
                    existing = (
                        pd.concat([existing, fresh])
                        .drop_duplicates("timestamp", keep="last")
                        .sort_values("timestamp")
                        .tail(self.lookback)
                        .reset_index(drop=True)
                    )
                from btcc.data.candles import save_candles
                save_candles(existing, path)
                if sym == btc_sym:
                    self.btc = existing
                else:
                    self.panels[sym] = existing
                    # keep native map in sync
                    for base, mkt in (self.cfg.get("universe", {}).get("btc_markets") or {}).items():
                        if mkt == sym:
                            self.native_btc_panels[base] = existing
            except Exception as e:
                logger.warning("Refresh failed %s: %s → DATA_UNAVAILABLE", sym, e)
                self.unavailable.add(sym)

    def _completed_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop potentially incomplete last candle (decision uses closed bars only)."""
        if df is None or len(df) < 3:
            return df
        return df.iloc[:-1].copy()

    def run_cycle(self) -> list[dict[str, Any]]:
        self.refresh_latest()

        if self.btc is None or self.btc.empty:
            logger.error("DATA_UNAVAILABLE: BTCUSDT")
            self.runtime.mark_cycle_failure("DATA_UNAVAILABLE: BTCUSDT")
            return []

        btc = self._completed_frame(self.btc)
        ts = pd.Timestamp(btc["timestamp"].iloc[-1])
        ts_key = str(ts)
        if self.runtime.already_processed(ts_key):
            logger.info(
                "Decision candle %s already processed — skip (resume from next closed bar)",
                ts_key,
            )
            self.last_data_meta = {
                "decision_candle_ts": ts_key,
                "skipped_duplicate": True,
                "btc_age": None,
                "unavailable": sorted(self.unavailable),
            }
            return []

        decision_dt = ts.to_pydatetime()
        if decision_dt.tzinfo is None:
            decision_dt = decision_dt.replace(tzinfo=timezone.utc)

        # Relative BTC.D — same definition as backtest; stamp / resolve at decision_ts
        # so a future observation cannot enter an earlier 15m prediction.
        self.dominance.fetch(force=True, as_of=decision_dt)
        dom_snap, dom_obs_ts, dom_status = self.dominance.observation_at(decision_dt)
        dominance_pct = dom_snap.btc_dominance_pct if dom_snap else None
        dom = dom_snap  # compat for later getattr
        dom_raw = {
            1: self.dominance.change(1, as_of=decision_dt),
            4: self.dominance.change(4, as_of=decision_dt),
            12: self.dominance.change(12, as_of=decision_dt),
            24: self.dominance.change(24, as_of=decision_dt),
        }
        # Pass numeric changes (or None) into factors; keep statuses for audit/Telegram
        dom_changes = {h: v for h, (v, _s) in dom_raw.items()}
        dom_statuses = {h: s for h, (_v, s) in dom_raw.items()}
        self.last_dominance_summary = {
            **self.dominance.summary(),
            "btc_dominance_pct": dominance_pct,
            "asof": dom_obs_ts.isoformat() if dom_obs_ts else None,
            "observation_status": dom_status,
            "decision_candle_ts": ts_key,
            "change_1h_pp": dom_raw[1][0],
            "change_1h_status": dom_raw[1][1],
            "change_12h_pp": dom_raw[12][0],
            "change_12h_status": dom_raw[12][1],
            "change_4h_pp": dom_raw[4][0],
            "change_4h_status": dom_raw[4][1],
            "change_24h_pp": dom_raw[24][0],
            "change_24h_status": dom_raw[24][1],
        }

        now = datetime.now(timezone.utc)
        btc_age = now - ts.to_pydatetime().replace(tzinfo=timezone.utc) if ts.tzinfo is None else now - ts.to_pydatetime()
        rows = []
        alt_btc_panels = {}

        for base in self.cfg["universe"]["bases"]:
            meta = resolve_data_market(base, self.cfg)
            resolved = meta["resolved_market"]
            usdt = resolve_usdt_symbol(base, self.cfg)
            btc_mkt = resolve_btc_market(base, self.cfg)
            mode = self.construction.get(base, meta["construction"])
            warnings: list[str] = []
            rel = None
            alt_for_volume = None
            sym_key = resolved

            if resolved in self.panels:
                alt = self._completed_frame(self.panels[resolved])
                alt_for_volume = alt
                if meta["mode"] == "native_btc" or base in self.native_btc_panels:
                    rel = native_btc_as_relative(alt)
                    mode = f"native {resolved}"
                else:
                    rel = build_alt_btc(alt, btc)
                    mode = f"{resolved} / BTCUSDT"
            elif usdt in self.panels:
                alt = self._completed_frame(self.panels[usdt])
                alt_for_volume = alt
                rel = build_alt_btc(alt, btc)
                sym_key = usdt
                mode = f"{usdt} / BTCUSDT"
            elif base in self.native_btc_panels:
                native = self._completed_frame(self.native_btc_panels[base])
                rel = native_btc_as_relative(native)
                alt_for_volume = native  # volume already in BTC-market units
                sym_key = btc_mkt
                mode = f"native {btc_mkt}"
                warnings.append(f"USDT_PAIR_MISSING used_native={btc_mkt}")
            else:
                logger.warning("DATA_UNAVAILABLE: %s — skipped", base)
                continue

            self.construction[base] = mode
            if rel is None or len(rel) < 100:
                logger.warning("DATA_UNAVAILABLE: %s relative series", base)
                continue
            if len(rel) < 200:
                warnings.append(f"short_relative_history n={len(rel)}")
            for h, status in dom_statuses.items():
                if status != "OK":
                    warnings.append(f"BTC.D_{h}h={status}")
                    break
            alt_btc_panels[sym_key] = rel
            factors = compute_all_factors(
                rel,
                alt_for_volume,
                btc,
                dominance_pct,
                dom_changes,
                self.cfg,
                self.interval,
                factor_weights=self.champion_weights,
            )
            # Prefer Champion calibration when present; else live calib file
            calib = (
                self.champion.calibration
                if self.champion is not None and self.champion.calibration
                else self.calib
            )
            probs = horizons_probabilities(factors["signal_score"], self.cfg, calib)
            late = late_entry_score(rel, factors, self.cfg, self.interval)
            sc = classify_signal(probs["p_4h"], late["late_entry_score"], self.cfg)
            for name in ("momentum", "trend", "volume", "volatility", "rsi", "structure"):
                if factors[name].get("insufficient_data"):
                    warnings.append(f"{name}_INSUFFICIENT_DATA")
            alt_usdt_px = None
            if meta["mode"] == "synthetic_usdt" and resolved in self.panels and "close" in self.panels[resolved].columns:
                try:
                    alt_usdt_px = float(self._completed_frame(self.panels[resolved])["close"].iloc[-1])
                except Exception:
                    alt_usdt_px = None
            rows.append({
                "timestamp": ts,
                "symbol": sym_key,
                "base": base,
                "logical_pair": meta["logical_pair"],
                "resolved_market": sym_key,
                "p_1h": probs["p_1h"],
                "p_4h": probs["p_4h"],
                "p_8h": probs["p_8h"],
                "p_12h": probs["p_12h"],
                "p_24h": probs["p_24h"],
                "p_1h_status": probs["p_1h_status"],
                "p_4h_status": probs["p_4h_status"],
                "p_8h_status": probs["p_8h_status"],
                "p_12h_status": probs["p_12h_status"],
                "p_24h_status": probs["p_24h_status"],
                "baseline_model_p_4h": probs.get("baseline_model_p_4h"),
                "probability_kind": probs["probability_kind"],
                "probability_disclaimer": probs["disclaimer"],
                "signal_score": factors["signal_score"],
                "factors": factors,
                "late_entry": late,
                "signal_class": sc,
                "data_warnings": warnings,
                "construction": mode,
                "alt_btc_price": float(rel["close"].iloc[-1]),
                "btc_price": float(btc["close"].iloc[-1]),
                "alt_usdt_price": alt_usdt_px,
                "btc_dominance": dominance_pct,
                "n_relative_bars": len(rel),
                "model_version": self.champion.version if self.champion else "config",
                "model_type": "champion" if self.champion else "config_baseline",
            })

        ranked = rank_signals(rows, self.cfg["probability"]["primary_rank_horizon"])
        self.last_universe_audit = build_universe_audit(
            self.cfg, self.panels, self.unavailable, self.construction
        )
        self.last_data_meta = {
            "decision_candle_ts": str(ts),
            "btc_age": str(btc_age),
            "unavailable": sorted(self.unavailable),
            "probability_kind": ranked[0]["probability_kind"] if ranked else "n/a",
            "model_version": self.champion.version if self.champion else "config",
        }

        # Persist legacy predictions (compat)
        store_rows = []
        for r in ranked:
            store_rows.append({
                "timestamp": r["timestamp"],
                "symbol": r["symbol"],
                "probability_1h": r["p_1h"],
                "probability_4h": r["p_4h"],
                "probability_8h": r["p_8h"],
                "probability_12h": r["p_12h"],
                "probability_24h": r["p_24h"],
                "probability_kind": r["probability_kind"],
                "signal_score": r["signal_score"],
                "signal_class": r["signal_class"]["signal_class"],
                "momentum_score": r["factors"]["momentum"]["score"],
                "trend_score": r["factors"]["trend"]["score"],
                "btc_regime_score": r["factors"]["btc_regime"]["score"],
                "volume_score": r["factors"]["volume"]["score"],
                "volatility_score": r["factors"]["volatility"]["score"],
                "rsi_score": r["factors"]["rsi"]["score"],
                "structure_score": r["factors"]["structure"]["score"],
                "late_entry_score": r["late_entry"]["late_entry_score"],
                "late_entry_class": r["late_entry"]["classification"],
                "alt_btc_price": r["alt_btc_price"],
                "btc_price": r["btc_price"],
                "btc_dominance": r["btc_dominance"],
            })
        self.store.append_many(store_rows)
        self.store.backfill_outcomes(alt_btc_panels, self.interval)

        # Adaptive store: every symbol, full indicators, delayed labels
        if self.adaptive_enabled and self.adaptive_store is not None:
            self._persist_adaptive(ranked, ts, btc_age, dom)
            self.adaptive_store.backfill_outcomes(alt_btc_panels, self.interval)
            self._maybe_run_checkpoint()

        # Telegram — main Top-5 then separate indicator breakdown (same ranking)
        if self.cfg["telegram"].get("send_ranking_every_cycle", True):
            self.tg.send_ranking_and_breakdown(
                ranked,
                self.last_dominance_summary,
                top_n=int(self.cfg["experiment"]["top_n_rank"]),
                detail_n=int(self.cfg["experiment"]["top_n_detail"]),
                ts=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else datetime.now(timezone.utc),
                data_meta=self.last_data_meta,
            )

        if self.cfg["telegram"].get("send_exhaustion_alerts", True):
            self.tg.maybe_send_exhaustion(
                ranked,
                float(self.cfg["late_entry"]["alert_threshold"]),
                int(self.cfg["late_entry"]["alert_cooldown_minutes"]),
            )

        # Selector E-v1 or Adaptive V2 paper sim
        if self.selector_live_engine is not None and ranked:
            try:
                dom_ts = dom_obs_ts
                dom_src = getattr(dom, "source", None) if dom is not None else None
                self.selector_live_engine.process_ranked_cycle(
                    ranked,
                    decision_ts=ts,
                    rel_panels=alt_btc_panels,
                    btc_df=btc,
                    dominance_pct=dominance_pct,
                    dominance_ts=dom_ts,
                    dominance_source=dom_src,
                )
            except Exception as e:
                logger.exception("Selector E-v1 live cycle failed: %s", e)
        elif self.sim_engine is not None and ranked:
            try:
                dom_ts = dom_obs_ts
                dom_src = getattr(dom, "source", None) if dom is not None else None
                self.sim_engine.process_ranked_cycle(
                    ranked,
                    decision_ts=ts,
                    rel_panels=alt_btc_panels,
                    btc_df=btc,
                    dominance_pct=dominance_pct,
                    dominance_ts=dom_ts,
                    dominance_source=dom_src,
                )
            except Exception as e:
                logger.exception("Adaptive V2 sim cycle failed: %s", e)
                try:
                    self.sim_engine.alerter.runtime_error(f"sim_cycle: {e}")
                except Exception:
                    pass

        wver = None
        health_ok = None
        if self.sim_engine is not None:
            try:
                wver = self.sim_engine.weights_version()
            except Exception:
                wver = None
        self.runtime.mark_cycle_ok(
            decision_candle_ts=ts_key,
            n_ranked=len(ranked),
            weights_version=wver,
            health_ok=health_ok,
        )
        self.runtime.update(last_successful_data_timestamp=ts_key)

        return ranked

    def _persist_adaptive(self, ranked: list[dict[str, Any]], ts, btc_age, dom) -> None:
        tag = self.adaptive_cfg.get("data_source_tag_live", "LIVE_OBSERVATION_DATA")
        dom_ts = None
        dom_age = None
        if dom is not None:
            dom_ts = getattr(dom, "timestamp", None)
            if dom_ts is not None:
                try:
                    now = datetime.now(timezone.utc)
                    dt = dom_ts if dom_ts.tzinfo else dom_ts.replace(tzinfo=timezone.utc)
                    dom_age = (now - dt).total_seconds() / 3600.0
                except Exception:
                    dom_age = None
        rows = []
        for r in ranked:
            flat = flatten_indicators_and_factors(r["factors"])
            rows.append({
                "timestamp": r["timestamp"],
                "symbol": r["symbol"],
                "base": r.get("base"),
                "rank": r.get("rank"),
                "model_version": r.get("model_version", "config"),
                "model_type": r.get("model_type", "champion"),
                "data_source": tag,
                "probability_kind": r["probability_kind"],
                "signal_score": r["signal_score"],
                **flat,
                "probability_1h": r["p_1h"],
                "probability_4h": r["p_4h"],
                "probability_8h": r["p_8h"],
                "probability_12h": r["p_12h"],
                "probability_24h": r["p_24h"],
                "late_entry_score": r["late_entry"]["late_entry_score"],
                "late_entry_class": r["late_entry"]["classification"],
                "alt_btc_price": r["alt_btc_price"],
                "btc_price": r["btc_price"],
                "alt_usdt_price": r.get("alt_usdt_price"),
                "btc_dominance": r.get("btc_dominance"),
                "btc_dominance_obs_ts": str(dom_ts) if dom_ts is not None else None,
                "btc_dominance_age_hours": dom_age,
                "candle_timestamp": str(ts),
                "data_age": str(btc_age),
            })
        n = self.adaptive_store.append_rows(rows)
        logger.info("Adaptive store appended %d LIVE_OBSERVATION_DATA rows", n)

    def _maybe_run_checkpoint(self) -> None:
        """24h checkpoint = check for Challenger, not automatic weight update."""
        hours = float(self.adaptive_cfg.get("checkpoint_hours", 24))
        now = datetime.now(timezone.utc)
        last = None
        if self._checkpoint_state_path.exists():
            try:
                import json
                data = json.loads(self._checkpoint_state_path.read_text(encoding="utf-8"))
                last = datetime.fromisoformat(data["last_utc"])
            except Exception:
                last = None
        if last is not None and (now - last).total_seconds() < hours * 3600:
            return
        logger.info("Adaptive checkpoint due (every %.0fh) — evaluating Challenger eligibility", hours)
        result = run_checkpoint(
            self.adaptive_store,
            self.models_dir,
            self.reports_dir,
            self.cfg,
            self.adaptive_cfg,
        )
        self.models_dir.mkdir(parents=True, exist_ok=True)
        import json
        self._checkpoint_state_path.write_text(
            json.dumps({"last_utc": now.isoformat(), "result_action": result.get("action")}, indent=2),
            encoding="utf-8",
        )
        # Reload champion if promoted
        if result.get("action") == "PROMOTE_CHALLENGER" or result.get("promote"):
            self.champion = get_champion(self.models_dir)
            self.champion_weights = (
                self.champion.normalized_weights() if self.champion else None
            )
            logger.info("Champion updated to %s", self.champion.version if self.champion else None)
        else:
            logger.info(
                "Checkpoint: %s (mature_4h=%s)",
                result.get("action") or result.get("decision"),
                result.get("mature_4h_observations"),
            )
