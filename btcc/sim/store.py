"""Append-only research storage for Adaptive V2 (predictions, opportunities, weights, state).

The rolling 90-day window controls LEARNING only — historical rows are never deleted.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from btcc.sim.state_machine import CrossingStateMachine

logger = logging.getLogger(__name__)

PRED_COLS = [
    "prediction_id", "timestamp", "symbol", "base", "construction",
    "alt_btc_price", "btc_price", "btc_usdt",
    "S", "long_threshold", "signal_generated", "trade_opened", "rejection_reason",
    "zone_state", "crossed_into", "opportunity_id",
    "horizon_hours", "model_weights_version",
    "signed_momentum", "signed_trend", "signed_btc_regime", "signed_volume",
    "signed_volatility", "signed_rsi", "signed_structure",
    "weight_momentum", "weight_trend", "weight_btc_regime", "weight_volume",
    "weight_volatility", "weight_rsi", "weight_structure",
    "factor_momentum", "factor_trend", "factor_btc_regime", "factor_volume",
    "factor_volatility", "factor_rsi", "factor_structure",
    "signal_score_legacy", "p_4h_legacy", "late_entry_score", "late_entry_class",
    "btc_dominance", "btc_d_status", "btc_d_age_seconds", "btc_d_available",
    "health_ok", "health_allow_new_trades", "health_reasons",
    "bot_version", "config_version",
    # Outcomes filled later — never overwrite prediction fields
    "outcome_ready", "future_alt_btc_price", "future_return_4h",
    "direction_actual", "prediction_correct", "prediction_score",
    "outcome_ts",
]

OPP_COLS = [
    "opportunity_id", "opened_ts", "signal_timestamp", "symbol", "base", "S", "threshold",
    "entry_fill_ts", "entry_alt_btc_mid", "entry_btc_usdt", "notional_usd",
    "status", "closed_ts", "n_legs_closed", "rejection_on_signal",
    "weight_version_id", "weights_calculated_at", "weights_effective_from",
    "weight_momentum", "weight_trend", "weight_btc_regime", "weight_volume",
    "weight_volatility", "weight_rsi", "weight_structure",
]

WEIGHT_COLS = [
    "update_timestamp", "update_id", "timezone", "learning_window_start",
    "learning_window_end", "n_samples", "indicator", "old_weight", "new_weight",
    "indicator_ic", "indicator_n", "update_status", "notes",
]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class SimStore:
    def __init__(self, sim_cfg: dict[str, Any]):
        storage = sim_cfg.get("storage") or {}
        self.predictions_path = Path(storage["predictions_path"])
        self.opportunities_path = Path(storage["opportunities_path"])
        self.strategy_legs_path = Path(storage["strategy_legs_path"])
        self.weight_history_path = Path(storage["weight_history_path"])
        self.learning_metrics_path = Path(storage["learning_metrics_path"])
        self.state_path = Path(storage["state_path"])
        self.daily_update_log_path = Path(storage["daily_update_log_path"])
        for p in (
            self.predictions_path,
            self.opportunities_path,
            self.strategy_legs_path,
            self.weight_history_path,
            self.learning_metrics_path,
            self.state_path,
            self.daily_update_log_path,
        ):
            p.parent.mkdir(parents=True, exist_ok=True)

    # --- CSV helpers ---
    def _read(self, path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path)

    def append_rows(self, path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
        if not rows:
            return
        new = pd.DataFrame(rows)
        if columns:
            for c in columns:
                if c not in new.columns:
                    new[c] = None
            new = new[columns]
        old = self._read(path)
        if old.empty:
            _atomic_write_df(path, new)
        else:
            # Align columns
            for c in new.columns:
                if c not in old.columns:
                    old[c] = None
            for c in old.columns:
                if c not in new.columns:
                    new[c] = None
            combined = pd.concat([old, new[old.columns]], ignore_index=True)
            _atomic_write_df(path, combined)

    def append_predictions(self, rows: list[dict[str, Any]]) -> None:
        """Append prediction rows; skip duplicates of (timestamp, symbol)."""
        if not rows:
            return
        existing = self.load_predictions()
        if not existing.empty and "timestamp" in existing.columns and "symbol" in existing.columns:
            keys = {
                (str(t), str(s))
                for t, s in zip(existing["timestamp"].astype(str), existing["symbol"].astype(str))
            }
            before = len(rows)
            rows = [r for r in rows if (str(r.get("timestamp")), str(r.get("symbol"))) not in keys]
            skipped = before - len(rows)
            if skipped:
                logger.info("Skipped %d duplicate sim predictions (timestamp,symbol)", skipped)
        if rows:
            self.append_rows(self.predictions_path, rows, PRED_COLS)

    def append_opportunities(self, rows: list[dict[str, Any]]) -> None:
        """Append opportunity rows; skip duplicate opportunity_id."""
        if not rows:
            return
        existing = self._read(self.opportunities_path)
        if not existing.empty and "opportunity_id" in existing.columns:
            seen = set(existing["opportunity_id"].astype(str))
            before = len(rows)
            rows = [r for r in rows if str(r.get("opportunity_id")) not in seen]
            skipped = before - len(rows)
            if skipped:
                logger.info("Skipped %d duplicate opportunities (opportunity_id)", skipped)
        if rows:
            self.append_rows(self.opportunities_path, rows, OPP_COLS)

    def append_strategy_legs(self, rows: list[dict[str, Any]]) -> None:
        self.append_rows(self.strategy_legs_path, rows, None)

    def append_weight_history(self, rows: list[dict[str, Any]]) -> None:
        self.append_rows(self.weight_history_path, rows, WEIGHT_COLS)

    def append_learning_metrics(self, rows: list[dict[str, Any]]) -> None:
        self.append_rows(self.learning_metrics_path, rows, None)

    def append_daily_update_log(self, record: dict[str, Any]) -> None:
        self.daily_update_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.daily_update_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def load_predictions(self) -> pd.DataFrame:
        df = self._read(self.predictions_path)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def load_weight_history(self) -> pd.DataFrame:
        return self._read(self.weight_history_path)

    # --- State (restart safety) ---
    def load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Failed to load sim state: %s", e)
            return {}

    def save_state(self, state: dict[str, Any]) -> None:
        _atomic_write_text(self.state_path, json.dumps(state, indent=2, default=str))

    def load_state_machine(self, sim_cfg: dict[str, Any]) -> CrossingStateMachine:
        state = self.load_state()
        raw = state.get("crossing") or {}
        if raw:
            return CrossingStateMachine.from_dict(raw)
        return CrossingStateMachine(
            long_threshold=float(sim_cfg.get("long_threshold", 0.60)),
            max_open=int(sim_cfg.get("max_open_opportunities", 10)),
            one_per_pair=bool(sim_cfg.get("one_opportunity_per_pair", True)),
        )

    def save_state_machine(self, sm: CrossingStateMachine, extra: dict[str, Any] | None = None) -> None:
        state = self.load_state()
        state["crossing"] = sm.to_dict()
        if extra:
            state.update(extra)
        self.save_state(state)

    def get_current_weights(self, fallback: dict[str, float]) -> dict[str, float]:
        state = self.load_state()
        w = state.get("current_weights")
        if isinstance(w, dict) and w:
            return {k: float(v) for k, v in w.items()}
        return dict(fallback)

    def set_current_weights(self, weights: dict[str, float], meta: dict[str, Any] | None = None) -> None:
        state = self.load_state()
        state["current_weights"] = dict(weights)
        state["weights_meta"] = meta or {}
        self.save_state(state)

    def last_daily_update_date(self) -> str | None:
        """Return America/Sao_Paulo calendar date string of last successful update."""
        return self.load_state().get("last_daily_update_local_date")

    def mark_daily_update(self, local_date: str, update_id: str) -> None:
        state = self.load_state()
        state["last_daily_update_local_date"] = local_date
        state["last_daily_update_id"] = update_id
        self.save_state(state)

    def open_opportunities_state(self) -> list[dict[str, Any]]:
        return list(self.load_state().get("open_opportunities") or [])

    def set_open_opportunities_state(self, rows: list[dict[str, Any]]) -> None:
        state = self.load_state()
        state["open_opportunities"] = rows
        self.save_state(state)

    def recent_open_timestamps(self) -> list[str]:
        return list(self.load_state().get("recent_open_timestamps") or [])

    def push_open_timestamp(self, iso_ts: str, keep: int = 50) -> list[str]:
        state = self.load_state()
        arr = list(state.get("recent_open_timestamps") or [])
        arr.append(iso_ts)
        arr = arr[-keep:]
        state["recent_open_timestamps"] = arr
        self.save_state(state)
        return arr
