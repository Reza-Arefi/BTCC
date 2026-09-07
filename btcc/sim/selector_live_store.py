"""Persistent append-only store for Selector E-v1 live engine."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from btcc.sim.state_machine import CrossingStateMachine

logger = logging.getLogger(__name__)


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


class SelectorLiveStore:
    def __init__(self, sim_cfg: dict[str, Any]):
        storage = sim_cfg.get("storage") or {}
        self.retention_days = int(storage.get("retention_days", 365))
        self.predictions_path = Path(storage["predictions_path"])
        self.opportunities_path = Path(storage["opportunities_path"])
        self.strategy_legs_path = Path(storage["strategy_legs_path"])
        self.selection_audit_path = Path(storage["selection_audit_path"])
        self.runtime_state_path = Path(storage["runtime_state_path"])
        self.state_path = Path(storage.get("state_path") or storage["root_dir"] + "/state.json")
        self.analytics_dir = Path(storage.get("analytics_dir") or "data/selector_live/analytics")
        for p in (
            self.predictions_path,
            self.opportunities_path,
            self.strategy_legs_path,
            self.selection_audit_path,
            self.runtime_state_path,
            self.state_path,
        ):
            p.parent.mkdir(parents=True, exist_ok=True)
        self.analytics_dir.mkdir(parents=True, exist_ok=True)

    def _append_rows(self, path: Path, rows: list[dict[str, Any]], dedupe_cols: list[str] | None = None) -> None:
        if not rows:
            return
        new_df = pd.DataFrame(rows)
        if path.exists() and path.stat().st_size > 0:
            old = pd.read_csv(path, low_memory=False)
            df = pd.concat([old, new_df], ignore_index=True)
            if dedupe_cols:
                cols = [c for c in dedupe_cols if c in df.columns]
                if cols:
                    df = df.drop_duplicates(subset=cols, keep="last")
        else:
            df = new_df
        df = self._apply_retention(df)
        _atomic_write_df(path, df)

    def _apply_retention(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or self.retention_days <= 0:
            return df
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        for col in ("exit_ts", "entry_ts", "timestamp", "opened_ts", "signal_timestamp"):
            if col in df.columns:
                ts = pd.to_datetime(df[col], utc=True, errors="coerce")
                keep = ts.isna() | (ts >= cutoff)
                if keep.any() and (~keep).any():
                    return df[keep].copy()
        return df

    def append_predictions(self, rows: list[dict[str, Any]]) -> None:
        self._append_rows(self.predictions_path, rows, dedupe_cols=["timestamp", "symbol"])

    def append_opportunities(self, rows: list[dict[str, Any]]) -> None:
        self._append_rows(self.opportunities_path, rows, dedupe_cols=["opportunity_id"])

    def append_strategy_legs(self, rows: list[dict[str, Any]]) -> None:
        self._append_rows(
            self.strategy_legs_path,
            rows,
            dedupe_cols=["opportunity_id", "arm_key", "strategy_key", "exit_ts"],
        )

    def append_selection_audit(self, rows: list[dict[str, Any]]) -> None:
        self._append_rows(self.selection_audit_path, rows, dedupe_cols=["opportunity_id", "selector_id"])

    def save_runtime_state(self, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload["written_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_text(self.runtime_state_path, json.dumps(payload, indent=2, default=str))

    def load_runtime_state(self) -> dict[str, Any] | None:
        if not self.runtime_state_path.exists():
            return None
        return json.loads(self.runtime_state_path.read_text(encoding="utf-8"))

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def save_state(self, state: dict[str, Any]) -> None:
        _atomic_write_text(self.state_path, json.dumps(state, indent=2, default=str))

    def save_state_machine(self, sm: CrossingStateMachine, *, extra: dict[str, Any] | None = None) -> None:
        state = self.load_state()
        state["state_machine"] = sm.to_dict()
        if extra:
            state.update(extra)
        self.save_state(state)

    def load_state_machine(self, sim_cfg: dict[str, Any]) -> CrossingStateMachine:
        state = self.load_state()
        raw = state.get("state_machine")
        if raw:
            return CrossingStateMachine.from_dict(raw)
        return CrossingStateMachine(
            long_threshold=float(sim_cfg.get("long_threshold", 0.60)),
            upper_threshold=sim_cfg.get("upper_threshold"),
            max_open=int(sim_cfg.get("max_open_opportunities", 10)),
            one_per_pair=bool(sim_cfg.get("one_opportunity_per_pair", True)),
        )

    def set_open_opportunities(self, opps: list[dict[str, Any]]) -> None:
        state = self.load_state()
        state["open_opportunities"] = opps
        self.save_state(state)

    def open_opportunities(self) -> list[dict[str, Any]]:
        return list(self.load_state().get("open_opportunities") or [])

    def read_legs(self) -> pd.DataFrame:
        if self.strategy_legs_path.exists() and self.strategy_legs_path.stat().st_size > 0:
            return pd.read_csv(self.strategy_legs_path, low_memory=False)
        return pd.DataFrame()

    def read_selection_audit(self) -> pd.DataFrame:
        if self.selection_audit_path.exists() and self.selection_audit_path.stat().st_size > 0:
            return pd.read_csv(self.selection_audit_path, low_memory=False)
        return pd.DataFrame()

    def read_opportunities(self) -> pd.DataFrame:
        if self.opportunities_path.exists() and self.opportunities_path.stat().st_size > 0:
            return pd.read_csv(self.opportunities_path, low_memory=False)
        return pd.DataFrame()
