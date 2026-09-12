"""Immutable daily live-trade archive (signals / orders / trades / summaries).

Writes under results/live_archive/YYYY-MM-DD/ and never overwrites prior days.
Research / walk-forward selection may consume this later — no auto-switching here.
"""

from __future__ import annotations

import csv
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SIGNAL_FIELDS = [
    "archived_at",
    "timestamp",
    "symbol",
    "timeframe",
    "signal_id",
    "previous_S",
    "current_S",
    "threshold",
    "genuine_new_cross",
    "strategy",
    "entry_profile",
    "config_version",
    "late_entry_score",
    "late_entry_status",
    "equity_btc_before",
    "classification",
    "relative_price",
    "payload_json",
]

ORDER_FIELDS = [
    "archived_at",
    "timestamp",
    "order_id",
    "client_order_id",
    "symbol",
    "side",
    "order_type",
    "quantity",
    "requested_price",
    "executed_quantity",
    "average_execution_price",
    "status",
    "fees",
    "fee_asset",
    "execution_timestamps",
    "cancellation_reason",
    "trade_id",
    "config_version",
    "payload_json",
]

TRADE_FIELDS = [
    "archived_at",
    "trade_id",
    "signal_id",
    "symbol",
    "strategy",
    "selector",
    "entry_timestamp",
    "exit_timestamp",
    "entry_price",
    "exit_price",
    "quantity",
    "gross_pnl",
    "fees",
    "net_pnl",
    "pnl_btc",
    "pnl_pct",
    "holding_time",
    "outcome",
    "exit_reason",
    "stop_loss",
    "trail_activation",
    "mfe",
    "mae",
    "equity_btc_before",
    "equity_btc_after",
    "config_version",
    "status",
    "payload_json",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _day_key(ts: datetime | None = None) -> str:
    return (ts or _utc_now()).strftime("%Y-%m-%d")


class LiveDailyArchive:
    """Append-only daily CSV/JSON archive. Safe for dry-run and live."""

    def __init__(
        self,
        root: Path | str,
        *,
        fingerprint: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> None:
        self.root = Path(root)
        self.fingerprint = dict(fingerprint or {})
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self.config_version = str(self.fingerprint.get("config_version") or "T30_E2_T65_v1")

    def _day_dir(self, day: str | None = None) -> Path:
        d = self.root / (day or _day_key())
        d.mkdir(parents=True, exist_ok=True)
        meta = d / "metadata.json"
        if not meta.exists():
            meta.write_text(
                json.dumps(
                    {
                        "date": d.name,
                        "created_at": _utc_now().isoformat(),
                        "config_version": self.config_version,
                        "fingerprint": self.fingerprint,
                        "note": "Immutable daily archive — do not overwrite historical rows",
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        return d

    def _append_csv(self, path: Path, fields: list[str], row: dict[str, Any]) -> None:
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k) for k in fields})

    def record_signal(self, row: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            with self._lock:
                d = self._day_dir()
                payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
                out = {
                    "archived_at": _utc_now().isoformat(),
                    "timestamp": row.get("timestamp") or row.get("created_at") or _utc_now().isoformat(),
                    "symbol": row.get("symbol"),
                    "timeframe": row.get("timeframe") or payload.get("timeframe") or "15m",
                    "signal_id": row.get("signal_id") or payload.get("signal_id"),
                    "previous_S": row.get("previous_S") or payload.get("previous_S") or payload.get("S_prev"),
                    "current_S": row.get("score") or payload.get("S") or payload.get("S_curr"),
                    "threshold": row.get("threshold")
                    or payload.get("threshold")
                    or self.fingerprint.get("threshold"),
                    "genuine_new_cross": row.get("genuine_new_cross")
                    if row.get("genuine_new_cross") is not None
                    else payload.get("genuine_new_cross"),
                    "strategy": row.get("strategy") or self.fingerprint.get("strategy"),
                    "entry_profile": self.fingerprint.get("entry_profile"),
                    "config_version": self.config_version,
                    "late_entry_score": payload.get("late_entry_score"),
                    "late_entry_status": payload.get("late_entry_status") or payload.get("late_entry_class"),
                    "equity_btc_before": row.get("equity_btc_before") or payload.get("equity_btc_before"),
                    "classification": row.get("classification"),
                    "relative_price": row.get("relative_price"),
                    "payload_json": json.dumps(payload or {}, default=str, sort_keys=True),
                }
                self._append_csv(d / "signals.csv", SIGNAL_FIELDS, out)
                with (d / "entry_factors.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"signal": out, "factors": payload}, default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.warning("live archive signal write failed: %s", e)

    def record_order(self, row: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            with self._lock:
                d = self._day_dir()
                out = {
                    "archived_at": _utc_now().isoformat(),
                    "timestamp": row.get("timestamp") or row.get("created_at") or _utc_now().isoformat(),
                    "order_id": row.get("order_id") or row.get("binance_order_id"),
                    "client_order_id": row.get("client_order_id"),
                    "symbol": row.get("symbol"),
                    "side": row.get("side"),
                    "order_type": row.get("order_type") or row.get("type"),
                    "quantity": row.get("quantity") or row.get("qty"),
                    "requested_price": row.get("requested_price") or row.get("price"),
                    "executed_quantity": row.get("executed_quantity") or row.get("executed_qty"),
                    "average_execution_price": row.get("average_execution_price") or row.get("avg_price"),
                    "status": row.get("status"),
                    "fees": row.get("fees") or row.get("fee"),
                    "fee_asset": row.get("fee_asset"),
                    "execution_timestamps": row.get("execution_timestamps"),
                    "cancellation_reason": row.get("cancellation_reason") or row.get("cancel_reason"),
                    "trade_id": row.get("trade_id"),
                    "config_version": self.config_version,
                    "payload_json": json.dumps(row, default=str, sort_keys=True),
                }
                self._append_csv(d / "orders.csv", ORDER_FIELDS, out)
        except Exception as e:  # noqa: BLE001
            logger.warning("live archive order write failed: %s", e)

    def record_trade(self, row: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            with self._lock:
                d = self._day_dir()
                pnl_btc = row.get("realized_pnl_btc")
                if pnl_btc is None:
                    pnl_btc = row.get("pnl_btc")
                outcome = row.get("outcome")
                if outcome is None and pnl_btc is not None:
                    try:
                        v = float(pnl_btc)
                        outcome = "WIN" if v > 1e-12 else ("LOSS" if v < -1e-12 else "FLAT")
                    except (TypeError, ValueError):
                        outcome = None
                out = {
                    "archived_at": _utc_now().isoformat(),
                    "trade_id": row.get("trade_id"),
                    "signal_id": row.get("signal_id"),
                    "symbol": row.get("symbol"),
                    "strategy": row.get("strategy") or self.fingerprint.get("strategy"),
                    "selector": row.get("selector"),
                    "entry_timestamp": row.get("entry_time") or row.get("entry_timestamp"),
                    "exit_timestamp": row.get("exit_time") or row.get("exit_timestamp"),
                    "entry_price": row.get("entry_price"),
                    "exit_price": row.get("exit_price"),
                    "quantity": row.get("quantity"),
                    "gross_pnl": row.get("gross_pnl"),
                    "fees": row.get("fees_btc") or row.get("fees"),
                    "net_pnl": row.get("net_pnl") or pnl_btc,
                    "pnl_btc": pnl_btc,
                    "pnl_pct": row.get("pnl_pct") or row.get("realized_pnl_pct"),
                    "holding_time": row.get("holding_time"),
                    "outcome": outcome,
                    "exit_reason": row.get("exit_reason"),
                    "stop_loss": row.get("stop_loss") or row.get("initial_sl"),
                    "trail_activation": row.get("trail_activation") or row.get("configured_activation"),
                    "mfe": row.get("mfe") or row.get("mfe_pct"),
                    "mae": row.get("mae") or row.get("mae_pct"),
                    "equity_btc_before": row.get("equity_btc_before"),
                    "equity_btc_after": row.get("equity_btc_after"),
                    "config_version": self.config_version,
                    "status": row.get("status"),
                    "payload_json": json.dumps(row, default=str, sort_keys=True),
                }
                self._append_csv(d / "trades.csv", TRADE_FIELDS, out)
        except Exception as e:  # noqa: BLE001
            logger.warning("live archive trade write failed: %s", e)

    def write_daily_summary(self, summary: dict[str, Any], *, day: str | None = None) -> Path | None:
        if not self.enabled:
            return None
        with self._lock:
            d = self._day_dir(day)
            path = d / "daily_summary.json"
            payload = {
                "date": d.name,
                "written_at": _utc_now().isoformat(),
                "strategy": self.fingerprint.get("strategy"),
                "config_version": self.config_version,
                **summary,
            }
            if path.exists():
                try:
                    prev = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(prev, dict):
                        prev.update(payload)
                        payload = prev
                except Exception:  # noqa: BLE001
                    pass
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            return path


def attach_archive_to_database(db: Any, archive: LiveDailyArchive) -> None:
    """Wrap insert_* methods so every DB write is mirrored to the daily archive."""
    if getattr(db, "_live_archive_attached", False):
        return

    _insert_signal = db.insert_signal
    _insert_order = db.insert_order
    _insert_trade = db.insert_trade
    _update_trade = db.update_trade

    def insert_signal(**row: Any) -> None:
        _insert_signal(**row)
        archive.record_signal(row)

    def insert_order(**row: Any) -> None:
        _insert_order(**row)
        archive.record_order(row)

    def insert_trade(row: dict[str, Any]) -> None:
        _insert_trade(row)
        archive.record_trade(row)

    def update_trade(trade_id: str, **fields: Any) -> None:
        _update_trade(trade_id, **fields)
        if any(k in fields for k in ("status", "exit_time", "exit_price", "realized_pnl_btc")):
            archive.record_trade({"trade_id": trade_id, **fields})

    db.insert_signal = insert_signal  # type: ignore[method-assign]
    db.insert_order = insert_order  # type: ignore[method-assign]
    db.insert_trade = insert_trade  # type: ignore[method-assign]
    db.update_trade = update_trade  # type: ignore[method-assign]
    db._live_archive = archive
    db._live_archive_attached = True
