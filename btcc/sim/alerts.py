"""Operational Telegram alerts for Adaptive V2 (separate from ranking messages)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


class OpsAlerter:
    """Thin wrapper — does not embed prediction/strategy logic."""

    def __init__(self, send_fn: Callable[[str], bool], enabled: bool = True, cfg: dict[str, Any] | None = None):
        self.send_fn = send_fn
        self.enabled = enabled
        self.cfg = cfg or {}

    def _send(self, text: str) -> bool:
        if not self.enabled:
            logger.info("Ops alert (disabled): %s", text[:200])
            return False
        try:
            return bool(self.send_fn(text))
        except Exception as e:
            logger.error("Ops alert send failed: %s", e)
            return False

    def startup(self, meta: dict[str, Any] | None = None) -> None:
        if not self.cfg.get("send_startup", True):
            return
        meta = meta or {}
        lines = [
            "BTCC Adaptive V2 — STARTUP",
            f"UTC: {datetime.now(timezone.utc).isoformat()}",
            f"sim_enabled: {meta.get('sim_enabled')}",
            f"long_threshold: {meta.get('long_threshold')}",
            f"max_open: {meta.get('max_open')}",
            f"weights_version: {meta.get('weights_version')}",
            "",
            "SIGNAL / PAPER SIM ONLY — NO EXCHANGE ORDERS",
        ]
        self._send("\n".join(lines))

    def shutdown(self, reason: str = "user_stop") -> None:
        if not self.cfg.get("send_shutdown", True):
            return
        self._send(
            f"BTCC Adaptive V2 — SHUTDOWN\nUTC: {datetime.now(timezone.utc).isoformat()}\nreason: {reason}"
        )

    def data_failure(self, detail: str) -> None:
        if not self.cfg.get("send_data_failure", True):
            return
        self._send(f"BTCC DATA FAILURE\nUTC: {datetime.now(timezone.utc).isoformat()}\n{detail}")

    def btcd_failure(self, detail: str) -> None:
        if not self.cfg.get("send_btcd_failure", True):
            return
        self._send(f"BTCC BTC.D FAILURE (new trades blocked)\nUTC: {datetime.now(timezone.utc).isoformat()}\n{detail}")

    def runtime_error(self, detail: str) -> None:
        if not self.cfg.get("send_runtime_error", True):
            return
        self._send(f"BTCC CRITICAL RUNTIME ERROR\nUTC: {datetime.now(timezone.utc).isoformat()}\n{detail}")

    def activity_burst(
        self,
        *,
        opens: list[dict[str, Any]],
        n_open: int,
        slots_remaining: int,
    ) -> None:
        if not self.cfg.get("send_activity_burst", True):
            return
        lines = [
            "BTCC ACTIVITY WARNING — 3+ opportunities in 60 minutes",
            f"UTC: {datetime.now(timezone.utc).isoformat()}",
            f"opens_in_window: {len(opens)}",
            f"current_open_opportunities: {n_open}",
            f"slots_remaining: {slots_remaining}",
            "",
        ]
        for o in opens[-10:]:
            lines.append(
                f"- {o.get('ts')} {o.get('symbol')} S={o.get('S')} dir=LONG_ALT_BTC"
            )
        self._send("\n".join(lines))

    def capacity_saturation(self, *, n_open: int, rejects: int) -> None:
        if not self.cfg.get("send_capacity_saturation", True):
            return
        self._send(
            "BTCC CAPACITY SATURATION\n"
            f"UTC: {datetime.now(timezone.utc).isoformat()}\n"
            f"open={n_open}/10 MAX_OPEN rejects (recent)={rejects}"
        )

    def daily_summary(self, text: str) -> None:
        if not self.cfg.get("send_daily_summary", True):
            return
        self._send(text[:3500])
