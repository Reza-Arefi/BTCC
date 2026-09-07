"""Independent safety layer — strategy cannot bypass HALT."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class SafetyState(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    HALT = "HALT"


NotifyFn = Callable[..., Any]


@dataclass
class SafetySystem:
    state: SafetyState = SafetyState.NORMAL
    reasons: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    on_notify: NotifyFn | None = None
    # Optional operator/runtime gate (pause/stop/corrupt). Does not replace HALT.
    _entries_blocker: Callable[[], bool] | None = field(default=None, repr=False)
    # Dedupe Telegram/operator spam for the same unresolved condition.
    _warn_last_notified: dict[str, float] = field(default_factory=dict, repr=False)
    warn_notify_cooldown_sec: float = 900.0

    def set_entries_blocker(self, fn: Callable[[], bool] | None) -> None:
        """Block NEW entries when fn() is True (e.g. Telegram PAUSED/STOPPED)."""
        self._entries_blocker = fn

    def allow_new_entries(self) -> bool:
        if self.state == SafetyState.HALT:
            return False
        if self._entries_blocker is not None:
            try:
                if bool(self._entries_blocker()):
                    return False
            except Exception:  # noqa: BLE001 — fail closed
                return False
        return True

    @staticmethod
    def _warn_key(reason: str, details: dict[str, Any]) -> str:
        return f"{reason}|{details.get('symbol') or ''}|{details.get('trade_id') or ''}"

    def clear_warn(self, reason: str, **details: Any) -> None:
        """Clear dedupe so a resolved-then-recurring condition can alarm again."""
        self._warn_last_notified.pop(self._warn_key(reason, details), None)

    def warn(self, reason: str, **details: Any) -> None:
        if self.state == SafetyState.HALT:
            self._emit("WARNING_WHILE_HALTED", reason, details)
            return
        if self.state == SafetyState.NORMAL:
            self.state = SafetyState.WARNING
        self.reasons.append(reason)
        self._emit("WARNING", reason, details)
        key = self._warn_key(reason, details)
        now = time.time()
        last = self._warn_last_notified.get(key)
        if last is not None and (now - last) < float(self.warn_notify_cooldown_sec):
            self._emit(
                "WARNING_NOTIFY_SUPPRESSED",
                reason,
                {**details, "cooldown_sec": self.warn_notify_cooldown_sec, "since_sec": now - last},
            )
            return
        self._warn_last_notified[key] = now
        self._safe_notify("WARNING", reason, details)

    def halt(self, reason: str, **details: Any) -> None:
        self.state = SafetyState.HALT
        self.reasons.append(reason)
        self._emit("HALT", reason, details)
        self._safe_notify("HALT", reason, details)

    def _safe_notify(self, event: str, reason: str, details: dict[str, Any]) -> None:
        if not self.on_notify:
            return
        try:
            if event == "HALT":
                self.on_notify(
                    "HALT",
                    f"BOT HALTED: {reason}",
                    severity="CRITICAL",
                    reason=reason,
                    details=details,
                )
            else:
                self.on_notify(
                    event,
                    f"Safety warning: {reason}",
                    severity="WARNING",
                    reason=reason,
                    details=details,
                )
        except Exception:  # noqa: BLE001 — notifications must never crash safety
            return

    def clear_warning_if_healthy(self) -> None:
        if self.state == SafetyState.WARNING:
            self.state = SafetyState.NORMAL
            self.reasons.clear()
            self._emit("RECOVERED", "warnings_cleared", {})

    def assert_pretrade(
        self,
        *,
        symbol_ok: bool,
        balance_ok: bool,
        filters_ok: bool,
        duplicate: bool,
        stale_data: bool,
        api_ok: bool,
        reconciliation_ok: bool,
        max_risk_ok: bool,
        max_exposure_ok: bool,
        leverage_ok: bool = True,
    ) -> bool:
        """Return True if new order may proceed; HALT/WARNING otherwise."""
        if not leverage_ok:
            self.halt("LEVERAGE_FORBIDDEN")
            return False
        if not api_ok:
            self.halt("API_FAILURE")
            return False
        if not reconciliation_ok:
            self.halt("RECONCILIATION_MISMATCH")
            return False
        if duplicate:
            self.halt("DUPLICATE_ORDER_PROTECTION")
            return False
        if stale_data:
            self.halt("STALE_DATA")
            return False
        if not symbol_ok:
            self.halt("SYMBOL_VALIDATION_FAILED")
            return False
        if not balance_ok:
            self.warn("BALANCE_VALIDATION_FAILED")
            return False
        if not filters_ok:
            self.halt("BINANCE_FILTER_VALIDATION_FAILED")
            return False
        if not max_risk_ok:
            self.halt("MAX_RISK_EXCEEDED")
            return False
        if not max_exposure_ok:
            self.halt("MAX_EXPOSURE_EXCEEDED")
            return False
        return self.allow_new_entries()

    def _emit(self, event: str, reason: str, details: dict[str, Any]) -> None:
        self.events.append({"event": event, "reason": reason, "state": self.state.value, **details})
