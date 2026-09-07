"""Notification manager — Telegram for ops, SMS for critical escalation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from binance_btc_bot.notifications.base import (
    NotificationChannel,
    NotificationEvent,
    NotificationResult,
    Severity,
)
from binance_btc_bot.notifications.sms import SMSNotifier, build_sms_provider_from_env
from binance_btc_bot.notifications.telegram import TelegramNotifier
from binance_btc_bot.secrets import scrub_obj, scrub_text

logger = logging.getLogger(__name__)

# Canonical event names
NORMAL_EVENTS = {
    "BOT_STARTED",
    "BOT_STOPPED",
    "SIGNAL_DETECTED",
    "ENTRY_SUBMITTED",
    "ENTRY_FILLED",
    "TRAILING_OCO_SUBMITTED",
    "TRAILING_ACTIVATED",
    "EXIT_FILLED",
    "DAILY_SUMMARY",
    "HOURLY_REPORT",
}
IMPORTANT_EVENTS = {
    "RECOVERY",
    "API_ERROR",
    "ORDER_ERROR",
    "CANCEL_ERROR",
}
CRITICAL_EVENTS = {
    "HALT",
    "UNPROTECTED_POSITION",
    "OCO_FAILURE",
    "PROTECTION_FAILED",
    "PARTIAL_FILL_UNSAFE",
    "STATE_RECONCILIATION_FAILURE",
    "UNEXPECTED_ACCOUNT_STATE",
    "BOT_HALTED",
    "RECOVERY_FAILURE",
    "CRITICAL_API_FAILURE",
    "BINANCE_ACCOUNT_ORDER_STATE_ERROR",
    "OPERATOR_EMERGENCY",
}

DEFAULT_POLICY = {
    "INFO": ["telegram"],
    "WARNING": ["telegram"],
    "ERROR": ["telegram"],
    "CRITICAL": ["telegram", "sms"],
}


@dataclass
class NotificationManager:
    telegram: NotificationChannel
    sms: NotificationChannel
    policy: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_POLICY))
    enabled: bool = True
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None = None) -> "NotificationManager":
        ncfg = (cfg or {}).get("notifications") or {}
        policy = dict(DEFAULT_POLICY)
        raw_policy = ncfg.get("policy") or {}
        for sev, channels in raw_policy.items():
            policy[str(sev).upper()] = [str(c).lower() for c in channels]
        tg_enabled = bool((ncfg.get("telegram") or {}).get("enabled", True))
        sms_enabled = bool((ncfg.get("sms") or {}).get("enabled", True))
        return cls(
            telegram=TelegramNotifier(enabled=tg_enabled),
            sms=SMSNotifier(provider=build_sms_provider_from_env(enabled=sms_enabled), enabled=sms_enabled),
            policy=policy,
            enabled=bool(ncfg.get("enabled", True)),
        )

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "policy": self.policy,
            "telegram": self.telegram.status(),
            "sms": self.sms.status(),
        }

    def notify_info(self, event: str, message: str, **kwargs: Any) -> list[NotificationResult]:
        return self._notify(event, message, Severity.INFO, **kwargs)

    def notify_warning(self, event: str, message: str, **kwargs: Any) -> list[NotificationResult]:
        return self._notify(event, message, Severity.WARNING, **kwargs)

    def notify_error(self, event: str, message: str, **kwargs: Any) -> list[NotificationResult]:
        return self._notify(event, message, Severity.ERROR, **kwargs)

    def notify_critical(self, event: str, message: str, **kwargs: Any) -> list[NotificationResult]:
        return self._notify(event, message, Severity.CRITICAL, **kwargs)

    def notify_event(self, event: str, message: str, **kwargs: Any) -> list[NotificationResult]:
        """Route by canonical event class."""
        name = str(event).upper()
        if name in CRITICAL_EVENTS or name in {"BOT HALTED", "UNPROTECTED POSITION", "OCO FAILURE"}:
            return self.notify_critical(name, message, **kwargs)
        if name in IMPORTANT_EVENTS:
            return self.notify_error(name, message, **kwargs)
        return self.notify_info(name, message, **kwargs)

    def _notify(
        self,
        event: str,
        message: str,
        severity: Severity,
        *,
        symbol: str | None = None,
        trade_id: str | None = None,
        order_id: str | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> list[NotificationResult]:
        results: list[NotificationResult] = []
        if not self.enabled:
            return [NotificationResult(ok=True, channel="manager", reason="DISABLED", skipped=True)]
        try:
            details_clean = scrub_obj(details or {})
            message_clean = scrub_text(message)
            from binance_btc_bot.notifications.telegram_reports import format_notification_message

            fancy = format_notification_message(str(event).upper(), details_clean)
            if fancy:
                message_clean = fancy
            payload = NotificationEvent(
                event=str(event).upper(),
                severity=severity,
                message=message_clean,
                symbol=symbol,
                trade_id=trade_id,
                order_id=order_id,
                reason=scrub_text(reason) if reason else None,
                details=details_clean,
            )
            channels = self.policy.get(severity.value, ["telegram"])
            for ch in channels:
                try:
                    if ch == "telegram":
                        results.append(self.telegram.send(payload))
                    elif ch == "sms":
                        results.append(self.sms.send(payload))
                    else:
                        results.append(
                            NotificationResult(ok=False, channel=ch, reason="UNKNOWN_CHANNEL", skipped=True)
                        )
                except Exception as e:  # noqa: BLE001 — isolate channel failures
                    logger.warning("Notification channel %s failed: %s", ch, scrub_text(str(e)))
                    results.append(
                        NotificationResult(ok=False, channel=ch, reason=scrub_text(f"{type(e).__name__}: {e}"))
                    )
            self.history.append(
                {
                    "event": payload.event,
                    "severity": severity.value,
                    "results": [r.__dict__ for r in results],
                }
            )
            if len(self.history) > 200:
                self.history = self.history[-200:]
        except Exception as e:  # noqa: BLE001 — never crash trading engine
            logger.warning("NotificationManager failure: %s", scrub_text(str(e)))
            results.append(
                NotificationResult(ok=False, channel="manager", reason=scrub_text(f"{type(e).__name__}: {e}"))
            )
        return results
