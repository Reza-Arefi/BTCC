"""Telegram notification channel — credentials from environment only."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from binance_btc_bot.notifications.base import NotificationChannel, NotificationEvent, NotificationResult
from binance_btc_bot.secrets import scrub_text

logger = logging.getLogger(__name__)


class TelegramNotifier(NotificationChannel):
    name = "telegram"

    def __init__(
        self,
        *,
        bot_token: str | None = None,
        chat_id: str | None = None,
        enabled: bool = True,
        timeout_sec: float = 10.0,
    ) -> None:
        self.bot_token = (bot_token if bot_token is not None else os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
        self.chat_id = (chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
        self.enabled = bool(enabled)
        self.timeout_sec = float(timeout_sec)

    def configured(self) -> bool:
        return bool(self.enabled and self.bot_token and self.chat_id)

    def status(self) -> dict[str, Any]:
        return {
            "channel": self.name,
            "enabled": self.enabled,
            "configured": self.configured(),
            "bot_token": "SET" if self.bot_token else "MISSING",
            "chat_id": "SET" if self.chat_id else "MISSING",
        }

    def send(self, event: NotificationEvent) -> NotificationResult:
        if not self.enabled:
            return NotificationResult(ok=True, channel=self.name, reason="DISABLED", skipped=True)
        if not self.bot_token or not self.chat_id:
            return NotificationResult(ok=False, channel=self.name, reason="MISSING_CONFIG", skipped=True)
        text = _format_telegram(event)
        text = scrub_text(text, extra_secrets=[self.bot_token])
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            body = urllib.parse.urlencode(
                {"chat_id": self.chat_id, "text": text, "disable_web_page_preview": "true"}
            ).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "binance-btc-bot/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw else {}
            if not data.get("ok", False):
                reason = scrub_text(str(data.get("description") or "TELEGRAM_API_ERROR"))
                logger.warning("Telegram send failed: %s", reason)
                return NotificationResult(ok=False, channel=self.name, reason=reason)
            return NotificationResult(ok=True, channel=self.name, reason="OK")
        except Exception as e:  # noqa: BLE001 — never crash the bot
            reason = scrub_text(f"{type(e).__name__}: {e}", extra_secrets=[self.bot_token])
            logger.warning("Telegram send exception: %s", reason)
            return NotificationResult(ok=False, channel=self.name, reason=reason)


def _format_telegram(event: NotificationEvent) -> str:
    from binance_btc_bot.notifications.telegram_reports import format_notification_message

    formatted = format_notification_message(event.event, event.details)
    if formatted:
        return formatted
    lines = [
        f"[{event.severity.value}] {event.event}",
        event.message,
    ]
    if event.symbol:
        lines.append(f"symbol={event.symbol}")
    if event.trade_id:
        lines.append(f"trade_id={event.trade_id}")
    if event.order_id:
        lines.append(f"order_id={event.order_id}")
    if event.reason:
        lines.append(f"reason={event.reason}")
    return "\n".join(lines)
