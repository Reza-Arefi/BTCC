"""SMS notification abstraction — critical escalation only.

Trading logic must not hard-code a vendor. TwilioSMSProvider is one implementation.
Credentials come from environment variables only.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from binance_btc_bot.notifications.base import NotificationChannel, NotificationEvent, NotificationResult
from binance_btc_bot.secrets import scrub_text

logger = logging.getLogger(__name__)


class SMSProvider(ABC):
    """Vendor-neutral SMS sender."""

    @abstractmethod
    def configured(self) -> bool: ...

    @abstractmethod
    def send_sms(self, body: str) -> NotificationResult: ...

    @abstractmethod
    def status(self) -> dict[str, Any]: ...


class NullSMSProvider(SMSProvider):
    def configured(self) -> bool:
        return False

    def send_sms(self, body: str) -> NotificationResult:
        return NotificationResult(ok=True, channel="sms", reason="NOT_CONFIGURED", skipped=True)

    def status(self) -> dict[str, Any]:
        return {"provider": "null", "configured": False}


class TwilioSMSProvider(SMSProvider):
    """Twilio REST SMS — uses TWILIO_* or SMS_* env vars."""

    def __init__(
        self,
        *,
        account_sid: str | None = None,
        auth_token: str | None = None,
        from_number: str | None = None,
        to_number: str | None = None,
        enabled: bool = True,
        timeout_sec: float = 10.0,
    ) -> None:
        self.account_sid = (
            account_sid
            if account_sid is not None
            else (os.environ.get("TWILIO_ACCOUNT_SID") or os.environ.get("SMS_ACCOUNT_SID") or "")
        ).strip()
        self.auth_token = (
            auth_token
            if auth_token is not None
            else (os.environ.get("TWILIO_AUTH_TOKEN") or os.environ.get("SMS_AUTH_TOKEN") or "")
        ).strip()
        self.from_number = (
            from_number
            if from_number is not None
            else (os.environ.get("TWILIO_FROM_NUMBER") or os.environ.get("SMS_FROM_NUMBER") or "")
        ).strip()
        self.to_number = (
            to_number
            if to_number is not None
            else (os.environ.get("TWILIO_TO_NUMBER") or os.environ.get("SMS_TO_NUMBER") or "")
        ).strip()
        self.enabled = bool(enabled)
        self.timeout_sec = float(timeout_sec)

    def configured(self) -> bool:
        return bool(
            self.enabled and self.account_sid and self.auth_token and self.from_number and self.to_number
        )

    def status(self) -> dict[str, Any]:
        return {
            "provider": "twilio",
            "enabled": self.enabled,
            "configured": self.configured(),
            "account_sid": "SET" if self.account_sid else "MISSING",
            "auth_token": "SET" if self.auth_token else "MISSING",
            "from_number": "SET" if self.from_number else "MISSING",
            "to_number": "SET" if self.to_number else "MISSING",
        }

    def send_sms(self, body: str) -> NotificationResult:
        if not self.enabled:
            return NotificationResult(ok=True, channel="sms", reason="DISABLED", skipped=True)
        if not self.configured():
            return NotificationResult(ok=False, channel="sms", reason="MISSING_CONFIG", skipped=True)
        safe_body = scrub_text(body, extra_secrets=[self.auth_token, self.account_sid])
        try:
            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
            form = urllib.parse.urlencode(
                {"To": self.to_number, "From": self.from_number, "Body": safe_body[:1500]}
            ).encode("utf-8")
            token = base64.b64encode(f"{self.account_sid}:{self.auth_token}".encode("utf-8")).decode("ascii")
            req = urllib.request.Request(
                url,
                data=form,
                method="POST",
                headers={
                    "Authorization": f"Basic {token}",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": "binance-btc-bot/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw else {}
            if data.get("error_code") or data.get("code") in (20003, 21211, 21608):
                reason = scrub_text(str(data.get("message") or data.get("error_message") or "SMS_API_ERROR"))
                logger.warning("SMS send failed: %s", reason)
                return NotificationResult(ok=False, channel="sms", reason=reason)
            return NotificationResult(ok=True, channel="sms", reason="OK")
        except Exception as e:  # noqa: BLE001
            reason = scrub_text(
                f"{type(e).__name__}: {e}",
                extra_secrets=[self.auth_token, self.account_sid],
            )
            logger.warning("SMS send exception: %s", reason)
            return NotificationResult(ok=False, channel="sms", reason=reason)


class SMSNotifier(NotificationChannel):
    """NotificationChannel wrapper around an SMSProvider (critical events only)."""

    name = "sms"

    def __init__(self, provider: SMSProvider | None = None, *, enabled: bool = True) -> None:
        self.provider = provider or build_sms_provider_from_env(enabled=enabled)
        self.enabled = bool(enabled)

    def configured(self) -> bool:
        return bool(self.enabled and self.provider.configured())

    def status(self) -> dict[str, Any]:
        st = dict(self.provider.status())
        st["channel"] = self.name
        st["enabled"] = self.enabled
        st["configured"] = self.configured()
        return st

    def send(self, event: NotificationEvent) -> NotificationResult:
        if not self.enabled:
            return NotificationResult(ok=True, channel=self.name, reason="DISABLED", skipped=True)
        body = f"BINANCE BTC BOT CRITICAL\n{event.event}\n{event.message}"
        if event.reason:
            body += f"\nreason={event.reason}"
        if event.symbol:
            body += f"\nsymbol={event.symbol}"
        return self.provider.send_sms(body)


class MockSMSProvider(SMSProvider):
    """Safe test/mock SMS provider — records messages; never touches a network."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.sent: list[str] = []

    def configured(self) -> bool:
        return bool(self.enabled)

    def send_sms(self, body: str) -> NotificationResult:
        if not self.enabled:
            return NotificationResult(ok=True, channel="sms", reason="DISABLED", skipped=True)
        safe = scrub_text(body)
        self.sent.append(safe)
        return NotificationResult(ok=True, channel="sms", reason="MOCK_OK")

    def status(self) -> dict[str, Any]:
        return {
            "provider": "mock",
            "enabled": self.enabled,
            "configured": self.configured(),
            "sent_count": len(self.sent),
        }


def build_sms_provider_from_env(*, enabled: bool = True) -> SMSProvider:
    provider_name = (os.environ.get("SMS_PROVIDER") or "twilio").strip().lower()
    if provider_name in {"", "none", "null", "off"}:
        return NullSMSProvider()
    if provider_name in {"mock", "test"}:
        return MockSMSProvider(enabled=enabled)
    if provider_name == "twilio":
        return TwilioSMSProvider(enabled=enabled)
    logger.warning("Unknown SMS_PROVIDER=%s; using NullSMSProvider", provider_name)
    return NullSMSProvider()
