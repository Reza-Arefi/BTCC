"""Secret scrubbing — never leak API keys/tokens/PEM into logs or notifications."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Iterable

SECRET_ENV_NAMES = (
    "BINANCE_API_KEY",
    "BINANCE_API_SECRET",  # legacy; scrub if present
    "BINANCE_ED25519_PRIVATE_KEY_PATH",
    "BINANCE_ED25519_PRIVATE_KEY_PASSPHRASE",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "SMS_ACCOUNT_SID",
    "SMS_AUTH_TOKEN",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "SMS_FROM_NUMBER",
    "SMS_TO_NUMBER",
)

_REDACTED = "***REDACTED***"
_RUNTIME_SECRETS: list[str] = []

_PEM_PRIVATE_BLOCK = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
_SECRETISH_PATTERNS = (
    re.compile(
        r"(api[_-]?key|api[_-]?secret|auth[_-]?token|bot[_-]?token|passphrase)\s*[:=]\s*['\"]?([^\s'\"]+)",
        re.I,
    ),
    re.compile(r"(Bearer\s+)([A-Za-z0-9._\-]{8,})", re.I),
    re.compile(r"(signature=)([A-Za-z0-9+/=_%\-]{16,})", re.I),
)


def register_runtime_secret(value: str | None) -> None:
    """Register sensitive material loaded at runtime (e.g. PEM body) for scrubbing."""
    if not value:
        return
    text = str(value).strip()
    if len(text) < 4:
        return
    if text not in _RUNTIME_SECRETS:
        _RUNTIME_SECRETS.append(text)
        # Also register individual non-empty PEM lines (except headers) for safety.
        for line in text.splitlines():
            line = line.strip()
            if (
                line
                and not line.startswith("-----")
                and len(line) >= 8
                and line not in _RUNTIME_SECRETS
            ):
                _RUNTIME_SECRETS.append(line)


def clear_runtime_secrets() -> None:
    _RUNTIME_SECRETS.clear()


def collect_secret_values(extra: Iterable[str] | None = None) -> list[str]:
    values: list[str] = list(_RUNTIME_SECRETS)
    for name in SECRET_ENV_NAMES:
        raw = os.environ.get(name)
        if raw and str(raw).strip():
            values.append(str(raw).strip())
    if extra:
        for v in extra:
            if v and str(v).strip():
                values.append(str(v).strip())
    values.sort(key=len, reverse=True)
    return values


def scrub_text(text: str, *, extra_secrets: Iterable[str] | None = None) -> str:
    if not text:
        return text
    out = str(text)
    out = _PEM_PRIVATE_BLOCK.sub(
        "-----BEGIN PRIVATE KEY-----***REDACTED***-----END PRIVATE KEY-----",
        out,
    )
    for secret in collect_secret_values(extra_secrets):
        if len(secret) >= 4:
            out = out.replace(secret, _REDACTED)
    for pat in _SECRETISH_PATTERNS:
        out = pat.sub(lambda m: f"{m.group(1)}{_REDACTED}", out)
    return out


def scrub_obj(obj: Any, *, extra_secrets: Iterable[str] | None = None) -> Any:
    if obj is None:
        return None
    if isinstance(obj, str):
        return scrub_text(obj, extra_secrets=extra_secrets)
    if isinstance(obj, dict):
        cleaned: dict[Any, Any] = {}
        for k, v in obj.items():
            key = str(k)
            if any(
                s in key.lower()
                for s in ("secret", "token", "password", "passphrase", "private_key", "api_key", "apikey", "auth", "pem")
            ):
                cleaned[k] = _REDACTED
            else:
                cleaned[k] = scrub_obj(v, extra_secrets=extra_secrets)
        return cleaned
    if isinstance(obj, (list, tuple)):
        return [scrub_obj(x, extra_secrets=extra_secrets) for x in obj]
    return obj


def scrub_exception(exc: BaseException, *, extra_secrets: Iterable[str] | None = None) -> str:
    return scrub_text(f"{type(exc).__name__}: {exc}", extra_secrets=extra_secrets)


class ScrubbingFilter(logging.Filter):
    """Logging filter that redacts secrets from LogRecord messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # Scrub argument values first (format placeholders stay intact).
            if record.args:
                if isinstance(record.args, dict):
                    record.args = scrub_obj(record.args)  # type: ignore[assignment]
                elif isinstance(record.args, tuple):
                    record.args = tuple(
                        scrub_text(a) if isinstance(a, str) else a for a in record.args
                    )
            if isinstance(record.msg, str):
                # Exact secret / PEM scrubbing only — avoid rewriting "api_key=%s" templates.
                msg = str(record.msg)
                msg = _PEM_PRIVATE_BLOCK.sub(
                    "-----BEGIN PRIVATE KEY-----***REDACTED***-----END PRIVATE KEY-----",
                    msg,
                )
                for secret in collect_secret_values():
                    if len(secret) >= 4:
                        msg = msg.replace(secret, _REDACTED)
                record.msg = msg
        except Exception:  # noqa: BLE001 — never break logging
            return True
        return True
