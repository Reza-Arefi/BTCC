"""Errors for MEXC read-only access — fail closed, never invent empty accounts."""

from __future__ import annotations


class MexcReadError(RuntimeError):
    """Base class for read-path failures."""

    def __init__(self, message: str, *, code: str | None = None, http_status: int | None = None):
        super().__init__(message)
        self.code = code
        self.http_status = http_status

    def __str__(self) -> str:
        # Never include request secrets; callers must redact separately.
        base = super().__str__()
        bits = [base]
        if self.code:
            bits.append(f"code={self.code}")
        if self.http_status is not None:
            bits.append(f"http={self.http_status}")
        return " | ".join(bits)


class AccountDataUnavailable(MexcReadError):
    """Account/balances could not be established safely."""


class SymbolMetadataUnavailable(MexcReadError):
    """Required symbol metadata missing or incomplete for REAL readiness."""


class OrderDataUnavailable(MexcReadError):
    """Open orders / order status / fills unavailable."""


class MexcAuthError(MexcReadError):
    """Authentication / signature / permission failure."""


class MexcRateLimitError(MexcReadError):
    """Exchange rate limit."""


class MexcTimeoutError(MexcReadError):
    """Network / read timeout."""


class MexcMalformedResponseError(MexcReadError):
    """Response JSON missing required fields or wrong type."""
