"""Explicit real-write enablement gate.

Default: CLOSED. Write code paths cannot fire unless allow_trading is true AND
exactly one session arm is set:

  allow_trading ∧ canary_writes_armed     → single-shot canary
  allow_trading ∧ bounded_6h_writes_armed → bounded 6h session

protection_api_confirmed is retained for legacy EXCHANGE_RESIDENT adapter tests
only — production T1 does NOT require it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class WriteGate:
    """Fail-closed gate for any MEXC order POST/DELETE."""

    allow_trading: bool = False
    canary_writes_armed: bool = False  # separate explicit arm for single-shot canary
    protection_api_confirmed: bool = False
    bounded_6h_writes_armed: bool = False  # distinct arm for REAL_BOUNDED_6H

    @property
    def market_writes_allowed(self) -> bool:
        return bool(
            self.allow_trading
            and (self.canary_writes_armed or self.bounded_6h_writes_armed)
        )

    @property
    def protection_writes_allowed(self) -> bool:
        return bool(self.market_writes_allowed and self.protection_api_confirmed)

    def assert_market_write(self, action: str) -> None:
        from btcc.safety.no_trading import TradingForbiddenError

        if not self.market_writes_allowed:
            raise TradingForbiddenError(
                f"WriteGate CLOSED for {action}: allow_trading={self.allow_trading} "
                f"canary_writes_armed={self.canary_writes_armed} "
                f"bounded_6h_writes_armed={self.bounded_6h_writes_armed}"
            )

    def assert_protection_write(self, action: str) -> None:
        from btcc.safety.no_trading import TradingForbiddenError

        self.assert_market_write(action)
        if not self.protection_api_confirmed:
            raise TradingForbiddenError(
                f"WriteGate CLOSED for {action}: MEXC Spot protection API not confirmed "
                "(set protection_api_confirmed only after official confirmation)"
            )


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes"}


def load_write_gate_from_env(*, allow_trading: bool = False) -> WriteGate:
    """Env arming is opt-in; defaults keep everything closed."""
    armed = _env_truthy("BTCC_CANARY_WRITES_ARMED") or _env_truthy("CANARY_WRITES_ARMED")
    bounded = _env_truthy("BTCC_REAL_6H_ARMED")
    prot = _env_truthy("BTCC_MEXC_PROTECTION_API_CONFIRMED")
    # Never infer allow_trading=true from env alone —
    # caller must pass explicit allow_trading (tests / future enablement).
    return WriteGate(
        allow_trading=bool(allow_trading),
        canary_writes_armed=armed and bool(allow_trading),
        protection_api_confirmed=prot and bool(allow_trading),
        bounded_6h_writes_armed=bounded and bool(allow_trading),
    )


CLOSED_WRITE_GATE = WriteGate()
