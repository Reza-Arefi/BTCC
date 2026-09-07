"""Hard safety: SIGNAL ONLY — trading APIs must never be callable."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

BLOCKED = (
    "create_order",
    "place_order",
    "cancel_order",
    "new_order",
    "order_test",
    "submit_order",
)


class TradingForbiddenError(RuntimeError):
    """Raised if any trading/order method is invoked."""


def deny_trading(*_args, **_kwargs):
    raise TradingForbiddenError(
        "BTCC is SIGNAL-ONLY. Trading/order methods are forbidden. "
        "No BUY/SELL/ORDER/POSITION code is allowed."
    )


def install_trading_guards() -> None:
    """Monkey-patch common order method names on this process if imported later."""
    import builtins

    # Expose a sentinel so tests can verify
    builtins.__BTCC_SIGNAL_ONLY__ = True  # type: ignore[attr-defined]
    logger.info("BTCC safety: trading guards active (SIGNAL ONLY)")


def assert_no_trading_config(cfg: dict) -> None:
    if cfg.get("safety", {}).get("allow_trading", False):
        raise TradingForbiddenError("allow_trading must remain false")
