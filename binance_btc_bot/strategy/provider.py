"""Strategy provider abstraction — execution asks provider, not hard-coded T1."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from binance_btc_bot.strategy.trails import TrailStrategy, get_strategy


class StrategyProvider(ABC):
    """Supplies the active trail strategy for live/dry execution."""

    @abstractmethod
    def get_strategy(self, context: Mapping[str, Any] | None = None) -> TrailStrategy:
        ...

    @abstractmethod
    def strategy_key(self) -> str:
        ...

    @abstractmethod
    def selector_key(self) -> str | None:
        """None when no live selector is active."""
        ...


class FixedStrategyProvider(StrategyProvider):
    """Initial live provider: fixed trail strategy from config (currently T1).

    Selectors A–F are intentionally not used. A future SelectorStrategyProvider
    can replace this without changing the execution engine.
    """

    def __init__(
        self,
        strategy_key: str = "T1",
        *,
        strategies_cfg: Mapping[str, Any] | None = None,
        selector_key: str | None = None,
    ) -> None:
        if selector_key not in (None, "", "null", "NONE"):
            raise ValueError(
                "FixedStrategyProvider does not activate selectors; selector_key must be null"
            )
        self._key = str(strategy_key).upper()
        self._strategies_cfg = strategies_cfg
        # Resolve once to fail fast on unknown keys / frozen mismatch.
        self._cached = get_strategy(self._key, strategies_cfg)

    def get_strategy(self, context: Mapping[str, Any] | None = None) -> TrailStrategy:
        _ = context  # reserved for future selector context
        return self._cached

    def strategy_key(self) -> str:
        return self._key

    def selector_key(self) -> str | None:
        return None


def build_strategy_provider(cfg: Mapping[str, Any]) -> StrategyProvider:
    """Factory from bot config.

    Default disk YAML remains Fixed T1 / selector null. When runtime control is
    enabled (``live.runtime_control=true`` or ``_runtime_control`` overlay),
    returns a mutable RuntimeStrategyProvider for NEW entries only.
    """
    live = cfg.get("live") or {}
    key = str(live.get("strategy") or "T1").upper()
    selector = live.get("selector")
    if selector in ("null", "NONE", ""):
        selector = None
    runtime = bool(live.get("runtime_control") or cfg.get("_runtime_control"))
    if runtime:
        from binance_btc_bot.control.runtime import RuntimeStrategyProvider

        return RuntimeStrategyProvider(
            strategy_key=key,
            selector_key=selector,
            strategies_cfg=cfg.get("strategies"),
        )
    return FixedStrategyProvider(
        strategy_key=key,
        strategies_cfg=cfg.get("strategies"),
        selector_key=selector,
    )
