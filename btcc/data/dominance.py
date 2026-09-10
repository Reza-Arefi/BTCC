"""Live BTC dominance feed stub (signal-bot path; unused by offline multi-arm)."""

from __future__ import annotations

from typing import Any


class DominanceFeed:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def observation(self) -> tuple[float | None, Any, str]:
        return None, None, "DISABLED"
