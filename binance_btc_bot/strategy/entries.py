"""Entry gate — uses strategy key from StrategyProvider (no hard-wired T1).

Selectors remain inactive unless a future SelectorStrategyProvider is installed.

Entry rule (live layer):
  NEW CROSS INTO S >= long_threshold (default 0.65)
  Requires a prior observation of S < threshold on this symbol.
  Stay-in-zone does NOT open another trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from binance_btc_bot.portfolio.manager import PortfolioManager, ReserveResult

CLS_ELIGIBLE = "ELIGIBLE"
CLS_BELOW_THRESHOLD = "BELOW_THRESHOLD"
CLS_ABOVE_THRESHOLD = "ABOVE_THRESHOLD"
CLS_SIGNAL_CONTINUATION = "SIGNAL_CONTINUATION"
CLS_SAME_PAIR_OPEN = "SAME_PAIR_ALREADY_OPEN"
CLS_MAX_OPEN = "MAX_OPEN_TRADES"
CLS_HALTED = "SAFETY_HALT"
CLS_NEED_PRIOR_BELOW = "NEED_PRIOR_BELOW"


@dataclass
class CrossingState:
    """Per-symbol cross-into memory.

    Trade suggested only when:
      prior S < long_threshold  AND  current S >= long_threshold
    Cold start in-band (never observed below) does NOT fire.
    """

    ever_below: bool = False
    in_band: bool = False

    def update(
        self,
        score: float,
        *,
        long_threshold: float,
        upper_threshold: float | None,
    ) -> dict[str, Any]:
        score = float(score)
        if score < long_threshold:
            self.ever_below = True
            self.in_band = False
            return {
                "trade_suggested": False,
                "rejection_reason": "BELOW_THRESHOLD",
                "score": score,
                "crossed_into": False,
            }
        if upper_threshold is not None and score >= float(upper_threshold):
            self.in_band = True
            return {
                "trade_suggested": False,
                "rejection_reason": "ABOVE_THRESHOLD",
                "score": score,
                "crossed_into": False,
            }

        # score >= long_threshold (in trade zone)
        if self.in_band:
            return {
                "trade_suggested": False,
                "rejection_reason": "SIGNAL_CONTINUATION",
                "score": score,
                "crossed_into": False,
            }
        if not self.ever_below:
            self.in_band = True
            return {
                "trade_suggested": False,
                "rejection_reason": "NEED_PRIOR_BELOW",
                "score": score,
                "crossed_into": False,
            }

        # Genuine cross: was below, now entering the band.
        self.in_band = True
        self.ever_below = False
        return {
            "trade_suggested": True,
            "rejection_reason": None,
            "score": score,
            "crossed_into": True,
        }


@dataclass(frozen=True)
class EntryDecision:
    trade_suggested: bool
    symbol: str
    strategy: str
    selector: str | None
    score: float | None
    classification: str
    rejection_reason: str | None
    relative_price: float | None = None
    reservation_id: str | None = None
    crossed_into: bool = False


class LiveEntryEngine:
    """Live entry gate. Strategy/selector labels come from StrategyProvider."""

    def __init__(
        self,
        *,
        long_threshold: float = 0.65,
        upper_threshold: float | None = None,
        strategy_key: str,
        selector_key: str | None = None,
        max_open: int = 8,
        one_per_symbol: bool = True,
        portfolio: PortfolioManager | None = None,
    ) -> None:
        self.long_threshold = float(long_threshold)
        self.upper_threshold = float(upper_threshold) if upper_threshold is not None else None
        self.strategy_key = str(strategy_key).upper()
        self.selector_key = None if selector_key in (None, "", "null", "NONE") else str(selector_key).upper()
        self.max_open = int(max_open)
        self.one_per_symbol = bool(one_per_symbol)
        self.portfolio = portfolio
        self._cross: dict[str, CrossingState] = {}

    def evaluate(
        self,
        *,
        symbol: str,
        score: float,
        open_symbols: set[str] | list[str] | None = None,
        open_count: int | None = None,
        safety_allows: bool = True,
        relative_price: float | None = None,
        strategy_key: str | None = None,
        selector_key: str | None = None,
        reserve_slot: bool = False,
    ) -> EntryDecision:
        sym = symbol.upper()
        strat = str(strategy_key or self.strategy_key).upper()
        sel = self.selector_key if selector_key is None else (
            None if selector_key in ("", "null", "NONE") else str(selector_key).upper()
        )
        if not safety_allows:
            return EntryDecision(False, sym, strat, sel, score, CLS_HALTED, CLS_HALTED, relative_price)

        st = self._cross.setdefault(sym, CrossingState())
        sm = st.update(
            float(score),
            long_threshold=self.long_threshold,
            upper_threshold=self.upper_threshold,
        )
        if not sm["trade_suggested"]:
            reason = str(sm["rejection_reason"])
            cls = {
                "BELOW_THRESHOLD": CLS_BELOW_THRESHOLD,
                "ABOVE_THRESHOLD": CLS_ABOVE_THRESHOLD,
                "SIGNAL_CONTINUATION": CLS_SIGNAL_CONTINUATION,
                "NEED_PRIOR_BELOW": CLS_NEED_PRIOR_BELOW,
            }.get(reason, reason)
            return EntryDecision(
                False, sym, strat, sel, score, cls, reason, relative_price, crossed_into=False
            )

        if self.portfolio is not None and reserve_slot:
            res: ReserveResult = self.portfolio.try_reserve(sym)
            if not res.ok:
                reason = res.reason
                cls = {
                    "SAME_PAIR_ALREADY_OPEN": CLS_SAME_PAIR_OPEN,
                    "MAX_OPEN_TRADES": CLS_MAX_OPEN,
                    "MAX_TOTAL_ALLOCATION": CLS_MAX_OPEN,
                }.get(reason, reason)
                return EntryDecision(
                    False, sym, strat, sel, score, cls, reason, relative_price, crossed_into=True
                )
            return EntryDecision(
                True,
                sym,
                strat,
                sel,
                score,
                CLS_ELIGIBLE,
                None,
                relative_price,
                reservation_id=res.reservation.reservation_id if res.reservation else None,
                crossed_into=True,
            )

        open_set = {s.upper() for s in (open_symbols or [])}
        if self.one_per_symbol and sym in open_set:
            return EntryDecision(
                False, sym, strat, sel, score, CLS_SAME_PAIR_OPEN, CLS_SAME_PAIR_OPEN,
                relative_price, crossed_into=True,
            )
        count = int(open_count if open_count is not None else len(open_set))
        if count >= self.max_open:
            return EntryDecision(
                False, sym, strat, sel, score, CLS_MAX_OPEN, CLS_MAX_OPEN,
                relative_price, crossed_into=True,
            )

        return EntryDecision(
            True, sym, strat, sel, score, CLS_ELIGIBLE, None, relative_price, crossed_into=True
        )
