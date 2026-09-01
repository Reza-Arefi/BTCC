"""Telegram notifications for Selector E live trades."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable


class SelectorTradeNotifier:
    def __init__(self, send_fn: Callable[[str], bool], *, enabled: bool = True):
        self.send_fn = send_fn
        self.enabled = enabled

    def _send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            return bool(self.send_fn(text))
        except Exception:
            return False

    def trade_opened(self, opp: dict[str, Any], pick: dict[str, Any], *, version: dict[str, Any] | None = None) -> None:
        v = version or {}
        lines = [
            "🟢 TRADE OPENED",
            "",
            f"Pair: {opp.get('symbol')}",
            f"Selector: {pick.get('arm_label', 'E')}",
            f"Selected strategy: {pick.get('selected_arm_label')}",
            "",
            f"Entry fill: {opp.get('entry_alt_btc_mid')}",
            f"Initial SL: {pick.get('initial_sl') or 'trail spec'}",
            f"Position notional: ${float(opp.get('notional_usd', 0)):.0f}",
            "",
            f"Market regime: {opp.get('regime')}",
            f"S-band: {opp.get('s_band')}",
            f"S score: {opp.get('S')}",
            "",
            f"Selector score (selected): {pick.get('selected_score')}",
            f"Second best: {pick.get('second_best_arm_label')} ({pick.get('second_best_score')})",
            "",
            f"Timestamp: {opp.get('entry_fill_ts') or opp.get('opened_ts')}",
            f"Bot: {v.get('bot_version')} | Selector: {v.get('selector_version')}",
        ]
        self._send("\n".join(str(x) for x in lines if x is not None))

    def trade_closed(
        self,
        opp: dict[str, Any],
        e_leg: dict[str, Any],
        *,
        best_cf_arm: str | None = None,
        best_cf_pnl_pct: float | None = None,
        regret_pct: float | None = None,
        version: dict[str, Any] | None = None,
    ) -> None:
        v = version or {}
        pnl_pct = float(e_leg.get("pnl_pct") or 0) * 100
        lines = [
            "🔴 TRADE CLOSED",
            "",
            f"Pair: {opp.get('symbol')}",
            f"Strategy: {e_leg.get('strategy_name') or e_leg.get('strategy_key')}",
            f"Selector: E → {opp.get('selected_arm_label')}",
            "",
            f"Entry: {e_leg.get('entry_fill_price')}",
            f"Exit: {e_leg.get('exit_fill_price')}",
            "",
            f"P/L: ${float(e_leg.get('pnl_usd_equiv') or 0):.2f}",
            f"P/L %: {pnl_pct:.3f}%",
            f"Duration (h): {e_leg.get('holding_hours')}",
            "",
            f"MFE: {float(e_leg.get('mfe_pct') or 0)*100:.3f}%",
            f"MAE: {float(e_leg.get('mae_pct') or 0)*100:.3f}%",
            "",
            f"Exit reason: {e_leg.get('exit_reason')}",
            f"Fees (BTC approx): {e_leg.get('fees_btc')}",
            "",
            f"Best counterfactual: {best_cf_arm} ({(best_cf_pnl_pct or 0)*100:.3f}%)",
            f"Selection regret: {(regret_pct or 0)*100:.3f}%",
            "",
            f"Closed: {e_leg.get('exit_ts')}",
            f"Bot: {v.get('bot_version')} | Selector: {v.get('selector_version')}",
        ]
        self._send("\n".join(str(x) for x in lines if x is not None))

    def safety_state_change(self, old: str, new: str, reasons: list[str]) -> None:
        self._send(
            f"⚠️ SAFETY STATE {old} → {new}\nUTC: {datetime.now(timezone.utc).isoformat()}\n"
            + "\n".join(reasons)
        )
