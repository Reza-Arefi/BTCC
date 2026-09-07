"""Telegram notifications for Selector E live paper trades."""

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

    def paper_startup(
        self,
        *,
        equity_btc: float,
        allocation_pct: float,
        max_positions: int,
        cf_trades: int,
        version: dict[str, Any] | None = None,
    ) -> None:
        v = version or {}
        lines = [
            "📋 PAPER ENGINE STARTED",
            "",
            "REAL ORDERS: DISABLED",
            "ALLOW_TRADING: FALSE",
            "",
            f"Selector: E / {v.get('selector_kind', 'rank_ewma')}",
            f"Allocation: {allocation_pct * 100:.0f}% of current equity",
            "Max total exposure: 100%",
            f"Max positions: {max_positions}",
            "Leverage: NONE",
            "",
            f"Current equity: {equity_btc:.8f} BTC",
            f"CF warmup trades loaded: {cf_trades}",
            "",
            f"Bot: {v.get('bot_version')} | Selector: {v.get('selector_version')}",
            f"UTC: {datetime.now(timezone.utc).isoformat()}",
        ]
        self._send("\n".join(lines))

    def trade_opened(
        self,
        opp: dict[str, Any],
        pick: dict[str, Any],
        *,
        equity_btc: float | None = None,
        version: dict[str, Any] | None = None,
    ) -> None:
        v = version or {}
        pos_btc = opp.get("position_btc")
        requested = float(opp.get("requested_allocation_pct", opp.get("allocation_pct", 0)) or 0)
        actual = float(opp.get("actual_allocation_pct", opp.get("allocation_pct", 0)) or 0)
        exposure_limited = bool(opp.get("exposure_limited"))
        rank_lines = []
        for k in sorted((pick.get("scores") or {}).keys()):
            rank_lines.append(f"{_label_for_key(k)}: {pick['scores'][k]:.4f}")
        lines = [
            "🟢 PAPER E TRADE",
            "",
            "Selector: E",
            f"Selected exit: {pick.get('selected_arm_label')}",
            f"Pair: {opp.get('symbol')}",
            "",
            f"Requested allocation: {requested * 100:.0f}%",
            f"Actual allocation: {actual * 100:.2f}%",
            f"Current equity: {float(equity_btc if equity_btc is not None else opp.get('equity_btc_at_entry') or 0):.8f} BTC",
            f"Position size: {float(pos_btc or 0):.8f} BTC",
            f"Entry price: {opp.get('entry_alt_btc_mid')}",
            f"Regime: {opp.get('regime')}",
            f"S-score: {opp.get('S')}",
        ]
        if exposure_limited:
            lines.extend([
                "",
                "EXPOSURE LIMITED",
                f"Requested: {requested * 100:.0f}%",
                f"Actual: {actual * 100:.2f}%",
                f"Remaining exposure at sizing: {float(opp.get('remaining_exposure_pct') or 0) * 100:.2f}%",
            ])
        lines.extend([
            "",
            f"Initial SL: {pick.get('initial_sl') or 'trail spec'}",
            "",
            "Selector ranking:",
            *rank_lines,
            "",
            f"Timestamp: {opp.get('entry_fill_ts') or opp.get('opened_ts')}",
            f"Bot: {v.get('bot_version')} | Selector: {v.get('selector_version')}",
        ])
        self._send("\n".join(str(x) for x in lines if x is not None))

    def trade_closed(
        self,
        opp: dict[str, Any],
        e_leg: dict[str, Any],
        *,
        best_cf_arm: str | None = None,
        best_cf_pnl_pct: float | None = None,
        regret_pct: float | None = None,
        equity_btc: float | None = None,
        version: dict[str, Any] | None = None,
    ) -> None:
        v = version or {}
        pnl_pct = float(e_leg.get("pnl_pct") or 0) * 100
        acct_impact = float(e_leg.get("account_impact_pct") or 0)
        lines = [
            "🔴 PAPER E TRADE CLOSED",
            "",
            f"Pair: {opp.get('symbol')}",
            f"Selector: E",
            f"Strategy: {opp.get('selected_arm_label') or e_leg.get('strategy_name') or e_leg.get('strategy_key')}",
            "",
            f"Entry: {e_leg.get('entry_fill_price')}",
            f"Exit: {e_leg.get('exit_fill_price')}",
            "",
            f"P/L: {pnl_pct:+.3f}%",
            f"Account impact: {acct_impact:+.4f}%",
            "",
            f"Equity: {float(equity_btc if equity_btc is not None else e_leg.get('equity_btc_after') or 0):.8f} BTC",
            "",
            f"Duration: {e_leg.get('holding_hours')}h",
            f"MFE: {float(e_leg.get('mfe_pct') or 0) * 100:.3f}%",
            f"MAE: {float(e_leg.get('mae_pct') or 0) * 100:.3f}%",
            "",
            f"Exit reason: {e_leg.get('exit_reason')}",
            f"Fees: {e_leg.get('fees_btc')}",
            f"Slippage: {e_leg.get('slippage_btc_approx')}",
            "",
            f"Best counterfactual: {best_cf_arm} ({(best_cf_pnl_pct or 0) * 100:+.3f}%)",
            f"Selector regret: {(regret_pct or 0) * 100:.3f}%",
            "",
            f"Timestamp: {e_leg.get('exit_ts')}",
            f"Bot: {v.get('bot_version')} | Selector: {v.get('selector_version')}",
        ]
        self._send("\n".join(str(x) for x in lines if x is not None))

    def safety_state_change(self, old: str, new: str, reasons: list[str]) -> None:
        self._send(
            f"⚠️ PAPER SAFETY {old} → {new}\nUTC: {datetime.now(timezone.utc).isoformat()}\n"
            + "\n".join(reasons)
        )

    def hourly_summary(self, stats: dict[str, Any], *, version: dict[str, Any] | None = None) -> bool:
        from btcc.telegram.hourly_summary import format_hourly_paper_message

        return self._send(format_hourly_paper_message(stats, version=version))


def _label_for_key(key: str) -> str:
    if key.startswith("trail_"):
        return f"T{key.split('_')[1]}"
    return key
