"""Hourly Telegram operational report (presentation layer)."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from binance_btc_bot.control.runtime import RuntimeController
from binance_btc_bot.notifications.telegram_reports import format_hourly_report
from binance_btc_bot.secrets import scrub_text

logger = logging.getLogger(__name__)

SendFn = Callable[[str], None]
ViewFn = Callable[[], dict[str, Any]]


def _win_rate(wins: int, losses: int, closed: int) -> str:
    if closed <= 0:
        return "N/A"
    total = wins + losses
    if total <= 0:
        return "N/A"
    return f"{(100.0 * wins / total):.1f}%"


def build_hourly_report_data(ctrl: RuntimeController, view: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble authoritative hour bundle from runtime counters + engine view (no P/L invention)."""
    st = ctrl.state
    counters = dict(st.hour_counters or {})
    now = time.time()
    window_end = now
    window_start = float(st.hour_window_start) if st.hour_window_start is not None else (now - 3600.0)

    signals = int(counters.get("signals") or 0)
    crosses = int(counters.get("crosses") or 0)
    entries = int(counters.get("entries") or 0)
    exits = int(counters.get("exits") or 0)
    rejected = int(counters.get("rejected_entries") or 0)
    closed = int(counters.get("closed_trades") or view.get("hour_closed_trades") or 0)
    wins = int(counters.get("wins") or view.get("hour_wins") or 0)
    losses = int(counters.get("losses") or view.get("hour_losses") or 0)

    end_eq = view.get("equity_btc")
    start_eq = st.equity_btc_at_hour_start
    eq_delta = None
    eq_pct = None
    try:
        if end_eq is not None and start_eq is not None and float(start_eq) != 0:
            eq_delta = float(end_eq) - float(start_eq)
            eq_pct = 100.0 * eq_delta / float(start_eq)
        elif end_eq is not None and start_eq is not None:
            eq_delta = float(end_eq) - float(start_eq)
    except Exception:  # noqa: BLE001
        eq_delta = None
        eq_pct = None

    trading = {
        "signals": signals,
        "genuine_crosses": crosses,
        "entries": entries,
        "exits": exits,
        "rejected_entries": rejected,
        "closed_trades": closed,
        "wins": wins,
        "losses": losses,
        "win_rate": _win_rate(wins, losses, closed),
        "gross_pnl_btc": view.get("hour_gross_pnl_btc", counters.get("gross_pnl_btc")),
        "fees_btc": view.get("hour_fees_btc", counters.get("fees_btc")),
        "net_realized_pnl_btc": view.get("hour_net_realized_pnl_btc", counters.get("net_realized_pnl_btc")),
        "unrealized_pnl_btc": view.get("unrealized_pnl_btc"),
        "net_portfolio_change_btc": eq_delta,
        "activity_score": signals + crosses + entries + exits + rejected + closed,
    }

    return {
        "window_start": datetime.fromtimestamp(window_start, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_end": datetime.fromtimestamp(window_end, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "system": {
            "state": ctrl.display_mode(safety_halted=bool(view.get("safety_halted"))),
            "strategy": st.strategy,
            "selector": st.selector,
            "max_trades": st.max_simultaneous_trades,
            "open_count": view.get("open_count", 0),
            "max_simultaneous_trades": st.max_simultaneous_trades,
            "slots_remaining": view.get("slots_remaining"),
        },
        "portfolio": {
            "window_start": datetime.fromtimestamp(window_start, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "window_end": datetime.fromtimestamp(window_end, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "start_equity_btc": start_eq,
            "start_btc_free": st.btc_free_at_hour_start,
            "start_btc_locked": st.btc_locked_at_hour_start,
            "end_equity_btc": end_eq,
            "end_btc_free": view.get("btc_free"),
            "end_btc_locked": view.get("btc_locked"),
            "equity_change_btc": eq_delta,
            "equity_change_pct": eq_pct,
            "bnb_balance": view.get("bnb_free"),
            "bnb_fees_hour": view.get("hour_bnb_fees", counters.get("bnb_fees")),
        },
        "trading": trading,
        "positions": view.get("positions") or [],
        "events": view.get("hour_events") or st.hour_events or [],
        "safety": {
            "binance_rest": view.get("binance_rest"),
            "user_data_ws": view.get("user_data_ws"),
            "market_data": view.get("market_data"),
            "reconciliation": view.get("reconciliation"),
            "protection": view.get("protection_status"),
            "telegram": view.get("telegram"),
            "execution_engine": view.get("execution_state"),
            "last_heartbeat": view.get("last_heartbeat"),
            "errors": view.get("errors_count", 0),
            "warnings": view.get("warnings_count", 0),
            "protection_failures": view.get("protection_failures", 0),
            "unknown_order_states": view.get("unknown_order_states", 0),
            "reconciliation_failures": view.get("reconciliation_failures", 0),
            "halt_events": view.get("halt_events", 0),
        },
        "cumulative": view.get("performance") or {},
    }


def build_hourly_report(ctrl: RuntimeController, view: dict[str, Any]) -> str:
    data = build_hourly_report_data(ctrl, view)
    return scrub_text(format_hourly_report(view, data))


class HourlyReportScheduler:
    def __init__(
        self,
        controller: RuntimeController,
        *,
        view_fn: ViewFn,
        send_fn: SendFn,
        interval_sec: float = 3600.0,
    ) -> None:
        self.controller = controller
        self.view_fn = view_fn
        self.send_fn = send_fn
        self.interval_sec = float(interval_sec)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="hourly-report", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(min(60.0, self.interval_sec)):
            try:
                self.maybe_send()
            except Exception as e:  # noqa: BLE001
                logger.warning("hourly report failed: %s", e)

    def maybe_send(self, *, force: bool = False) -> str | None:
        now = time.time()
        last = self.controller.state.last_hourly_report_at
        if not force and last is not None and (now - float(last)) < self.interval_sec:
            return None
        view = self.view_fn() or {}
        text = build_hourly_report(self.controller, view)
        self.send_fn(text)
        eq = view.get("equity_btc")
        try:
            eq_f = float(eq) if eq is not None else None
        except Exception:  # noqa: BLE001
            eq_f = None
        try:
            free_f = float(view["btc_free"]) if view.get("btc_free") is not None else None
        except Exception:  # noqa: BLE001
            free_f = None
        try:
            locked_f = float(view["btc_locked"]) if view.get("btc_locked") is not None else None
        except Exception:  # noqa: BLE001
            locked_f = None
        self.controller.reset_hour_counters(
            equity_btc=eq_f,
            btc_free=free_f,
            btc_locked=locked_f,
        )
        return text
