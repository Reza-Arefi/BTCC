"""Telegram inbound control plane (getUpdates) + reply formatting."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from binance_btc_bot.control.runtime import HELP_TEXT, RuntimeController
from binance_btc_bot.notifications.telegram_reports import (
    format_performance_message,
    format_status_message,
)
from binance_btc_bot.notifications.timezone_brt import format_brt
from binance_btc_bot.secrets import scrub_exception, scrub_text

logger = logging.getLogger(__name__)


class TelegramAPI:
    """Minimal Telegram Bot API client — never logs token."""

    def __init__(self, bot_token: str, *, timeout_sec: float = 35.0) -> None:
        self._token = (bot_token or "").strip()
        self.timeout_sec = float(timeout_sec)

    @property
    def configured(self) -> bool:
        return bool(self._token)

    def _url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self._token}/{method}"

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN missing")
        data = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None}).encode("utf-8")
        req = urllib.request.Request(
            self._url(method),
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "binance-btc-bot-control/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            out = json.loads(raw) if raw else {}
            if not out.get("ok", False):
                raise RuntimeError(scrub_text(str(out.get("description") or "TELEGRAM_API_ERROR")))
            return out
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(scrub_exception(e)) from e

    def get_updates(self, *, offset: int | None = None, timeout: int = 25) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"timeout": int(timeout)}
        if offset is not None:
            params["offset"] = int(offset)
        data = self.call("getUpdates", params)
        return list(data.get("result") or [])

    def send_message(self, chat_id: str | int, text: str) -> None:
        # Chunk Telegram 4096 limit
        msg = scrub_text(text)
        for i in range(0, len(msg), 3500):
            self.call("sendMessage", {"chat_id": str(chat_id), "text": msg[i : i + 3500], "disable_web_page_preview": "true"})


EngineView = Callable[[], dict[str, Any]]
ReconcileFn = Callable[[], dict[str, Any]]
EmergencyFn = Callable[[], dict[str, Any]]


class TelegramControlPlane:
    """Poll Telegram commands, authorize by chat_id, dispatch to RuntimeController + engine views."""

    def __init__(
        self,
        controller: RuntimeController,
        *,
        bot_token: str | None = None,
        chat_id: str | None = None,
        engine_view: EngineView | None = None,
        reconcile_fn: ReconcileFn | None = None,
        emergency_fn: EmergencyFn | None = None,
        poll_interval_sec: float = 1.0,
        long_poll_sec: int = 25,
    ) -> None:
        self.controller = controller
        token = (bot_token if bot_token is not None else os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
        self.chat_id = (chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
        self.api = TelegramAPI(token)
        self.engine_view = engine_view
        self.reconcile_fn = reconcile_fn
        self.emergency_fn = emergency_fn
        self.poll_interval_sec = float(poll_interval_sec)
        self.long_poll_sec = int(long_poll_sec)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset: int | None = None
        self.last_heartbeat_at: float = time.time()
        self.last_error: str | None = None

    def configured(self) -> bool:
        return self.api.configured and bool(self.chat_id) and bool(self.controller.authorized_chat_id)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="telegram-control", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=8)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
                self.last_heartbeat_at = time.time()
                self.last_error = None
            except Exception as e:  # noqa: BLE001
                self.last_error = scrub_exception(e)
                logger.warning("telegram control poll failed: %s", self.last_error)
                time.sleep(self.poll_interval_sec)

    def poll_once(self) -> int:
        """Process one getUpdates batch. Returns number of handled messages."""
        if not self.configured():
            time.sleep(self.poll_interval_sec)
            return 0
        updates = self.api.get_updates(offset=self._offset, timeout=self.long_poll_sec)
        handled = 0
        for upd in updates:
            uid = int(upd.get("update_id") or 0)
            self._offset = uid + 1
            msg = upd.get("message") or upd.get("edited_message") or {}
            chat = (msg.get("chat") or {}).get("id")
            text = msg.get("text") or ""
            if chat is None or not text:
                # Still consume update_id via mark on empty? Skip without marking command.
                continue
            reply = self.dispatch(chat_id=str(chat), text=str(text), update_id=uid)
            try:
                self.api.send_message(chat, reply)
            except Exception as e:  # noqa: BLE001
                logger.warning("telegram reply failed: %s", scrub_exception(e))
            handled += 1
        return handled

    def dispatch(self, *, chat_id: str, text: str, update_id: int | None = None) -> str:
        result = self.controller.handle_text(chat_id=chat_id, text=text, update_id=update_id)
        if result.data.get("duplicate"):
            return result.message
        if not result.ok and result.message == "Unauthorized.":
            return "Unauthorized."
        dispatch = result.data.get("dispatch")
        if dispatch:
            return self._handle_dispatch(str(dispatch), chat_id=chat_id)
        if result.data.get("startup_checks"):
            return self._run_startup_checks(result.message)
        if result.data.get("emergency") and self.emergency_fn:
            try:
                er = self.emergency_fn()
                return result.message + "\n" + scrub_text(json.dumps(er, default=str)[:1500])
            except Exception as e:  # noqa: BLE001
                return result.message + f"\nEmergency side-effects error: {scrub_exception(e)}"
        # After confirm of strategy/selector/max, push to engine via on_apply already.
        return result.message

    def _run_startup_checks(self, base_message: str) -> str:
        """START/RESUME: reconcile before permitting new entries; fail closed on error."""
        if not self.reconcile_fn:
            return base_message + "\n(startup reconcile unavailable — remaining RUNNING subject to safety)"
        try:
            out = self.reconcile_fn()
            ok = bool(out.get("ok", True))
            if not ok:
                self.controller.fail_closed_halt("STARTUP_RECONCILE_FAILED")
                return (
                    "FAIL CLOSED: reconciliation failed after /start|/resume — HALTED.\n"
                    + scrub_text(json.dumps(out, default=str)[:1500])
                )
            return base_message + "\nStartup reconciliation: OK"
        except Exception as e:  # noqa: BLE001
            self.controller.fail_closed_halt(f"STARTUP_RECONCILE_ERROR:{scrub_exception(e)}")
            return f"FAIL CLOSED: reconciliation error after resume — HALTED.\n{scrub_exception(e)}"

    def _handle_dispatch(self, name: str, *, chat_id: str) -> str:
        if name == "help":
            return HELP_TEXT
        view = (self.engine_view() if self.engine_view else {}) or {}
        if name == "status":
            return format_status(view, self.controller)
        if name == "config":
            return format_config(view, self.controller)
        if name == "positions":
            return format_positions(view)
        if name == "balance":
            return format_balance(view)
        if name == "health":
            return format_health(view, self)
        if name == "performance":
            return format_performance(view)
        if name == "reconcile":
            if not self.reconcile_fn:
                return "Reconcile unavailable."
            try:
                out = self.reconcile_fn()
                ok = bool(out.get("ok", True))
                if not ok:
                    self.controller.fail_closed_halt("MANUAL_RECONCILE_FAILED")
                    return (
                        "RECONCILE FAILED — FAIL CLOSED (new entries blocked).\n"
                        + scrub_text(json.dumps(out, indent=2, default=str)[:3000])
                    )
                return "RECONCILE\n" + scrub_text(json.dumps(out, indent=2, default=str)[:3000])
            except Exception as e:  # noqa: BLE001
                self.controller.fail_closed_halt(f"MANUAL_RECONCILE_ERROR:{scrub_exception(e)}")
                return f"Reconcile failed — FAIL CLOSED: {scrub_exception(e)}"
        return f"Unhandled dispatch {name}"


def format_status(view: dict[str, Any], ctrl: RuntimeController) -> str:
    st = ctrl.state
    safety_halted = bool(view.get("safety_halted"))
    mode = ctrl.display_mode(safety_halted=safety_halted)
    return format_status_message(
        {
            **view,
            "last_signal": view.get("last_signal"),
            "last_trade": view.get("last_trade") or view.get("last_entry"),
            "last_reconciliation": view.get("last_reconciliation"),
        },
        {
            "mode": mode,
            "strategy": st.strategy,
            "selector": st.selector,
            "max_trades": st.max_simultaneous_trades,
            "max_simultaneous_trades": st.max_simultaneous_trades,
        },
    )


def format_performance(view: dict[str, Any]) -> str:
    perf = view.get("performance") or {}
    # Merge top-level equity/unrealized when performance bundle is sparse.
    merged = {
        "total_trades": perf.get("total_trades", view.get("total_trades")),
        "open_trades": perf.get("open_trades", view.get("open_count")),
        "wins": perf.get("wins"),
        "losses": perf.get("losses"),
        "win_rate": perf.get("win_rate"),
        "realized_pnl_btc": perf.get("realized_pnl_btc", view.get("realized_pnl_btc")),
        "unrealized_pnl_btc": perf.get("unrealized_pnl_btc", view.get("unrealized_pnl_btc")),
        "best_trade": perf.get("best_trade"),
        "worst_trade": perf.get("worst_trade"),
        "average_trade_btc": perf.get("average_trade_btc"),
        "total_fees_btc": perf.get("total_fees_btc", view.get("fees_btc")),
        "equity_btc": perf.get("equity_btc", view.get("equity_btc")),
        "cumulative_return_pct": perf.get("cumulative_return_pct"),
    }
    return format_performance_message(merged)


def format_config(view: dict[str, Any], ctrl: RuntimeController) -> str:
    st = ctrl.state
    return "\n".join(
        [
            "⚙️ CONFIG",
            "",
            "RUNTIME CHANGEABLE",
            f"  mode={st.mode}",
            f"  strategy={st.strategy}",
            f"  selector={st.selector}",
            f"  max_simultaneous_trades={st.max_simultaneous_trades}",
            "",
            "FROZEN (Telegram cannot change)",
            "  S threshold 0.65",
            "  new-cross-only entry gate",
            "  T1–T10 trail definitions (activation/SL/trail distances)",
            "  selector algorithms A–F (research definitions)",
            "  risk ceiling 0.5%",
            "  allocation_per_trade 12.5%",
            "  total allocation cap 100%",
            "  universe 37 BTC pairs",
            "  BTC accounting rules",
            "  spot / no leverage",
            "  scoring / factor logic",
            "  research / backtest configuration",
            "  Binance-native OCO protection architecture",
            f"  live.enabled={view.get('live_enabled', False)}",
            f"  dry_run={view.get('dry_run', True)}",
        ]
    )


def format_positions(view: dict[str, Any]) -> str:
    positions = view.get("positions") or []
    if not positions:
        return "POSITIONS\n(none open)"
    lines = ["POSITIONS"]
    for p in positions:
        lines.append(
            f"- {p.get('symbol')} status={p.get('status')} "
            f"strategy={p.get('strategy')} selector={p.get('selector')} "
            f"entry_time={format_brt(p.get('entry_time'))} entry={p.get('entry_price')} "
            f"current={p.get('current_price', 'N/A')} "
            f"unrealized_pnl_btc={p.get('unrealized_pnl_btc', 'N/A')} "
            f"protection={p.get('protection_state', p.get('status'))} "
            f"oco={p.get('binance_oco_list_id')}"
        )
    return "\n".join(lines)


def format_balance(view: dict[str, Any]) -> str:
    return "\n".join(
        [
            "💰 BALANCE (BTC accounting)",
            f"total_equity_btc={view.get('equity_btc', 'N/A')}",
            f"btc_free={view.get('btc_free', 'N/A')}",
            f"btc_locked={view.get('btc_locked', 'N/A')}",
            f"stables_as_btc={view.get('stables_as_btc', 'N/A')}",
            f"alts_as_btc={view.get('alts_as_btc', 'N/A')}",
            f"bnb_free={view.get('bnb_free', 'N/A')} (fee asset only — not a strategy position)",
            f"reconciliation={view.get('reconciliation', 'N/A')}",
            "Secrets never included.",
        ]
    )


def format_health(view: dict[str, Any], plane: TelegramControlPlane) -> str:
    hb = format_brt(plane.last_heartbeat_at)
    crit = view.get("critical_errors") or []
    return "\n".join(
        [
            "HEALTH",
            f"binance_rest={view.get('binance_rest', 'N/A')}",
            f"user_data_ws={view.get('user_data_ws', 'N/A')}",
            f"market_data={view.get('market_data', 'N/A')}",
            f"reconciliation={view.get('reconciliation', 'N/A')}",
            f"execution_engine={view.get('execution_state', 'N/A')}",
            f"telegram={'OK' if plane.configured() else 'NOT_CONFIGURED'}",
            f"safety_state={view.get('safety_state', 'N/A')}",
            f"last_heartbeat={hb}",
            f"control_error={plane.last_error or 'none'}",
            f"critical_errors={'; '.join(str(x) for x in crit[:5]) if crit else 'none'}",
        ]
    )
