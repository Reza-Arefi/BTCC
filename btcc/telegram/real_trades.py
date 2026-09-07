"""Telegram notifications for REAL / LIVE T1 canary trades.

Observability only — never gates or retries exchange orders.
Trader-facing messages (LIVE/REAL, never PAPER). Timestamps in BRT (UTC−3).
Internal audit logs keep UTC separately.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

BRT = timezone(timedelta(hours=-3))
SEP = "━━━━━━━━━━━━━━━━━━"


class TelegramSentLedger:
    """Durable idempotency keys so restart/reconcile cannot duplicate OPEN/CLOSE."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._keys: set[str] = set()
        if self.path and self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._keys = {str(k) for k, v in raw.items() if v}
                elif isinstance(raw, list):
                    self._keys = {str(k) for k in raw}
            except Exception as e:  # noqa: BLE001
                logger.warning("telegram ledger load failed: %s", e)

    def already_sent(self, key: str) -> bool:
        return str(key) in self._keys

    def mark_sent(self, key: str) -> None:
        k = str(key)
        if k in self._keys:
            return
        self._keys.add(k)
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {x: True for x in sorted(self._keys)}
            self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning("telegram ledger write failed: %s", e)


def to_brt(ts: datetime | str | None = None) -> datetime:
    """Convert a timestamp to Brazil time (UTC−3)."""
    if ts is None:
        dt = datetime.now(timezone.utc)
    elif isinstance(ts, datetime):
        dt = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    else:
        raw = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(BRT)


def format_brt(ts: datetime | str | None = None, *, with_date: bool = False) -> str:
    dt = to_brt(ts)
    if with_date:
        return dt.strftime("%Y-%m-%d %H:%M:%S") + " BRT (UTC−3)"
    return dt.strftime("%H:%M:%S") + " BRT (UTC−3)"


def format_pair(symbol: str | None) -> str:
    s = str(symbol or "").upper().replace("/", "").replace("-", "").replace("_", "")
    if s.endswith("BTC") and len(s) > 3:
        return f"{s[:-3]}/BTC"
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/USDT"
    return str(symbol or "n/a")


def base_asset(symbol: str | None) -> str:
    pair = format_pair(symbol)
    return pair.split("/", 1)[0] if "/" in pair else str(symbol or "")


def _pct_label(frac: float | None, *, signed: bool = True) -> str:
    if frac is None:
        return "n/a"
    pct = float(frac) * 100.0
    if signed:
        return f"{pct:+.2f}%"
    return f"{abs(pct):.2f}%"


def _fmt_price(x: Any, *, digits: int = 10) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{digits}f}".rstrip("0").rstrip(".") if digits else f"{float(x)}"
    except (TypeError, ValueError):
        return str(x)


def _fmt_btc(x: Any, *, digits: int = 8) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def _fmt_qty(x: Any) -> str:
    if x is None:
        return "n/a"
    try:
        q = float(x)
        if abs(q - round(q)) < 1e-12:
            return str(int(round(q)))
        s = f"{q:.8f}".rstrip("0").rstrip(".")
        return s
    except (TypeError, ValueError):
        return str(x)


def exit_kind_label(exit_reason: str | None, *, activated: bool | None = None) -> str:
    raw = str(exit_reason or "")
    if activated is True or "TRAILING" in raw.upper():
        return "TRAILING STOP"
    if "INITIAL" in raw.upper():
        return "INITIAL STOP LOSS"
    return raw or "T1 EXIT"


def format_real_open_message(payload: dict[str, Any]) -> str:
    t1 = payload.get("t1") or {}
    sl = float(t1.get("stop_loss_pct") if t1.get("stop_loss_pct") is not None else 0.0075)
    act = float(t1.get("activation_pct") if t1.get("activation_pct") is not None else 0.0075)
    trail = float(t1.get("trailing_pct") if t1.get("trailing_pct") is not None else 0.0025)
    pair = format_pair(str(payload.get("symbol") or ""))
    base = base_asset(str(payload.get("symbol") or ""))
    ts = payload.get("entry_time") or payload.get("asof_utc")
    lines = [
        "🟢 LIVE TRADE OPENED",
        SEP,
        f"Pair: {pair}",
        "Strategy: T1",
        "",
        "Entry",
        f"• Price: {_fmt_price(payload.get('entry_vwap'))} BTC",
        f"• Quantity: {_fmt_qty(payload.get('executed_quantity'))} {base}",
        f"• Capital: {_fmt_btc(payload.get('btc_allocated'))} BTC",
        "",
        "T1 Protection",
        f"• Initial SL: {_pct_label(-sl)}",
        f"• Stop Price: {_fmt_price(payload.get('initial_stop'))} BTC",
        f"• Activation: {_pct_label(+act)}",
        f"• Activation Price: {_fmt_price(payload.get('activation_price'))} BTC",
        f"• Trailing: {_pct_label(trail, signed=False)}",
        "",
        "Account",
        f"• Available BTC: {_fmt_btc(payload.get('btc_balance'))}",
        "",
        f"🕐 {format_brt(ts)}",
        "",
        (
            "Status: 🟢 PROTECTED"
            if bool(payload.get("protection_verified"))
            else "Status: ⚠️ PROTECTION NOT VERIFIED"
        ),
    ]
    return "\n".join(lines)


def format_real_close_message(payload: dict[str, Any]) -> str:
    pair = format_pair(str(payload.get("symbol") or ""))
    base = base_asset(str(payload.get("symbol") or ""))
    entry = float(payload.get("entry_vwap") or 0.0)
    exit_px = float(payload.get("exit_vwap") or 0.0)
    trigger_px = payload.get("trigger_price")
    if trigger_px is None:
        trigger_px = payload.get("stop_price")
    exec_pct = ((exit_px / entry) - 1.0) if entry > 0 else None
    if payload.get("trigger_pct") is not None:
        trig_pct = float(payload["trigger_pct"])
    elif trigger_px is not None and entry > 0:
        trig_pct = (float(trigger_px) / entry) - 1.0
    else:
        trig_pct = None
    slip_pp = None
    if exec_pct is not None and trig_pct is not None:
        slip_pp = (exec_pct - trig_pct) * 100.0

    kind = exit_kind_label(
        payload.get("exit_reason"),
        activated=payload.get("activated"),
    )
    # Configured stop distance label (not the actual execution %).
    t1 = payload.get("t1") or {}
    if kind == "INITIAL STOP LOSS":
        stop_cfg = _pct_label(-float(t1.get("stop_loss_pct") or 0.0075))
    elif kind == "TRAILING STOP":
        stop_cfg = f"trail {_pct_label(float(t1.get('trailing_pct') or 0.0025), signed=False)}"
    else:
        stop_cfg = "n/a"

    ts = payload.get("asof_utc") or payload.get("exit_time")
    lines = [
        "🔴 LIVE TRADE CLOSED",
        SEP,
        f"Pair: {pair}",
        "Strategy: T1",
        "",
        "Trade",
        f"• Entry: {_fmt_price(entry)} BTC",
        f"• Exit: {_fmt_price(exit_px)} BTC",
        f"• Quantity: {_fmt_qty(payload.get('executed_quantity'))} {base}",
        "",
        "Result",
        f"• Return: {_pct_label(exec_pct) if exec_pct is not None else 'n/a'}",
        f"• Gross P/L: {_fmt_btc(payload.get('gross_pnl_btc'), digits=9)} BTC",
        f"• Fees: {_fmt_btc(payload.get('fees_btc'), digits=9)} BTC",
        f"• Net P/L: {_fmt_btc(payload.get('net_pnl_btc'), digits=9)} BTC",
        "",
        "Exit",
        f"• Reason: {kind}",
        f"• Stop: {stop_cfg}",
        f"• Trigger: {_pct_label(trig_pct) if trig_pct is not None else 'n/a'}",
        f"• Trigger Price: {_fmt_price(trigger_px)} BTC",
        f"• Actual execution: {_pct_label(exec_pct) if exec_pct is not None else 'n/a'}",
        f"• Execution VWAP: {_fmt_price(exit_px)} BTC",
    ]
    if slip_pp is not None:
        lines.append(f"• Slippage vs trigger: {slip_pp:+.2f} pp")
    lines.extend(
        [
            "",
            "Account",
            f"• BTC before: {_fmt_btc(payload.get('btc_before'))}",
            f"• BTC after: {_fmt_btc(payload.get('btc_after'))}",
            "",
            f"🕐 {format_brt(ts)}",
            "",
            "Position: CLOSED",
        ]
    )
    return "\n".join(lines)


def format_hourly_real_message(stats: dict[str, Any], *, version: dict[str, Any] | None = None) -> str:
    v = version or {}
    asof = stats.get("asof_utc") or stats.get("asof_brt")
    brt = to_brt(asof if isinstance(asof, (datetime, str)) else None)
    open_n = int(stats.get("n_open") or 0)
    positions = list(stats.get("open_positions") or [])
    single = stats.get("open_position")
    # Single-position callers (canary / tests) may update open_position only —
    # treat it as authoritative when multi-list is empty or length ≤ 1.
    if single is not None and len(positions) <= 1:
        positions = [single]
    open_lines: list[str]
    if open_n <= 0 and not positions:
        open_lines = ["Open Positions: 0", "• NONE (flat)"]
    elif not positions:
        open_lines = ["Open Positions: 0", "• NONE (flat)"]
    else:
        open_n = max(open_n, len(positions))
        open_lines = [f"Open Positions: {open_n}"]
        for pos in positions:
            activated = bool(pos.get("activated"))
            t1_status = "ACTIVE" if activated else "INACTIVE"
            pair = format_pair(str(pos.get("symbol") or ""))
            base = base_asset(str(pos.get("symbol") or ""))
            sl = float(pos.get("stop_loss_pct") if pos.get("stop_loss_pct") is not None else 0.0075)
            act = float(pos.get("activation_pct") if pos.get("activation_pct") is not None else 0.0075)
            trail = float(pos.get("trailing_pct") if pos.get("trailing_pct") is not None else 0.0025)
            open_lines.extend(
                [
                    f"• {pair}",
                    f"• Qty: {_fmt_qty(pos.get('executed_quantity'))} {base}",
                    f"• Entry: {_fmt_price(pos.get('entry_vwap'))} BTC",
                    f"• Current T1 Status: {t1_status}",
                    f"• Initial SL: {_pct_label(-sl)}",
                    f"• Stop: {_fmt_price(pos.get('stop_price') or pos.get('initial_stop'))} BTC",
                    f"• Activation: {_pct_label(+act)}",
                    f"• Activation Price: {_fmt_price(pos.get('activation_price'))} BTC",
                    f"• Trail: {_pct_label(trail, signed=False)}",
                ]
            )
            if activated:
                open_lines.extend(
                    [
                        f"• Highest Price: {_fmt_price(pos.get('highest_price'))} BTC",
                        f"• Trailing Stop: {_fmt_price(pos.get('trailing_stop') or pos.get('stop_price'))} BTC",
                    ]
                )

    status = str(stats.get("execution_status") or "NORMAL")
    live = "🟢 LIVE" if status.upper() in {"MONITORING", "NORMAL", "LIVE"} else status
    lines = [
        "⏱ LIVE PORTFOLIO — HOURLY",
        "━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🕐 {brt.strftime('%H:%M')} BRT · UTC−3",
        "",
        "Account",
        f"• BTC Available: {_fmt_btc(stats.get('btc_balance'))} BTC",
        f"• Open Positions: {open_n}",
        f"• Realized P/L: {_fmt_btc(stats.get('realized_pnl_btc'))} BTC",
        "",
        *open_lines,
        "",
        "System",
        f"• Strategy: {v.get('strategy_version') or 'T1-ONLY'}",
        f"• Execution: {status}",
        f"• Status: {live}",
        "",
        "LIVE / REAL — real money",
    ]
    return "\n".join(lines)


def brt_hour_bucket(now: datetime | None = None) -> str:
    """Hourly bucket key in BRT (UTC−3), e.g. 2026-09-05T00."""
    dt = to_brt(now)
    return dt.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H")


def compute_hourly_real_stats(
    *,
    btc_balance: float | None,
    n_entered: int,
    n_closed: int,
    realized_pnl_btc: float | None,
    open_position: dict[str, Any] | None,
    uptime_s: float,
    execution_status: str,
    canary_phase: str,
    write_gate_armed: bool,
    now: datetime | None = None,
    open_positions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    positions = list(open_positions or [])
    if not positions and open_position:
        positions = [open_position]
    n_open = len(positions)
    return {
        "asof_utc": now,
        "asof_brt": to_brt(now),
        "hour_bucket": brt_hour_bucket(now),
        "btc_balance": btc_balance,
        "n_entered": int(n_entered),
        "n_closed": int(n_closed),
        "n_open": n_open,
        "realized_pnl_btc": realized_pnl_btc,
        "open_position": positions[0] if len(positions) == 1 else open_position,
        "open_positions": positions,
        "uptime_s": round(float(uptime_s), 3),
        "execution_status": execution_status,
        "canary_phase": canary_phase,
        "write_gate_armed": bool(write_gate_armed),
    }


class RealTradeNotifier:
    """Send LIVE/REAL trade Telegram messages. Failures never raise to callers."""

    def __init__(
        self,
        send_fn: Callable[[str], bool],
        *,
        enabled: bool = True,
        ledger: TelegramSentLedger | None = None,
    ) -> None:
        self.send_fn = send_fn
        self.enabled = enabled
        self.ledger = ledger or TelegramSentLedger()

    def _send(self, text: str) -> bool:
        if not self.enabled:
            logger.info("REAL Telegram disabled — message:\n%s", text)
            return False
        try:
            return bool(self.send_fn(text))
        except Exception as e:  # noqa: BLE001
            logger.error("REAL Telegram send failed (non-blocking): %s", e)
            return False

    def trade_opened(self, payload: dict[str, Any]) -> bool:
        key = f"OPEN:{payload.get('client_order_id') or payload.get('position_id')}"
        if self.ledger.already_sent(key):
            logger.info("REAL Telegram OPEN skipped (idempotent): %s", key)
            return False
        ok = self._send(format_real_open_message(payload))
        if ok or not self.enabled:
            self.ledger.mark_sent(key)
        return ok

    def trade_closed(self, payload: dict[str, Any]) -> bool:
        key = (
            f"CLOSE:{payload.get('exit_client_order_id') or payload.get('position_id')}"
        )
        if self.ledger.already_sent(key):
            logger.info("REAL Telegram CLOSE skipped (idempotent): %s", key)
            return False
        ok = self._send(format_real_close_message(payload))
        if ok or not self.enabled:
            self.ledger.mark_sent(key)
        return ok

    def hourly_summary(
        self,
        stats: dict[str, Any],
        *,
        version: dict[str, Any] | None = None,
        bucket_key: str | None = None,
    ) -> bool:
        key = f"HOURLY:{bucket_key or stats.get('hour_bucket')}"
        if self.ledger.already_sent(key):
            logger.info("REAL Telegram HOURLY skipped (idempotent): %s", key)
            return False
        ok = self._send(format_hourly_real_message(stats, version=version))
        if ok or not self.enabled:
            self.ledger.mark_sent(key)
        return ok


def build_real_telegram_send_from_env() -> Callable[[str], bool]:
    """Build send_fn from BTCC_TELEGRAM_* env (same credentials as paper)."""
    import os

    from btcc.telegram.notifier import TelegramNotifier

    token = os.getenv("BTCC_TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("BTCC_TELEGRAM_CHAT_ID", "")
    tg = TelegramNotifier(token, chat_id, enabled=True)
    return tg.send
