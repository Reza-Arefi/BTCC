"""Professional Telegram trade / hourly / status formatters.

Presentation only — displays authoritative fields from engine/accounting/DB
payloads. Never invents P/L, prices, or order IDs. Missing → N/A.
"""

from __future__ import annotations

from typing import Any, Mapping

from binance_btc_bot.notifications.timezone_brt import (
    format_brt,
    format_brt_hm,
    format_hourly_window,
    parse_utc,
)
from binance_btc_bot.secrets import scrub_text

NA = "N/A"

CLOSE_REASON_LABELS = {
    "TRAILING_EXIT": "T1 trailing exit",
    "HARD_SL": "Hard stop loss",
    "TAKE_PROFIT": "Take profit",
    "EMERGENCY_PROTECTION": "Emergency protection",
    "MANUAL_CLOSE": "Manual close",
    "RECONCILED_FLAT": "Reconciled flat",
    "UNKNOWN": "Unknown exit",
}


def na(value: Any) -> str:
    if value is None or value == "" or value == "n/a" or value == "none":
        return NA
    return str(value)


def fmt_btc(value: Any, *, digits: int = 8) -> str:
    if value is None or value == "" or value == "n/a":
        return NA
    try:
        # Fixed precision so small P/L is never hidden; no scientific notation.
        return f"{float(value):.{digits}f}"
    except Exception:  # noqa: BLE001
        return NA


def fmt_pct(value: Any, *, digits: int = 2) -> str:
    if value is None or value == "" or value == "n/a":
        return NA
    try:
        return f"{float(value):.{digits}f}%"
    except Exception:  # noqa: BLE001
        return NA


def fmt_price(value: Any) -> str:
    if value is None or value == "" or value == "n/a":
        return NA
    try:
        v = float(value)
        # Preserve small alt/BTC prices without scientific notation.
        if abs(v) >= 1:
            return f"{v:.8f}".rstrip("0").rstrip(".")
        return f"{v:.10f}".rstrip("0").rstrip(".")
    except Exception:  # noqa: BLE001
        return NA


def fmt_qty(value: Any) -> str:
    if value is None or value == "" or value == "n/a":
        return NA
    try:
        v = float(value)
        return f"{v:.8f}".rstrip("0").rstrip(".")
    except Exception:  # noqa: BLE001
        return NA


def close_reason_label(code: Any) -> str:
    if code is None or code == "":
        return CLOSE_REASON_LABELS["UNKNOWN"]
    key = str(code).upper()
    return CLOSE_REASON_LABELS.get(key, CLOSE_REASON_LABELS["UNKNOWN"])


def _sep() -> str:
    return "────────────"


def _portfolio_block(title: str, snap: Mapping[str, Any] | None, *, after: bool = False) -> list[str]:
    s = snap or {}
    lines = [
        title,
        f"BTC equity: {fmt_btc(s.get('equity_btc'))}",
        f"BTC free:   {fmt_btc(s.get('btc_free'))}",
        f"BTC locked: {fmt_btc(s.get('btc_locked'))}",
        f"Open trades:{na(s.get('open_trades', s.get('open_count')))}",
    ]
    if after:
        lines.append(f"Slots free: {na(s.get('available_slots', s.get('slots_remaining')))}")
    return lines


def format_commissions(fees: Mapping[str, Any] | None) -> list[str]:
    """Display commission legs from authoritative fee payload (no recalculation)."""
    f = fees or {}
    lines: list[str] = []
    legs = f.get("legs") or f.get("commissions") or []
    if isinstance(legs, list) and legs:
        for leg in legs:
            if not isinstance(leg, Mapping):
                continue
            amt = leg.get("amount", leg.get("commission"))
            asset = str(leg.get("asset", leg.get("commissionAsset")) or NA).upper()
            note = " (BNB fee path)" if asset == "BNB" else ""
            lines.append(f"  {fmt_qty(amt)} {asset}{note}")
    else:
        # Flat fields
        if f.get("commission_bnb") is not None:
            lines.append(f"  {fmt_qty(f.get('commission_bnb'))} BNB (BNB fee path)")
        if f.get("commission_base") is not None:
            asset = str(f.get("commission_base_asset") or "BASE").upper()
            lines.append(f"  {fmt_qty(f.get('commission_base'))} {asset}")
        if f.get("commission_btc") is not None or f.get("fees_btc") is not None:
            lines.append(f"  fees_btc={fmt_btc(f.get('commission_btc', f.get('fees_btc')))}")
        if f.get("commission_usdt") is not None or f.get("fees_usdt") is not None:
            lines.append(f"  fees_usdt={na(f.get('commission_usdt', f.get('fees_usdt')))}")
    if not lines:
        lines.append(f"  {NA}")
    return lines


def format_trade_open(data: Mapping[str, Any]) -> str:
    """🟢 TRADE OPENED — payload must supply all financial fields authoritatively."""
    signal = data.get("signal") or {}
    entry = data.get("entry") or {}
    fees = data.get("fees") or {}
    prot = data.get("protection") or {}
    before = data.get("portfolio_before") or {}
    after = data.get("portfolio_after") or {}

    genuine = signal.get("genuine_new_cross")
    cross_line = (
        "Genuine new cross: YES"
        if genuine is True
        else ("Genuine new cross: NO" if genuine is False else f"Genuine new cross: {NA}")
    )

    lines = [
        "🟢 TRADE OPENED",
        _sep(),
        f"Symbol:   {na(data.get('symbol'))}",
        f"Strategy: {na(data.get('strategy'))}",
        f"Selector: {na(data.get('selector') or 'NONE')}",
        "",
        "SIGNAL",
        f"Time:     {format_brt(signal.get('timestamp'))}",
        f"Prev S:   {na(signal.get('previous_s'))}",
        f"Curr S:   {na(signal.get('current_s'))}",
        f"Threshold:{na(signal.get('threshold'))}",
        cross_line,
        "",
        "ENTRY",
        f"Side:     {na(entry.get('side') or 'BUY')}",
        f"Qty:      {fmt_qty(entry.get('quantity'))} {na(entry.get('base_asset'))}",
        f"Fill avg: {fmt_price(entry.get('avg_price'))}",
        f"BTC invested: {fmt_btc(entry.get('btc_invested'))}",
        f"Allocation:   {fmt_pct(entry.get('actual_allocation_pct') if entry.get('actual_allocation_pct') is not None else None)}",
        f"Order ID: {na(entry.get('order_id'))}",
        f"Client ID:{na(entry.get('client_order_id'))}",
        "",
        "FEES",
        *format_commissions(fees if isinstance(fees, Mapping) else {}),
        "",
        "PROTECTION",
        f"Strategy: {na(prot.get('strategy') or data.get('strategy'))}",
        f"Activation: {na(prot.get('activation_display') or prot.get('activation'))}",
        f"Trail dist: {na(prot.get('trail_display') or prot.get('trail_distance'))}",
        f"Hard SL:    {na(prot.get('hard_sl_display') or prot.get('hard_sl'))}",
        f"OCO ID:     {na(prot.get('oco_id'))}",
        f"OCO status: {na(prot.get('oco_status'))}",
        f"Prot qty:   {fmt_qty(prot.get('protected_qty'))}",
        "",
        "PORTFOLIO",
        *_portfolio_block("BEFORE", before),
        *_portfolio_block("AFTER", after, after=True),
        "",
        "Note: BTC equity is total account equity (spot + mapped). Converting BTC→alt does not by itself change total equity.",
    ]
    return scrub_text("\n".join(lines))


def format_protection_event(data: Mapping[str, Any]) -> str:
    """Protection lifecycle alerts with severity emoji."""
    kind = str(data.get("kind") or data.get("event") or "").upper()
    severity = str(data.get("severity") or "").upper()
    emoji = {
        "OCO_ACCEPTED": "🟢",
        "OCO_REJECTED": "🔴",
        "PROTECTION_RETRY": "🟠",
        "EMERGENCY_PROTECTION": "🚨",
        "PROTECTION_FAILURE": "🔴",
        "PROTECTION_FAILED": "🔴",
        "RECONCILE_PROTECTION": "🔄",
        "UNPROTECTED_POSITION": "🚨",
    }.get(kind, "🛡")
    if severity == "CRITICAL" or kind in {"EMERGENCY_PROTECTION", "UNPROTECTED_POSITION"}:
        emoji = "🚨"
    elif severity == "WARNING":
        emoji = "🟠"
    elif severity in {"ERROR", "FAILURE"}:
        emoji = "🔴"

    title = {
        "OCO_ACCEPTED": "OCO ACCEPTED",
        "OCO_REJECTED": "OCO REJECTED",
        "PROTECTION_RETRY": "PROTECTION RETRY",
        "EMERGENCY_PROTECTION": "EMERGENCY PROTECTION",
        "PROTECTION_FAILURE": "PROTECTION FAILURE",
        "PROTECTION_FAILED": "PROTECTION FAILURE",
        "RECONCILE_PROTECTION": "RECONCILE → PROTECTION STATE",
        "UNPROTECTED_POSITION": "UNPROTECTED POSITION",
    }.get(kind, kind or "PROTECTION EVENT")

    verified = data.get("binance_verified")
    state = data.get("protection_state")
    # Never claim PROTECTED unless verified flag is True.
    if state and str(state).upper() in {"PROTECTED", "DRY_RUN_PROTECTED"} and verified is not True:
        state = f"{state} (unverified — not confirmed on Binance)"

    lines = [
        f"{emoji} {title}",
        _sep(),
        f"Symbol:     {na(data.get('symbol'))}",
        f"Quantity:   {fmt_qty(data.get('quantity'))}",
        f"Reason:     {na(data.get('reason'))}",
        f"Protection: {na(state)}",
        f"New entries:{na(data.get('new_entries_status'))}",
        f"OCO/ID:     {na(data.get('oco_id') or data.get('order_id'))}",
    ]
    if data.get("timestamp") is not None:
        lines.insert(2, f"Time:       {format_brt(data.get('timestamp'))}")
    return scrub_text("\n".join(lines))


def format_trade_close(data: Mapping[str, Any]) -> str:
    entry = data.get("entry") or {}
    exit_ = data.get("exit") or {}
    result = data.get("result") or {}
    trail = data.get("trailing") or {}
    before = data.get("portfolio_before") or {}
    after = data.get("portfolio_after") or {}
    recon = data.get("reconciliation") or {}
    reason_code = data.get("close_reason") or "UNKNOWN"

    lines = [
        "🔴 TRADE CLOSED",
        _sep(),
        f"Symbol:   {na(data.get('symbol'))}",
        f"Strategy: {na(data.get('strategy'))}",
        f"Selector: {na(data.get('selector') or 'NONE')}",
        f"Duration: {na(data.get('duration'))}",
        "",
        "ENTRY",
        f"Time:     {format_brt(entry.get('timestamp') or entry.get('entry_time'))}",
        f"Qty:      {fmt_qty(entry.get('quantity'))}",
        f"Fill avg: {fmt_price(entry.get('avg_price') or entry.get('entry_price'))}",
        f"BTC invested: {fmt_btc(entry.get('btc_invested') or entry.get('btc_value'))}",
        f"Order ID: {na(entry.get('order_id') or entry.get('binance_entry_order_id'))}",
        "",
        "EXIT",
        f"Time:     {format_brt(exit_.get('timestamp') or exit_.get('exit_time'))}",
        f"Qty:      {fmt_qty(exit_.get('quantity') or entry.get('quantity'))}",
        f"Fill avg: {fmt_price(exit_.get('avg_price') or exit_.get('exit_price'))}",
        f"Order ID: {na(exit_.get('order_id'))}",
        "",
        "CLOSE REASON",
        f"{close_reason_label(reason_code)} ({na(reason_code)})",
    ]
    if str(reason_code).upper() == "TRAILING_EXIT":
        lines.extend(
            [
                f"Activation: {na(trail.get('activation'))}",
                f"Trail dist: {na(trail.get('trail_distance'))}",
                f"Peak:       {fmt_price(trail.get('peak_price'))}",
                f"Trigger:    {fmt_price(trail.get('exit_trigger_price'))}",
            ]
        )
    lines.extend(
        [
            "",
            "RESULT",
            f"Price P/L %:     {fmt_pct(result.get('price_pnl_pct'))}",
            f"Gross P/L BTC:   {fmt_btc(result.get('gross_pnl_btc'))}",
            f"Fees BTC:        {fmt_btc(result.get('fees_btc'))}",
            f"Net realized BTC:{fmt_btc(result.get('net_realized_pnl_btc') or result.get('realized_pnl_btc'))}",
            f"Commission:      {na(result.get('commission_assets'))}",
            "",
            "PORTFOLIO",
            *_portfolio_block("BEFORE", before),
            *_portfolio_block("AFTER", after, after=True),
            "",
            "RECONCILIATION",
            f"Binance: {na(recon.get('binance'))}",
            f"Local DB:{na(recon.get('local_db'))}",
        ]
    )
    return scrub_text("\n".join(lines))


def format_hourly_report(ctrl_view: Mapping[str, Any], report: Mapping[str, Any]) -> str:
    """Full hourly report from authoritative view + hour report bundle."""
    sys_ = report.get("system") or ctrl_view
    port = report.get("portfolio") or {}
    trading = report.get("trading") or {}
    positions = report.get("positions") or ctrl_view.get("positions") or []
    events = report.get("events") or []
    safety = report.get("safety") or {}
    cum = report.get("cumulative") or {}

    start = port.get("window_start") or report.get("window_start")
    end = port.get("window_end") or report.get("window_end")

    lines = [
        "📊 BTC BOT — HOURLY REPORT",
        format_hourly_window(start, end),
        _sep(),
        "1️⃣ SYSTEM",
        f"State:     {na(sys_.get('state') or sys_.get('mode'))}",
        f"Strategy:  {na(sys_.get('strategy'))}",
        f"Selector:  {na(sys_.get('selector'))}",
        f"Max:       {na(sys_.get('max_trades') or sys_.get('max_simultaneous_trades'))}",
        f"Open/Max:  {na(sys_.get('open_count'))}/{na(sys_.get('max_trades') or sys_.get('max_simultaneous_trades'))}",
        f"Slots:     {na(sys_.get('slots_remaining') or sys_.get('available_slots'))}",
        "",
        "2️⃣ PORTFOLIO 💰",
        "START OF HOUR",
        f"  Equity: {fmt_btc(port.get('start_equity_btc'))}",
        f"  Free:   {fmt_btc(port.get('start_btc_free'))}",
        f"  Locked: {fmt_btc(port.get('start_btc_locked'))}",
        "END OF HOUR",
        f"  Equity: {fmt_btc(port.get('end_equity_btc') or ctrl_view.get('equity_btc'))}",
        f"  Free:   {fmt_btc(port.get('end_btc_free') or ctrl_view.get('btc_free'))}",
        f"  Locked: {fmt_btc(port.get('end_btc_locked') or ctrl_view.get('btc_locked'))}",
        "CHANGE",
        f"  BTC Δ:  {fmt_btc(port.get('equity_change_btc'))}",
        f"  % Δ:    {fmt_pct(port.get('equity_change_pct'))}",
        "BNB",
        f"  Balance:{fmt_qty(port.get('bnb_balance') or ctrl_view.get('bnb_free'))}",
        f"  Fees 1h:{na(port.get('bnb_fees_hour'))}",
        "",
        "3️⃣ TRADING 📈",
        f"Signals:        {na(trading.get('signals'))}",
        f"Genuine crosses:{na(trading.get('genuine_crosses'))}",
        f"Entries:        {na(trading.get('entries'))}",
        f"Exits:          {na(trading.get('exits'))}",
        f"Rejected:       {na(trading.get('rejected_entries'))}",
        f"Closed trades:  {na(trading.get('closed_trades'))}",
        f"Wins / Losses:  {na(trading.get('wins'))} / {na(trading.get('losses'))}",
        f"Win rate:       {na(trading.get('win_rate'))}",
        f"Gross P/L BTC:  {fmt_btc(trading.get('gross_pnl_btc'))}",
        f"Fees BTC:       {fmt_btc(trading.get('fees_btc'))}",
        f"Net realized:   {fmt_btc(trading.get('net_realized_pnl_btc'))}",
        f"Unrealized:     {fmt_btc(trading.get('unrealized_pnl_btc') if trading.get('unrealized_pnl_btc') is not None else ctrl_view.get('unrealized_pnl_btc'))}",
        f"Net port. Δ:    {fmt_btc(trading.get('net_portfolio_change_btc') or port.get('equity_change_btc'))}",
    ]
    activity = int(trading.get("activity_score") or 0)
    if activity == 0 and not any(
        int(trading.get(k) or 0)
        for k in ("signals", "genuine_crosses", "entries", "exits", "rejected_entries", "closed_trades")
    ):
        lines.append("No trading activity during the last hour.")

    lines.extend(["", "4️⃣ CURRENT POSITIONS"])
    if not positions:
        lines.append("(none open)")
    else:
        for p in positions:
            lines.append(
                f"• {na(p.get('symbol'))} | {na(p.get('strategy'))}/{na(p.get('selector') or 'NONE')} | "
                f"entry {format_brt(p.get('entry_time'), with_date=False)} @ {fmt_price(p.get('entry_price'))} | "
                f"now {fmt_price(p.get('current_price'))} | "
                f"uPnL {fmt_pct(p.get('unrealized_pnl_pct'))} / {fmt_btc(p.get('unrealized_pnl_btc'))} | "
                f"{na(p.get('protection_state') or p.get('status'))} "
                f"id={na(p.get('binance_oco_list_id') or p.get('oco_id'))}"
            )

    lines.extend(["", "5️⃣ LAST-HOUR EVENTS"])
    if not events:
        lines.append("(none)")
    else:
        for ev in events:
            if isinstance(ev, Mapping):
                lines.append(f"{format_brt_hm(ev.get('timestamp'))} {na(ev.get('text') or ev.get('event'))}")
            else:
                lines.append(str(ev))

    lines.extend(
        [
            "",
            "6️⃣ SAFETY",
            f"REST:     {na(safety.get('binance_rest') or ctrl_view.get('binance_rest'))}",
            f"User WS:  {na(safety.get('user_data_ws') or ctrl_view.get('user_data_ws'))}",
            f"Market:   {na(safety.get('market_data') or ctrl_view.get('market_data'))}",
            f"Reconcile:{na(safety.get('reconciliation') or ctrl_view.get('reconciliation'))}",
            f"Protect:  {na(safety.get('protection') or ctrl_view.get('protection_status'))}",
            f"Telegram: {na(safety.get('telegram') or ctrl_view.get('telegram'))}",
            f"Engine:   {na(safety.get('execution_engine') or ctrl_view.get('execution_state'))}",
            f"Heartbeat:{format_brt(safety.get('last_heartbeat') or ctrl_view.get('last_heartbeat'))}",
            f"Errors:   {int(safety.get('errors') or 0)}",
            f"Warnings: {int(safety.get('warnings') or 0)}",
            f"Prot fail:{int(safety.get('protection_failures') or ctrl_view.get('protection_failures') or 0)}",
            f"Unknown:  {int(safety.get('unknown_order_states') or ctrl_view.get('unknown_order_states') or 0)}",
            f"Recon fail:{int(safety.get('reconciliation_failures') or ctrl_view.get('reconciliation_failures') or 0)}",
            f"HALT evts:{int(safety.get('halt_events') or 0)}",
            "",
            "7️⃣ CUMULATIVE",
            f"Closed:   {na(cum.get('closed_trades'))}",
            f"W/L:      {na(cum.get('wins'))}/{na(cum.get('losses'))}",
            f"Win rate: {na(cum.get('win_rate'))}",
            f"Realized: {fmt_btc(cum.get('realized_pnl_btc'))}",
            f"Unrealzd: {fmt_btc(cum.get('unrealized_pnl_btc'))}",
            f"Equity:   {fmt_btc(cum.get('equity_btc') or ctrl_view.get('equity_btc'))}",
            f"Return:   {fmt_pct(cum.get('cumulative_return_pct'))}",
        ]
    )
    return scrub_text("\n".join(lines))


def format_status_message(view: Mapping[str, Any], runtime: Mapping[str, Any]) -> str:
    lines = [
        "⚙️ STATUS",
        _sep(),
        f"State:    {na(runtime.get('mode') or view.get('mode'))}",
        f"Strategy: {na(runtime.get('strategy'))}",
        f"Selector: {na(runtime.get('selector'))}",
        f"Open/Max: {na(view.get('open_count'))}/{na(runtime.get('max_trades') or runtime.get('max_simultaneous_trades'))}",
        f"Equity:   {fmt_btc(view.get('equity_btc'))} BTC",
        f"Free:     {fmt_btc(view.get('btc_free'))} BTC",
        f"BNB:      {fmt_qty(view.get('bnb_free'))}",
        f"Realized: {fmt_btc(view.get('realized_pnl_btc'))}",
        f"Unrealzd: {fmt_btc(view.get('unrealized_pnl_btc'))}",
        f"REST:     {na(view.get('binance_rest'))}",
        f"User WS:  {na(view.get('user_data_ws'))}",
        f"Market:   {na(view.get('market_data'))}",
        f"Reconcile:{na(view.get('reconciliation'))}",
        f"Protect:  {na(view.get('protection_status'))}",
        f"Signal:   {na(view.get('last_signal'))}",
        f"Trade:    {na(view.get('last_trade') or view.get('last_entry'))}",
        f"Last recon:{format_brt(view.get('last_reconciliation'))}",
    ]
    return scrub_text("\n".join(lines))


def format_performance_message(perf: Mapping[str, Any]) -> str:
    lines = [
        "📈 PERFORMANCE",
        _sep(),
        f"Total trades:  {na(perf.get('total_trades'))}",
        f"Open trades:   {na(perf.get('open_trades'))}",
        f"Wins/Losses:   {na(perf.get('wins'))}/{na(perf.get('losses'))}",
        f"Win rate:      {na(perf.get('win_rate'))}",
        f"Realized BTC:  {fmt_btc(perf.get('realized_pnl_btc'))}",
        f"Unrealized BTC:{fmt_btc(perf.get('unrealized_pnl_btc'))}",
        f"Best trade:    {na(perf.get('best_trade'))}",
        f"Worst trade:   {na(perf.get('worst_trade'))}",
        f"Average trade: {fmt_btc(perf.get('average_trade_btc'))}",
        f"Total fees:    {fmt_btc(perf.get('total_fees_btc'))}",
        f"Equity:        {fmt_btc(perf.get('equity_btc'))}",
        f"Cum. return:   {fmt_pct(perf.get('cumulative_return_pct'))}",
    ]
    return scrub_text("\n".join(lines))


def format_notification_message(event: str, details: Mapping[str, Any] | None) -> str | None:
    """If details carries a presentation kind/payload, return formatted Telegram body."""
    d = details or {}
    kind = str(d.get("kind") or "").lower()
    payload = d.get("payload") if isinstance(d.get("payload"), Mapping) else d
    ev = str(event or "").upper()

    if kind in {"trade_open", "open"}:
        return format_trade_open(payload)  # type: ignore[arg-type]
    if kind in {"trade_close", "close"}:
        return format_trade_close(payload)  # type: ignore[arg-type]
    if kind in {"protection", "protection_event"}:
        body = dict(payload) if isinstance(payload, Mapping) else dict(d)
        if "kind" not in body and "event" not in body:
            body["kind"] = ev
        return format_protection_event(body)

    if ev in {"TRADE_OPENED"} and isinstance(payload, Mapping) and payload.get("entry"):
        return format_trade_open(payload)
    if ev in {"TRADE_CLOSED"} and isinstance(payload, Mapping) and (payload.get("exit") or payload.get("close_reason")):
        return format_trade_close(payload)
    if ev in {
        "OCO_ACCEPTED",
        "OCO_REJECTED",
        "PROTECTION_RETRY",
        "EMERGENCY_PROTECTION",
        "PROTECTION_FAILED",
        "PROTECTION_FAILURE",
        "RECONCILE_PROTECTION",
        "UNPROTECTED_POSITION",
    } and (
        d.get("protection_state") is not None
        or d.get("binance_verified") is not None
        or kind.startswith("protection")
        or d.get("symbol")
    ):
        body = dict(payload) if isinstance(payload, Mapping) else dict(d)
        body.setdefault("kind", ev)
        return format_protection_event(body)

    if d.get("telegram_message"):
        return scrub_text(str(d["telegram_message"]))
    return None


def compact_hour_event(timestamp: Any, text: str) -> dict[str, Any]:
    return {"timestamp": timestamp, "text": text}


# Avoid unused import warning for parse_utc in external tests
__all__ = [
    "CLOSE_REASON_LABELS",
    "close_reason_label",
    "compact_hour_event",
    "fmt_btc",
    "fmt_pct",
    "fmt_price",
    "fmt_qty",
    "format_commissions",
    "format_hourly_report",
    "format_notification_message",
    "format_performance_message",
    "format_protection_event",
    "format_status_message",
    "format_trade_close",
    "format_trade_open",
    "na",
    "parse_utc",
]
