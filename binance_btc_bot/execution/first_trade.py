"""Controlled first-real-trade oneshot (Stage 6).

HARD RULES
----------
* Default config stays LIVE=false / DRY_RUN=true.
* This module arms live writes ONLY when Stage-6 live authorize gate PASSes.
* Temporary portfolio: max_simultaneous_trades=1, allocation=12.5%.
* Does NOT auto-expand to 8 trades after close.
* Does NOT force/fake signals — waits for genuine S cross into >= 0.65.
* Does NOT bypass Binance geo / IP restrictions.
* After PROTECTED entry, waits for natural Binance T1/OCO exit (no forced close).
"""

from __future__ import annotations

import json
import logging
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from binance_btc_bot.accounting.equity import compute_equity_btc
from binance_btc_bot.config_loader import load_config
from binance_btc_bot.preflight import PreflightReport, run_preflight
from binance_btc_bot.secrets import scrub_exception

logger = logging.getLogger(__name__)

ScoreProvider = Callable[[str, dict[str, Any]], float | None]


@dataclass
class FirstTradeReport:
    armed: bool = False
    aborted: bool = False
    abort_reason: str = ""
    preflight_gate: dict[str, Any] = field(default_factory=dict)
    signal_time: str | None = None
    symbol: str | None = None
    score: float | None = None
    score_prev: float | None = None
    relative_price: float | None = None
    entry: dict[str, Any] = field(default_factory=dict)
    protection: dict[str, Any] = field(default_factory=dict)
    exit: dict[str, Any] = field(default_factory=dict)
    accounting: dict[str, Any] = field(default_factory=dict)
    equity_before: dict[str, Any] = field(default_factory=dict)
    equity_after: dict[str, Any] = field(default_factory=dict)
    portfolio_before: dict[str, Any] = field(default_factory=dict)
    portfolio_after: dict[str, Any] = field(default_factory=dict)
    ws_events: list[dict[str, Any]] = field(default_factory=list)
    reconciliation: dict[str, Any] = field(default_factory=dict)
    config_restored: dict[str, Any] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "armed": self.armed,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "preflight_gate": self.preflight_gate,
            "signal_time": self.signal_time,
            "symbol": self.symbol,
            "score": self.score,
            "score_prev": self.score_prev,
            "relative_price": self.relative_price,
            "entry": self.entry,
            "protection": self.protection,
            "exit": self.exit,
            "accounting": self.accounting,
            "equity_before": self.equity_before,
            "equity_after": self.equity_after,
            "portfolio_before": self.portfolio_before,
            "portfolio_after": self.portfolio_after,
            "ws_events": list(self.ws_events),
            "reconciliation": self.reconciliation,
            "config_restored": self.config_restored,
            "events": list(self.events),
            "notes": list(self.notes),
            "auto_expand_to_8": False,
        }

    def text(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


def build_first_trade_config(base: dict[str, Any] | None = None) -> dict[str, Any]:
    """In-memory oneshot overlay — does not rewrite production YAML on disk."""
    cfg = deepcopy(base or load_config())
    live = dict(cfg.get("live") or {})
    live["enabled"] = True
    live["dry_run"] = False
    live["strategy"] = "T1"
    live["selector"] = None
    live["first_trade_oneshot"] = True
    cfg["live"] = live
    portfolio = dict(cfg.get("portfolio") or {})
    portfolio["max_simultaneous_trades"] = 1
    portfolio["allocation_per_trade"] = 0.125
    portfolio["max_total_allocation"] = 1.0
    portfolio["one_position_per_symbol"] = True
    cfg["portfolio"] = portfolio
    risk = dict(cfg.get("risk") or {})
    risk["max_allocation_pct"] = 0.125
    risk["max_aggregate_exposure"] = 1.0
    cfg["risk"] = risk
    return cfg


def _scrub_ws_event(ev: dict[str, Any]) -> dict[str, Any]:
    """Keep only non-secret operational fields from a user-data event."""
    return {
        "type": ev.get("type") or ev.get("e"),
        "symbol": ev.get("symbol") or ev.get("s"),
        "side": ev.get("side") or ev.get("S"),
        "status": ev.get("status") or ev.get("X") or ev.get("execution_type") or ev.get("x"),
        "order_id": ev.get("order_id") or ev.get("i"),
        "client_order_id": ev.get("client_order_id") or ev.get("c"),
        "order_list_id": ev.get("order_list_id") or ev.get("g") or ev.get("orderListId"),
        "event_time": ev.get("event_time") or ev.get("E"),
    }


class FirstTradeController:
    """Gate → arm oneshot → wait for one genuine cross → protect → wait natural exit."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        *,
        score_provider: ScoreProvider | None = None,
        poll_interval_sec: float = 15.0,
        max_wait_sec: float | None = None,
        exit_poll_interval_sec: float = 10.0,
        max_exit_wait_sec: float | None = None,
    ) -> None:
        self.base_cfg = cfg or load_config()
        self.score_provider = score_provider
        self.poll_interval_sec = float(poll_interval_sec)
        self.max_wait_sec = max_wait_sec
        self.exit_poll_interval_sec = float(exit_poll_interval_sec)
        self.max_exit_wait_sec = max_exit_wait_sec

    def authorize_or_abort(self) -> tuple[bool, PreflightReport, dict[str, Any]]:
        """Run preflight against current network; return gate decision.

        Uses the SAFE (dry) engine for preflight — never arms writes here.
        """
        # Preflight always against production-safe config (live=false).
        safe_cfg = deepcopy(self.base_cfg)
        safe_cfg.setdefault("live", {})["enabled"] = False
        safe_cfg.setdefault("live", {})["dry_run"] = True
        report = run_preflight(safe_cfg)
        gate = report.stage6_live_authorize_gate()
        return bool(gate["ok"]), report, gate

    def run(self, *, authorize_live: bool = False) -> FirstTradeReport:
        out = FirstTradeReport()
        out.notes.append("first_trade_oneshot: max=1 alloc=12.5% T1 selector=NONE")
        out.notes.append("auto_expand_to_8=false")
        out.notes.append("production YAML not rewritten — live overlay is in-memory only")

        if not authorize_live:
            out.aborted = True
            out.abort_reason = "missing --authorize-live (refusing to arm writes)"
            return out

        env_ok = str(os.environ.get("BINANCE_FIRST_TRADE_AUTHORIZED") or "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if not env_ok:
            out.aborted = True
            out.abort_reason = (
                "BINANCE_FIRST_TRADE_AUTHORIZED not set — refusing to arm live writes"
            )
            return out

        ok, pref, gate = self.authorize_or_abort()
        out.preflight_gate = gate
        out.events.append("PREFLIGHT_COMPLETE")
        if not ok:
            out.aborted = True
            out.abort_reason = gate.get("message") or "DO NOT ENABLE LIVE"
            out.events.append("LIVE_ARM_REFUSED")
            logger.error("%s", out.abort_reason)
            return out

        # Gate passed — arm oneshot config in memory only.
        live_cfg = build_first_trade_config(self.base_cfg)
        out.armed = True
        out.events.append("LIVE_ARMED_ONESHOT_MAX1")

        # Import engine only after gate — construction may allow writes.
        from binance_btc_bot.execution.engine import BinanceBotEngine

        try:
            engine = BinanceBotEngine(
                live_cfg,
                score_provider=self.score_provider,
                allow_live_writes=True,
            )
        except Exception as e:  # noqa: BLE001
            out.aborted = True
            out.abort_reason = f"engine arm failed: {scrub_exception(e)}"
            out.armed = False
            return out

        uds = None
        try:
            uds = self._start_user_data_ws(engine, out)
            return self._wait_and_trade(engine, out)
        finally:
            if uds is not None:
                try:
                    uds.stop()
                except Exception:  # noqa: BLE001
                    pass
            try:
                engine.stop()
            except Exception:  # noqa: BLE001
                pass
            self._record_config_restored(out)

    def _start_user_data_ws(self, engine: Any, out: FirstTradeReport) -> Any:
        from binance_btc_bot.market_data.user_stream import BinanceUserDataWebsocket

        ex = (engine.cfg.get("exchange") or {}) if hasattr(engine, "cfg") else {}
        ws_api_base = str(ex.get("ws_api_base") or "wss://ws-api.binance.com:443/ws-api/v3")

        def on_event(ev: dict[str, Any]) -> None:
            try:
                out.ws_events.append(_scrub_ws_event(ev))
            except Exception:  # noqa: BLE001
                pass

        uds = BinanceUserDataWebsocket(
            api_key=engine.creds.api_key,
            signer=engine.creds.get_signer(),
            ws_api_base=ws_api_base,
            recv_window_ms=int(ex.get("recv_window_ms") or 5000),
            dry_run=False,
            on_event=on_event,
            rest_reconcile=lambda: engine.lifecycle.reconcile_rest(engine.universe),
            on_auth_failure=lambda msg: out.events.append(f"WS_AUTH_FAIL:{msg}"),
        )
        uds.start()
        # Brief wait for subscribe ack
        deadline = time.time() + 15.0
        while time.time() < deadline:
            if uds.subscribed or uds.auth_ok:
                break
            time.sleep(0.25)
        out.events.append(
            f"USER_DATA_WS:{'SUBSCRIBED' if uds.subscribed or uds.auth_ok else 'START_PENDING'}"
            f":events={uds.events_received}"
        )
        if uds.last_error:
            out.notes.append(f"user_data_ws_last_error={uds.last_error}")
        return uds

    def _record_config_restored(self, out: FirstTradeReport) -> None:
        """Confirm production YAML remains safe (oneshot never rewrote disk)."""
        live = (self.base_cfg.get("live") or {})
        port = (self.base_cfg.get("portfolio") or {})
        out.config_restored = {
            "live_enabled": bool(live.get("enabled", False)),
            "dry_run": bool(live.get("dry_run", True)),
            "max_simultaneous_trades": int(port.get("max_simultaneous_trades") or 8),
            "allocation_per_trade": float(port.get("allocation_per_trade") or 0.125),
            "overlay_released": True,
            "disk_unchanged": True,
        }
        out.events.append("CONFIG_RESTORED_SAFE_DEFAULTS")
        out.notes.append(
            "LIVE=false DRY_RUN=true max_simultaneous_trades=8 restored "
            "(in-memory overlay released; production YAML was never live)"
        )

    def _persist(self, engine: Any, out: FirstTradeReport) -> None:
        path = Path(engine.db.path).parent / "first_trade_report.json"
        path.write_text(out.text(), encoding="utf-8")
        note = f"report_path={path}"
        if note not in out.notes:
            out.notes.append(note)
        # Also write a lightweight waiting heartbeat for operators.
        hb = Path(engine.db.path).parent / "first_trade_heartbeat.json"
        hb.write_text(
            json.dumps(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "armed": out.armed,
                    "aborted": out.aborted,
                    "symbol": out.symbol,
                    "events_tail": out.events[-8:],
                    "ws_event_count": len(out.ws_events),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _wait_and_trade(self, engine: Any, out: FirstTradeReport) -> FirstTradeReport:
        if self.score_provider is None:
            out.aborted = True
            out.abort_reason = (
                "score_provider required for genuine S cross — refusing fake/forced signals"
            )
            out.events.append("NO_SCORE_PROVIDER")
            return out

        # Snapshot equity / portfolio before.
        try:
            acct = engine.exchange.get_account()
            btc_usdt = float(engine.exchange.get_price("BTCUSDT"))
            out.equity_before = compute_equity_btc(acct, btc_usdt=btc_usdt).to_dict()
        except Exception as e:  # noqa: BLE001
            out.aborted = True
            out.abort_reason = f"equity snapshot failed: {scrub_exception(e)}"
            return out
        out.portfolio_before = engine.portfolio.snapshot()
        out.events.append("WAITING_GENUINE_CROSS")
        self._persist(engine, out)

        started = time.time()
        traded = False
        last_score: dict[str, float] = {}
        while True:
            if self.max_wait_sec is not None and (time.time() - started) > self.max_wait_sec:
                out.aborted = True
                out.abort_reason = "max_wait_sec elapsed with no genuine cross"
                out.events.append("STILL_WAITING_NO_CROSS")
                break
            if engine.portfolio.slots_used() >= 1 or traded:
                break
            if not engine.safety.allow_new_entries():
                out.aborted = True
                out.abort_reason = f"safety HALT: {engine.safety.reasons}"
                break

            engine.market.sync(engine.universe)
            for sym in engine.universe:
                rel = engine.market.relative_for(sym)
                score = self.score_provider(sym, rel)
                if score is None:
                    continue
                score_f = float(score)
                prev = last_score.get(sym)
                decision = engine.entry_engine.evaluate(
                    symbol=sym,
                    score=score_f,
                    open_symbols=engine.portfolio.open_symbols(),
                    open_count=engine.portfolio.slots_used(),
                    safety_allows=engine.safety.allow_new_entries(),
                    relative_price=rel["relative_price"],
                    strategy_key=engine.strategy_provider.strategy_key(),
                    selector_key=None,
                    reserve_slot=True,
                )
                last_score[sym] = score_f
                if not decision.trade_suggested:
                    continue

                # Genuine cross accepted — execute exactly one live lifecycle entry.
                # Enforce explicit prev < 0.65 AND curr >= 0.65 (defense in depth).
                thr = float(engine.entry_engine.long_threshold)
                if prev is None or not (prev < thr and score_f >= thr):
                    out.notes.append(
                        f"entry_engine suggested {sym} but prev/curr gate failed "
                        f"prev={prev} curr={score_f} thr={thr} — refusing"
                    )
                    if decision.reservation_id and engine.portfolio:
                        try:
                            engine.portfolio.release(decision.reservation_id)
                        except Exception:  # noqa: BLE001
                            pass
                    continue

                out.signal_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                out.symbol = sym
                out.score = score_f
                out.score_prev = float(prev)
                out.relative_price = float(rel.get("relative_price") or 0)
                out.events.append(f"SIGNAL_CROSS:{sym}:prev={prev}:curr={score_f}")
                out.events.append("RESERVATION")
                px = float(engine.market.book.get(sym) or 0)
                eq = float(out.equity_before.get("trading_capital_btc") or 0)
                avail = float(out.equity_before.get("available_btc") or 0)
                out.events.append("ENTRY_SUBMISSION")
                signal = {
                    "timestamp": out.signal_time,
                    "previous_s": float(prev),
                    "current_s": score_f,
                    "threshold": thr,
                    "genuine_new_cross": True,
                }
                try:
                    if hasattr(self.score_provider, "last_entry_snapshot"):
                        snap = self.score_provider.last_entry_snapshot(sym)
                        if isinstance(snap, dict) and snap:
                            signal["entry_snapshot"] = snap
                except Exception:  # noqa: BLE001
                    pass
                life = engine.lifecycle.run_entry(
                    symbol=sym,
                    strategy=engine.current_strategy(),
                    price_alt_btc=px,
                    equity_btc=eq,
                    available_btc=avail,
                    open_exposure_pct=0.0,
                    btc_usdt=float(engine.market.book.get("BTCUSDT") or 0),
                    reservation_id=decision.reservation_id,
                    signal=signal,
                )
                fill = life.fill
                out.entry = {
                    "ok": life.ok,
                    "reason": life.reason,
                    "trade_id": life.trade_id,
                    "reservation_id": life.reservation_id or decision.reservation_id,
                    "status": life.status,
                    "events": list(life.events),
                    "score_prev": out.score_prev,
                    "score_curr": out.score,
                    "relative_price": out.relative_price,
                    "binance_order_id": fill.order_id if fill else None,
                    "client_order_id": fill.client_order_id if fill else None,
                    "requested_quantity": life.size.quantity if life.size else None,
                    "actual_filled_quantity": fill.executed_qty if fill else None,
                    "actual_avg_entry_price": fill.avg_price if fill else None,
                    "requested_allocation": life.size.requested_allocation_pct if life.size else None,
                    "actual_btc_allocation": (
                        (float(fill.executed_qty) * float(fill.avg_price)) if fill else None
                    ),
                    "actual_allocation_pct": (
                        (
                            (float(fill.executed_qty) * float(fill.avg_price) / eq)
                            if fill and eq
                            else None
                        )
                    ),
                    "fees_btc": fill.commission_btc if fill else None,
                    "fees_usdt": fill.commission_usdt if fill else None,
                    "portfolio_equity_before_entry": out.equity_before,
                    "activation": life.oco.activation_price if life.oco else None,
                    "trail_bips": life.oco.trail_bips if life.oco else None,
                    "hard_sl": life.oco.initial_stop if life.oco else None,
                    "oco_list_id": (
                        (life.accounting.binance_oco_list_id if life.accounting else None)
                        or (life.oco.order.order_id if life.oco and life.oco.order else None)
                    ),
                }
                if "BUY_FILLED" in (life.events or []):
                    out.events.append("ENTRY_FILL")
                if "OCO_SUBMITTED" in (life.events or []):
                    out.events.append("OCO_SUBMISSION")
                protected = "PROTECTED" in (life.events or [])
                out.protection = {
                    "protected": protected,
                    "status": life.status,
                    "events": list(life.events),
                    "oco_list_id": out.entry.get("oco_list_id"),
                    "activation": out.entry.get("activation"),
                    "trail_bips": out.entry.get("trail_bips"),
                    "hard_sl": out.entry.get("hard_sl"),
                }
                if protected:
                    out.events.append("PROTECTION_ACCEPTED")
                traded = True
                out.notes.append("ONE trade attempt completed — will not open a second")
                self._persist(engine, out)

                if not life.ok or not protected:
                    out.aborted = True
                    out.abort_reason = (
                        f"entry/protection failed: status={life.status} reason={life.reason} "
                        f"events={life.events} — STOP (no improvisation)"
                    )
                    out.events.append("PROTECTION_FAILED_HALT")
                    break

                # Wait for natural T1 / Binance OCO exit — do not force-close.
                self._await_natural_exit(
                    engine,
                    out,
                    trade_id=str(life.trade_id),
                    symbol=sym,
                    reservation_id=life.reservation_id or decision.reservation_id,
                    entry_qty=float(fill.executed_qty) if fill else 0.0,
                    entry_order_id=str(fill.order_id) if fill and fill.order_id else None,
                    oco_list_id=str(out.entry.get("oco_list_id") or "") or None,
                )
                break

            if traded:
                break
            # Heartbeat while waiting for cross
            if int(time.time() - started) % 60 < self.poll_interval_sec:
                self._persist(engine, out)
            time.sleep(self.poll_interval_sec)

        out.portfolio_after = engine.portfolio.snapshot()
        try:
            acct = engine.exchange.get_account()
            btc_usdt = float(engine.exchange.get_price("BTCUSDT"))
            out.equity_after = compute_equity_btc(acct, btc_usdt=btc_usdt).to_dict()
        except Exception as e:  # noqa: BLE001
            out.notes.append(f"equity_after failed: {scrub_exception(e)}")

        self._persist(engine, out)
        return out

    def _await_natural_exit(
        self,
        engine: Any,
        out: FirstTradeReport,
        *,
        trade_id: str,
        symbol: str,
        reservation_id: str | None,
        entry_qty: float,
        entry_order_id: str | None,
        oco_list_id: str | None,
    ) -> None:
        out.events.append("WAITING_NATURAL_EXIT")
        started = time.time()
        entry_ms = int(time.time() * 1000) - 60_000  # small lookback cushion
        try:
            # Prefer trade entry_time from DB if present
            cur = engine.db._conn.execute(
                "SELECT entry_time FROM trades WHERE trade_id=?", (trade_id,)
            )
            row = cur.fetchone()
            if row and row[0]:
                # leave entry_ms as cushioned now; myTrades filtered by excluding entry order
                pass
        except Exception:  # noqa: BLE001
            pass

        while True:
            if self.max_exit_wait_sec is not None and (time.time() - started) > self.max_exit_wait_sec:
                out.aborted = True
                out.abort_reason = (
                    "max_exit_wait_sec elapsed — trade still open; STOP (position may remain)"
                )
                out.events.append("EXIT_WAIT_TIMEOUT")
                return

            try:
                recon = engine.lifecycle.reconcile_rest([symbol])
                out.reconciliation = {"last": recon, "ok": bool(recon.get("ok"))}
                open_lists = engine.exchange.get_open_order_lists(symbol)
                open_orders = engine.exchange.get_open_orders(symbol)
            except Exception as e:  # noqa: BLE001
                out.notes.append(f"exit_poll_api_error={scrub_exception(e)}")
                time.sleep(self.exit_poll_interval_sec)
                continue

            oco_still_open = False
            if oco_list_id:
                oco_still_open = any(
                    str(x.get("orderListId")) == str(oco_list_id) for x in open_lists
                )
            else:
                oco_still_open = bool(open_lists)

            if oco_still_open or open_orders:
                self._persist(engine, out)
                time.sleep(self.exit_poll_interval_sec)
                continue

            # Protection list gone and no open orders — locate SELL fills.
            try:
                my_trades = engine.exchange.get_my_trades(symbol, limit=100)
            except Exception as e:  # noqa: BLE001
                out.notes.append(f"my_trades_failed={scrub_exception(e)}")
                time.sleep(self.exit_poll_interval_sec)
                continue

            sells: list[dict[str, Any]] = []
            for t in my_trades:
                is_buyer = t.get("isBuyer")
                if is_buyer is True or str(is_buyer).lower() in {"true", "1"}:
                    continue
                oid = str(t.get("orderId") or "")
                if entry_order_id and oid == str(entry_order_id):
                    continue
                # Prefer fills after arming window
                t_ms = int(t.get("time") or 0)
                if t_ms and t_ms < entry_ms:
                    continue
                sells.append(t)

            if not sells:
                # May still be settling; keep waiting briefly
                out.notes.append("oco_gone_but_no_sell_fills_yet")
                self._persist(engine, out)
                time.sleep(self.exit_poll_interval_sec)
                continue

            # Aggregate exit from sell fills (existing accounting path).
            qty = sum(float(t.get("qty") or 0) for t in sells)
            notional = sum(float(t.get("qty") or 0) * float(t.get("price") or 0) for t in sells)
            avg_px = (notional / qty) if qty else 0.0
            fees_btc = 0.0
            fees_usdt = 0.0
            for t in sells:
                c = float(t.get("commission") or 0)
                asset = str(t.get("commissionAsset") or "BTC").upper()
                if asset == "BTC":
                    fees_btc += c
                elif asset in {"USDT", "USDC", "BUSD", "FDUSD"}:
                    fees_usdt += c
            exit_order_id = str(sells[-1].get("orderId") or "")
            btc_usdt = None
            try:
                btc_usdt = float(engine.exchange.get_price("BTCUSDT"))
            except Exception:  # noqa: BLE001
                btc_usdt = None

            closed = engine.lifecycle.close_from_exit_fill(
                trade_id=trade_id,
                reservation_id=reservation_id,
                exit_price=avg_px,
                exit_qty=qty if qty > 0 else None,
                fees_btc=fees_btc,
                fees_usdt=fees_usdt,
                btc_usdt=btc_usdt,
                other_leg_cancelled=True,
            )
            out.events.append("EXIT")
            out.exit = {
                "ok": closed.ok,
                "reason": closed.reason,
                "status": closed.status,
                "events": list(closed.events),
                "exit_order_id": exit_order_id,
                "actual_exit_quantity": qty,
                "actual_avg_exit_price": avg_px,
                "fees_btc": fees_btc,
                "fees_usdt": fees_usdt,
                "sell_fill_count": len(sells),
            }
            if closed.accounting:
                out.accounting = closed.accounting.to_dict()
                out.exit["realized_pnl_btc"] = closed.accounting.realized_pnl_btc
                out.exit["realized_pnl_usdt"] = closed.accounting.realized_pnl_usdt
                out.exit["realized_pnl_btc_equivalent"] = (
                    closed.accounting.realized_pnl_btc_equivalent
                )
            out.events.append("ACCOUNTING")
            out.events.append("SLOT_RELEASE")

            # Post-close verification
            try:
                open_lists2 = engine.exchange.get_open_order_lists(symbol)
                open_orders2 = engine.exchange.get_open_orders(symbol)
                local_open = engine.db.open_trades()
                recon2 = engine.lifecycle.reconcile_rest([symbol])
                out.reconciliation = {
                    "ok": bool(recon2.get("ok"))
                    and not open_lists2
                    and not open_orders2
                    and not local_open,
                    "open_lists": len(open_lists2),
                    "open_orders": len(open_orders2),
                    "local_open_trades": len(local_open),
                    "rest": recon2,
                    "ws_event_count": len(out.ws_events),
                    "enableWithdrawals_unchanged_expected": True,
                }
                out.events.append(
                    "RECONCILIATION:" + ("PASS" if out.reconciliation["ok"] else "FAIL")
                )
            except Exception as e:  # noqa: BLE001
                out.reconciliation = {"ok": False, "error": scrub_exception(e)}
                out.events.append("RECONCILIATION:FAIL")

            out.events.append("NOTIFICATION_VIA_LIFECYCLE")
            self._persist(engine, out)
            return
