"""Runtime glue for the portfolio SafetyManager (monitoring, stop replacement, commands).

Exchange side effects happen only here and only when the engine is live-enabled:

* ``replace_stop``: cancel the trade's W2 OCO and re-place it with a tighter
  STOP_LOSS leg (same W2 trailing TAKE_PROFIT delta). If the re-place fails the
  position is sold at market; if that also fails the trade is marked
  PROTECTION_FAILED and everything halts.
* ``exit_position``: cancel OCO → MARKET SELL → close through the lifecycle.
  If the sell fails, the previous protection level is re-placed.

All protection actions are serialized with ``lifecycle.protection_lock`` so REST
reconciliation never observes the brief cancel→re-place window.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any

from binance_btc_bot.exchange.base import OrderRequest, TrailingOcoRequest
from binance_btc_bot.risk.safety_manager import (
    ACTION_EXIT,
    ACTION_REPLACE,
    PositionView,
    SafetyManager,
)
from binance_btc_bot.secrets import scrub_exception

logger = logging.getLogger(__name__)

MANAGED_STATUSES = frozenset({"PROTECTED", "DRY_RUN_PROTECTED"})
PENDING_STATUSES = frozenset({"ENTRY_PENDING", "ENTRY_FILLED", "PROTECTION_PENDING", "EXIT_PENDING"})
TAKER_FEE = 0.001


def _floor_tick(px: float, tick: float) -> float:
    if tick <= 0:
        return float(px)
    return math.floor(float(px) / tick + 1e-9) * tick


def _ceil_tick(px: float, tick: float) -> float:
    if tick <= 0:
        return float(px)
    return math.ceil(float(px) / tick - 1e-9) * tick


def _floor_step(qty: float, step: float) -> float:
    if step <= 0:
        return float(qty)
    return math.floor(float(qty) / step + 1e-9) * step


def _trade_cfg(trade: dict[str, Any]) -> dict[str, Any]:
    raw = trade.get("strategy_config_json")
    try:
        return json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    except Exception:  # noqa: BLE001
        return {}


def base_stop_for_trade(trade: dict[str, Any], default_sl: float = 0.05) -> float | None:
    entry = float(trade.get("entry_price") or 0)
    if entry <= 0:
        return None
    sl = float(_trade_cfg(trade).get("arm_sl_activation_trail") or default_sl)
    return entry * (1.0 - sl)


class SafetyRuntime:
    def __init__(self, engine: Any, manager: SafetyManager) -> None:
        self.engine = engine
        self.manager = manager
        self._last_cycle_at = 0.0

    # ------------------------------------------------------------ helpers
    @property
    def _live(self) -> bool:
        return bool(self.engine.live_enabled) and not bool(self.engine.dry_run)

    def _lock(self):
        return self.engine.lifecycle.protection_lock

    def _notify(self, severity: str, event: str, message: str) -> None:
        n = getattr(self.engine, "notifications", None)
        if n is None:
            return
        method = {
            "CRITICAL": "notify_critical",
            "ERROR": "notify_error",
            "WARNING": "notify_warning",
        }.get(severity.upper(), "notify_info")
        try:
            getattr(n, method)(event, message)
        except Exception:  # noqa: BLE001
            logger.warning("safety runtime notify failed (ignored)")

    def open_positions(self, prices: dict[str, float] | None = None) -> tuple[list[PositionView], list[dict[str, Any]]]:
        trades = [t for t in self.engine.db.open_trades() if float(t.get("entry_price") or 0) > 0]
        symbols = sorted({str(t["symbol"]).upper() for t in trades})
        if prices is None:
            prices = self.engine.exchange.get_prices(symbols) if symbols else {}
        views: list[PositionView] = []
        for t in trades:
            sym = str(t["symbol"]).upper()
            px = prices.get(sym)
            if px is None or float(px) <= 0:
                raise RuntimeError(f"missing price for open position {sym}")
            tp = self.manager.protection_for(str(t["trade_id"]))
            stop = tp.stop if tp and tp.stop else base_stop_for_trade(t)
            views.append(PositionView(str(t["trade_id"]), sym, float(t["entry_price"]), float(px), stop))
        return views, trades

    # ------------------------------------------------------------ monitoring
    def cycle(self, *, force: bool = False) -> dict[str, Any]:
        """One monitoring pass: verify prices, detect defensive mode, apply defensive protection."""
        if not self.manager.config.enabled:
            return {"enabled": False}
        now = time.time()
        if not force and (now - self._last_cycle_at) < self.manager.config.monitor_interval_sec:
            return {"skipped": True}
        self._last_cycle_at = now
        try:
            views, trades = self.open_positions()
        except Exception as e:  # noqa: BLE001 — fail closed on unverifiable state
            self.manager.mark_monitor_failed(f"position/price check failed: {scrub_exception(e)}")
            return {"ok": False, "error": scrub_exception(e)}
        self.manager.mark_monitor_ok()
        for v, t in zip(views, trades):
            self.manager.register_position(v.trade_id, v.symbol, v.entry_price, base_stop_for_trade(t))
        newly = self.manager.evaluate_portfolio(views)
        by_id = {str(t["trade_id"]): t for t in trades}
        managed = [v for v in views if str(by_id[v.trade_id].get("status") or "").upper() in MANAGED_STATUSES]
        results: list[dict[str, Any]] = []
        for act in self.manager.plan_protection(managed):
            trade = by_id[act.trade_id]
            price = next(v.current_price for v in managed if v.trade_id == act.trade_id)
            if act.action == ACTION_REPLACE and act.new_stop is not None:
                results.append(self.replace_stop(trade, act.new_stop, price, act.stage))
            elif act.action == ACTION_EXIT:
                self._notify("WARNING", "SAFETY_DEFENSIVE_EXIT", f"🛡 PROTECTION:\n{act.reason}\nClosing position.")
                results.append(self.exit_position(trade, reason="SAFETY_EXIT", current_price=price))
        return {"ok": True, "positions": len(views), "defensive_activated": newly, "actions": results}

    # ------------------------------------------------------- stop replacement
    def _open_lists_for(self, symbol: str) -> list[Any]:
        from binance_btc_bot.execution.order_lists import parse_order_list

        views = [parse_order_list(r) for r in self.engine.exchange.get_open_order_lists(symbol)]
        return [v for v in views if v.is_open]

    def _sellable_qty(self, trade: dict[str, Any], step: float) -> float:
        base = str(trade["symbol"]).upper()[:-3]
        bal = self.engine.exchange.get_balance(base)
        return _floor_step(min(float(bal.free), float(trade.get("quantity") or 0)), step)

    def _cancel_oco(self, symbol: str, oco_id: str) -> bool:
        """Cancel and verify. Returns True once the list is confirmed not open."""
        try:
            self.engine.exchange.cancel_order_list(symbol, order_list_id=str(oco_id))
        except Exception as e:  # noqa: BLE001
            logger.warning("cancel OCO %s %s raised: %s", symbol, oco_id, scrub_exception(e))
        return not any(str(v.order_list_id) == str(oco_id) for v in self._open_lists_for(symbol))

    def _place_oco(self, trade: dict[str, Any], stop: float, price: float, qty: float, info: Any) -> Any:
        cfg = _trade_cfg(trade)
        entry = float(trade["entry_price"])
        act = float(cfg.get("activation") or trade.get("configured_activation") or 0.04)
        trail = float(cfg.get("trail_distance") or trade.get("configured_trailing_distance") or 0.02)
        above = _ceil_tick(max(entry * (1.0 + act), price * 1.002), info.price_tick)
        n = int(time.time()) % 100000
        req = TrailingOcoRequest(
            symbol=str(trade["symbol"]).upper(),
            side="SELL",
            quantity=qty,
            above_type="TAKE_PROFIT",
            above_stop_price=above,
            above_trailing_delta=int(round(trail * 10_000)),
            below_type="STOP_LOSS",
            below_stop_price=_floor_tick(stop, info.price_tick),
            list_client_order_id=f"s{n}_{str(trade['trade_id'])[:16]}",
        )
        return self.engine.exchange.place_trailing_exit(req)

    def replace_stop(self, trade: dict[str, Any], new_stop: float, price: float, stage: str) -> dict[str, Any]:
        tid = str(trade["trade_id"])
        sym = str(trade["symbol"]).upper()
        if not self._live:
            self.manager.confirm_stop(tid, new_stop, stage)
            self.engine.db.insert_event("SAFETY_STOP_REPLACED_DRY", symbol=sym, trade_id=tid,
                                        payload={"new_stop": new_stop, "stage": stage, "price": price})
            return {"trade_id": tid, "result": "DRY_RUN_TIGHTENED", "stop": new_stop}
        with self._lock():
            fresh = self.engine.db.get_trade(tid) or trade
            if str(fresh.get("status") or "").upper() not in MANAGED_STATUSES:
                return {"trade_id": tid, "result": "SKIPPED_STATUS", "status": fresh.get("status")}
            oco = fresh.get("binance_oco_list_id")
            if not oco:
                return {"trade_id": tid, "result": "SKIPPED_NO_OCO"}
            info = self.engine.exchange.get_symbol_info(sym)
            open_lists = self._open_lists_for(sym)
            if len(open_lists) > 1:
                self.manager.enter_halt(f"DUPLICATE_PROTECTION_LISTS:{sym}")
                return {"trade_id": tid, "result": "DUPLICATE_LISTS_HALT"}
            if not open_lists or str(open_lists[0].order_list_id) != str(oco):
                return {"trade_id": tid, "result": "SKIPPED_OCO_MISMATCH"}
            stop_px = _floor_tick(new_stop, info.price_tick)
            if stop_px >= price:
                return self._exit_locked(fresh, "SAFETY_EXIT", price, info)
            if not self._cancel_oco(sym, str(oco)):
                return {"trade_id": tid, "result": "CANCEL_FAILED_OLD_OCO_KEPT"}
            qty = self._sellable_qty(fresh, info.quantity_step)
            if qty < float(info.min_quantity or 0) or qty <= 0:
                return {"trade_id": tid, "result": "FLAT_AFTER_CANCEL"}
            res = None
            try:
                res = self._place_oco(fresh, stop_px, price, qty, info)
            except Exception as e:  # noqa: BLE001
                logger.error("replacement OCO failed %s: %s", sym, scrub_exception(e))
            if res is not None and res.ok and res.order_id:
                self.engine.db.update_trade(tid, binance_oco_list_id=str(res.order_id))
                self.engine.db.insert_order(trade_id=tid, symbol=sym, order_id=None,
                                            client_order_id=res.client_order_id, order_list_id=str(res.order_id),
                                            side="SELL", order_type="OCO_TRAILING", status=res.status,
                                            payload={"safety_stage": stage, "stop": stop_px, "replaced": oco})
                self.engine.db.insert_event("SAFETY_STOP_REPLACED", symbol=sym, trade_id=tid,
                                            order_id=str(res.order_id),
                                            payload={"old_oco": oco, "new_stop": stop_px, "stage": stage})
                self.manager.confirm_stop(tid, stop_px, stage)
                return {"trade_id": tid, "result": "TIGHTENED", "stop": stop_px, "oco": res.order_id}
            self._notify("CRITICAL", "SAFETY_REPLACE_FAILED",
                         f"🚨 {sym}: tighter stop could not be placed — selling at market.")
            tp = self.manager.protection_for(tid)
            prev_stop = tp.stop if tp and tp.stop else base_stop_for_trade(fresh)
            return self._market_sell_locked(fresh, qty, "SAFETY_EXIT", previous_stop=prev_stop, info=info, price=price)

    # --------------------------------------------------------------- exits
    def exit_position(self, trade: dict[str, Any], *, reason: str, current_price: float | None = None) -> dict[str, Any]:
        tid = str(trade["trade_id"])
        sym = str(trade["symbol"]).upper()
        if not self._live:
            px = current_price if current_price else float(self.engine.exchange.get_price(sym))
            self.engine.lifecycle.close_from_exit_fill(
                trade_id=tid, reservation_id=None, exit_price=px,
                fees_btc=px * float(trade.get("quantity") or 0) * TAKER_FEE, close_reason=reason)
            return {"trade_id": tid, "result": "DRY_RUN_CLOSED", "price": px}
        with self._lock():
            fresh = self.engine.db.get_trade(tid) or trade
            if str(fresh.get("status") or "").upper() == "CLOSED":
                return {"trade_id": tid, "result": "ALREADY_CLOSED"}
            info = self.engine.exchange.get_symbol_info(sym)
            px = current_price if current_price else float(self.engine.exchange.get_price(sym))
            return self._exit_locked(fresh, reason, px, info)

    def _exit_locked(self, trade: dict[str, Any], reason: str, price: float, info: Any) -> dict[str, Any]:
        tid = str(trade["trade_id"])
        sym = str(trade["symbol"]).upper()
        oco = trade.get("binance_oco_list_id")
        tp = self.manager.protection_for(tid)
        prev_stop = tp.stop if tp and tp.stop else base_stop_for_trade(trade)
        if oco and not self._cancel_oco(sym, str(oco)):
            return {"trade_id": tid, "result": "CANCEL_FAILED_OLD_OCO_KEPT"}
        qty = self._sellable_qty(trade, info.quantity_step)
        if qty < float(info.min_quantity or 0) or qty <= 0:
            return {"trade_id": tid, "result": "FLAT_AFTER_CANCEL"}
        return self._market_sell_locked(trade, qty, reason, previous_stop=prev_stop if oco else None,
                                        info=info, price=price)

    def _market_sell_locked(
        self,
        trade: dict[str, Any],
        qty: float,
        reason: str,
        *,
        previous_stop: float | None,
        info: Any,
        price: float,
    ) -> dict[str, Any]:
        tid = str(trade["trade_id"])
        sym = str(trade["symbol"]).upper()
        res = None
        try:
            res = self.engine.exchange.place_order(OrderRequest(
                symbol=sym, side="SELL", order_type="MARKET", quantity=qty,
                client_order_id=f"x{int(time.time()) % 100000}_{tid[:16]}"))
        except Exception as e:  # noqa: BLE001
            logger.error("safety market sell failed %s: %s", sym, scrub_exception(e))
        executed = float(getattr(res, "executed_qty", 0) or 0) if res is not None else 0.0
        if res is not None and res.ok and executed > 0:
            quote = float(res.cumulative_quote_qty or 0)
            avg = quote / executed if quote > 0 else price
            self.engine.lifecycle.close_from_exit_fill(
                trade_id=tid, reservation_id=None, exit_price=avg, exit_qty=executed,
                fees_btc=avg * executed * TAKER_FEE, close_reason=reason,
                exit_order_id=str(res.order_id) if res.order_id else None)
            return {"trade_id": tid, "result": "CLOSED", "price": avg, "qty": executed}
        if previous_stop is not None and previous_stop < price:
            try:
                back = self._place_oco(trade, previous_stop, price, qty, info)
                if back.ok and back.order_id:
                    self.engine.db.update_trade(tid, binance_oco_list_id=str(back.order_id))
                    self._notify("ERROR", "SAFETY_CLOSE_FAILED",
                                 f"⚠️ {sym}: market close failed — previous protection re-placed.")
                    return {"trade_id": tid, "result": "CLOSE_FAILED_REPROTECTED", "oco": back.order_id}
            except Exception as e:  # noqa: BLE001
                logger.error("re-protect failed %s: %s", sym, scrub_exception(e))
        self.engine.db.update_trade(tid, status="PROTECTION_FAILED")
        self.manager.enter_halt(f"PROTECTION_FAILED:{sym}")
        self.engine.safety.halt("SAFETY_PROTECTION_FAILED", symbol=sym, trade_id=tid)
        self._notify("CRITICAL", "SAFETY_UNPROTECTED",
                     f"🚨 {sym} is UNPROTECTED — market exit and re-protection both failed. Manual action required.")
        return {"trade_id": tid, "result": "UNPROTECTED_HALTED"}

    def close(self, symbol: str | None) -> list[dict[str, Any]]:
        trades = self.engine.db.open_trades()
        if symbol:
            sym = symbol.upper()
            trades = [t for t in trades if str(t.get("symbol") or "").upper() in {sym, f"{sym}BTC"}]
        out = []
        for t in trades:
            try:
                out.append(self.exit_position(t, reason="MANUAL_CLOSE"))
            except Exception as e:  # noqa: BLE001
                out.append({"trade_id": t.get("trade_id"), "result": "ERROR", "error": scrub_exception(e)})
        return out

    # -------------------------------------------------------------- restart
    def restart_checks(self) -> dict[str, tuple[bool, str]]:
        eng = self.engine
        checks: dict[str, tuple[bool, str]] = {}

        def run(name: str, fn) -> Any:
            try:
                val = fn()
            except Exception as e:  # noqa: BLE001
                checks[name] = (False, scrub_exception(e)[:160])
                return None
            checks[name] = (True, "")
            return val

        run("api_connection", eng.exchange.ping)
        run("account_balance", eng.exchange.get_account)
        book = run("exchange_connection", lambda: eng.market.sync(eng.universe))
        if book is not None:
            fresh = not eng.market.is_stale()
            checks["market_data_fresh"] = (fresh, "" if fresh else f"age {eng.market.book.age_sec():.0f}s")
            missing = []
            for sym in eng.universe:
                try:
                    eng.market.relative_for(sym)
                except Exception:  # noqa: BLE001
                    missing.append(sym)
            checks["market_data_available"] = (not missing, ",".join(missing[:8]))
        else:
            checks["market_data_fresh"] = (False, "market sync failed")
            checks["market_data_available"] = (False, "market sync failed")

        rec = run("positions_reconciled", lambda: eng.lifecycle.reconcile_rest(eng.universe))
        if rec is not None:
            ok = bool(rec.get("ok", True))
            checks["positions_reconciled"] = (ok, "" if ok else str(rec.get("error") or rec.get("notes") or "")[:160])

        trades = eng.db.open_trades()
        pending = [f"{t.get('symbol')}:{t.get('status')}" for t in trades
                   if str(t.get("status") or "").upper() in PENDING_STATUSES]
        stale_orders = run("no_unresolved_orders", lambda: [
            o for o in eng.exchange.get_open_orders()
            if str(o.get("side") or "").upper() == "BUY"
        ])
        if stale_orders is not None:
            bad = pending + [f"open BUY {o.get('symbol')}" for o in stale_orders]
            checks["no_unresolved_orders"] = (not bad, ",".join(bad[:8]))

        failed_prot = [str(t.get("symbol")) for t in trades if str(t.get("status") or "").upper() == "PROTECTION_FAILED"]
        if self._live:
            lists = run("protective_stops_exist", eng.exchange.get_open_order_lists)
            if lists is not None:
                from binance_btc_bot.execution.order_lists import parse_order_list

                open_ids = {str(parse_order_list(r).order_list_id) for r in lists if parse_order_list(r).is_open}
                missing = [str(t.get("symbol")) for t in trades
                           if str(t.get("status") or "").upper() == "PROTECTED"
                           and str(t.get("binance_oco_list_id")) not in open_ids]
                checks["protective_stops_exist"] = (not missing and not failed_prot,
                                                    ",".join(missing + failed_prot))
                syms = {str(t.get("symbol")).upper() for t in trades}
                orphans = [str(parse_order_list(r).symbol) for r in lists
                           if parse_order_list(r).is_open and parse_order_list(r).symbol.upper() not in syms]
                checks["local_matches_exchange"] = (not orphans, ",".join(orphans[:8]))
        else:
            checks["protective_stops_exist"] = (not failed_prot, ",".join(failed_prot))
            checks["local_matches_exchange"] = (True, "dry-run")

        from binance_btc_bot.risk.safety import SafetyState

        halted = eng.safety.state == SafetyState.HALT
        checks["no_emergency_condition"] = (not halted,
                                            ",".join(eng.safety.reasons[-3:]) if halted else "")
        try:
            views, _ = self.open_positions()
            defensive_now = self.manager.would_trigger_defensive(views)
            checks["no_defensive_condition"] = (not defensive_now,
                                                "portfolio loss condition still true" if defensive_now else "")
        except Exception as e:  # noqa: BLE001
            checks["no_defensive_condition"] = (False, scrub_exception(e)[:160])
        return checks

    def restart(self) -> str:
        checks = self.restart_checks()
        ok, failed = self.manager.restart(checks)
        if not ok:
            msg = "🚨 RESTART BLOCKED\n\nFailed checks:\n" + "\n".join(f"- {f}" for f in failed) + \
                  "\n\nNew entries remain BLOCKED."
            self._notify("WARNING", "SAFETY_RESTART_BLOCKED", msg)
            return msg
        self.manager.mark_monitor_ok()
        ctrl = getattr(self.engine, "_runtime_controller", None)
        if ctrl is not None:
            ctrl.resume_after_safety_restart()
        open_n = len(self.engine.db.open_trades())
        cap = int(self.engine.portfolio.max_simultaneous_trades)
        msg = f"🟢 BOT RESTARTED\n\nSafety checks: PASSED\nOpen positions: {open_n}/{cap}\nNew entries: ENABLED"
        self._notify("INFO", "SAFETY_RESTARTED", msg)
        return msg

    # --------------------------------------------------------------- views
    def status_text(self, view: dict[str, Any], ctrl: Any | None) -> str:
        st = self.manager.status()
        cap = int(self.engine.portfolio.max_simultaneous_trades)
        positions = view.get("positions") or []
        mode = st["mode"]
        op_mode = getattr(getattr(ctrl, "state", None), "mode", None) if ctrl is not None else None
        if mode == "NORMAL" and op_mode and str(op_mode).upper() != "RUNNING":
            mode = str(op_mode).upper()
        reason = st["halt_reason"] or st["defensive_reason"] or st["blocking_reason"] or "-"
        entries_ok = self.engine.safety.allow_new_entries()
        eq = view.get("equity_btc")
        free = view.get("btc_free")
        unreal = view.get("unrealized_pnl_btc")
        invested = sum(float(p.get("entry_price") or 0) * float(p.get("quantity") or 0) for p in positions)
        unreal_pct = (100.0 * float(unreal) / invested) if isinstance(unreal, (int, float)) and invested > 0 else None
        lines = [
            "BOT STATUS",
            "",
            f"Mode: {mode}",
            f"Reason: {reason}",
            f"Safety halt: {'YES' if st['safety_halt'] else 'no'}",
            f"Defensive mode: {'YES' if st['defensive_mode'] else 'no'}",
            "",
            f"Open positions: {len(positions)}/{cap}",
            f"Available balance: {free:.8f} BTC" if isinstance(free, (int, float)) else "Available balance: n/a",
            f"Equity: {eq:.8f} BTC" if isinstance(eq, (int, float)) else "Equity: n/a",
            (f"Unrealized P&L: {unreal:+.8f} BTC ({unreal_pct:+.2f}%)" if unreal_pct is not None
             else f"Unrealized P&L: {unreal}"),
            f"Consecutive SLs: {st['consecutive_stop_losses']}/{st['sl_halt_count']}",
            "",
            f"New entries: {'ALLOWED' if entries_ok else 'BLOCKED'}",
        ]
        if positions:
            lines += ["", "Positions:"]
            for p in positions:
                pct = p.get("unrealized_pnl_pct")
                pct_s = f"{float(pct):+.2f}%" if isinstance(pct, (int, float)) else "n/a"
                lines.append(f"{p.get('symbol')} {pct_s}")
        if st["safety_halt"] or st["defensive_mode"]:
            lines += ["", "Use /restart to resume after safety checks."]
        return "\n".join(lines)

    def risk_text(self) -> str:
        st = self.manager.status()
        c = self.manager.config
        lines = [
            "RISK / SAFETY MANAGER",
            f"Enabled: {c.enabled}",
            f"Mode: {st['mode']}",
            f"Blocking: {st['blocking_reason'] or 'none'}",
            f"Consecutive SLs: {st['consecutive_stop_losses']}/{c.sl_halt_count}",
            f"Last SL: {st['last_sl_symbol'] or '-'}",
            f"Defensive trigger: {c.defensive_min_open} open AND >= {c.defensive_loss_count} at <= {c.defensive_loss_pct * 100:.1f}%",
            f"Defensive loss levels: {c.loss_tighten_pct * 100:.1f}% trail {c.loss_tighten_trail_pct * 100:.1f}% | "
            f"{c.loss_strong_pct * 100:.1f}% trail {c.loss_strong_trail_pct * 100:.1f}% | exit {c.loss_hard_exit_pct * 100:.1f}%",
            f"Defensive profit levels: +{c.profit_breakeven_pct * 100:.1f}% → BE+{c.breakeven_offset_pct * 100:.2f}% | "
            f"+{c.profit_lock_pct * 100:.1f}% → lock +{c.profit_lock_stop_pct * 100:.2f}% | "
            f"+{c.profit_trail_pct * 100:.1f}% → trail {c.profit_trail_distance_pct * 100:.1f}%",
            f"Monitor: {'OK' if st['monitor_ok'] else 'FAILED ' + str(st['monitor_error'])}",
        ]
        if st["protection"]:
            lines.append("Tracked stops:")
            for tid, p in st["protection"].items():
                stop = p.get("stop")
                lines.append(f"  {p.get('symbol')} stop={stop:.10g} stage={p.get('stage')}" if stop
                             else f"  {p.get('symbol')} stop=n/a stage={p.get('stage')}")
        return "\n".join(lines)
