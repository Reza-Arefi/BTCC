"""Safety hardening: emergency protection, OCO order-list reconcile, dust (A–N)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from binance_btc_bot.exchange.base import AccountSnapshot, Balance, OrderRequest, OrderResult, SymbolInfo
from binance_btc_bot.exchange.binance import BinanceAPIError, BinanceExchange
from binance_btc_bot.execution.fills import aggregate_fills_from_order
from binance_btc_bot.execution.lifecycle import OrderLifecycle
from binance_btc_bot.execution.order_lists import (
    find_matching_open_list,
    is_binance_param_error,
    parse_order_list,
    require_order_list_object,
    require_order_list_rows,
)
from binance_btc_bot.execution.protection_qty import (
    classify_residual_base,
    compute_protectable_quantity,
)
from binance_btc_bot.execution.trailing import TrailingExecutor
from binance_btc_bot.notifications.manager import NotificationManager
from binance_btc_bot.portfolio.manager import PortfolioConfig, PortfolioManager
from binance_btc_bot.portfolio.trade_state import TradeStatus
from binance_btc_bot.risk.safety import SafetyState, SafetySystem
from binance_btc_bot.storage.database import BotDatabase
from binance_btc_bot.strategy.trails import get_strategy
from binance_btc_bot.tests.test_order_lifecycle_layer2 import RecordingChannel


def _meta(
    symbol: str = "UNIBTC",
    *,
    step: float = 0.01,
    min_qty: float = 0.01,
    min_notional: float = 0.0001,
    base: str = "UNI",
) -> SymbolInfo:
    return SymbolInfo(
        symbol=symbol,
        status="TRADING",
        base_asset=base,
        quote_asset="BTC",
        quantity_step=step,
        min_quantity=min_qty,
        max_quantity=1_000_000.0,
        price_tick=1e-7,
        min_notional=min_notional,
        order_types=("LIMIT", "MARKET", "STOP_LOSS", "TAKE_PROFIT"),
        oco_allowed=True,
        min_trailing_above_delta=10,
        max_trailing_above_delta=2000,
        min_trailing_below_delta=10,
        max_trailing_below_delta=2000,
    )


def _fill_uni(*, commission_asset: str = "BNB", commission: float = 0.0001, qty: float = 10.97, price: float = 9.11e-5):
    return aggregate_fills_from_order(
        symbol="UNIBTC",
        side="BUY",
        order_id="1",
        client_order_id="c1",
        status="FILLED",
        executed_qty=qty,
        cumulative_quote_qty=qty * price,
        fills_raw=[
            {
                "price": str(price),
                "qty": str(qty),
                "commission": str(commission),
                "commissionAsset": commission_asset,
            }
        ],
    )


class TestEmergencyHardening(unittest.TestCase):
    def _life(self, exchange: MagicMock):
        tmp = tempfile.mkdtemp()
        db = BotDatabase(Path(tmp) / "bot.sqlite3")
        safety = SafetySystem()
        portfolio = PortfolioManager(
            PortfolioConfig(max_simultaneous_trades=8, allocation_per_trade=0.125, max_total_allocation=1.0)
        )
        notifications = NotificationManager(
            telegram=RecordingChannel("telegram"), sms=RecordingChannel("sms")
        )
        trailing = TrailingExecutor(exchange, db, notifications=notifications)
        life = OrderLifecycle(
            exchange,
            db,
            safety,
            portfolio=portfolio,
            trailing=trailing,
            notifications=notifications,
            dry_broker=None,
            live_enabled=True,
            dry_run=False,
        )
        return life, db, safety, portfolio, notifications

    def _ex(self, *, free: float = 10.97):
        ex = MagicMock()
        ex.get_symbol_info.side_effect = lambda s: _meta(str(s).upper(), step=0.00001, min_qty=0.00001)
        ex.get_account.return_value = AccountSnapshot(balances={"UNI": Balance("UNI", free, 0.0)})
        ex.get_open_order_lists.return_value = []
        ex.get_open_orders.return_value = []
        ex.get_order_list.side_effect = BinanceAPIError("not found")
        ex.place_protective_sell.return_value = OrderResult(ok=False, reason="NOT_CONFIGURED")
        return ex

    def _run_entry(self, life, fill):
        with patch.object(life, "_wait_for_buy_fill", return_value=fill), patch.object(
            life,
            "_submit_buy",
            return_value=OrderResult(ok=True, status="FILLED", order_id="buy1", dry_run=False),
        ):
            return life.run_entry(
                symbol="UNIBTC",
                strategy=get_strategy("T1"),
                price_alt_btc=9.11e-5,
                equity_btc=0.008,
                available_btc=0.008,
            )

    # A — initial OCO failure
    def test_a_initial_oco_failure(self):
        ex = self._ex()
        calls = {"n": 0}

        def place_trailing(req):
            calls["n"] += 1
            raise RuntimeError("API_FAILURE: first OCO reject")

        ex.place_trailing_exit.side_effect = place_trailing
        life, db, safety, pm, n = self._life(ex)
        r = self._run_entry(life, _fill_uni())
        self.assertFalse(r.ok)
        self.assertIn("OCO_SUBMIT_FAILED", r.events)
        self.assertGreaterEqual(calls["n"], 1)

    # B — corrected OCO retry
    def test_b_corrected_oco_retry(self):
        ex = self._ex()
        free1, free2 = 10.97, 10.95
        ex.get_account.side_effect = [
            AccountSnapshot(balances={"UNI": Balance("UNI", free1, 0.0)}),
            AccountSnapshot(balances={"UNI": Balance("UNI", free2, 0.0)}),
            AccountSnapshot(balances={"UNI": Balance("UNI", free2, 0.0)}),
        ]
        calls = {"n": 0}

        def place_trailing(req):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError(
                    "BinanceAPIError: {'code': -2010, 'msg': 'Account has insufficient balance'}"
                )
            return OrderResult(
                ok=True,
                order_id="oco_retry",
                client_order_id=req.list_client_order_id,
                status="EXECUTING",
                symbol=req.symbol,
                quantity=req.quantity,
            )

        ex.place_trailing_exit.side_effect = place_trailing
        ex.get_open_order_lists.return_value = [
            {"orderListId": "oco_retry", "symbol": "UNIBTC", "listOrderStatus": "EXECUTING", "listStatusType": "EXEC_STARTED"}
        ]
        life, db, safety, pm, n = self._life(ex)
        r = self._run_entry(life, _fill_uni())
        self.assertTrue(r.ok, r.events)
        self.assertIn("OCO_RETRY_OK", r.events)
        self.assertEqual(calls["n"], 2)
        self.assertNotEqual(r.status, TradeStatus.PROTECTION_FAILED.value)

    # C — second OCO failure → real emergency path
    def test_c_second_oco_failure_triggers_emergency(self):
        ex = self._ex(free=10.95)
        oco_calls = {"n": 0}

        def place_trailing(req):
            oco_calls["n"] += 1
            raise RuntimeError(
                "BinanceAPIError: {'code': -2010, 'msg': 'Account has insufficient balance'}"
            )

        ex.place_trailing_exit.side_effect = place_trailing
        # Different frees so retry is attempted, then emergency OCO also fails → STOP path.
        ex.get_account.side_effect = [
            AccountSnapshot(balances={"UNI": Balance("UNI", 10.97, 0.0)}),
            AccountSnapshot(balances={"UNI": Balance("UNI", 10.95, 0.0)}),
            AccountSnapshot(balances={"UNI": Balance("UNI", 10.95, 0.0)}),
            AccountSnapshot(balances={"UNI": Balance("UNI", 10.95, 0.0)}),
            AccountSnapshot(balances={"UNI": Balance("UNI", 10.95, 0.0)}),
        ]
        stop_order = OrderResult(
            ok=True,
            order_id="stop1",
            client_order_id="es_x",
            status="NEW",
            symbol="UNIBTC",
            side="SELL",
            order_type="STOP_LOSS",
            quantity=10.95,
            dry_run=False,
        )
        ex.place_protective_sell.return_value = stop_order

        def get_order(*_a, **_k):
            if ex.place_protective_sell.called:
                return stop_order
            return OrderResult(ok=False, status="UNKNOWN")

        ex.get_order.side_effect = get_order
        # Enough account snapshots for protect + retry + emergency refresh + dust.
        acct_hi = AccountSnapshot(balances={"UNI": Balance("UNI", 10.97, 0.0)})
        acct_lo = AccountSnapshot(balances={"UNI": Balance("UNI", 10.95, 0.0)})
        ex.get_account.side_effect = [acct_hi, acct_lo] + [acct_lo] * 20
        life, db, safety, pm, n = self._life(ex)
        r = self._run_entry(life, _fill_uni())
        self.assertIn("EMERGENCY_OK", r.events)
        self.assertEqual(r.status, "PROTECTED_EMERGENCY")
        self.assertGreaterEqual(oco_calls["n"], 2)  # initial + retry (+ emergency OCO)
        self.assertTrue(ex.place_protective_sell.called)
        req = ex.place_protective_sell.call_args.args[0]
        self.assertIsInstance(req, OrderRequest)
        self.assertEqual(req.order_type, "STOP_LOSS")
        self.assertEqual(req.side, "SELL")
        self.assertNotEqual(req.order_type, "MARKET")

    # D — emergency protection success
    def test_d_emergency_protection_success(self):
        ex = self._ex(free=10.97)
        ex.place_trailing_exit.side_effect = RuntimeError("OCO down")
        stop_order = OrderResult(
            ok=True,
            order_id="stop_ok",
            client_order_id="es_y",
            status="NEW",
            symbol="UNIBTC",
            side="SELL",
            order_type="STOP_LOSS",
            dry_run=False,
        )
        ex.place_protective_sell.return_value = stop_order

        def get_order(*_a, **_k):
            if ex.place_protective_sell.called:
                return stop_order
            return OrderResult(ok=False, status="UNKNOWN")

        ex.get_order.side_effect = get_order
        life, db, safety, pm, n = self._life(ex)
        r = self._run_entry(life, _fill_uni())
        self.assertEqual(r.status, "PROTECTED_EMERGENCY")
        self.assertIn("EMERGENCY_OK", r.events)
        self.assertEqual(safety.state, SafetyState.HALT)
        self.assertNotEqual(r.status, TradeStatus.PROTECTED.value)

    # E — emergency protection failure → HALT
    def test_e_emergency_failure_halt(self):
        ex = self._ex(free=10.97)
        ex.place_trailing_exit.side_effect = RuntimeError("OCO fail")
        ex.place_protective_sell.return_value = OrderResult(ok=False, reason="STOP_REJECTED", dry_run=False)
        life, db, safety, pm, n = self._life(ex)
        r = self._run_entry(life, _fill_uni())
        self.assertEqual(r.status, TradeStatus.PROTECTION_FAILED.value)
        self.assertIn("EMERGENCY_FAILED", r.events)
        self.assertEqual(safety.state, SafetyState.HALT)
        self.assertFalse(safety.allow_new_entries())
        self.assertTrue(any(x["event"] == "UNPROTECTED_POSITION" for x in n.telegram.sent))

    # F — reconciliation after emergency protection
    def test_f_reconcile_after_emergency(self):
        ex = self._ex(free=10.97)
        ex.place_trailing_exit.side_effect = RuntimeError("OCO fail")
        stop_order = OrderResult(
            ok=True, order_id="stop_f", status="NEW", symbol="UNIBTC", dry_run=False
        )
        ex.place_protective_sell.return_value = stop_order

        def get_order(*_a, **_k):
            if ex.place_protective_sell.called:
                return stop_order
            return OrderResult(ok=False, status="UNKNOWN")

        ex.get_order.side_effect = get_order
        ex.get_open_order_lists.return_value = []
        ex.get_open_orders.return_value = [
            {"orderId": "stop_f", "symbol": "UNIBTC", "side": "SELL", "type": "STOP_LOSS"}
        ]
        life, db, safety, pm, n = self._life(ex)
        r = self._run_entry(life, _fill_uni())
        self.assertIn("RECONCILE_AFTER_EMERGENCY", r.events)
        self.assertEqual(r.status, "PROTECTED_EMERGENCY")

    # G — OCO order-list lookup
    def test_g_oco_order_list_lookup(self):
        rows = [
            {
                "orderListId": 31,
                "symbol": "ETHBTC",
                "contingencyType": "OCO",
                "listStatusType": "EXEC_STARTED",
                "listOrderStatus": "EXECUTING",
                "listClientOrderId": "abc",
                "orders": [
                    {"symbol": "ETHBTC", "orderId": 1, "clientOrderId": "a"},
                    {"symbol": "ETHBTC", "orderId": 2, "clientOrderId": "b"},
                ],
            },
            {
                "orderListId": 32,
                "symbol": "UNIBTC",
                "listStatusType": "EXEC_STARTED",
                "listOrderStatus": "EXECUTING",
                "listClientOrderId": "uni1",
            },
        ]
        hit = find_matching_open_list(rows, symbol="UNIBTC", order_list_id="32")
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.symbol, "UNIBTC")
        self.assertTrue(hit.is_open)
        self.assertEqual(hit.list_client_order_id, "uni1")
        view = parse_order_list(rows[0])
        self.assertEqual(len(view.child_orders), 2)

    # H — malformed / ambiguous order-list response
    def test_h_malformed_order_list_response(self):
        with self.assertRaises(ValueError):
            require_order_list_rows({"orderListId": 1}, context="openOrderList")
        with self.assertRaises(ValueError):
            require_order_list_object([1, 2], context="orderList")
        with self.assertRaises(ValueError):
            require_order_list_object({"foo": 1}, context="orderList")

        ex = BinanceExchange(dry_run=True, live_enabled=False)
        with patch.object(ex, "_request", return_value={"not": "a list"}):
            with self.assertRaises(BinanceAPIError):
                ex.get_open_order_lists()

    # I — Binance parameter errors
    def test_i_binance_parameter_errors(self):
        self.assertTrue(is_binance_param_error("Binance HTTP 400: {'code': -1102}"))
        self.assertTrue(is_binance_param_error("too many parameters"))
        self.assertFalse(is_binance_param_error("insufficient balance -2010"))

        # Adapter must never send symbol to openOrderList.
        ex = BinanceExchange(dry_run=True, live_enabled=False)
        seen: list[dict] = []

        def fake_request(method, path, *, params=None, signed=False, use_private=None):
            seen.append({"path": path, "params": dict(params or {})})
            return [
                {
                    "orderListId": 1,
                    "symbol": "UNIBTC",
                    "listOrderStatus": "EXECUTING",
                    "listStatusType": "EXEC_STARTED",
                },
                {
                    "orderListId": 2,
                    "symbol": "ETHBTC",
                    "listOrderStatus": "EXECUTING",
                    "listStatusType": "EXEC_STARTED",
                },
            ]

        with patch.object(ex, "_request", side_effect=fake_request):
            filtered = ex.get_open_order_lists("UNIBTC")
        self.assertEqual(seen[0]["path"], "/api/v3/openOrderList")
        self.assertNotIn("symbol", seen[0]["params"])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["symbol"], "UNIBTC")

    # J — residual dust after floor
    def test_j_residual_dust(self):
        fill = _fill_uni(commission_asset="UNI", commission=0.01097, qty=10.97)
        meta = _meta(step=0.01, min_qty=0.01)
        free = 10.95903
        pq = compute_protectable_quantity(
            fill=fill, base_asset="UNI", free_base=free, meta=meta, ref_price=9.11e-5
        )
        self.assertTrue(pq.ok)
        dust = classify_residual_base(
            fill=fill,
            base_asset="UNI",
            free_base=free,
            meta=meta,
            ref_price=9.11e-5,
            protected_qty=pq.quantity,
        )
        self.assertGreater(dust.residual_unprotected, 0)
        self.assertEqual(dust.category, "LOT_SIZE_DUST")
        self.assertTrue(dust.counts_as_trading_position)  # main position protected

    # K — dust below minQty
    def test_k_dust_below_min_qty(self):
        fill = _fill_uni(qty=0.005, commission_asset="BNB", commission=0.0)
        meta = _meta(step=0.01, min_qty=0.01)
        dust = classify_residual_base(
            fill=fill, base_asset="UNI", free_base=0.005, meta=meta, ref_price=9.11e-5
        )
        self.assertEqual(dust.category, "DUST_BELOW_MIN_QTY")
        self.assertFalse(dust.counts_as_trading_position)
        self.assertAlmostEqual(dust.residual_unprotected, 0.005)

    # L — no false PROTECTED state
    def test_l_no_false_protected_state(self):
        ex = self._ex(free=10.97)
        ex.place_trailing_exit.side_effect = RuntimeError("fail")
        ex.place_protective_sell.return_value = OrderResult(ok=False, reason="nope")
        life, db, safety, pm, n = self._life(ex)
        r = self._run_entry(life, _fill_uni())
        self.assertNotEqual(r.status, TradeStatus.PROTECTED.value)
        self.assertNotEqual(r.status, "DRY_RUN_PROTECTED")
        self.assertEqual(r.status, TradeStatus.PROTECTION_FAILED.value)
        tr = db.get_trade(r.trade_id)
        self.assertNotEqual(tr["status"], "PROTECTED")

    # M — no duplicate protection
    def test_m_no_duplicate_protection(self):
        ex = self._ex(free=10.97)
        # Pretend emergency OCO already open.
        ex.get_open_order_lists.return_value = [
            {
                "orderListId": "already",
                "symbol": "UNIBTC",
                "listClientOrderId": "em_tradeidxxxxxxxxxxxx"[:21],  # will not match exact — set after
                "listOrderStatus": "EXECUTING",
                "listStatusType": "EXEC_STARTED",
            }
        ]
        life, db, safety, pm, n = self._life(ex)
        fill = _fill_uni()
        # Direct emergency with matching client id after we know trade id format.
        trade_id = "abcdefghijklmnopqrstuv"
        em_cid = life._emergency_client_oco_id(trade_id)
        ex.get_open_order_lists.return_value = [
            {
                "orderListId": "already",
                "symbol": "UNIBTC",
                "listClientOrderId": em_cid,
                "listOrderStatus": "EXECUTING",
                "listStatusType": "EXEC_STARTED",
            }
        ]
        ok = life._emergency_protect(
            symbol="UNIBTC", trade_id=trade_id, qty=10.97, entry=9.11e-5, strategy=get_strategy("T1"), fill=fill
        )
        self.assertTrue(ok)
        ex.place_trailing_exit.assert_not_called()
        ex.place_protective_sell.assert_not_called()

    # N — no duplicate emergency order
    def test_n_no_duplicate_emergency_order(self):
        ex = self._ex(free=10.97)
        stop_order = OrderResult(
            ok=True, order_id="stop_n", status="NEW", symbol="UNIBTC", dry_run=False
        )
        ex.place_trailing_exit.side_effect = RuntimeError("oco fail")
        ex.place_protective_sell.return_value = stop_order

        def get_order(*_a, **_k):
            if ex.place_protective_sell.called:
                return stop_order
            return OrderResult(ok=False, status="UNKNOWN")

        ex.get_order.side_effect = get_order
        life, db, safety, pm, n = self._life(ex)
        fill = _fill_uni()
        trade_id = "dupemtrade00000000001"
        ok1 = life._emergency_protect(
            symbol="UNIBTC",
            trade_id=trade_id,
            qty=10.97,
            entry=9.11e-5,
            strategy=get_strategy("T1"),
            fill=fill,
        )
        self.assertTrue(ok1)
        place_count = ex.place_protective_sell.call_count
        # Second call must not place again.
        ex.get_open_order_lists.return_value = []
        ex.get_open_orders.return_value = [
            {
                "orderId": "stop_n",
                "clientOrderId": life._emergency_client_stop_id(trade_id),
                "side": "SELL",
                "symbol": "UNIBTC",
            }
        ]
        ok2 = life._emergency_protect(
            symbol="UNIBTC",
            trade_id=trade_id,
            qty=10.97,
            entry=9.11e-5,
            strategy=get_strategy("T1"),
            fill=fill,
        )
        self.assertTrue(ok2)
        self.assertEqual(ex.place_protective_sell.call_count, place_count)


class TestProtectionQtyRetryStillHalted(unittest.TestCase):
    """Regression: both OCO attempts + emergency fail → PROTECTION_FAILED (not PROTECTED)."""

    def test_retry_and_emergency_fail(self):
        tmp = tempfile.mkdtemp()
        db = BotDatabase(Path(tmp) / "bot.sqlite3")
        safety = SafetySystem()
        portfolio = PortfolioManager(
            PortfolioConfig(max_simultaneous_trades=8, allocation_per_trade=0.125, max_total_allocation=1.0)
        )
        notifications = NotificationManager(
            telegram=RecordingChannel("telegram"), sms=RecordingChannel("sms")
        )
        ex = MagicMock()
        ex.get_symbol_info.side_effect = lambda s: _meta(str(s).upper(), step=0.00001, min_qty=0.00001)
        ex.get_account.return_value = AccountSnapshot(
            balances={"UNI": Balance("UNI", 10.95903, 0.0)}
        )
        ex.place_trailing_exit.side_effect = RuntimeError(
            "BinanceAPIError: {'code': -2010, 'msg': 'Account has insufficient balance'}"
        )
        ex.place_protective_sell.return_value = OrderResult(ok=False, reason="STOP_FAIL")
        ex.get_open_order_lists.return_value = []
        ex.get_open_orders.return_value = []
        ex.get_account.side_effect = [
            AccountSnapshot(balances={"UNI": Balance("UNI", 10.97, 0.0)}),
            AccountSnapshot(balances={"UNI": Balance("UNI", 10.95903, 0.0)}),
            AccountSnapshot(balances={"UNI": Balance("UNI", 10.95903, 0.0)}),
            AccountSnapshot(balances={"UNI": Balance("UNI", 10.95903, 0.0)}),
            AccountSnapshot(balances={"UNI": Balance("UNI", 10.95903, 0.0)}),
        ]
        trailing = TrailingExecutor(ex, db, notifications=notifications)
        life = OrderLifecycle(
            ex,
            db,
            safety,
            portfolio=portfolio,
            trailing=trailing,
            notifications=notifications,
            dry_broker=None,
            live_enabled=True,
            dry_run=False,
        )
        fill = _fill_uni(commission_asset="UNI", commission=0.01097)
        with patch.object(life, "_wait_for_buy_fill", return_value=fill), patch.object(
            life,
            "_submit_buy",
            return_value=OrderResult(ok=True, status="FILLED", order_id="b", dry_run=False),
        ):
            r = life.run_entry(
                symbol="UNIBTC",
                strategy=get_strategy("T1"),
                price_alt_btc=9.11e-5,
                equity_btc=0.008,
                available_btc=0.008,
            )
        self.assertFalse(r.ok)
        self.assertEqual(r.status, TradeStatus.PROTECTION_FAILED.value)
        self.assertFalse(safety.allow_new_entries())
        self.assertNotEqual(r.status, TradeStatus.PROTECTED.value)


class TestEmergencyVsEntrySemantics(unittest.TestCase):
    def test_protective_sell_refuses_buy_and_market(self):
        from binance_btc_bot.exchange.binance import BinanceExchange
        from binance_btc_bot.exchange.base import OrderRequest

        ex = BinanceExchange(dry_run=True, live_enabled=False)
        buy = ex.place_protective_sell(
            OrderRequest(symbol="ETHBTC", side="BUY", order_type="STOP_LOSS", quantity=1, stop_price=0.01)
        )
        self.assertFalse(buy.ok)
        self.assertIn("SIDE_SELL", buy.reason or "")
        mkt = ex.place_protective_sell(
            OrderRequest(symbol="ETHBTC", side="SELL", order_type="MARKET", quantity=1)
        )
        self.assertFalse(mkt.ok)
        self.assertIn("MARKET", mkt.reason or "")

    def test_emergency_sell_does_not_call_place_entry_or_reserve_slot(self):
        tmp = tempfile.mkdtemp()
        db = BotDatabase(Path(tmp) / "bot.sqlite3")
        safety = SafetySystem()
        portfolio = PortfolioManager(
            PortfolioConfig(max_simultaneous_trades=8, allocation_per_trade=0.125, max_total_allocation=1.0)
        )
        slots_before = portfolio.slots_used()
        ex = MagicMock()
        ex.get_symbol_info.side_effect = lambda s: _meta(str(s).upper(), step=0.00001, min_qty=0.00001)
        ex.get_account.return_value = AccountSnapshot(balances={"UNI": Balance("UNI", 10.97, 0.0)})
        ex.get_open_order_lists.return_value = []
        ex.get_open_orders.return_value = []
        ex.place_trailing_exit.side_effect = RuntimeError("oco fail")
        stop = OrderResult(ok=True, order_id="s1", status="NEW", symbol="UNIBTC", dry_run=False)
        ex.place_protective_sell.return_value = stop

        def get_order(*_a, **_k):
            if ex.place_protective_sell.called:
                return stop
            return OrderResult(ok=False, status="UNKNOWN")

        ex.get_order.side_effect = get_order
        life = OrderLifecycle(
            ex, db, safety, portfolio=portfolio, trailing=TrailingExecutor(ex, db),
            dry_broker=None, live_enabled=True, dry_run=False,
        )
        fill = _fill_uni()
        ok = life._emergency_protect(
            symbol="UNIBTC",
            trade_id="entrybypass000000001",
            qty=10.97,
            entry=9.11e-5,
            strategy=get_strategy("T1"),
            fill=fill,
        )
        self.assertTrue(ok)
        ex.place_entry.assert_not_called()
        self.assertTrue(ex.place_protective_sell.called)
        req = ex.place_protective_sell.call_args.args[0]
        self.assertEqual(req.side, "SELL")
        self.assertEqual(req.order_type, "STOP_LOSS")
        self.assertTrue(str(req.client_order_id).startswith("es_"))
        self.assertFalse(str(req.client_order_id).startswith("e_"))  # not entry id
        self.assertEqual(portfolio.slots_used(), slots_before)

    def test_request_construction_oco_and_stop(self):
        from binance_btc_bot.execution.protection_requests import (
            build_emergency_stop_request,
            build_t1_oco_request_params,
        )

        meta = _meta(step=0.00001, min_qty=0.00001)
        strat = get_strategy("T1")
        oco = build_t1_oco_request_params(
            strategy=strat,
            symbol="UNIBTC",
            entry_price=9.11e-5,
            quantity=10.95,
            symbol_info=meta,
            list_client_order_id="em_test",
        )
        self.assertTrue(oco["ok"])
        self.assertEqual(oco["endpoint"], "POST /api/v3/orderList/oco")
        self.assertEqual(oco["params"]["side"], "SELL")
        self.assertEqual(oco["params"]["aboveType"], "TAKE_PROFIT")
        self.assertEqual(oco["params"]["belowType"], "STOP_LOSS")
        self.assertEqual(oco["params"]["aboveTrailingDelta"], 25)
        stop = build_emergency_stop_request(
            symbol="UNIBTC",
            quantity=10.95,
            entry_price=9.11e-5,
            strategy=strat,
            symbol_info=meta,
            client_order_id="es_test",
        )
        self.assertTrue(stop["ok"])
        self.assertEqual(stop["endpoint"], "POST /api/v3/order")
        self.assertEqual(stop["params"]["side"], "SELL")
        self.assertEqual(stop["params"]["type"], "STOP_LOSS")
        self.assertEqual(stop["adapter"], "place_protective_sell")

    def test_reconcile_cannot_falsely_mark_protected(self):
        tmp = tempfile.mkdtemp()
        db = BotDatabase(Path(tmp) / "bot.sqlite3")
        safety = SafetySystem()
        ex = MagicMock()
        ex.get_open_orders.return_value = []
        ex.get_open_order_lists.return_value = []
        life = OrderLifecycle(
            ex, db, safety, dry_broker=None, live_enabled=True, dry_run=False
        )
        db.insert_trade(
            {
                "trade_id": "unprot1",
                "symbol": "UNIBTC",
                "strategy": "T1",
                "entry_price": 9.11e-5,
                "quantity": 10.0,
                "status": "PROTECTION_FAILED",
            }
        )
        # Also a falsely "PROTECTED" local without exchange protection
        db.insert_trade(
            {
                "trade_id": "fakeprot",
                "symbol": "ETHBTC",
                "strategy": "T1",
                "entry_price": 0.05,
                "quantity": 1.0,
                "status": "PROTECTED",
                "binance_oco_list_id": "missing",
            }
        )
        out = life.reconcile_rest()
        self.assertFalse(out["ok"])
        self.assertTrue(out.get("fail_closed"))
        # Status must not have been upgraded
        self.assertEqual(db.get_trade("unprot1")["status"], "PROTECTION_FAILED")
        self.assertEqual(db.get_trade("fakeprot")["status"], "PROTECTED")  # unchanged, not "fixed" upward
        notes = " ".join(out["notes"])
        self.assertIn("OPEN_UNPROTECTED", notes)
        self.assertIn("FAIL_CLOSED", notes)
        self.assertNotIn("status_mutated\": true", notes.lower())


if __name__ == "__main__":
    unittest.main()
