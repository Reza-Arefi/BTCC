"""Regression: fee-aware / balance-aware OCO protection quantity (UNIBTC bug)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from binance_btc_bot.config_loader import FROZEN_STRATEGIES, load_config
from binance_btc_bot.exchange.base import AccountSnapshot, Balance, OrderResult, SymbolInfo
from binance_btc_bot.execution.dry_broker import DryRunBroker
from binance_btc_bot.execution.fills import aggregate_fills_from_order
from binance_btc_bot.execution.lifecycle import OrderLifecycle
from binance_btc_bot.execution.protection_qty import (
    compute_protectable_quantity,
    is_insufficient_balance_reason,
    sellable_qty_from_fills,
)
from binance_btc_bot.execution.trailing import TrailingExecutor, TrailingSubmitResult
from binance_btc_bot.notifications.manager import NotificationManager
from binance_btc_bot.portfolio.manager import PortfolioConfig, PortfolioManager
from binance_btc_bot.portfolio.trade_state import TradeStatus
from binance_btc_bot.risk.safety import SafetySystem
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


def _fill_uni(*, commission_asset: str, commission: float, qty: float = 10.97, price: float = 9.11e-5):
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


class TestProtectableQuantity(unittest.TestCase):
    # A
    def test_a_bnb_commission_full_base_protectable(self):
        fill = _fill_uni(commission_asset="BNB", commission=0.0001)
        self.assertAlmostEqual(sellable_qty_from_fills(fill, "UNI"), 10.97)
        meta = _meta()
        pq = compute_protectable_quantity(
            fill=fill, base_asset="UNI", free_base=10.97, meta=meta, ref_price=9.11e-5
        )
        self.assertTrue(pq.ok)
        self.assertAlmostEqual(pq.quantity, 10.97)
        self.assertIn("BNB", pq.commissions_by_asset)

    # B
    def test_b_base_asset_commission_reduces(self):
        fill = _fill_uni(commission_asset="UNI", commission=0.01097)
        self.assertAlmostEqual(sellable_qty_from_fills(fill, "UNI"), 10.97 - 0.01097)
        meta = _meta(step=0.00001, min_qty=0.00001)
        pq = compute_protectable_quantity(
            fill=fill,
            base_asset="UNI",
            free_base=10.95903,
            meta=meta,
            ref_price=9.11e-5,
        )
        self.assertTrue(pq.ok)
        self.assertLess(pq.quantity, 10.97)
        self.assertLessEqual(pq.quantity, 10.95903 + 1e-12)

    # C
    def test_c_quote_asset_commission(self):
        fill = _fill_uni(commission_asset="BTC", commission=0.000001)
        self.assertAlmostEqual(sellable_qty_from_fills(fill, "UNI"), 10.97)

    # D
    def test_d_multiple_fills(self):
        fill = aggregate_fills_from_order(
            symbol="UNIBTC",
            side="BUY",
            order_id="1",
            client_order_id="c1",
            status="FILLED",
            executed_qty=None,
            cumulative_quote_qty=None,
            fills_raw=[
                {"price": "0.000091", "qty": "5", "commission": "0.005", "commissionAsset": "UNI"},
                {"price": "0.000092", "qty": "5.97", "commission": "0.001", "commissionAsset": "BNB"},
            ],
        )
        # Only UNI commission reduces base: 10.97 - 0.005
        self.assertAlmostEqual(sellable_qty_from_fills(fill, "UNI"), 10.965)

    # E
    def test_e_partial_fills_status(self):
        fill = aggregate_fills_from_order(
            symbol="UNIBTC",
            side="BUY",
            order_id="1",
            client_order_id="c1",
            status="PARTIALLY_FILLED",
            executed_qty=5.0,
            cumulative_quote_qty=0.00045,
            fills_raw=[
                {"price": "0.00009", "qty": "5", "commission": "0.01", "commissionAsset": "UNI"},
            ],
        )
        self.assertTrue(fill.partially_filled)
        self.assertAlmostEqual(sellable_qty_from_fills(fill, "UNI"), 4.99)

    # F
    def test_f_balance_lower_than_calculated(self):
        fill = _fill_uni(commission_asset="BNB", commission=0.0001)
        meta = _meta(step=0.01)
        pq = compute_protectable_quantity(
            fill=fill, base_asset="UNI", free_base=5.0, meta=meta, ref_price=9.11e-5
        )
        self.assertTrue(pq.ok)
        self.assertAlmostEqual(pq.quantity, 5.0)

    # G
    def test_g_balance_higher_than_calculated(self):
        fill = _fill_uni(commission_asset="UNI", commission=0.01097)
        meta = _meta(step=0.00001, min_qty=0.00001)
        pq = compute_protectable_quantity(
            fill=fill, base_asset="UNI", free_base=100.0, meta=meta, ref_price=9.11e-5
        )
        self.assertTrue(pq.ok)
        self.assertAlmostEqual(pq.quantity, 10.95903)

    # H
    def test_h_lot_size_flooring(self):
        fill = _fill_uni(commission_asset="UNI", commission=0.01097)
        meta = _meta(step=0.1, min_qty=0.1)
        pq = compute_protectable_quantity(
            fill=fill, base_asset="UNI", free_base=10.95903, meta=meta, ref_price=9.11e-5
        )
        self.assertTrue(pq.ok)
        self.assertAlmostEqual(pq.quantity, 10.9)  # floor, never up

    # I
    def test_i_min_qty_failure(self):
        fill = _fill_uni(commission_asset="UNI", commission=10.96, qty=10.97)
        meta = _meta(step=0.01, min_qty=1.0)
        pq = compute_protectable_quantity(
            fill=fill, base_asset="UNI", free_base=0.01, meta=meta, ref_price=9.11e-5
        )
        self.assertFalse(pq.ok)
        self.assertIn(pq.reason, {"MIN_QTY", "PROTECT_QTY_ZERO"})

    # J
    def test_j_min_notional_failure(self):
        fill = _fill_uni(commission_asset="BNB", commission=0.0, qty=0.02, price=0.00001)
        meta = _meta(step=0.01, min_qty=0.01, min_notional=1.0)
        pq = compute_protectable_quantity(
            fill=fill, base_asset="UNI", free_base=0.02, meta=meta, ref_price=0.00001
        )
        self.assertFalse(pq.ok)
        self.assertEqual(pq.reason, "MIN_NOTIONAL")

    # O
    def test_o_commission_asset_never_assumed(self):
        fill = aggregate_fills_from_order(
            symbol="UNIBTC",
            side="BUY",
            order_id="1",
            client_order_id="c1",
            status="FILLED",
            executed_qty=10.0,
            cumulative_quote_qty=0.001,
            fills_raw=[{"price": "0.0001", "qty": "10", "commission": "0.1"}],  # no asset
        )
        # Missing asset → do not subtract from base
        self.assertAlmostEqual(sellable_qty_from_fills(fill, "UNI"), 10.0)
        self.assertEqual(fill.fills[0].commission_asset, "")

    # P
    def test_p_bnb_not_manually_deducted(self):
        fill = _fill_uni(commission_asset="BNB", commission=0.5)
        # Bot must not invent a UNI reduction from BNB fee amount
        self.assertAlmostEqual(sellable_qty_from_fills(fill, "UNI"), 10.97)


class TestOcoRetryLifecycle(unittest.TestCase):
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
        return life, db, safety, portfolio

    def _ex_base(self):
        ex = MagicMock()
        ex.get_symbol_info.side_effect = lambda s: _meta(str(s).upper(), step=0.00001, min_qty=0.00001)
        return ex

    # K
    def test_k_insufficient_then_retry_succeeds(self):
        ex = self._ex_base()
        fill_qty = 10.97
        # Stale free overstates inventory; retry sees true free < sellable.
        free1 = 10.97
        free2 = 10.95
        acct1 = AccountSnapshot(balances={"UNI": Balance("UNI", free1, 0.0)})
        acct2 = AccountSnapshot(balances={"UNI": Balance("UNI", free2, 0.0)})
        ex.get_account.side_effect = [acct1, acct2, acct2, acct2]
        calls = {"n": 0}

        def place_trailing(req):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError(
                    "BinanceAPIError: Binance HTTP 400: {'code': -2010, 'msg': 'Account has insufficient balance for requested action.'}"
                )
            return OrderResult(
                ok=True,
                order_id="oco999",
                client_order_id=req.list_client_order_id,
                status="EXECUTING",
                symbol=req.symbol,
                side="SELL",
                order_type="OCO_TRAILING",
                quantity=req.quantity,
                dry_run=False,
            )

        ex.place_trailing_exit.side_effect = place_trailing
        ex.get_open_order_lists.return_value = [
            {
                "orderListId": "oco999",
                "symbol": "UNIBTC",
                "listOrderStatus": "EXECUTING",
                "listStatusType": "EXEC_STARTED",
            }
        ]
        # BNB fee → fill-derived sellable stays full 10.97; free balance is the binding constraint.
        fill = _fill_uni(commission_asset="BNB", commission=0.0001, qty=fill_qty)
        life, db, safety, pm = self._life(ex)
        with patch.object(life, "_wait_for_buy_fill", return_value=fill), patch.object(
            life,
            "_submit_buy",
            return_value=OrderResult(ok=True, status="FILLED", order_id="buy1", dry_run=False),
        ):
            r = life.run_entry(
                symbol="UNIBTC",
                strategy=get_strategy("T1"),
                price_alt_btc=9.11e-5,
                equity_btc=0.008,
                available_btc=0.008,
            )
        self.assertTrue(r.ok, f"{r.reason} events={r.events}")
        self.assertIn("OCO_RETRY_OK", r.events)
        self.assertIn("PROTECTED", r.events)
        self.assertEqual(calls["n"], 2)
        second_req = ex.place_trailing_exit.call_args_list[1].args[0]
        self.assertLess(second_req.quantity, fill_qty)
        self.assertLessEqual(second_req.quantity, free2 + 1e-12)

    # L
    def test_l_retry_fails_protection_failed_halt(self):
        ex = self._ex_base()
        fill_qty = 10.97
        ex.get_account.return_value = AccountSnapshot(
            balances={"UNI": Balance("UNI", 10.95903, 0.0)}
        )

        def place_trailing(req):
            raise RuntimeError(
                "BinanceAPIError: Binance HTTP 400: {'code': -2010, 'msg': 'Account has insufficient balance for requested action.'}"
            )

        ex.place_trailing_exit.side_effect = place_trailing
        ex.get_open_order_lists.return_value = []
        ex.place_protective_sell.return_value = OrderResult(ok=False, reason="STOP_FAIL", dry_run=False)
        life, db, safety, pm = self._life(ex)
        fill = _fill_uni(commission_asset="UNI", commission=0.01097)
        with patch.object(life, "_wait_for_buy_fill", return_value=fill), patch.object(
            life,
            "_submit_buy",
            return_value=OrderResult(ok=True, status="FILLED", order_id="b", dry_run=False),
        ):
            # Force first qty attempt then retry path by making free identical so retry skipped
            # → still PROTECTION_FAILED. Use two different frees that both fail place.
            ex.get_account.side_effect = [
                AccountSnapshot(balances={"UNI": Balance("UNI", 10.97, 0.0)}),
                AccountSnapshot(balances={"UNI": Balance("UNI", 10.95903, 0.0)}),
                AccountSnapshot(balances={"UNI": Balance("UNI", 10.95903, 0.0)}),
            ]
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

    # M / N
    def test_m_n_no_duplicate_oco_and_qty_le_free(self):
        fill = _fill_uni(commission_asset="UNI", commission=0.01097)
        meta = _meta(step=0.00001, min_qty=0.00001)
        free = 10.95903
        pq = compute_protectable_quantity(
            fill=fill, base_asset="UNI", free_base=free, meta=meta, ref_price=9.11e-5
        )
        self.assertTrue(pq.ok)
        self.assertLessEqual(pq.quantity, free + 1e-12)
        self.assertTrue(is_insufficient_balance_reason("API_FAILURE:... code': -2010 ..."))

    # Q
    def test_q_strategy_risk_config_invariants(self):
        cfg = load_config()
        self.assertFalse(cfg["live"]["enabled"])
        self.assertTrue(cfg["live"]["dry_run"])
        self.assertEqual(cfg["live"]["strategy"], "T1")
        self.assertIn(cfg["live"]["selector"], (None, "null", ""))
        self.assertAlmostEqual(float(cfg["risk"]["max_loss_per_trade"]), 0.005)
        self.assertAlmostEqual(float(cfg["portfolio"]["allocation_per_trade"]), 0.125)
        self.assertEqual(int(cfg["portfolio"]["max_simultaneous_trades"]), 8)
        self.assertEqual(len((cfg.get("universe") or {}).get("btc_pairs") or []), 38)
        # T1–T10 frozen tuple unchanged
        self.assertEqual(FROZEN_STRATEGIES["T1"], (0.0075, 0.0075, 0.0025))
        # Bot must not manage BNB as a fee-debiting strategy asset
        from binance_btc_bot.execution import protection_qty as pqmod

        src = Path(pqmod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("commission_bnb", src.lower().replace("commissionasset", ""))
        self.assertIn("never assume", src.lower())


if __name__ == "__main__":
    unittest.main()
