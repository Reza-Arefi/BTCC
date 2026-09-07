"""Unit tests for economically-flat dust classification (reconcile safety)."""

from __future__ import annotations

import unittest

from binance_btc_bot.exchange.base import SymbolInfo
from binance_btc_bot.execution.protection_qty import is_economically_flat_base
from binance_btc_bot.risk.safety import SafetySystem


def _meta(**kwargs) -> SymbolInfo:
    base = dict(
        symbol="ICPBTC",
        status="TRADING",
        base_asset="ICP",
        quote_asset="BTC",
        quantity_step=0.01,
        min_quantity=0.01,
        max_quantity=1e6,
        price_tick=1e-7,
        min_notional=0.0001,
        order_types=("LIMIT", "MARKET", "STOP_LOSS", "TAKE_PROFIT"),
        oco_allowed=True,
        min_trailing_above_delta=10,
        max_trailing_above_delta=2000,
        min_trailing_below_delta=10,
        max_trailing_below_delta=2000,
    )
    base.update(kwargs)
    return SymbolInfo(**base)


class TestEconomicFlat(unittest.TestCase):
    def test_zero(self):
        ok, reason = is_economically_flat_base(free=0.0, locked=0.0, meta=_meta(), ref_price=0.00004)
        self.assertTrue(ok)
        self.assertEqual(reason, "ZERO")

    def test_locked_not_flat(self):
        ok, reason = is_economically_flat_base(free=0.0, locked=1.0, meta=_meta(), ref_price=0.00004)
        self.assertFalse(ok)
        self.assertEqual(reason, "LOCKED")

    def test_icp_style_min_notional_dust(self):
        # 0.01 ICP * 0.0000392 = 3.92e-7 << minNotional 0.0001
        ok, reason = is_economically_flat_base(
            free=0.01, locked=0.0, meta=_meta(), ref_price=0.0000392
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "DUST_BELOW_MIN_NOTIONAL")

    def test_sellable_inventory(self):
        ok, reason = is_economically_flat_base(
            free=10.0, locked=0.0, meta=_meta(), ref_price=0.00015
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "SELLABLE")


class TestWarnDedupe(unittest.TestCase):
    def test_cooldown_suppresses_telegram_not_reasons(self):
        s = SafetySystem(warn_notify_cooldown_sec=1000.0)
        sent: list[str] = []
        s.on_notify = lambda event, message, **kw: sent.append(kw.get("reason") or event)
        s.warn("RECONCILE_PROTECTION_MISSING", symbol="ICPBTC", trade_id="t1")
        s.warn("RECONCILE_PROTECTION_MISSING", symbol="ICPBTC", trade_id="t1")
        s.warn("RECONCILE_PROTECTION_MISSING", symbol="ICPBTC", trade_id="t1")
        self.assertEqual(sent.count("RECONCILE_PROTECTION_MISSING"), 1)
        self.assertEqual(s.reasons.count("RECONCILE_PROTECTION_MISSING"), 3)
        s.clear_warn("RECONCILE_PROTECTION_MISSING", symbol="ICPBTC", trade_id="t1")
        s.warn("RECONCILE_PROTECTION_MISSING", symbol="ICPBTC", trade_id="t1")
        self.assertEqual(sent.count("RECONCILE_PROTECTION_MISSING"), 2)


if __name__ == "__main__":
    unittest.main()
