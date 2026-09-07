"""REAL / REAL_CANARY broker — reads via MexcReadOnlyClient; writes gated.

Default: WriteGate CLOSED → all trading methods raise TradingForbiddenError.
MARKET BUY/SELL fire only when WriteGate.market_writes_allowed
(allow_trading ∧ (canary_writes_armed ∨ bounded_6h_writes_armed)).
Bot-managed T1 does not require protection_api_confirmed.
Optional MexcSpotProtectionAdapter remains legacy-only.
"""

from __future__ import annotations

from typing import Any

from btcc.execution.modes import ExecutionMode
from btcc.execution.mexc.client import MexcReadOnlyClient
from btcc.execution.mexc.credentials import MexcCredentials, load_mexc_credentials
from btcc.execution.mexc.errors import AccountDataUnavailable, MexcReadError, OrderDataUnavailable
from btcc.execution.mexc.models import AssetBalance, ExchangeFill, ExchangeOrder
from btcc.execution.mexc.protection_adapter import MexcSpotProtectionAdapter
from btcc.execution.mexc.write_client import MexcWriteClient
from btcc.execution.safety_state import ExecutionSafetyState, REAL_TRADING_DISABLED
from btcc.execution.symbols import SymbolMeta
from btcc.execution.types import AccountState, FillReport, OrderRequest
from btcc.execution.write_gate import WriteGate, CLOSED_WRITE_GATE
from btcc.safety.no_trading import TradingForbiddenError


_TRADE_DISABLED = (
    "RealBroker trading writes are gated closed (WriteGate). "
    "Open WriteGate (allow_trading + canary_writes_armed or bounded_6h_writes_armed) "
    "only at explicit enablement. Refusing to fall back to PAPER."
)


class RealBroker:
    """REAL / canary broker: reads OK; writes only when WriteGate open."""

    def __init__(
        self,
        *args: Any,
        client: MexcReadOnlyClient | None = None,
        write_client: MexcWriteClient | None = None,
        protection: MexcSpotProtectionAdapter | None = None,
        credentials: MexcCredentials | None = None,
        load_env_credentials: bool = False,
        safety: ExecutionSafetyState | None = None,
        write_gate: WriteGate | None = None,
        mode: ExecutionMode = ExecutionMode.REAL,
        **kwargs: Any,
    ) -> None:
        # Reject legacy "enable trading via kwargs" without going through WriteGate.
        if kwargs.get("allow_trading") or kwargs.get("enable_trading") or kwargs.get("trading_enabled"):
            raise TradingForbiddenError(
                "RealBroker refuses allow_trading / enable_trading kwargs — use write_gate=WriteGate(...)"
            )
        if kwargs.get("api_key") or kwargs.get("api_secret"):
            raise TradingForbiddenError(
                "pass credentials=MexcCredentials(...) or client=MexcReadOnlyClient(...) "
                "— raw api_key/api_secret kwargs are rejected"
            )

        if mode not in (
            ExecutionMode.REAL,
            ExecutionMode.REAL_CANARY_SINGLE_SHOT,
            ExecutionMode.REAL_BOUNDED_6H,
        ):
            raise TradingForbiddenError(
                f"RealBroker mode must be REAL, REAL_CANARY_SINGLE_SHOT, or REAL_BOUNDED_6H, got {mode}"
            )

        self._mode = mode
        self.gate = write_gate or CLOSED_WRITE_GATE
        self._client = client
        self._write = write_client
        self._protection = protection

        if self._client is None and credentials is not None:
            self._client = MexcReadOnlyClient(credentials)
        if self._client is None and load_env_credentials:
            creds = load_mexc_credentials(require=True)
            assert creds is not None
            self._client = MexcReadOnlyClient(creds)
            if self._write is None and self.gate.market_writes_allowed:
                self._write = MexcWriteClient(creds, gate=self.gate)
            if self._protection is None and self.gate.protection_writes_allowed:
                self._protection = MexcSpotProtectionAdapter(
                    write_client=self._write, gate=self.gate
                )

        # If write_client provided, sync gate reference
        if self._write is not None and self._write.gate is CLOSED_WRITE_GATE and self.gate is not CLOSED_WRITE_GATE:
            self._write.gate = self.gate
        if self._protection is not None:
            self._protection.gate = self.gate

        self.safety = safety or ExecutionSafetyState(
            reason_code=REAL_TRADING_DISABLED,
            detail="REAL trading gated; default write gate CLOSED",
            real_entries_blocked=not self.gate.market_writes_allowed,
        )

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    @property
    def name(self) -> str:
        if self.gate.market_writes_allowed:
            return f"RealBroker({self._mode.value}/WRITES_ARMED)"
        if self._client is not None:
            return f"RealBroker({self._mode.value}/READ_ONLY)"
        return f"RealBroker({self._mode.value}/DISABLED)"

    def _require_client(self) -> MexcReadOnlyClient:
        if self._client is None:
            raise AccountDataUnavailable(
                "MEXC read-only credentials/client not configured (set MEXC_API_KEY/MEXC_API_SECRET)"
            )
        return self._client

    def _deny_trade(self, action: str) -> None:
        raise TradingForbiddenError(f"{_TRADE_DISABLED} action={action}")

    def _require_write(self, action: str) -> MexcWriteClient:
        if not self.gate.market_writes_allowed:
            self._deny_trade(action)
        if self._write is None:
            raise TradingForbiddenError(f"{_TRADE_DISABLED} write_client missing action={action}")
        self.gate.assert_market_write(action)
        return self._write

    # --- READ path ------------------------------------------------------

    def get_account_state(self) -> AccountState:
        client = self._require_client()
        try:
            snap = client.get_account_snapshot(quote_asset="BTC")
            opens = client.get_open_orders()
        except MexcReadError as e:
            self.safety.apply_read_error(e)
            raise
        return AccountState(
            mode=self._mode.value,
            equity=float(snap.total_balance),
            available=float(snap.available_balance),
            reserved=float(snap.locked_balance),
            currency=snap.currency,
            open_positions=0,
            open_orders=len(opens),
            meta={"source": snap.source, "assets_returned": snap.meta.get("assets_returned")},
        )

    def get_balances(self) -> dict[str, float]:
        bals = self.get_balances_detailed()
        return {b.asset: float(b.free) for b in bals}

    def get_balances_detailed(self) -> list[AssetBalance]:
        client = self._require_client()
        try:
            return client.get_balances()
        except MexcReadError as e:
            self.safety.apply_read_error(e)
            raise

    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return [self._order_to_dict(o) for o in self.get_open_orders_typed(symbol=symbol)]

    def get_open_orders_typed(self, symbol: str | None = None) -> list[ExchangeOrder]:
        client = self._require_client()
        try:
            return client.get_open_orders(symbol=symbol)
        except MexcReadError as e:
            self.safety.apply_read_error(e)
            raise

    def get_open_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        bals = self.get_balances_detailed()
        out = []
        for b in bals:
            if b.total == 0:
                continue
            if symbol and b.asset.upper() not in str(symbol).upper():
                continue
            out.append(
                {
                    "asset": b.asset,
                    "free": str(b.free),
                    "locked": str(b.locked),
                    "total": str(b.total),
                }
            )
        return out

    def get_order(self, order_id: str, *, symbol: str | None = None) -> dict[str, Any]:
        if not symbol:
            raise OrderDataUnavailable("symbol required for MEXC order query")
        client = self._require_client()
        try:
            o = client.get_order_status(symbol=symbol, order_id=order_id)
        except MexcReadError as e:
            self.safety.apply_read_error(e)
            raise
        return self._order_to_dict(o)

    def get_order_status(
        self,
        *,
        symbol: str,
        order_id: str | None = None,
        orig_client_order_id: str | None = None,
    ) -> ExchangeOrder:
        client = self._require_client()
        try:
            return client.get_order_status(
                symbol=symbol,
                order_id=order_id,
                orig_client_order_id=orig_client_order_id,
            )
        except MexcReadError as e:
            self.safety.apply_read_error(e)
            raise

    def get_recent_fills(self, symbol: str | None = None, *, limit: int = 50) -> list[dict[str, Any]]:
        if not symbol:
            raise OrderDataUnavailable("symbol required for MEXC myTrades")
        return [self._fill_to_dict(f) for f in self.get_recent_fills_typed(symbol, limit=limit)]

    def get_recent_fills_typed(self, symbol: str, *, limit: int = 50) -> list[ExchangeFill]:
        client = self._require_client()
        try:
            return client.get_recent_fills(symbol, limit=limit)
        except MexcReadError as e:
            self.safety.apply_read_error(e)
            raise

    def get_symbol_metadata(self, symbol: str, *, use_cache: bool = True) -> SymbolMeta:
        """Fetch SymbolMeta. use_cache=False forces fresh exchangeInfo (T1 capability)."""
        client = self._require_client()
        try:
            return client.get_symbol_metadata(symbol, use_cache=use_cache)
        except TypeError:
            # Older clients without use_cache kwarg.
            return client.get_symbol_metadata(symbol)
        except MexcReadError as e:
            self.safety.apply_read_error(e)
            raise

    def get_server_time(self) -> int:
        return self._require_client().get_server_time()

    # --- WRITE path (gated) ---------------------------------------------

    def submit_order(self, request: OrderRequest | Any) -> FillReport:
        """Submit MARKET order. Prefer SpotMarketOrderRequest; OrderRequest lacks quantity."""
        from btcc.execution.symbols import normalize_order_quantity
        from btcc.execution.types import SpotMarketOrderRequest

        wc = self._require_write("submit_order")
        if isinstance(request, SpotMarketOrderRequest):
            symbol = request.symbol
            side = str(request.side).upper()
            qty = float(request.quantity)
            coid = str(request.client_order_id)
            intent_id = request.intent_id
            quote_qty = request.quote_order_qty
            order_type = str(request.order_type).upper()
            meta_hint = (request.meta or {}).get("symbol_meta")
        else:
            # Legacy OrderRequest — quantity must be in meta
            symbol = request.symbol
            side = str(request.side).upper()
            meta = getattr(request, "meta", {}) or {}
            qty = float(meta.get("quantity") or 0)
            coid = str(request.client_order_id)
            intent_id = str(request.intent_id)
            quote_qty = meta.get("quote_order_qty")
            order_type = str(meta.get("order_type") or "MARKET").upper()
            meta_hint = meta.get("symbol_meta")

        if order_type not in ("MARKET", "MKT"):
            raise TradingForbiddenError(f"unsupported real order type: {order_type}")

        # Always evaluate MARKET capability from fresh exchangeInfo when a read
        # client exists. Unit-test stacks without a client use request meta_hint.
        from btcc.execution.symbols import t1_market_exit_capability

        if self._client is None:
            if meta_hint is None:
                return FillReport(
                    ok=False,
                    broker_name=self.name,
                    client_order_id=coid,
                    requested_quantity=qty,
                    executed_quantity=0.0,
                    average_price=0.0,
                    fee=0.0,
                    order_status="REJECTED",
                    raw={"intent_id": intent_id, "side": side},
                    rejection_reason="SYMBOL_META_UNAVAILABLE:no_client",
                )
            fresh_meta = meta_hint
        else:
            try:
                try:
                    fresh_meta = self.get_symbol_metadata(symbol, use_cache=False)
                except TypeError:
                    # Older / test stubs without use_cache kwarg.
                    fresh_meta = self.get_symbol_metadata(symbol)
            except Exception as e:  # noqa: BLE001
                return FillReport(
                    ok=False,
                    broker_name=self.name,
                    client_order_id=coid,
                    requested_quantity=qty,
                    executed_quantity=0.0,
                    average_price=0.0,
                    fee=0.0,
                    order_status="REJECTED",
                    raw={"intent_id": intent_id, "side": side},
                    rejection_reason=f"SYMBOL_META_UNAVAILABLE:{e}",
                )

        ok_mkt, mkt_reason = t1_market_exit_capability(fresh_meta)
        if not ok_mkt:
            return FillReport(
                ok=False,
                broker_name=self.name,
                client_order_id=coid,
                requested_quantity=qty,
                executed_quantity=0.0,
                average_price=0.0,
                fee=0.0,
                order_status="REJECTED",
                raw={
                    "intent_id": intent_id,
                    "side": side,
                    "symbol": symbol,
                    "order_types": list(getattr(fresh_meta, "order_types", None) or ()),
                },
                rejection_reason=mkt_reason,
            )

        # Exchange-metadata-driven quantity normalization (BUY and SELL).
        # quoteOrderQty path skips base-qty normalize.
        qty_serialized: str | None = None
        sym_meta = meta_hint if meta_hint is not None else fresh_meta
        if quote_qty is None:
            norm = normalize_order_quantity(qty, sym_meta)
            if not norm.ok:
                return FillReport(
                    ok=False,
                    broker_name=self.name,
                    client_order_id=coid,
                    requested_quantity=qty,
                    executed_quantity=0.0,
                    average_price=0.0,
                    fee=0.0,
                    order_status="REJECTED",
                    raw={
                        "intent_id": intent_id,
                        "side": side,
                        "raw_quantity": norm.raw_quantity,
                        "normalized_quantity": norm.quantity,
                        "serialized": norm.serialized,
                    },
                    rejection_reason=f"QTY_NORMALIZE:{norm.reason}",
                )
            qty = float(norm.quantity)
            qty_serialized = norm.serialized

        if side == "BUY":
            ack = wc.place_market_buy(
                symbol=symbol,
                quantity=qty_serialized if quote_qty is None else None,
                quote_order_qty=float(quote_qty) if quote_qty is not None else None,
                client_order_id=coid,
            )
        elif side == "SELL":
            ack = wc.place_market_sell(
                symbol=symbol,
                quantity=qty_serialized if qty_serialized is not None else qty,
                client_order_id=coid,
            )
        else:
            raise TradingForbiddenError(f"unsupported real order side: {side}")

        report = self._resolve_market_fill_report(
            ack=ack if isinstance(ack, dict) else {},
            symbol=symbol,
            side=side,
            qty=qty,
            coid=coid,
            intent_id=intent_id,
        )
        # Attach wire serialization for audits (requested vs submitted).
        if isinstance(report.raw, dict):
            report.raw = dict(report.raw)
            report.raw["submitted_quantity"] = qty
            if qty_serialized is not None:
                report.raw["submitted_quantity_serialized"] = qty_serialized
        return report

    def _vwap_from_order(self, order: ExchangeOrder) -> float:
        if order.average_price is not None and float(order.average_price) > 0:
            return float(order.average_price)
        raw = getattr(order, "raw", None) or {}
        executed = float(order.executed_quantity or 0)
        cq = raw.get("cummulativeQuoteQty") or raw.get("cumulativeQuoteQty")
        if cq is not None and executed > 0:
            return float(cq) / executed
        return 0.0

    def _ack_incomplete(self, order: ExchangeOrder, *, requested_qty: float) -> bool:
        """True when ACK cannot be trusted as a conclusive fill result."""
        executed = float(order.executed_quantity or 0)
        avg = self._vwap_from_order(order)
        status = str(order.status or "").upper()
        terminal_empty = status in {
            "CANCELED",
            "CANCELLED",
            "REJECTED",
            "EXPIRED",
            "EXPIRED_IN_MATCH",
        }
        # Conclusive unfilled terminal states are complete even with qty=0.
        if terminal_empty and executed <= 0:
            return False
        if executed <= 0:
            return True
        if avg <= 0:
            return True
        # Accepted but not yet showing terminal fill detail.
        if status in {"", "NEW", "NEW_ORDER", "PENDING", "SUBMITTED", "ACK"}:
            return True
        if status in {"FILLED", "PARTIALLY_FILLED"} and executed + 1e-15 < float(requested_qty):
            # Partial may be conclusive; still OK to use without forcing unknown.
            return False
        return False

    def _resolve_market_fill_report(
        self,
        *,
        ack: dict[str, Any],
        symbol: str,
        side: str,
        qty: float,
        coid: str,
        intent_id: str | None,
    ) -> FillReport:
        """Map place-order ACK → FillReport with mandatory clientOrderId reconcile.

        Invariant: an incomplete/ambiguous ACK is NEVER reported as ZERO_FILL.
        ZERO_FILL is reserved for conclusive exchange states with executedQty=0
        (CANCELED / REJECTED / EXPIRED / EXPIRED_IN_MATCH).
        """
        from btcc.execution.mexc.write_client import parse_order_ack

        order = parse_order_ack(ack)
        executed = float(order.executed_quantity or 0)
        avg = self._vwap_from_order(order)
        remaining = float(order.remaining_quantity or max(0.0, qty - executed))
        exchange_oid = order.order_id or str(ack.get("orderId") or "")
        status = str(order.status or "").upper()
        reconciled_raw: dict[str, Any] | None = None
        accepted = bool(exchange_oid) or bool(ack.get("orderId") or ack.get("clientOrderId"))

        need_reconcile = self._ack_incomplete(order, requested_qty=qty)
        if need_reconcile:
            try:
                ex = self.get_order_status(symbol=symbol, orig_client_order_id=coid)
                reconciled_raw = dict(getattr(ex, "raw", None) or {})
                executed = float(ex.executed_quantity or 0)
                remaining = float(ex.remaining_quantity or max(0.0, qty - executed))
                exchange_oid = ex.order_id or exchange_oid
                avg = self._vwap_from_order(ex)
                status = str(ex.status or "").upper()
                order = ex
            except Exception as recon_err:  # noqa: BLE001
                # Order may already exist on the exchange — never call this ZERO_FILL.
                return FillReport(
                    ok=False,
                    broker_name=self.name,
                    entry_mid=None,
                    exchange_order_id=exchange_oid or None,
                    client_order_id=coid,
                    requested_quantity=qty,
                    executed_quantity=0.0,
                    average_price=0.0,
                    fee=0.0,
                    order_status="RECONCILE_UNKNOWN",
                    raw={
                        "ack": ack,
                        "reconcile_error": str(recon_err),
                        "intent_id": intent_id,
                        "side": side,
                    },
                    rejection_reason=f"ACK_INCOMPLETE_RECONCILE_FAILED:{recon_err}",
                )

        terminal_empty = status in {
            "CANCELED",
            "CANCELLED",
            "REJECTED",
            "EXPIRED",
            "EXPIRED_IN_MATCH",
        }
        working = status in {"NEW", "NEW_ORDER", "PENDING", "SUBMITTED", "ACK", "PARTIALLY_FILLED"}

        if executed > 0:
            if avg <= 0:
                # Qty known but VWAP unknown — never invent a fill price; HALT upstream.
                return FillReport(
                    ok=False,
                    broker_name=self.name,
                    entry_mid=None,
                    exchange_order_id=exchange_oid or None,
                    client_order_id=order.client_order_id or coid,
                    requested_quantity=qty,
                    executed_quantity=executed,
                    average_price=0.0,
                    fee=float(ack.get("fee") or 0.0),
                    order_status="RECONCILE_UNKNOWN",
                    raw={"ack": ack, "reconciled": reconciled_raw, "intent_id": intent_id, "side": side},
                    rejection_reason="FILL_QTY_OK_VWAP_UNCONFIRMED",
                )
            if remaining <= 1e-15 or status == "FILLED":
                out_status = "FILLED"
            else:
                out_status = "PARTIALLY_FILLED"
            return FillReport(
                ok=True,
                broker_name=self.name,
                entry_mid=avg,
                exchange_order_id=exchange_oid or None,
                client_order_id=order.client_order_id or coid,
                requested_quantity=qty,
                executed_quantity=executed,
                average_price=avg,
                fee=float(ack.get("fee") or 0.0),
                order_status=out_status,
                raw={"ack": ack, "reconciled": reconciled_raw, "intent_id": intent_id, "side": side},
                rejection_reason=None,
            )

        # executed == 0
        if terminal_empty:
            return FillReport(
                ok=False,
                broker_name=self.name,
                entry_mid=None,
                exchange_order_id=exchange_oid or None,
                client_order_id=coid,
                requested_quantity=qty,
                executed_quantity=0.0,
                average_price=0.0,
                fee=0.0,
                order_status=status or "CANCELED",
                raw={"ack": ack, "reconciled": reconciled_raw, "intent_id": intent_id, "side": side},
                rejection_reason="ZERO_FILL_OR_UNFILLED",
            )

        if accepted or working or need_reconcile:
            # Ambiguous / working order with no fill yet — NOT zero fill.
            return FillReport(
                ok=False,
                broker_name=self.name,
                entry_mid=None,
                exchange_order_id=exchange_oid or None,
                client_order_id=coid,
                requested_quantity=qty,
                executed_quantity=0.0,
                average_price=0.0,
                fee=0.0,
                order_status="RECONCILE_UNKNOWN" if status in {"", "UNKNOWN"} else status or "RECONCILE_UNKNOWN",
                raw={"ack": ack, "reconciled": reconciled_raw, "intent_id": intent_id, "side": side},
                rejection_reason="ORDER_ACCEPTED_FILL_UNCONFIRMED",
            )

        # No acceptance evidence and no fill — treat as unfilled/rejected.
        return FillReport(
            ok=False,
            broker_name=self.name,
            entry_mid=None,
            exchange_order_id=None,
            client_order_id=coid,
            requested_quantity=qty,
            executed_quantity=0.0,
            average_price=0.0,
            fee=0.0,
            order_status=status or "UNKNOWN",
            raw={"ack": ack, "reconciled": reconciled_raw, "intent_id": intent_id, "side": side},
            rejection_reason="ZERO_FILL_OR_UNFILLED",
        )

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> dict[str, Any]:
        wc = self._require_write("cancel_order")
        if not symbol:
            raise OrderDataUnavailable("symbol required for cancel")
        return wc.cancel_order(symbol=symbol, order_id=order_id)

    def place_protective_stop(
        self,
        position_id: str,
        *,
        stop_price: float,
        meta: dict[str, Any] | None = None,
        symbol: str | None = None,
        quantity: float | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        if self._protection is None:
            self._deny_trade("place_protective_stop")
        assert self._protection is not None
        meta = meta or {}
        return self._protection.place_protective_stop(
            symbol=str(symbol or meta.get("symbol") or ""),
            quantity=float(quantity if quantity is not None else meta.get("quantity") or 0),
            stop_price=float(stop_price),
            client_order_id=str(client_order_id or meta.get("client_order_id") or position_id),
            meta=meta,
        )

    def cancel_protective_stop(self, *, symbol: str, protection_id: str) -> dict[str, Any]:
        if self._protection is None:
            self._deny_trade("cancel_protective_stop")
        assert self._protection is not None
        return self._protection.cancel_protective_stop(symbol=symbol, protection_id=protection_id)

    def replace_protective_stop(
        self,
        *,
        symbol: str,
        old_protection_id: str,
        quantity: float,
        stop_price: float,
        client_order_id: str,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._protection is None:
            self._deny_trade("replace_protective_stop")
        assert self._protection is not None
        return self._protection.replace_protective_stop(
            symbol=symbol,
            old_protection_id=old_protection_id,
            quantity=quantity,
            stop_price=stop_price,
            client_order_id=client_order_id,
            meta=meta,
        )

    def open_long_legs(
        self,
        *,
        alt_btc_entry_mid: float,
        btc_usdt: float,
        notional_usd: float,
        specs: list[Any],
        entry_ts: Any,
    ) -> list[Any]:
        self._deny_trade("open_long_legs")
        raise AssertionError("unreachable")

    # --- helpers --------------------------------------------------------

    @staticmethod
    def _order_to_dict(o: ExchangeOrder) -> dict[str, Any]:
        return {
            "order_id": o.order_id,
            "client_order_id": o.client_order_id,
            "symbol": o.symbol,
            "side": o.side,
            "type": o.type,
            "status": o.status,
            "original_quantity": str(o.original_quantity),
            "executed_quantity": str(o.executed_quantity),
            "remaining_quantity": str(o.remaining_quantity),
            "price": str(o.price) if o.price is not None else None,
            "average_price": str(o.average_price) if o.average_price is not None else None,
            "stop_price": str(o.stop_price) if o.stop_price is not None else None,
            "time": o.time,
            "update_time": o.update_time,
        }

    @staticmethod
    def _fill_to_dict(f: ExchangeFill) -> dict[str, Any]:
        return {
            "trade_id": f.trade_id,
            "order_id": f.order_id,
            "symbol": f.symbol,
            "quantity": str(f.quantity),
            "price": str(f.price),
            "fee": str(f.fee) if f.fee is not None else None,
            "fee_asset": f.fee_asset,
            "timestamp": f.timestamp,
        }
