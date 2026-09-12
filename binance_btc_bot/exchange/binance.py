"""Binance spot exchange adapter — native trailing via OCO + trailingDelta.

IMPORTANT
---------
Trailing behavior after submission is owned by Binance. This adapter only
places / cancels / queries orders. It does NOT simulate trail ratchets.

Authentication: Ed25519 (asymmetric). HMAC secrets are not used.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any
from urllib.request import Request

from binance_btc_bot.exchange.base import (
    AccountSnapshot,
    Balance,
    ExchangeAdapter,
    OrderRequest,
    OrderResult,
    SymbolInfo,
    TrailingOcoRequest,
)
from binance_btc_bot.exchange.signing import (
    Ed25519RequestSigner,
    build_signature_payload,
    encode_signature_for_query,
)
from binance_btc_bot.secrets import scrub_obj, scrub_text

logger = logging.getLogger(__name__)


class BinanceAPIError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, payload: Any = None):
        safe_msg = scrub_text(message)
        super().__init__(safe_msg)
        self.status = status
        self.payload = scrub_obj(payload) if payload is not None else None


class BinanceExchange(ExchangeAdapter):
    """Spot Binance REST adapter with Ed25519 signed private endpoints."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        signer: Ed25519RequestSigner | None = None,
        public_rest_base: str = "https://data-api.binance.vision",
        private_rest_base: str = "https://api.binance.com",
        recv_window_ms: int = 5000,
        dry_run: bool = True,
        live_enabled: bool = False,
        timeout_sec: float = 20.0,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.signer = signer
        self.public_rest_base = public_rest_base.rstrip("/")
        self.private_rest_base = private_rest_base.rstrip("/")
        self.recv_window_ms = int(recv_window_ms)
        self.dry_run = bool(dry_run)
        self.live_enabled = bool(live_enabled)
        self.timeout_sec = float(timeout_sec)
        self._symbol_cache: dict[str, SymbolInfo] = {}

    # ------------------------------------------------------------------ HTTP
    def _sign_query(self, params: dict[str, Any]) -> str:
        """Build query string including Ed25519 signature (percent-encoded)."""
        if not self.signer:
            raise BinanceAPIError("Ed25519 signer required for signed request")
        # Payload excludes signature; must be percent-encoded before signing (Binance rule).
        payload = build_signature_payload(params)
        signature_b64 = self.signer.sign_payload(payload)
        # Append percent-encoded base64 signature to the already-encoded payload.
        return f"{payload}&signature={encode_signature_for_query(signature_b64)}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        use_private: bool | None = None,
    ) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        private = self.private_rest_base if (use_private if use_private is not None else signed) else self.public_rest_base
        headers = {"User-Agent": "binance-btc-bot/1.0", "Accept": "application/json"}
        if signed:
            if not self.api_key or not self.signer:
                raise BinanceAPIError("API key and Ed25519 signer required for signed request")
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = self.recv_window_ms
            query = self._sign_query(params)
            headers["X-MBX-APIKEY"] = self.api_key
            url = f"{private}{path}?{query}"
            body = None
        else:
            query = urllib.parse.urlencode(params, doseq=True)
            url = f"{private if use_private else self.public_rest_base}{path}"
            if method.upper() == "GET":
                if query:
                    url = f"{url}?{query}"
                body = None
            else:
                body = query.encode("utf-8") if query else None
                headers["Content-Type"] = "application/x-www-form-urlencoded"
                if self.api_key:
                    headers["X-MBX-APIKEY"] = self.api_key

        req = Request(url, data=body, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = {"raw": raw}
            raise BinanceAPIError(
                scrub_text(f"Binance HTTP {e.code}: {payload}"),
                status=e.code,
                payload=payload,
            ) from e

    def _writes_allowed(self) -> tuple[bool, str]:
        # Hard kill switch — default false. Accidental process start cannot submit.
        from binance_btc_bot.config_loader import env_live_trading_enabled

        if not env_live_trading_enabled():
            return False, "LIVE_TRADING_ENABLED_FALSE"
        if self.dry_run:
            return False, "DRY_RUN"
        if not self.live_enabled:
            return False, "LIVE_DISABLED"
        if not self.api_key or not self.signer:
            return False, "MISSING_API_CREDENTIALS"
        return True, "OK"

    # -------------------------------------------------------------- public API
    def ping(self) -> bool:
        self._request("GET", "/api/v3/ping")
        return True

    def get_price(self, symbol: str) -> float:
        data = self._request("GET", "/api/v3/ticker/price", params={"symbol": symbol.upper()})
        return float(data["price"])

    def get_klines(
        self,
        symbol: str,
        *,
        interval: str = "15m",
        limit: int = 1000,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list:
        """Public klines via public_rest_base (Vision). No credentials required."""
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": str(interval),
            "limit": int(min(max(limit, 1), 1000)),
        }
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)
        data = self._request("GET", "/api/v3/klines", params=params, signed=False)
        if not isinstance(data, list):
            raise BinanceAPIError("unexpected klines payload", payload=data)
        return data

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}
        # Batch endpoint expects JSON array string
        payload = json.dumps([s.upper() for s in symbols], separators=(",", ":"))
        data = self._request("GET", "/api/v3/ticker/price", params={"symbols": payload})
        if isinstance(data, dict):
            return {str(data["symbol"]).upper(): float(data["price"])}
        return {str(row["symbol"]).upper(): float(row["price"]) for row in data}

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        sym = symbol.upper()
        if sym in self._symbol_cache:
            return self._symbol_cache[sym]
        data = self._request("GET", "/api/v3/exchangeInfo", params={"symbol": sym})
        rows = data.get("symbols") or []
        if not rows:
            raise BinanceAPIError(f"symbol not found: {sym}", payload=data)
        info = _parse_symbol_info(rows[0])
        self._symbol_cache[sym] = info
        return info

    def get_symbol_infos(self, symbols: list[str]) -> dict[str, SymbolInfo]:
        """Batch exchangeInfo lookup; populates the symbol cache."""
        wanted = [s.upper() for s in symbols]
        missing = [s for s in wanted if s not in self._symbol_cache]
        if missing:
            payload = json.dumps(missing, separators=(",", ":"))
            data = self._request("GET", "/api/v3/exchangeInfo", params={"symbols": payload})
            for row in data.get("symbols") or []:
                info = _parse_symbol_info(row)
                self._symbol_cache[info.symbol] = info
        out: dict[str, SymbolInfo] = {}
        for s in wanted:
            if s not in self._symbol_cache:
                raise BinanceAPIError(f"symbol not found: {s}")
            out[s] = self._symbol_cache[s]
        return out

    def get_account(self) -> AccountSnapshot:
        data = self._request("GET", "/api/v3/account", signed=True)
        balances: dict[str, Balance] = {}
        for row in data.get("balances") or []:
            asset = str(row["asset"]).upper()
            free = float(row.get("free") or 0)
            locked = float(row.get("locked") or 0)
            if free == 0 and locked == 0:
                continue
            balances[asset] = Balance(asset=asset, free=free, locked=locked)
        return AccountSnapshot(balances=balances, raw=data)

    def get_api_key_restrictions(self) -> dict[str, Any]:
        """API-key permission flags (USER_DATA).

        Uses ``GET /sapi/v1/account/apiRestrictions``.
        This is the authoritative source for key-level ``enableWithdrawals``.
        Do not confuse with account-level ``canWithdraw`` from ``/api/v3/account``.
        """
        data = self._request("GET", "/sapi/v1/account/apiRestrictions", params={}, signed=True)
        if not isinstance(data, dict):
            raise BinanceAPIError("unexpected apiRestrictions payload", payload=data)
        return scrub_obj(data)

    def get_bnb_burn_status(self) -> dict[str, Any]:
        """Read-only BNB fee-payment / burn status (USER_DATA).

        Uses ``GET /sapi/v1/bnbBurn``. Does not enable/disable or transfer BNB.
        ``spotBNBBurn=true`` means the account is set to pay Spot fees in BNB when possible.
        Actual fee asset on each fill remains whatever Binance reports in commissionAsset.
        """
        data = self._request("GET", "/sapi/v1/bnbBurn", params={}, signed=True)
        if not isinstance(data, dict):
            raise BinanceAPIError("unexpected bnbBurn payload", payload=data)
        return scrub_obj(data)

    def get_balance(self, asset: str) -> Balance:
        acct = self.get_account()
        a = asset.upper()
        return acct.balances.get(a) or Balance(asset=a, free=0.0, locked=0.0)

    def place_entry(self, request: OrderRequest) -> OrderResult:
        """Entry path — MARKET BUY (and similar). Delegates to ``place_order``."""
        return self.place_order(request)

    def place_protective_sell(self, request: OrderRequest) -> OrderResult:
        """Emergency / protective SELL — explicit SELL semantics, never BUY/MARKET.

        Shares the low-level ``POST /api/v3/order`` adapter with entries but refuses
        entry-side misuse (BUY, MARKET flatten).
        """
        side = str(request.side or "").upper()
        otype = str(request.order_type or "").upper()
        if side != "SELL":
            return OrderResult(
                ok=False,
                reason="PROTECTIVE_SELL_REQUIRES_SIDE_SELL",
                symbol=request.symbol,
                side=side,
                order_type=otype,
            )
        if otype == "MARKET":
            return OrderResult(
                ok=False,
                reason="PROTECTIVE_SELL_REFUSES_MARKET",
                symbol=request.symbol,
                side=side,
                order_type=otype,
            )
        allowed_types = {
            "STOP_LOSS",
            "STOP_LOSS_LIMIT",
            "TAKE_PROFIT",
            "TAKE_PROFIT_LIMIT",
            "LIMIT_MAKER",
            "LIMIT",
        }
        if otype not in allowed_types:
            return OrderResult(
                ok=False,
                reason=f"PROTECTIVE_SELL_UNSUPPORTED_TYPE:{otype}",
                symbol=request.symbol,
                side=side,
                order_type=otype,
            )
        return self.place_order(request)

    def place_order(self, request: OrderRequest) -> OrderResult:
        """Low-level ``POST /api/v3/order`` — no portfolio/signal/allocation logic."""
        allowed, reason = self._writes_allowed()
        params = {
            "symbol": request.symbol.upper(),
            "side": request.side.upper(),
            "type": request.order_type.upper(),
        }
        if request.quantity is not None:
            params["quantity"] = _fmt_decimal(request.quantity)
        if request.quote_order_qty is not None:
            params["quoteOrderQty"] = _fmt_decimal(request.quote_order_qty)
        if request.price is not None:
            params["price"] = _fmt_decimal(request.price)
        if request.stop_price is not None:
            params["stopPrice"] = _fmt_decimal(request.stop_price)
        if request.trailing_delta is not None:
            params["trailingDelta"] = int(request.trailing_delta)
        if request.time_in_force:
            params["timeInForce"] = request.time_in_force
        if request.client_order_id:
            params["newClientOrderId"] = request.client_order_id
        params.update(request.extra or {})

        if not allowed:
            logger.info(
                "ORDER dry-blocked reason=%s side=%s type=%s params=%s",
                reason,
                params.get("side"),
                params.get("type"),
                {k: params[k] for k in params if k != "signature"},
            )
            return OrderResult(
                ok=True,
                order_id=None,
                client_order_id=request.client_order_id,
                status="DRY_RUN",
                symbol=request.symbol.upper(),
                side=request.side.upper(),
                order_type=request.order_type.upper(),
                quantity=request.quantity,
                reason=reason,
                dry_run=True,
                raw={"params": params, "blocked_reason": reason},
            )

        data = self._request("POST", "/api/v3/order", params=params, signed=True)
        return _order_result_from_response(data)

    def place_trailing_exit(self, request: TrailingOcoRequest) -> OrderResult:
        """Submit Binance-native OCO trailing exit (exchange owns trail after this)."""
        allowed, reason = self._writes_allowed()
        params: dict[str, Any] = {
            "symbol": request.symbol.upper(),
            "side": request.side.upper(),
            "quantity": _fmt_decimal(request.quantity),
            "aboveType": request.above_type.upper(),
            "aboveStopPrice": _fmt_decimal(request.above_stop_price),
            "aboveTrailingDelta": int(request.above_trailing_delta),
            "belowType": request.below_type.upper(),
            "belowStopPrice": _fmt_decimal(request.below_stop_price),
            "newOrderRespType": request.new_order_resp_type,
        }
        if request.list_client_order_id:
            params["listClientOrderId"] = request.list_client_order_id
        if request.above_price is not None:
            params["abovePrice"] = _fmt_decimal(request.above_price)
        if request.below_price is not None:
            params["belowPrice"] = _fmt_decimal(request.below_price)
        if request.above_time_in_force:
            params["aboveTimeInForce"] = request.above_time_in_force
        if request.below_time_in_force:
            params["belowTimeInForce"] = request.below_time_in_force

        if not allowed:
            logger.info(
                "TRAILING_ORDER dry-blocked reason=%s mode=BINANCE_NATIVE params=%s",
                reason,
                params,
            )
            return OrderResult(
                ok=True,
                order_id=None,
                client_order_id=request.list_client_order_id,
                status="DRY_RUN",
                symbol=request.symbol.upper(),
                side=request.side.upper(),
                order_type="OCO_TRAILING",
                quantity=request.quantity,
                reason=reason,
                dry_run=True,
                raw={"params": params, "blocked_reason": reason, "mode": "BINANCE_NATIVE"},
            )

        data = self._request("POST", "/api/v3/orderList/oco", params=params, signed=True)
        order_list_id = str(data.get("orderListId")) if data.get("orderListId") is not None else None
        return OrderResult(
            ok=True,
            order_id=order_list_id,
            client_order_id=data.get("listClientOrderId"),
            status=str(data.get("listOrderStatus") or data.get("listStatusType") or "NEW"),
            symbol=request.symbol.upper(),
            side=request.side.upper(),
            order_type="OCO_TRAILING",
            quantity=request.quantity,
            raw=data,
            dry_run=False,
        )

    def cancel_order(
        self,
        symbol: str,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> OrderResult:
        allowed, reason = self._writes_allowed()
        params = {"symbol": symbol.upper()}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        if not allowed:
            return OrderResult(ok=True, status="DRY_RUN", reason=reason, dry_run=True, raw={"params": params})
        data = self._request("DELETE", "/api/v3/order", params=params, signed=True)
        return _order_result_from_response(data)

    def cancel_order_list(
        self,
        symbol: str,
        order_list_id: str | None = None,
        list_client_order_id: str | None = None,
    ) -> OrderResult:
        allowed, reason = self._writes_allowed()
        params: dict[str, Any] = {"symbol": symbol.upper()}
        if order_list_id:
            params["orderListId"] = order_list_id
        if list_client_order_id:
            params["listClientOrderId"] = list_client_order_id
        if not allowed:
            return OrderResult(ok=True, status="DRY_RUN", reason=reason, dry_run=True, raw={"params": params})
        data = self._request("DELETE", "/api/v3/orderList", params=params, signed=True)
        return OrderResult(
            ok=True,
            order_id=str(data.get("orderListId")) if data.get("orderListId") is not None else None,
            client_order_id=data.get("listClientOrderId"),
            status=str(data.get("listOrderStatus") or "CANCELED"),
            symbol=symbol.upper(),
            raw=data,
        )

    def get_order(
        self,
        symbol: str,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> OrderResult:
        params: dict[str, Any] = {"symbol": symbol.upper()}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        data = self._request("GET", "/api/v3/order", params=params, signed=True)
        return _order_result_from_response(data)

    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        data = self._request("GET", "/api/v3/openOrders", params=params, signed=True)
        return list(data or [])

    def get_open_order_lists(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Query open order lists.

        Binance ``GET /api/v3/openOrderList`` accepts only timestamp/recvWindow.
        Passing ``symbol`` causes parameter errors (-1102 / too many parameters).
        When ``symbol`` is provided, filter client-side after a successful fetch.
        Ambiguous / non-list responses fail closed (raise).
        """
        from binance_btc_bot.execution.order_lists import (
            filter_order_lists_by_symbol,
            require_order_list_rows,
        )

        # Never send symbol to Binance — unsupported parameter.
        data = self._request("GET", "/api/v3/openOrderList", params={}, signed=True)
        try:
            rows = require_order_list_rows(data, context="openOrderList")
        except ValueError as e:
            raise BinanceAPIError(str(e), payload=data) from e
        return filter_order_lists_by_symbol(rows, symbol)

    def get_order_list(
        self,
        *,
        order_list_id: str | None = None,
        orig_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Authoritative query for one order list (``GET /api/v3/orderList``)."""
        from binance_btc_bot.execution.order_lists import require_order_list_object

        if not order_list_id and not orig_client_order_id:
            raise BinanceAPIError("orderListId or origClientOrderId required")
        params: dict[str, Any] = {}
        if order_list_id:
            params["orderListId"] = order_list_id
        if orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
        data = self._request("GET", "/api/v3/orderList", params=params, signed=True)
        try:
            return require_order_list_object(data, context="orderList")
        except ValueError as e:
            raise BinanceAPIError(str(e), payload=data) from e

    def get_my_trades(self, symbol: str, limit: int = 50) -> list[dict[str, Any]]:
        data = self._request(
            "GET",
            "/api/v3/myTrades",
            params={"symbol": symbol.upper(), "limit": int(limit)},
            signed=True,
        )
        return list(data or [])


def _fmt_decimal(value: float | int | str) -> str:
    """Format without scientific notation; strip trailing zeros carefully."""
    if isinstance(value, int):
        return str(value)
    s = f"{float(value):.12f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _parse_symbol_info(row: dict[str, Any]) -> SymbolInfo:
    filters = {f["filterType"]: f for f in (row.get("filters") or [])}
    lot = filters.get("LOT_SIZE") or {}
    price = filters.get("PRICE_FILTER") or {}
    notion = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
    trail = filters.get("TRAILING_DELTA") or {}
    min_notional = float(notion.get("minNotional") or notion.get("notional") or 0)
    max_qty = lot.get("maxQty")
    return SymbolInfo(
        symbol=str(row["symbol"]).upper(),
        status=str(row.get("status") or "UNKNOWN"),
        base_asset=str(row.get("baseAsset") or "").upper(),
        quote_asset=str(row.get("quoteAsset") or "").upper(),
        quantity_step=float(lot.get("stepSize") or 0),
        min_quantity=float(lot.get("minQty") or 0),
        max_quantity=float(max_qty) if max_qty not in (None, "") else None,
        price_tick=float(price.get("tickSize") or 0),
        min_notional=min_notional,
        order_types=tuple(str(x) for x in (row.get("orderTypes") or [])),
        oco_allowed=bool(row.get("ocoAllowed", True)),
        min_trailing_above_delta=int(trail["minTrailingAboveDelta"]) if "minTrailingAboveDelta" in trail else None,
        max_trailing_above_delta=int(trail["maxTrailingAboveDelta"]) if "maxTrailingAboveDelta" in trail else None,
        min_trailing_below_delta=int(trail["minTrailingBelowDelta"]) if "minTrailingBelowDelta" in trail else None,
        max_trailing_below_delta=int(trail["maxTrailingBelowDelta"]) if "maxTrailingBelowDelta" in trail else None,
        raw_filters=filters,
    )


def _order_result_from_response(data: dict[str, Any]) -> OrderResult:
    return OrderResult(
        ok=True,
        order_id=str(data["orderId"]) if data.get("orderId") is not None else None,
        client_order_id=data.get("clientOrderId"),
        status=str(data.get("status") or "UNKNOWN"),
        symbol=str(data.get("symbol") or "").upper() or None,
        side=data.get("side"),
        order_type=data.get("type"),
        price=float(data["price"]) if data.get("price") not in (None, "") else None,
        quantity=float(data["origQty"]) if data.get("origQty") not in (None, "") else None,
        executed_qty=float(data["executedQty"]) if data.get("executedQty") not in (None, "") else None,
        cumulative_quote_qty=(
            float(data["cummulativeQuoteQty"]) if data.get("cummulativeQuoteQty") not in (None, "") else None
        ),
        raw=data,
        dry_run=False,
    )
