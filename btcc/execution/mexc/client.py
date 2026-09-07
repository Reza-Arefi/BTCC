"""MEXC Spot v3 authenticated READ-ONLY HTTP client.

Official endpoints used (SPOT only):
  GET /api/v3/time                 (public)
  GET /api/v3/exchangeInfo         (public)
  GET /api/v3/account              (SIGNED, SPOT_ACCOUNT_READ)
  GET /api/v3/openOrders           (SIGNED, SPOT_DEAL_READ)
  GET /api/v3/order                (SIGNED, SPOT_DEAL_READ)  # query only
  GET /api/v3/myTrades             (SIGNED, SPOT_ACCOUNT_READ)

Explicitly NOT implemented:
  any order-create HTTP verb
  any order-cancel HTTP verb
  any stop/trailing/plan placement
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from btcc.execution.account import AccountSnapshot
from btcc.execution.mexc.auth import build_signed_query, server_timestamp_ms
from btcc.execution.mexc.credentials import MexcCredentials, redact_secrets
from btcc.execution.mexc.errors import (
    AccountDataUnavailable,
    MexcAuthError,
    MexcMalformedResponseError,
    MexcRateLimitError,
    MexcReadError,
    MexcTimeoutError,
    OrderDataUnavailable,
    SymbolMetadataUnavailable,
)
from btcc.execution.mexc.models import AssetBalance, ExchangeFill, ExchangeOrder
from btcc.execution.mexc.symbols_map import map_exchange_info_symbol
from btcc.execution.modes import ExecutionMode
from btcc.execution.symbols import SymbolMeta

logger = logging.getLogger(__name__)

# Only GET is allowed. POST/PUT/DELETE/PATCH are hard-blocked (no trading).
_TRADING_METHODS = frozenset({"POST", "PUT", "DELETE", "PATCH"})
_ALLOWED_GET_PATHS = frozenset(
    {
        "/api/v3/time",
        "/api/v3/exchangeInfo",
        "/api/v3/account",
        "/api/v3/openOrders",
        "/api/v3/order",  # GET query only — POST/DELETE never issued
        "/api/v3/myTrades",
    }
)


class MexcReadOnlyClient:
    """Authenticated Spot read client. Cannot place/cancel/protect orders."""

    def __init__(
        self,
        credentials: MexcCredentials,
        *,
        timeout_s: float = 15.0,
        opener: Any | None = None,
    ) -> None:
        self._creds = credentials
        self._timeout = float(timeout_s)
        self._opener = opener  # injectable for tests: callable(url, headers) -> bytes/str/dict
        self._symbol_cache: dict[str, SymbolMeta] = {}

    @property
    def base_url(self) -> str:
        return self._creds.base_url

    def get_server_time(self) -> int:
        data = self._request("GET", "/api/v3/time", signed=False)
        if not isinstance(data, dict) or "serverTime" not in data:
            raise MexcMalformedResponseError("time response missing serverTime")
        return int(data["serverTime"])

    def get_balances(self) -> list[AssetBalance]:
        data = self._request("GET", "/api/v3/account", signed=True)
        if not isinstance(data, dict) or "balances" not in data:
            raise AccountDataUnavailable("account response missing balances")
        balances = data["balances"]
        if not isinstance(balances, list):
            raise MexcMalformedResponseError("balances is not a list")
        out: list[AssetBalance] = []
        try:
            for row in balances:
                out.append(AssetBalance.from_mexc(row))
        except ValueError as e:
            raise MexcMalformedResponseError(f"malformed balance row: {e}") from e
        return out

    def get_account_snapshot(self, *, quote_asset: str = "BTC") -> AccountSnapshot:
        """Map free/locked for a focus asset into AccountSnapshot (REAL, not paper)."""
        try:
            bals = self.get_balances()
        except MexcReadError:
            raise
        except Exception as e:  # noqa: BLE001
            raise AccountDataUnavailable(redact_secrets(str(e), self._creds)) from e

        focus = next((b for b in bals if b.asset.upper() == quote_asset.upper()), None)
        if focus is None:
            free = locked = 0.0
        else:
            free = float(focus.free)
            locked = float(focus.locked)
        return AccountSnapshot(
            mode=ExecutionMode.REAL.value,
            available_balance=free,
            locked_balance=locked,
            total_balance=free + locked,
            currency=quote_asset.upper(),
            open_positions=0,
            reserved_order_exposure=locked,
            source="mexc_account",
            meta={
                "assets_returned": len(bals),
                "can_trade_field_ignored_for_stage3": True,
            },
        )

    def get_open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        data = self._request("GET", "/api/v3/openOrders", signed=True, params=params)
        if not isinstance(data, list):
            raise OrderDataUnavailable("openOrders response is not a list")
        try:
            return [ExchangeOrder.from_mexc(row) for row in data]
        except ValueError as e:
            raise MexcMalformedResponseError(f"malformed open order: {e}") from e

    def get_order_status(
        self,
        *,
        symbol: str,
        order_id: str | None = None,
        orig_client_order_id: str | None = None,
    ) -> ExchangeOrder:
        if not order_id and not orig_client_order_id:
            raise OrderDataUnavailable("order_id or orig_client_order_id required")
        params: dict[str, Any] = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
        data = self._request("GET", "/api/v3/order", signed=True, params=params)
        if not isinstance(data, dict):
            raise OrderDataUnavailable("order response is not an object")
        try:
            return ExchangeOrder.from_mexc(data)
        except ValueError as e:
            raise MexcMalformedResponseError(f"malformed order: {e}") from e

    def get_recent_fills(self, symbol: str, *, limit: int = 50, order_id: str | None = None) -> list[ExchangeFill]:
        params: dict[str, Any] = {"symbol": symbol, "limit": int(limit)}
        if order_id:
            params["orderId"] = order_id
        data = self._request("GET", "/api/v3/myTrades", signed=True, params=params)
        if not isinstance(data, list):
            raise OrderDataUnavailable("myTrades response is not a list")
        try:
            return [ExchangeFill.from_mexc(row) for row in data]
        except ValueError as e:
            raise MexcMalformedResponseError(f"malformed fill: {e}") from e

    def get_symbol_metadata(self, symbol: str, *, use_cache: bool = True) -> SymbolMeta:
        sym = symbol.upper()
        if use_cache and sym in self._symbol_cache:
            return self._symbol_cache[sym]
        data = self._request("GET", "/api/v3/exchangeInfo", signed=False, params={"symbol": sym})
        if not isinstance(data, dict) or "symbols" not in data:
            raise SymbolMetadataUnavailable("exchangeInfo missing symbols")
        rows = data["symbols"]
        if not isinstance(rows, list) or not rows:
            raise SymbolMetadataUnavailable(f"symbol not found: {sym}")
        meta = map_exchange_info_symbol(rows[0])
        self._symbol_cache[sym] = meta
        return meta

    # --- HTTP layer -----------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        signed: bool,
        params: dict[str, Any] | None = None,
    ) -> Any:
        method = method.upper()
        if method in _TRADING_METHODS:
            raise MexcReadError(f"Stage 3 forbids HTTP {method} — read-only client")
        if method != "GET":
            raise MexcReadError(f"unsupported method {method}")
        if path not in _ALLOWED_GET_PATHS:
            raise MexcReadError(f"path not allowlisted for Stage 3 reads: {path}")

        params = dict(params or {})
        if signed:
            query = build_signed_query(
                params,
                api_secret=self._creds.api_secret,
                recv_window_ms=self._creds.recv_window_ms,
                timestamp_ms=server_timestamp_ms(),
            )
            url = f"{self._creds.base_url}{path}?{query}"
            headers = {"X-MEXC-APIKEY": self._creds.api_key}
        else:
            from urllib.parse import urlencode

            q = urlencode({k: str(v) for k, v in params.items()}) if params else ""
            url = f"{self._creds.base_url}{path}" + (f"?{q}" if q else "")
            headers = {}

        try:
            raw = self._http_get(url, headers=headers)
        except MexcReadError:
            raise
        except Exception as e:  # noqa: BLE001
            raise MexcReadError(redact_secrets(f"request failed: {e}", self._creds)) from e

        try:
            if isinstance(raw, (dict, list)):
                data = raw
            else:
                data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise MexcMalformedResponseError("response is not JSON") from e

        if (
            isinstance(data, dict)
            and "code" in data
            and "msg" in data
            and "balances" not in data
            and "symbols" not in data
            and "serverTime" not in data
        ):
            code = str(data.get("code"))
            msg = redact_secrets(str(data.get("msg")), self._creds)
            if code in {"700001", "700002", "700006", "700007"}:
                raise MexcAuthError(msg, code=code)
            if code in {"429"} or "rate" in msg.lower() or "too many" in msg.lower():
                raise MexcRateLimitError(msg, code=code)
            # 700003 is often timestamp / recvWindow
            if code == "700003":
                raise MexcAuthError(msg, code=code)
            raise MexcReadError(msg, code=code)
        return data

    def _http_get(self, url: str, *, headers: dict[str, str]) -> Any:
        if self._opener is not None:
            return self._opener(url, headers)

        safe_url = redact_secrets(url, self._creds)
        req = Request(url, headers=headers, method="GET")
        try:
            with urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
                return body
        except TimeoutError as e:
            raise MexcTimeoutError(f"timeout contacting MEXC ({safe_url.split('?')[0]})") from e
        except HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            body = redact_secrets(body, self._creds)
            if e.code == 429:
                raise MexcRateLimitError(f"HTTP 429 {body}", http_status=429) from e
            if e.code in (401, 403):
                raise MexcAuthError(f"HTTP {e.code} {body}", http_status=e.code) from e
            raise MexcReadError(f"HTTP {e.code} {body}", http_status=e.code) from e
        except URLError as e:
            raise MexcReadError(redact_secrets(f"network error: {e}", self._creds)) from e
