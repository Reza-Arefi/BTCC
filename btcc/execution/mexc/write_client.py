"""MEXC Spot write client — MARKET entry + cancel only.

Official (documented) write endpoints used when WriteGate is open:
  POST   /api/v3/order   type=MARKET
  DELETE /api/v3/order

Protective stops are NOT placed here — see protection_adapter.py (ASSUMPTION).

Writes are impossible while WriteGate is closed (default Stage 7 state).
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from btcc.execution.mexc.auth import build_signed_query, server_timestamp_ms
from btcc.execution.mexc.credentials import MexcCredentials, redact_secrets
from btcc.execution.mexc.errors import (
    MexcAuthError,
    MexcMalformedResponseError,
    MexcRateLimitError,
    MexcReadError,
    MexcTimeoutError,
)
from btcc.execution.mexc.models import ExchangeOrder
from btcc.execution.write_gate import WriteGate, CLOSED_WRITE_GATE

logger = logging.getLogger(__name__)

_ALLOWED_POST_PATHS = frozenset({"/api/v3/order"})
_ALLOWED_DELETE_PATHS = frozenset({"/api/v3/order"})


class MexcWriteTimeout(MexcTimeoutError):
    """Timeout on a write — caller MUST reconcile, never blind-retry."""


class MexcWriteClient:
    """Gated Spot write client. Default gate = CLOSED."""

    def __init__(
        self,
        credentials: MexcCredentials,
        *,
        gate: WriteGate | None = None,
        timeout_s: float = 15.0,
        opener: Any | None = None,
    ) -> None:
        self._creds = credentials
        self.gate = gate or CLOSED_WRITE_GATE
        self._timeout = float(timeout_s)
        self._opener = opener

    def place_market_buy(
        self,
        *,
        symbol: str,
        quantity: float | str | None = None,
        quote_order_qty: float | str | None = None,
        client_order_id: str,
    ) -> dict[str, Any]:
        """MARKET BUY ALT/BTC. Prefer quantity (base) or quoteOrderQty (BTC).

        Prefer pre-normalized *string* quantities from SymbolMeta formatting so
        integer-lot symbols serialize as "1343" rather than "1343.0".
        """
        self.gate.assert_market_write("place_market_buy")
        if quantity is None and quote_order_qty is None:
            raise ValueError("quantity or quote_order_qty required")
        params: dict[str, Any] = {
            "symbol": str(symbol).upper().replace("/", "").replace("-", "").replace("_", ""),
            "side": "BUY",
            "type": "MARKET",
            "newClientOrderId": client_order_id,
        }
        if quantity is not None:
            params["quantity"] = quantity
        if quote_order_qty is not None:
            params["quoteOrderQty"] = quote_order_qty
        return self._request("POST", "/api/v3/order", signed=True, params=params)

    def place_market_sell(
        self,
        *,
        symbol: str,
        quantity: float | str,
        client_order_id: str,
    ) -> dict[str, Any]:
        """Emergency MARKET SELL — only via safety layer when gate open."""
        self.gate.assert_market_write("place_market_sell")
        params = {
            "symbol": str(symbol).upper().replace("/", "").replace("-", "").replace("_", ""),
            "side": "SELL",
            "type": "MARKET",
            "quantity": quantity,
            "newClientOrderId": client_order_id,
        }
        return self._request("POST", "/api/v3/order", signed=True, params=params)

    def cancel_order(
        self,
        *,
        symbol: str,
        order_id: str | None = None,
        orig_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        self.gate.assert_market_write("cancel_order")
        if not order_id and not orig_client_order_id:
            raise ValueError("order_id or orig_client_order_id required")
        params: dict[str, Any] = {
            "symbol": str(symbol).upper().replace("/", "").replace("-", "").replace("_", ""),
        }
        if order_id:
            params["orderId"] = order_id
        if orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
        return self._request("DELETE", "/api/v3/order", signed=True, params=params)

    def _request(
        self,
        method: str,
        path: str,
        *,
        signed: bool,
        params: dict[str, Any] | None = None,
    ) -> Any:
        method = method.upper()
        if method == "POST" and path not in _ALLOWED_POST_PATHS:
            raise MexcReadError(f"POST path not allowlisted: {path}")
        if method == "DELETE" and path not in _ALLOWED_DELETE_PATHS:
            raise MexcReadError(f"DELETE path not allowlisted: {path}")
        if method not in {"POST", "DELETE"}:
            raise MexcReadError(f"MexcWriteClient does not issue {method}")

        params = dict(params or {})
        if not signed:
            raise MexcReadError("writes must be signed")
        query = build_signed_query(
            params,
            api_secret=self._creds.api_secret,
            recv_window_ms=self._creds.recv_window_ms,
            timestamp_ms=server_timestamp_ms(),
        )
        url = f"{self._creds.base_url}{path}?{query}"
        # MEXC Spot rejects POSTs without an explicit Content-Type (error 700013).
        headers = {
            "X-MEXC-APIKEY": self._creds.api_key,
            "Content-Type": "application/json",
            "User-Agent": "BTCC-mexc-write/0.1",
        }

        try:
            raw = self._http(method, url, headers=headers)
        except MexcTimeoutError:
            raise
        except MexcReadError:
            raise
        except Exception as e:  # noqa: BLE001
            raise MexcReadError(redact_secrets(f"write failed: {e}", self._creds)) from e

        try:
            if isinstance(raw, (dict, list)):
                data = raw
            else:
                data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise MexcMalformedResponseError("write response is not JSON") from e

        if isinstance(data, dict) and "code" in data and "msg" in data and "orderId" not in data:
            code = str(data.get("code"))
            msg = redact_secrets(str(data.get("msg")), self._creds)
            if code in {"700001", "700002", "700003", "700006", "700007"}:
                raise MexcAuthError(msg, code=code)
            if code in {"429"}:
                raise MexcRateLimitError(msg, code=code)
            raise MexcReadError(msg, code=code)
        return data

    def _http(self, method: str, url: str, *, headers: dict[str, str]) -> Any:
        if self._opener is not None:
            return self._opener(method, url, headers)

        safe_url = redact_secrets(url, self._creds)
        # Query-string signed params (unchanged). Body stays empty so it is not
        # part of totalParams; Content-Type must still be set (MEXC 700013).
        req = Request(url, headers=headers, method=method, data=b"")
        try:
            with urlopen(req, timeout=self._timeout) as resp:
                return resp.read().decode("utf-8")
        except TimeoutError as e:
            raise MexcWriteTimeout(f"timeout on {method} ({safe_url.split('?')[0]})") from e
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


def parse_order_ack(data: dict[str, Any]) -> ExchangeOrder:
    """Map MEXC order ack/query JSON into ExchangeOrder (best-effort)."""
    from decimal import Decimal

    from btcc.execution.mexc.models import ExchangeOrder as EO

    executed = Decimal(str(data.get("executedQty") or data.get("executed_quantity") or 0))
    orig_raw = data.get("origQty") or data.get("original_quantity")
    orig = Decimal(str(orig_raw if orig_raw is not None else executed))
    avg = data.get("avgPrice") or data.get("price")
    avg_d = Decimal(str(avg)) if avg not in (None, "", "0") else None
    # MEXC market fills often omit avgPrice; derive VWAP from quote notional.
    if (avg_d is None or avg_d <= 0) and executed > 0:
        cq = data.get("cummulativeQuoteQty") or data.get("cumulativeQuoteQty")
        if cq not in (None, "", "0"):
            try:
                avg_d = Decimal(str(cq)) / executed
            except Exception:
                avg_d = None
    stop = data.get("stopPrice")
    return EO(
        order_id=str(data.get("orderId") or data.get("order_id") or ""),
        client_order_id=(
            str(data.get("clientOrderId") or data.get("client_order_id"))
            if (data.get("clientOrderId") or data.get("client_order_id"))
            else None
        ),
        symbol=str(data.get("symbol") or ""),
        side=str(data.get("side") or ""),
        type=str(data.get("type") or ""),
        status=str(data.get("status") or ""),
        original_quantity=orig,
        executed_quantity=executed,
        remaining_quantity=max(Decimal("0"), orig - executed),
        price=avg_d,
        average_price=avg_d,
        stop_price=Decimal(str(stop)) if stop not in (None, "") else None,
        time=int(data["time"]) if data.get("time") is not None else None,
        update_time=int(data["updateTime"]) if data.get("updateTime") is not None else None,
        raw=dict(data),
    )
