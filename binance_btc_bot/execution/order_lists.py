"""OCO / order-list reconciliation helpers (Binance Spot).

Binance ``GET /api/v3/openOrderList`` does **not** accept ``symbol``.
Callers must fetch open lists without that parameter and filter client-side.
``GET /api/v3/orderList`` is the authoritative single-list query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


# listOrderStatus values that mean the list is still open / working.
OPEN_LIST_ORDER_STATUSES = frozenset({"EXECUTING", "RESPONSE"})
# listStatusType values commonly seen while open.
OPEN_LIST_STATUS_TYPES = frozenset({"EXEC_STARTED", "RESPONSE", "TRAILING_ACTIVE"})


@dataclass(frozen=True)
class OrderListView:
    order_list_id: str | None
    symbol: str
    contingency_type: str
    list_status_type: str
    list_order_status: str
    list_client_order_id: str | None
    quantity: float | None
    child_orders: tuple[dict[str, Any], ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        los = str(self.list_order_status or "").upper()
        lst = str(self.list_status_type or "").upper()
        if los in {"ALL_DONE", "REJECT", "REJECTED"}:
            return False
        if lst in {"ALL_DONE"}:
            return False
        if los in OPEN_LIST_ORDER_STATUSES:
            return True
        if lst in OPEN_LIST_STATUS_TYPES:
            return True
        # GET /api/v3/openOrderList only returns open lists — empty status ⇒ open.
        if not los and not lst:
            return True
        return False


def require_order_list_rows(data: Any, *, context: str = "openOrderList") -> list[dict[str, Any]]:
    """Fail closed: response must be a list of objects."""
    if not isinstance(data, list):
        raise ValueError(f"ambiguous {context} response: expected list, got {type(data).__name__}")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(data):
        if not isinstance(row, Mapping):
            raise ValueError(f"ambiguous {context} row[{i}]: expected object")
        out.append(dict(row))
    return out


def require_order_list_object(data: Any, *, context: str = "orderList") -> dict[str, Any]:
    """Fail closed: single order-list payload must be an object with orderListId."""
    if not isinstance(data, Mapping):
        raise ValueError(f"ambiguous {context} response: expected object, got {type(data).__name__}")
    if data.get("orderListId") is None and data.get("listClientOrderId") is None:
        raise ValueError(f"ambiguous {context} response: missing orderListId/listClientOrderId")
    return dict(data)


def filter_order_lists_by_symbol(
    rows: Sequence[Mapping[str, Any]],
    symbol: str | None,
) -> list[dict[str, Any]]:
    """Client-side symbol filter (Binance openOrderList has no symbol param)."""
    if not symbol:
        return [dict(r) for r in rows]
    sym = str(symbol).upper()
    return [dict(r) for r in rows if str(r.get("symbol") or "").upper() == sym]


def parse_order_list(row: Mapping[str, Any]) -> OrderListView:
    orders_raw = row.get("orders") or ()
    children: list[dict[str, Any]] = []
    if isinstance(orders_raw, Sequence) and not isinstance(orders_raw, (str, bytes)):
        for o in orders_raw:
            if isinstance(o, Mapping):
                children.append(dict(o))
    qty = row.get("quantity")
    if qty is None and children:
        # Some payloads omit list qty; leave None rather than inventing.
        qty = None
    try:
        qty_f = float(qty) if qty is not None and qty != "" else None
    except (TypeError, ValueError):
        qty_f = None
    oid = row.get("orderListId")
    return OrderListView(
        order_list_id=str(oid) if oid is not None else None,
        symbol=str(row.get("symbol") or "").upper(),
        contingency_type=str(row.get("contingencyType") or ""),
        list_status_type=str(row.get("listStatusType") or ""),
        list_order_status=str(row.get("listOrderStatus") or ""),
        list_client_order_id=(
            str(row["listClientOrderId"]) if row.get("listClientOrderId") is not None else None
        ),
        quantity=qty_f,
        child_orders=tuple(children),
        raw=dict(row),
    )


def find_matching_open_list(
    rows: Sequence[Mapping[str, Any]],
    *,
    symbol: str | None = None,
    order_list_id: str | None = None,
    list_client_order_id: str | None = None,
) -> OrderListView | None:
    """Find an open list matching id and/or client id; optionally scoped by symbol."""
    filtered = filter_order_lists_by_symbol(rows, symbol)
    for row in filtered:
        view = parse_order_list(row)
        if order_list_id and str(view.order_list_id) == str(order_list_id):
            return view if view.is_open else None
        if list_client_order_id and view.list_client_order_id == str(list_client_order_id):
            return view if view.is_open else None
    return None


def is_binance_param_error(reason: str | None) -> bool:
    """Detect Binance unexpected/mandatory parameter errors (e.g. -1102, -1104)."""
    r = str(reason or "").lower()
    return any(
        x in r
        for x in (
            "-1102",
            "-1104",
            "-1101",
            "mandatory parameter",
            "unknown parameter",
            "too many parameters",
            "illegal characters",
        )
    )


def protection_state_from_list(view: OrderListView | None) -> str:
    """Map order-list view → coarse protection state for reconciliation."""
    if view is None:
        return "NO_OPEN_LIST"
    if not view.is_open:
        return "LIST_NOT_OPEN"
    return "PROTECTED_OPEN_LIST"
