"""SQLite persistence for trades, orders, signals, snapshots, events."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
  trade_id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  strategy TEXT NOT NULL,
  selector TEXT,
  entry_time TEXT,
  entry_price REAL,
  quantity REAL,
  btc_value REAL,
  usdt_value REAL,
  configured_activation REAL,
  configured_trailing_distance REAL,
  strategy_config_json TEXT NOT NULL,
  binance_entry_order_id TEXT,
  binance_oco_list_id TEXT,
  exit_time TEXT,
  exit_price REAL,
  fees_btc REAL,
  fees_usdt REAL,
  realized_pnl_btc REAL,
  realized_pnl_usdt REAL,
  realized_pnl_btc_equivalent REAL,
  status TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  trade_id TEXT,
  symbol TEXT,
  order_id TEXT,
  client_order_id TEXT,
  order_list_id TEXT,
  side TEXT,
  order_type TEXT,
  status TEXT,
  payload_json TEXT,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol TEXT,
  score REAL,
  relative_price REAL,
  strategy TEXT,
  classification TEXT,
  payload_json TEXT,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS account_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  equity_btc REAL,
  btc_free REAL,
  usdt_free REAL,
  payload_json TEXT,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS bot_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event TEXT NOT NULL,
  symbol TEXT,
  trade_id TEXT,
  order_id TEXT,
  reason TEXT,
  payload_json TEXT,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_configuration (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  strategy TEXT NOT NULL,
  config_json TEXT NOT NULL,
  frozen_at REAL NOT NULL
);
"""


class BotDatabase:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def new_trade_id(self) -> str:
        return uuid.uuid4().hex

    def freeze_strategy_config(self, strategy: str, config: dict[str, Any]) -> None:
        """Snapshot strategy params — never overwrite historical rows."""
        self._conn.execute(
            "INSERT INTO strategy_configuration(strategy, config_json, frozen_at) VALUES (?,?,?)",
            (strategy, json.dumps(config, sort_keys=True), time.time()),
        )
        self._conn.commit()

    def insert_trade(self, row: dict[str, Any]) -> None:
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO trades(
              trade_id, symbol, strategy, selector, entry_time, entry_price, quantity,
              btc_value, usdt_value, configured_activation, configured_trailing_distance,
              strategy_config_json, binance_entry_order_id, binance_oco_list_id,
              exit_time, exit_price, fees_btc, fees_usdt, realized_pnl_btc, realized_pnl_usdt,
              realized_pnl_btc_equivalent, status, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row["trade_id"],
                row["symbol"],
                row["strategy"],
                row.get("selector"),
                row.get("entry_time"),
                row.get("entry_price"),
                row.get("quantity"),
                row.get("btc_value"),
                row.get("usdt_value"),
                row.get("configured_activation"),
                row.get("configured_trailing_distance"),
                json.dumps(row.get("strategy_config") or {}, sort_keys=True),
                row.get("binance_entry_order_id"),
                row.get("binance_oco_list_id"),
                row.get("exit_time"),
                row.get("exit_price"),
                row.get("fees_btc", 0.0),
                row.get("fees_usdt", 0.0),
                row.get("realized_pnl_btc"),
                row.get("realized_pnl_usdt"),
                row.get("realized_pnl_btc_equivalent"),
                row.get("status", "OPEN"),
                now,
                now,
            ),
        )
        self._conn.commit()

    def update_trade(self, trade_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = []
        vals: list[Any] = []
        for k, v in fields.items():
            if k == "strategy_config":
                cols.append("strategy_config_json=?")
                vals.append(json.dumps(v, sort_keys=True))
            else:
                cols.append(f"{k}=?")
                vals.append(v)
        cols.append("updated_at=?")
        vals.append(time.time())
        vals.append(trade_id)
        self._conn.execute(f"UPDATE trades SET {', '.join(cols)} WHERE trade_id=?", vals)
        self._conn.commit()

    def open_trades(self) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT * FROM trades WHERE status IN (
              'OPEN','PROTECTED','DRY_RUN_PROTECTED','ENTRY_FILLED','ENTRY_PENDING',
              'PROTECTION_PENDING','PROTECTED_EMERGENCY','PROTECTION_FAILED','EXIT_PENDING','DRY_RUN'
            )
            """
        )
        return [dict(r) for r in cur.fetchall()]

    def find_open_trade_for_symbol(self, symbol: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            """
            SELECT * FROM trades WHERE symbol=? AND status IN (
              'OPEN','PROTECTED','DRY_RUN_PROTECTED','ENTRY_FILLED','ENTRY_PENDING',
              'PROTECTION_PENDING','PROTECTED_EMERGENCY','PROTECTION_FAILED','EXIT_PENDING','DRY_RUN'
            ) LIMIT 1
            """,
            (symbol.upper(),),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def update_order_status(
        self,
        *,
        order_id: str | None = None,
        client_order_id: str | None = None,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append a status snapshot row (orders table is append-only history)."""
        self.insert_order(
            order_id=order_id,
            client_order_id=client_order_id,
            status=status,
            payload=payload or {},
        )

    def get_trade(self, trade_id: str) -> dict[str, Any] | None:
        cur = self._conn.execute("SELECT * FROM trades WHERE trade_id=?", (trade_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def insert_order(self, **row: Any) -> None:
        self._conn.execute(
            """
            INSERT INTO orders(trade_id, symbol, order_id, client_order_id, order_list_id,
              side, order_type, status, payload_json, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row.get("trade_id"),
                row.get("symbol"),
                row.get("order_id"),
                row.get("client_order_id"),
                row.get("order_list_id"),
                row.get("side"),
                row.get("order_type"),
                row.get("status"),
                json.dumps(row.get("payload") or {}, sort_keys=True, default=str),
                time.time(),
            ),
        )
        self._conn.commit()

    def insert_signal(self, **row: Any) -> None:
        self._conn.execute(
            """
            INSERT INTO signals(symbol, score, relative_price, strategy, classification, payload_json, created_at)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                row.get("symbol"),
                row.get("score"),
                row.get("relative_price"),
                row.get("strategy"),
                row.get("classification"),
                json.dumps(row.get("payload") or {}, sort_keys=True, default=str),
                time.time(),
            ),
        )
        self._conn.commit()

    def insert_snapshot(self, **row: Any) -> None:
        self._conn.execute(
            """
            INSERT INTO account_snapshots(equity_btc, btc_free, usdt_free, payload_json, created_at)
            VALUES (?,?,?,?,?)
            """,
            (
                row.get("equity_btc"),
                row.get("btc_free"),
                row.get("usdt_free"),
                json.dumps(row.get("payload") or {}, sort_keys=True, default=str),
                time.time(),
            ),
        )
        self._conn.commit()

    def insert_event(self, event: str, **row: Any) -> None:
        self._conn.execute(
            """
            INSERT INTO bot_events(event, symbol, trade_id, order_id, reason, payload_json, created_at)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                event,
                row.get("symbol"),
                row.get("trade_id"),
                row.get("order_id"),
                row.get("reason"),
                json.dumps(row.get("payload") or {}, sort_keys=True, default=str),
                time.time(),
            ),
        )
        self._conn.commit()
