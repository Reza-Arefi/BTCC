"""Universe resolution + audit table for the fixed 20-pair research set."""

from __future__ import annotations

from typing import Any

import pandas as pd


def resolve_usdt_symbol(base: str, cfg: dict[str, Any]) -> str:
    overrides = cfg.get("universe", {}).get("symbol_overrides") or {}
    if base in overrides:
        return str(overrides[base])
    return f"{base}{cfg['universe']['quote']}"


def resolve_btc_market(base: str, cfg: dict[str, Any]) -> str:
    markets = cfg.get("universe", {}).get("btc_markets") or {}
    if base in markets:
        return str(markets[base])
    return f"{base}BTC"


def usdt_symbols(cfg: dict[str, Any]) -> list[str]:
    return [resolve_usdt_symbol(b, cfg) for b in cfg["universe"]["bases"]]


def build_universe_audit(
    cfg: dict[str, Any],
    panels: dict[str, pd.DataFrame],
    unavailable: set[str] | None = None,
    construction: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """One row per configured base with coverage / construction metadata."""
    unavailable = unavailable or set()
    construction = construction or {}
    rows = []
    for base in cfg["universe"]["bases"]:
        usdt = resolve_usdt_symbol(base, cfg)
        btc_mkt = resolve_btc_market(base, cfg)
        mode = construction.get(base, f"{usdt} / BTCUSDT")
        # Prefer panel keyed by USDT; native BTC fallbacks stored under btc market key
        df = panels.get(usdt)
        if df is None:
            df = panels.get(btc_mkt)
        n = int(len(df)) if df is not None else 0
        first = last = None
        if df is not None and n > 0 and "timestamp" in df.columns:
            first = str(df["timestamp"].iloc[0])
            last = str(df["timestamp"].iloc[-1])
        status = "OK"
        if usdt in unavailable and btc_mkt in unavailable:
            status = "DATA_UNAVAILABLE"
        elif df is None or n == 0:
            status = "DATA_UNAVAILABLE"
        elif n < 100:
            status = "INSUFFICIENT_BARS"
        rows.append({
            "symbol": base,
            "mexc_btc_market": btc_mkt,
            "usdt_pair": usdt,
            "btc_relative_construction": mode,
            "n_15m_candles": n,
            "first_candle": first,
            "last_candle": last,
            "status": status,
        })
    return rows


def format_universe_audit(rows: list[dict[str, Any]]) -> str:
    lines = [
        "=== UNIVERSE AUDIT (exact 20) ===",
        f"{'Symbol':<8} {'MEXC BTC':<12} {'USDT pair':<14} {'Construction':<32} {'N15m':>6} Status",
    ]
    for r in rows:
        lines.append(
            f"{r['symbol']:<8} {r['mexc_btc_market']:<12} {r['usdt_pair']:<14} "
            f"{r['btc_relative_construction']:<32} {r['n_15m_candles']:>6} {r['status']}"
        )
    return "\n".join(lines)
