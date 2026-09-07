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


def uses_native_btc_market(base: str, cfg: dict[str, Any]) -> bool:
    """True when the base has no USDT listing and must use the native *BTC market."""
    native_only = cfg.get("universe", {}).get("native_btc_only") or []
    return base in {str(x) for x in native_only}


def resolve_data_market(base: str, cfg: dict[str, Any]) -> dict[str, str]:
    """Canonical OHLCV fetch target for a logical research pair.

    Shared by historical download, download_panels, backtest, validator, and live
    bootstrap. Native-only bases resolve to the BTC market *before* any API call
    (never probe an invalid USDT symbol).

    Returns:
      logical_pair: research base (e.g. CKBTC)
      resolved_market: exchange symbol to fetch (e.g. CKBTCBTC)
      construction: how ALT/BTC relative series is built
      mode: ``native_btc`` | ``synthetic_usdt``
    """
    logical = str(base)
    btc_mkt = resolve_btc_market(logical, cfg)
    if uses_native_btc_market(logical, cfg):
        return {
            "logical_pair": logical,
            "resolved_market": btc_mkt,
            "construction": f"native {btc_mkt}",
            "mode": "native_btc",
        }
    usdt = resolve_usdt_symbol(logical, cfg)
    return {
        "logical_pair": logical,
        "resolved_market": usdt,
        "construction": f"{usdt} / BTCUSDT",
        "mode": "synthetic_usdt",
    }


def usdt_symbols(cfg: dict[str, Any]) -> list[str]:
    """USDT (or native) fetch symbols for the universe — one per base."""
    return [resolve_data_market(b, cfg)["resolved_market"] for b in cfg["universe"]["bases"]]


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
        meta = resolve_data_market(base, cfg)
        usdt = resolve_usdt_symbol(base, cfg)
        btc_mkt = resolve_btc_market(base, cfg)
        mode = construction.get(base, meta["construction"])
        # Prefer panel keyed by resolved market; fall back to USDT / BTC keys
        df = panels.get(meta["resolved_market"])
        if df is None:
            df = panels.get(usdt)
        if df is None:
            df = panels.get(btc_mkt)
        n = int(len(df)) if df is not None else 0
        first = last = None
        if df is not None and n > 0 and "timestamp" in df.columns:
            first = str(df["timestamp"].iloc[0])
            last = str(df["timestamp"].iloc[-1])
        status = "OK"
        if meta["resolved_market"] in unavailable or (
            usdt in unavailable and btc_mkt in unavailable
        ):
            status = "DATA_UNAVAILABLE"
        elif df is None or n == 0:
            status = "DATA_UNAVAILABLE"
        elif n < 100:
            status = "INSUFFICIENT_BARS"
        rows.append({
            "symbol": base,
            "logical_pair": meta["logical_pair"],
            "resolved_market": meta["resolved_market"],
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
        f"{'Symbol':<8} {'Resolved':<12} {'Construction':<32} {'N15m':>6} Status",
    ]
    for r in rows:
        lines.append(
            f"{r['symbol']:<8} {r.get('resolved_market', r.get('mexc_btc_market', '')):<12} "
            f"{r['btc_relative_construction']:<32} {r['n_15m_candles']:>6} {r['status']}"
        )
    return "\n".join(lines)
