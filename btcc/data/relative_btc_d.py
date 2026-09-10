"""Relative BTC.D proxy from top-coin market caps (CoinGecko free-tier path)."""

from __future__ import annotations

from typing import Any

TOP_COIN_IDS: list[str] = [
    "bitcoin",
    "ethereum",
    "tether",
    "binancecoin",
    "solana",
    "ripple",
    "usd-coin",
    "staked-ether",
    "dogecoin",
    "cardano",
    "tron",
    "avalanche-2",
    "chainlink",
    "the-open-network",
    "shiba-inu",
    "polkadot",
    "bitcoin-cash",
    "near",
    "uniswap",
    "litecoin",
]


def compute_relative_btc_d_pct(caps: dict[str, float]) -> tuple[float | None, dict[str, Any]]:
    """BTC.D_relative = 100 * BTC_cap / sum(available top-N caps)."""
    btc = caps.get("bitcoin")
    if btc is None or float(btc) <= 0:
        return None, {"status": "missing_bitcoin", "n_coins": 0}
    total = 0.0
    used = 0
    for _k, v in caps.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > 0:
            total += fv
            used += 1
    if total <= 0:
        return None, {"status": "empty_caps", "n_coins": 0}
    pct = 100.0 * float(btc) / total
    return pct, {"status": "ok", "n_coins": used, "total_cap": total}
