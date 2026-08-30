"""Relative BTC.D proxy — shared by backtest and live.

Definition (identical in both paths):

    BTC.D_relative(t) = 100 * BTC_market_cap(t) / sum_i market_cap_i(t)

where the denominator is the sum of available market caps among a fixed
top-N CoinGecko id list (skip missing ids; require bitcoin present).

- NOT absolute global BTC dominance.
- NO present-day /global level calibration.
- NO interpolation.
- Missing coin → skip that coin (same rule live and historical).
"""

from __future__ import annotations

from typing import Iterable

# Fixed universe for the relative proxy (must stay identical live vs backtest).
# Order: BTC first, then large caps. Invalid / renamed ids are skipped at fetch.
TOP_COIN_IDS: tuple[str, ...] = (
    "bitcoin",
    "ethereum",
    "tether",
    "ripple",
    "binancecoin",
    "solana",
    "usd-coin",
    "dogecoin",
    "cardano",
    "tron",
    "chainlink",
    "avalanche-2",
    "bitcoin-cash",
    "litecoin",
    "polkadot",
    "uniswap",
    "stellar",
    "hyperliquid",
    "sui",
    "toncoin",
    "the-open-network",
)

RELATIVE_SOURCE = "coingecko_top_coins_relative"
RELATIVE_REPRESENTATION = "relative_btc_share_of_top_n"
MIN_COINS_FOR_VALID = 8  # including bitcoin


def compute_relative_btc_d_pct(
    market_caps: dict[str, float],
    *,
    min_coins: int = MIN_COINS_FOR_VALID,
) -> tuple[float | None, dict]:
    """Compute relative BTC.D from a {coin_id: market_cap_usd} map.

    Returns (pct_or_None, meta).
    """
    caps = {
        k: float(v)
        for k, v in market_caps.items()
        if v is not None and float(v) > 0
    }
    meta: dict = {
        "source": RELATIVE_SOURCE,
        "representation": RELATIVE_REPRESENTATION,
        "calibration": "none_no_present_day_scaling",
        "coins_requested": list(TOP_COIN_IDS),
        "coins_used": sorted(caps.keys()),
        "n_coins_used": len(caps),
        "bitcoin_present": "bitcoin" in caps,
    }
    if "bitcoin" not in caps:
        meta["status"] = "BTC_MCAP_MISSING"
        return None, meta
    if len(caps) < min_coins:
        meta["status"] = f"TOO_FEW_COINS_{len(caps)}<{min_coins}"
        return None, meta
    total = sum(caps.values())
    if total <= 0:
        meta["status"] = "ZERO_TOTAL_CAP"
        return None, meta
    pct = 100.0 * caps["bitcoin"] / total
    meta["status"] = "OK"
    meta["btc_market_cap"] = caps["bitcoin"]
    meta["total_top_n_cap"] = total
    meta["btc_dominance_pct"] = pct
    return pct, meta


def missing_top_coin_ids(present: Iterable[str]) -> list[str]:
    have = set(present)
    return [c for c in TOP_COIN_IDS if c not in have]
