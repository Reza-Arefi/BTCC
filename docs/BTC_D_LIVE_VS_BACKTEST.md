# BTC.D live vs backtest — parity verification

Generated: `2026-08-29T23:02:26.414300+00:00`

## Verdict

- Historical contamination (present-day scale): **fixed**
- Absolute BTC.D: **unavailable** (free tier)
- Relative proxy: **implemented**
- Live uses same definition as backtest: **`True`**
- 15m alignment / no future leak: **`True`**
- Stale BTC.D blocks new simulated trades: **`True`**

## Formula (identical)

```
BTC.D_relative(t) = 100 * BTC_market_cap(t) / sum_i market_cap_i(t) over available coins in fixed TOP_COIN_IDS; calibration=none
```

- Shared helper: `btcc.data.relative_btc_d.compute_relative_btc_d_pct`
- Shared universe size: 21 CoinGecko ids
- Calibration: `none_no_present_day_scaling`

## Answers

| Question | Answer |
|----------|--------|
| Historical formula | relative top-N proxy |
| Live formula | relative top-N proxy (same) |
| Identical? | **yes** |
| Live data source | `CoinGecko GET /api/v3/coins/markets?ids=<TOP_COIN_IDS>&vs_currency=usd` |
| Live update frequency | each 15m cycle; soft poll 300s |
| Daily→15m mapping | last obs with `timestamp <= decision_ts` |
| Future obs can enter earlier prediction? | **no** |
| Missing top-N mcap | skip coin; require BTC + ≥8 coins |
| API failure | keep last ≤ t; else unavailable → block new trades |
| Stale max age (`BTC_D_MAX_AGE`) | **7200 s (2 h)** — observation age > 7200s ⇒ STALE ⇒ block NEW trades |
| Stale blocks new trades? | **yes** (`require_for_new_trades`) |
| Stored on every prediction? | **yes** (`btc_d_status`, `btc_d_age_seconds`, `btc_dominance`) |

## Historical vs live resolution

| Context | Resolution | Notes |
|---------|------------|-------|
| **1-year backtest** | **Daily** relative proxy (~365 points) | Acceptable for primary experiment |
| **Live / short backtest** | **~Hourly** relative proxy | Higher-frequency same formula |

Both paths: `observation_at(t)` returns last obs with `timestamp <= t`; **no interpolation**; **no present-day calibration**.
Both are **relative BTC dominance proxy**, not official absolute BTC.D.

## Live probe

```json
{
  "ok": true,
  "pct": 63.45665387610847,
  "obs_ts": "2026-08-29T23:00:00+00:00",
  "source": "coingecko_top_coins_relative",
  "meta": {
    "source": "coingecko_top_coins_relative",
    "representation": "relative_btc_share_of_top_n",
    "calibration": "none_no_present_day_scaling",
    "coins_requested": [
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
      "the-open-network"
    ],
    "coins_used": [
      "avalanche-2",
      "binancecoin",
      "bitcoin",
      "bitcoin-cash",
      "cardano",
      "chainlink",
      "dogecoin",
      "ethereum",
      "hyperliquid",
      "litecoin",
      "polkadot",
      "ripple",
      "solana",
      "stellar",
      "sui",
      "tether",
      "the-open-network",
      "tron",
      "uniswap",
      "usd-coin"
    ],
    "n_coins_used": 20,
    "bitcoin_present": true,
    "status": "OK",
    "btc_market_cap": 1570518817920.0,
    "total_top_n_cap": 2474947419992.0,
    "btc_dominance_pct": 63.45665387610847
  }
}
```

## Historical cache compare

```json
{
  "available": true,
  "cache_days": 365,
  "source": "coingecko_top_coins_relative",
  "calibration": "none_no_present_day_scaling",
  "representation": "relative_btc_share_of_top_n",
  "last_ts": "2026-08-29 00:00:00+00:00",
  "last_pct": 63.43318092529729,
  "live_pct": 63.45665387610847,
  "abs_diff_pp": 0.023472950811175508,
  "note": "Live (/markets now) vs last historical market_chart point can differ by hours/days of market move; definition/source/calibration must match."
}
```

## Optional next experiment

On the 1-year walk-forward, compare:

- **Model A**: adaptive indicators + BTC.D proxy
- **Model B**: adaptive indicators **without** BTC.D

Same everything else → measure whether the proxy helps, is neutral, or hurts.
