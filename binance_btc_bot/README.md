# Binance BTC Compounding Bot

Clean, self-contained Binance **spot** execution bot for the BTC-compounding strategy.

- **Live strategy (now):** `T1` only  
- **Live selector:** `null` (A–F research-only)  
- **Trailing:** Binance-native OCO (`TAKE_PROFIT` + `trailingDelta` + hard `STOP_LOSS`)  
- **Real trading:** disabled (`live.enabled: false`)

This package does **not** copy MEXC exchange code. Strategy geometry stays exchange-neutral; Binance details live under `exchange/` and `execution/`.

## Quick start (dry-run)

```bash
cd /path/to/BTCC
DRY_RUN=true python -m binance_btc_bot.main --stage all --symbol ETHBTC --demo-score 0.70
```

## Layout

```text
binance_btc_bot/
  config/binance_bot.yaml
  exchange/          # ExchangeAdapter + BinanceExchange
  market_data/       # REST + websocket
  strategy/          # relative price, entries, trails T1–T10, selectors A–F
  risk/              # sizing + independent safety NORMAL/WARNING/HALT
  execution/         # entry, native trailing, recovery, engine
  accounting/        # BTC / USDT / BTC-equivalent P&L
  storage/           # SQLite trades/orders/signals/events
  docs/READINESS_REPORT.md
  tests/
  main.py
```

## T1 → Binance mapping

See `docs/READINESS_REPORT.md`.

```text
aboveType=TAKE_PROFIT  aboveStopPrice=entry*(1+0.75%)  aboveTrailingDelta=25
belowType=STOP_LOSS    belowStopPrice=entry*(1-0.75%)
```

After submission, **Binance owns the trail**. The bot monitors fills / recovers state; it does not simulate trailing locally.

## Safety

- Spot only, no leverage  
- Max planned loss 0.5% equity at hard SL (allocation-capped)  
- Duplicate-order protection + restart recovery (Binance = source of truth for live orders)  
- Engine refuses write-armed construction in this deliverable

## Research compatibility

T1–T10 and selectors A–F remain in config for the existing `btcc` backtest/selector experiments. Live execution never calls selectors until explicitly enabled later.
