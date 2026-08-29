# BTCC — BTC Relative-Strength Signal Bot

**SIGNAL ONLY.** No buy, sell, order, position, stop, take-profit, or leverage.  
This project is a **prediction / research signal system**, not an execution bot.  
There are **no MEXC trading API keys** — market data is public REST only.

---

## Requirements

- **Python 3.11+** (developed on 3.11)
- Network access to:
  - MEXC public REST (`https://api.mexc.com`)
  - CoinGecko global (BTC dominance)
  - Telegram Bot API (optional, for notifications)

---

## Install

```powershell
cd BTCC
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt

# Create a local .env (never commit it) with:
#   BTCC_TELEGRAM_BOT_TOKEN=...
#   BTCC_TELEGRAM_CHAT_ID=...
# Optional:
#   BTCC_COINGECKO_API_KEY=...
```

Dev/tests only:

```powershell
pip install -r requirements-dev.txt
pytest tests/ -q
```

---

## Environment variables

Put these in a local **`.env`** file in `BTCC/` (never commit `.env`).

| Variable | Required | Purpose |
|----------|----------|---------|
| `BTCC_TELEGRAM_BOT_TOKEN` | For Telegram | Telegram bot token |
| `BTCC_TELEGRAM_CHAT_ID` | For Telegram | Destination chat id |
| `BTCC_COINGECKO_API_KEY` | Optional | Historical BTC.D backtests (Demo/Pro) |
| `COINGECKO_API_KEY` | Optional | Alias for the same CoinGecko key |

Without Telegram variables, the bot still runs; messages are logged locally and Telegram stays disabled.

**There are no BTCC/MEXC trading API key env vars** — trading is forbidden by design.

---

## Run

From the `BTCC/` directory (venv activated):

```powershell
# Preflight (secrets hygiene + trading disabled)
python scripts/check_deploy.py

# Download / refresh candle history + universe audit
python -m btcc.main bootstrap

# Seed Champion v1 + import 90d historical labels (once, if using adaptive learning)
python -m btcc.main adaptive-bootstrap

# Single 15-minute cycle (audit + optional Telegram)
python -m btcc.main once

# Continuous loop (every 15m) — run exactly ONE process
python -m btcc.main run

# Manual adaptive checkpoint (does not blindly change weights)
python -m btcc.main adaptive-checkpoint

# Research report / calibration fit from stored predictions
python -m btcc.main report
```

---

## What it does (every 15 minutes)

1. Updates local 15m candles from **MEXC** (public REST)
2. Builds **ALT/BTC = ALTUSDT / BTCUSDT** for a fixed universe of 20 alts
3. Computes factor groups → **signal score** (not a probability)
4. Maps score → baseline / calibrated **P(outperform BTC)** for 1h / 4h / 8h / 12h / 24h
5. **Ranks by 4h probability**
6. Computes separate **Late Entry / Exhaustion** score
7. Sends Telegram ranking (+ exhaustion alerts when LateEntry ≥ 0.75)
8. Stores predictions for later calibration / adaptive learning

---

## Configuration vs secrets

| Kind | Location |
|------|----------|
| Strategy / factors / universe | `configs/signal_config.yaml` |
| Adaptive learning settings | `configs/adaptive_config.yaml` |
| Secrets | `.env` only (never commit) |

Do not put tokens or keys in YAML.

---

## Runtime data (not committed)

These are created on the host and are **gitignored**:

- `data/candles/` — OHLCV cache
- `data/predictions/` — prediction CSVs, dominance history, calibration
- `data/models/` — Champion / Challenger models
- `data/backtest_*` — offline research caches
- `logs/` — audits and adaptive reports
- `results/` — backtest outputs

---

## Private Git checklist

1. Confirm `.env` is **not** staged (`git status` must never show `.env`).
2. Run `python scripts/check_deploy.py`.
3. Rotate any Telegram token that was ever shared or committed historically.
4. Push only to a **private** remote.
5. Run a **single** `python -m btcc.main run` under a process supervisor (no duplicate instances).

---

## Safety

- `safety.allow_trading: false` is required
- `deny_trading()` / order method names are blocked
- No order placement code path
- Public market data + Telegram notifications only

---

## Architecture (high level)

```
MEXC public REST → candles → ALT/BTC series
  → Momentum / Trend / BTC Regime / Volume / Volatility / RSI / Structure
  → signal_score → probability (baseline or calibrated)
  → rank by 4h P
  → Telegram Top-5 + indicator breakdown
  → predictions (+ delayed ALT/BTC outcome labels)
  → optional Champion / Challenger adaptive learning
```

---

## Important

This is a **forecasting system**, not a trade-execution bot.

1. *How likely is this coin to outperform BTC over 1h–24h?*  
2. *Even if that probability is high, how extended / late is the entry?*
