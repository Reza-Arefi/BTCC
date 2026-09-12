# BTCC — Binance BTC Compounding Bot

Frozen production identity: **T30 + E2 + threshold 0.65 + NEW CROSS + late-entry OFF**.

**LIVE TRADING IS DISABLED BY DEFAULT. Do not enable without explicit authorization.**

## 1. Installation

```bash
git clone https://github.com/Reza-Arefi/BTCC.git
cd BTCC
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Python version

Python **3.11+** recommended (3.10+ should work).

## 3. Dependencies

See `requirements.txt` (pandas, numpy, PyYAML, requests, websockets, cryptography, matplotlib, …).

## 4. Environment variables

Copy the template (names only — no real secrets):

```bash
cp .env.example .env
```

Critical variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `LIVE_TRADING_ENABLED` | `false` | Hard kill switch. Must be `true` before any real order. |
| `DRY_RUN` | `true` | Force simulated execution. |
| `BINANCE_API_KEY` | empty | Binance API key id (Ed25519). |
| `BINANCE_ED25519_PRIVATE_KEY_PATH` | empty | Path to private PEM (never commit). |

Optional: Telegram / Twilio SMS — see `.env.example`.

## 5. Configuration

**Authoritative runtime config:**

`binance_btc_bot/config/binance_bot.yaml`

**Discovery / fingerprint document:**

`configs/live_production.yaml`

Frozen values (`config_version = T30_E2_T65_v1`):

| Setting | Value |
|---------|-------|
| Strategy | T30 |
| Entry profile | E2 |
| Threshold | 0.65 |
| Entry rule | NEW CROSS |
| Late-entry | OFF |
| Selector | null |
| T30 SL / act / trail | 3% / 1% / 0.25% |
| Max open | 8 × 12.5% |

Do **not** enable B1/D2, T60, intrabar entry, late-entry veto, or live selectors without a new validation cycle.

## 6. Dry-run / paper mode (safe)

Market data + simulated orders — **no exchange writes**:

```bash
python -m binance_btc_bot --stage dry
python -m binance_btc_bot --stage all
```

## 7. Market-data / diagnostics only

```bash
python -m binance_btc_bot --status
python -m binance_btc_bot --signal-diag
python -m binance_btc_bot --preflight
python -m binance_btc_bot --live3-preflight
```

## 8. Verify configuration (no orders)

```bash
python -m binance_btc_bot --check-config
```

Must report `STATUS: OK` and:

`LIVE TRADING DISABLED — NO ORDERS WILL BE SUBMITTED`

## 9. How to enable live trading later

**Do not do this until the operator explicitly authorizes.**

All of the following are required:

1. Set `LIVE_TRADING_ENABLED=true` in `.env`
2. Set `live.enabled: true` and `live.dry_run: false` in the YAML (or use the LIVE-3 arm path)
3. Set `BINANCE_LIVE3_AUTHORIZED=true`
4. Pass preflight: `python -m binance_btc_bot --live3-preflight`
5. Arm only after PASS:

```bash
python -m binance_btc_bot --live3-arm --authorize-live
```

## 10. How to stop the bot

- Ctrl+C in the terminal
- Telegram control (dry): `/pause`, `/stop`, `/emergency`
- Set `LIVE_TRADING_ENABLED=false` and restart (orders blocked even if YAML says enabled)

## 11. Data archive location

Immutable daily folders:

`results/live_archive/YYYY-MM-DD/`

Contents (when the bot runs):

- `signals.csv`, `entry_factors.jsonl`
- `orders.csv`, `trades.csv`
- `daily_summary.json`, `metadata.json`

SQLite operational DB (gitignored): `data/binance_btc_bot/bot.sqlite3`

## 12. Safety warnings

- Default construction **never** submits real orders.
- Order layer refuses writes unless `LIVE_TRADING_ENABLED=true` **and** dry_run is false **and** live is enabled **and** an explicit arm path is used.
- Never commit `.env`, PEMs, API secrets, or account dumps.
- Research tools under `binance_btc_bot/tools/` do not change live config.

## Quick commands

```bash
python -m binance_btc_bot --check-config
python -m binance_btc_bot --stage dry
python -m binance_btc_bot --status
```
