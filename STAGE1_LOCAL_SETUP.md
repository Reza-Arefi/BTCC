# Stage 1 — Local Binance bot copy (dry-run only)

Source AWS commit: `a5021286df21c6a58c84fc2bf9b95bf162e7ca8c`
Source tip: a502128 Fail closed on exchangeInfo gaps and refresh MARKET capability every entry.
Export time (UTC): 2026-09-06T13:42:38Z

## Frozen config (unchanged)
- LIVE=false / dry_run=true
- STRATEGY=T1 / SELECTOR=NONE
- MAX_SIMULTANEOUS_TRADES=8
- ALLOCATION_PER_TRADE=12.5%
- RISK CEILING=0.5%
- THRESHOLD=0.65

## Setup
```bash
cd BTCC
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r binance_btc_bot/requirements.txt
export PYTHONPATH=\"$(pwd):${PYTHONPATH}\"
python -m unittest discover -s binance_btc_bot/tests -q
DRY_RUN=true python -m binance_btc_bot.main --preflight
```

## Secrets
- No AWS `.env` or Ed25519 private key is included.
- Do not create/configure API keys in Stage 1.
