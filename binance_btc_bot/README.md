# Binance BTC Bot (package)

Authoritative production config: `config/binance_bot.yaml`  
Fingerprint: **T30_E2_T65_v1** (T30 + E2 + 0.65 NEW CROSS, late-entry OFF).

See the **repository root README.md** for installation, dry-run, kill switch, and archive docs.

```bash
python -m binance_btc_bot --check-config
python -m binance_btc_bot --stage dry
```

`LIVE_TRADING_ENABLED` defaults to **false**. No live orders without explicit authorization.
