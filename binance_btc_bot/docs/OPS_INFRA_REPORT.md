# Operational Infrastructure Report

**Date:** 2026-09-06  
**Phase:** Ops readiness (no Stage 6 / no micro-live)

## Final state (mandatory)

```text
LIVE ENABLED = FALSE
REAL ORDERS  = DISABLED
STRATEGY     = T1
SELECTOR     = NONE
PROVIDER     = FixedStrategyProvider
```

---

## Binance configuration

| Item | Value |
|------|--------|
| Credentials source | `BINANCE_API_KEY` / `BINANCE_API_SECRET` env only |
| YAML secrets | None |
| Startup validation | Required only for signed modes; dry/public path allows missing keys |
| Masked status | `SET` / `MISSING` only — never prints values |
| Withdrawal permission | Not required / not used (spot trading endpoints only) |
| Live with credentials | **Still impossible** while `live.enabled=false` |
| Write gate | `dry_run` OR `!live_enabled` → orders blocked (`LIVE_DISABLED` / `DRY_RUN`) |

Template: `binance_btc_bot/.env.example`

---

## Telegram configuration

| Item | Value |
|------|--------|
| Env | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` |
| Role | Normal + important + critical operational channel |
| Failure mode | Log warning; **never crash** trading engine |
| Current host status | `NOT_CONFIGURED` (env unset) — expected |

Events: `BOT_STARTED`, `BOT_STOPPED`, `SIGNAL_DETECTED`, `ENTRY_SUBMITTED`, `ENTRY_FILLED`, `TRAILING_OCO_SUBMITTED`, `TRAILING_ACTIVATED`, `EXIT_FILLED`, `DAILY_SUMMARY`, plus `RECOVERY` / `API_ERROR` / `ORDER_ERROR` / `CANCEL_ERROR`.

---

## SMS configuration

| Item | Value |
|------|--------|
| Abstraction | `SMSProvider` → `TwilioSMSProvider` (or `NullSMSProvider`) |
| Env | `SMS_PROVIDER=twilio`, `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, `TWILIO_TO_NUMBER` (or `SMS_*` aliases) |
| Role | **CRITICAL only** (Telegram + SMS) |
| Examples | `HALT`, `UNPROTECTED_POSITION`, `OCO_FAILURE`, `RECOVERY_FAILURE`, account/order state errors |
| Failure mode | Log warning; never crash engine |
| Stage-6 | **Optional** — SMS/Twilio credentials are **not** required for `STAGE6_READY` |

## Withdrawal least-privilege

| Field | Source | Meaning |
|------|--------|---------|
| `canWithdraw` | `GET /api/v3/account` | **Account** capability (often true) — **not** the API-key gate |
| `enableWithdrawals` | `GET /sapi/v1/account/apiRestrictions` | **API-key** Enable Withdrawals — must be **false** for trading keys |

Preflight Withdraw permission checks `enableWithdrawals` only (fail closed if unreadable).

Policy (config):

```yaml
notifications.policy:
  INFO: [telegram]
  WARNING: [telegram]
  ERROR: [telegram]
  CRITICAL: [telegram, sms]
```

---

## Strategy provider

```text
StrategyProvider (ABC)
  └── FixedStrategyProvider  →  get_strategy() = T1, selector = None
```

Execution engine calls only:

```python
strategy = strategy_provider.get_strategy(...)
```

Future `SelectorStrategyProvider` (A–F) can replace Fixed without rewriting execution.

Research/backtest T1–T10 / A–F definitions were **not modified**.

---

## Notification / safety / secret tests

```text
34 passed (binance_btc_bot/tests/)
```

Includes:

- missing key / missing secret / credentials present / never logged
- `live.enabled=false` blocks order submission
- Telegram: valid (mocked), API failure, missing config
- SMS: valid (mocked), API failure, missing config
- critical → Telegram + SMS; info → Telegram only
- notification channel exception does not crash manager
- `FixedStrategyProvider → T1` wired into entry decisions
- secret scrubbing for env tokens and `api_key=` patterns

---

## Dry-run result

Command:

```bash
DRY_RUN=true python -m binance_btc_bot.main --stage all --symbol ETHBTC --demo-score 0.70
```

**Result (this host):** all stages `ok=true`

| Stage | ok |
|-------|----|
| stage1_connectivity | true |
| stage2_market_data | true |
| stage3_signal | true |
| stage4_risk | true |
| stage5_order_validation_dry | true |
| recovery | true |
| dry_pipeline | true |

Stage 5 confirms: strategy=`T1`, selector=`None`, `trail_bips=25`, `live_enabled=false`, trailing blocked with `DRY_RUN` / `BINANCE_NATIVE`.

Status command:

```bash
python -m binance_btc_bot.main --status
```

Shows connection OK, live DISABLED, dry TRUE, strategy T1, selector NONE, Telegram/SMS NOT_CONFIGURED (until env set), `STATUS: SAFE`.

---

## Explicitly not done (per request)

- Stage 6 native trailing live fill/cancel
- Micro-live / real orders
- Enabling `live.enabled`
- Changing T1–T10 or selector research definitions
