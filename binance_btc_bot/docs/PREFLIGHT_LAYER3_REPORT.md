# Layer 3 — Production Preflight / Stage-6 Readiness Report

Generated from `python -m binance_btc_bot.main --preflight`.

**Posture (unchanged):** `LIVE=false` · `DRY_RUN=true` · `STRATEGY=T1` · `SELECTOR=NONE` · `MAX_TRADES=8` · `ALLOCATION=12.5%` · `RISK=0.5%`

**REAL ORDERS: DISABLED** — this preflight never submits BUY or OCO.

## Host constraints (this run)

| Item | Result |
|------|--------|
| Egress IP | `100.27.80.205` |
| Public Vision API | Reachable |
| `api.binance.com` private | **HTTP 451 geo-restricted** |
| Stage-6 ready | **FALSE** until re-run on an eligible network |

Do **not** assume Stage-6 uses `100.27.80.205`. Confirm the live host egress IP and set Binance API key IP allowlist accordingly (`preflight.expected_egress_ips` in config).

## Check summary (this host)

| Check | Status |
|-------|--------|
| Binance connectivity | RESTRICTED (public OK / private 451) |
| Ed25519 authentication | RESTRICTED (local sign OK; signed REST blocked) |
| Account access | RESTRICTED |
| Spot permission | RESTRICTED |
| Withdraw permission | Must be API-key `enableWithdrawals=FALSE` via `/sapi/v1/account/apiRestrictions` (not account `canWithdraw`) |
| 37 symbols | PASS |
| T1 configuration | PASS |
| 8-slot / 12.5% / 0.5% risk | PASS |
| Telegram | PASS |
| SMS | OPTIONAL / NOT CONFIGURED (not a Stage-6 blocker) |
| WebSocket | WARN (`websockets` package missing) |
| Database / Recovery / Protection failure / Fills / Emergency | PASS |
| Dry-run / REAL ORDERS | PASS / DISABLED |

### Withdrawal: account vs API key

- `GET /api/v3/account` → `canWithdraw` is an **account-level capability** field. `canWithdraw=true` is **not sufficient** to conclude this API key has withdrawal permission.
- `GET /sapi/v1/account/apiRestrictions` → `enableWithdrawals` is the **API-key** Enable Withdrawals flag. Stage-6 Withdraw permission requires `enableWithdrawals=false` (UI: Enable Withdrawals = OFF). Fail closed if unreadable.

## T1 OCO (verified against current Spot docs — not dry-run inference)

Endpoint: `POST /api/v3/orderList/oco`

```text
side=SELL
aboveType=TAKE_PROFIT
aboveStopPrice=entry × 1.0075
aboveTrailingDelta=25
belowType=STOP_LOSS
belowStopPrice=entry × 0.9925
quantity=actual filled qty
```

Semantics (Binance FAQ):

1. **Entry** — MARKET BUY fill (actual avg price / qty).
2. **Activation threshold** — trailing tracking starts when last price ≥ `aboveStopPrice`.
3. **Trailing movement** — after activation, a **25 BIP (0.25%) decrease from the post-activation high** triggers MARKET SELL on the above leg.
4. **Hard stop** — `belowStopPrice` STOP_LOSS SELL if price ≤ entry×0.9925.

### Research vs Binance (documented, research unchanged)

Research ratchets the stop after activation and effectively replaces the initial SL with the trail. **Binance OCO keeps both legs until one fills**; after activation the trail is the economic exit while the hard SL remains a crash safety net.

ETHBTC `TRAILING_DELTA` filter admits 25 BIPS (`minTrailingAboveDelta=10`, `max=2000`).

## Equity formula

```text
total_equity_btc = btc_free + btc_locked + stables_as_btc + mapped_alts_btc
available_btc    = btc_free
trading_capital  = total_equity_btc   # used for 12.5% alloc + 0.5% risk
```

**Forbidden:** using available USDT alone as total equity.

## Risk worked example (equity = 1 BTC)

| Quantity | Value |
|----------|-------|
| Risk budget (0.5%) | 0.005 BTC |
| T1 hard SL | 0.75% |
| Max notional by risk | 0.005 / 0.0075 ≈ 0.6667 BTC |
| Max notional by allocation (12.5% − fee buffer) | 0.12475 BTC |
| **Binding constraint** | **ALLOCATION** |
| Planned loss at hard SL | ≈ 0.000936 BTC ≈ **0.094% of equity** (< 0.5%) |
| 8 × 12.5% | 100% max allocation |

`max_loss_per_trade=0.005` means **maximum planned monetary loss at the hard SL**, **not** stop distance = 0.5%.

## Stage-6 gate

Re-run on an eligible network until:

```text
Ed25519 authentication     PASS
Account access             PASS
Spot permission            PASS
Withdraw permission        PASS (enableWithdrawals=FALSE)
STAGE6_READY               TRUE
```

Still keep `LIVE=false` / `DRY_RUN=true` until an explicit authorization to submit the first real order.
