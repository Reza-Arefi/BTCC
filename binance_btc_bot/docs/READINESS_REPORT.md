# Binance BTC Bot — Execution Readiness Report

**Date:** 2026-09-06  
**Live strategy:** `T1` only (`LIVE_SELECTOR = null`)  
**Real trading:** **DISABLED** (`live.enabled: false`, dry-run default)

This report does **not** declare the bot “ready for live trading.” It documents Stage 1–7 progress, the Binance-native trailing mapping, and remaining gated work (signed API from an eligible region + Stage 8 micro live test).

---

## 1. Binance API / order type used for native trailing

| Item | Value |
|------|--------|
| Market | Spot (no leverage) |
| Exit construct | `POST /api/v3/orderList/oco` |
| Above leg | `TAKE_PROFIT` SELL with `aboveStopPrice` + `aboveTrailingDelta` |
| Below leg | `STOP_LOSS` SELL with `belowStopPrice` |
| Trail ownership | **Binance** after OCO submission (bot does not ratchet stops locally) |
| Docs | [Trailing Stop FAQ](https://github.com/binance/binance-spot-api-docs/blob/master/faqs/trailing-stop-faq.md), REST `orderList/oco` |

`trailingDelta` is specified in **BIPS** (1 BIP = 0.01%).

Public market data uses `https://data-api.binance.vision` (reachable from this host).  
Signed trading REST `https://api.binance.com` returns **HTTP 451** from this AWS region — private Stage 6 fill/cancel tests must run from an eligible network.

---

## 2. Exact T1 → Binance parameter mapping

Research T1 (unchanged):

| Field | Value |
|-------|------:|
| `arm_sl_activation_trail` (hard SL) | 0.75% |
| `activation` | 0.75% |
| `trail_distance` | 0.25% |

After a MARKET BUY fill at `entry`:

```text
aboveType            = TAKE_PROFIT
aboveStopPrice       = entry * (1 + 0.0075)    # activation
aboveTrailingDelta   = 25                     # 0.25% in BIPS
belowType            = STOP_LOSS
belowStopPrice       = entry * (1 - 0.0075)    # initial hard SL
side                 = SELL
quantity             = filled base qty
```

Code: `binance_btc_bot/strategy/trails.py` → `map_trail_to_binance_oco()`  
Submit path: `binance_btc_bot/execution/trailing.py` (no local trailing engine).

---

## 3. Binance limitations (explicit — not silently patched)

1. **OCO keeps both legs** until one fills. Research T1 replaces the initial SL with the trail after activation; Binance leaves the hard SL as a crash safety net below the trail. Economics after activation are trail-led; geometry is documented, not rewritten in research configs.
2. **Integer BIPS only** — 0.25% → exactly 25 BIPS (no rounding error for T1).
3. **TRAILING_DELTA filter** on all 37 pairs: `min=10`, `max=2000` BIPS → T1’s 25 is in range. If a symbol ever rejected the delta, the bot **HALTs / rejects** rather than raising the trail to the exchange minimum.
4. **Tick snap** may adjust `stopPrice` by &lt;1 tick; noted in mapping `constraint_notes`.
5. **Geo restriction** on signed `api.binance.com` from this host (451) — cannot complete live Stage 6/8 here.
6. **Relative signal ≠ native `*BTC` mid** — small basis vs `BASEUSDT/BTCUSDT` is expected; execution still uses the `*BTC` book.

---

## 4. Universe validation (37 pairs)

**Stage 2 result (Vision public API):** `validated_symbols=37`, `failed=[]`, all required USDT legs present (`n_prices=75` including BTC pairs + USDT legs + BTCUSDT).

Relative price: `BASEUSDT / BTCUSDT` for every base.

---

## 5. Risk-sizing validation

**Stage 4 (ETHBTC, equity=1 BTC):**  
`risk_budget_btc=0.005`, allocation-capped notional ≈ `0.25 BTC`, `planned_loss_btc≈0.001875` (&lt; 0.5% budget). Unit tests cover uncapped risk math and allocation cap.

---

## 6. Dry-run result

**Stage 5 + dry pipeline:** PASS  

Example native trailing params (ETHBTC dry):

```text
aboveType=TAKE_PROFIT  aboveStopPrice=0.0316  aboveTrailingDelta=25
belowType=STOP_LOSS    belowStopPrice=0.03113
mode=BINANCE_NATIVE
blocked_reason=DRY_RUN
```

`live_enabled=false` confirmed in stage output.

Command:

```bash
cd /home/ubuntu/LCADAME/BTCC
DRY_RUN=true python -m binance_btc_bot.main --stage all --symbol ETHBTC --demo-score 0.70
```

---

## 7. Restart / recovery test

- Unit test: local OPEN trade + matching exchange OCO → `RECOVERY` / `MATCHED_OPEN_OCO` — **PASS**
- Unit test: duplicate OCO lists on one symbol → `HALT` / `DUPLICATE_ORDER_PROTECTION` — **PASS**
- Dry recovery skip of private endpoints — expected when unsigned / dry
- Live recovery against Binance private endpoints: **PENDING** (signed API geo-blocked on this host, HTTP 451)

Principle: **Binance is source of truth** for open orders; local DB is reconciled, not trusted blindly.

---

## 8. Duplicate-order protection test

**PASS (unit):** Safety halt + recovery duplicate OCO detection.

---

## 9. Native trailing-order test

| Check | Status |
|-------|--------|
| Mapping T1 → OCO params | **PASS** (unit) |
| Dry submit path / logging `mode=BINANCE_NATIVE` | **PASS** (unit + Stage 5) |
| Real order create / activate / trail / fill / cancel | **PENDING** (signed API geo-blocked) |

---

## 10. Confirmation: real trading remains disabled

```yaml
live:
  enabled: false
  strategy: T1
  selector: null
  dry_run: true
```

Stage 1 reports `live_enabled: false`. Engine construction refuses write-armed mode in this deliverable.  
Selectors A–F and T2–T10 remain in config for research/backtest only.

### Stage summary (this host)

| Stage | Result |
|-------|--------|
| 1 Connectivity (public) | PASS |
| 2 Market data (37 + USDT) | PASS |
| 3 Relative signal sample | PASS |
| 4 Risk sizing | PASS |
| 5 Order validation dry + native trail params | PASS |
| 6 Live native trail fill/cancel | PENDING (451) |
| 7 Dry-run pipeline | PASS |
| 8 Small live test | NOT RUN (gated) |

---

## Architecture

```text
Market data → Strategy/Entry (T1 fixed) → Risk/Safety → Live engine
                                              ↓
                                      Binance adapter
                                              ↓
                               Native OCO trailing (exchange-owned)
```

Package root: `binance_btc_bot/` (separate from MEXC execution under `btcc/execution/mexc/`).
