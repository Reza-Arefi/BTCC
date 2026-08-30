# Pre-backtest report — A/B/C arms (awaiting 1-year authorization)

**Status:** Implementation + unit tests complete. Entry-policy arm added — see [`PRE_BACKTEST_ENTRY_POLICY.md`](PRE_BACKTEST_ENTRY_POLICY.md). Full **1-year backtest NOT run** (awaiting your authorization after smoke).

---

## Checklist results

| Check | Result |
|-------|--------|
| 1. Unit tests Static weights | **PASS** |
| 2. Unit tests Equal weights | **PASS** |
| 3. Unit tests Adaptive weights | **PASS** |
| 4. Identical factor inputs (shared cache + same FACTOR_KEYS) | **PASS** |
| 5. A weights never change (runtime invariant) | **PASS** |
| 6. B weights remain 1/N (runtime invariant) | **PASS** |
| 7. C updates only on schedule (`effective_from` next bar) | **PASS** |
| 8. C uses only matured outcomes (`pred_ts+4h ≤ update_ts`) | **PASS** |
| 9. All arms threshold **0.60** | **PASS** |
| 10. Identical fee/slippage/SL-first/max-10 | **PASS** |
| 11. Separate result dirs per arm | **PASS** |
| 12. Small historical smoke (14d, all three arms) | **PASS** — `abc_compare_14d_thr0p60_20260829_232417` |

```
STATIC_OK
EQUAL_OK
ADAPTIVE_OK
FAIR_COMPARISON_OK
```

Smoke notes: factor caches matched exactly (abs diff 0); Adaptive produced 12 weight updates with `effective_from` strictly after `calculated_at`; opportunities differed by arm as expected (Static 55 / Equal 23 / Adaptive 32).
---

## Arms

| Arm | Method | Updates? |
|-----|--------|----------|
| **A Static** | `configs/signal_config.yaml` → `factors.weights` (momentum 0.20, trend 0.20, btc_regime 0.20, volume 0.12, volatility 0.10, rsi 0.08, structure 0.10) | **Never** |
| **B Equal** | \(w_i = 1/7\) over `FACTOR_KEYS` | **Never** |
| **C Adaptive** | 90d init → daily rolling 90d @ 23:00 America/Sao_Paulo; matured labels only; `effective_from` = next bar | **Yes** |

**Why this static vector:** It is the single pre-Adaptive V2 configured weight set in `signal_config.yaml` (`experiment.name: btcc_signal_baseline`). No other static config exists in-repo.

**Missing indicator rule (all arms):** Signed score → 0 (neutral); Equal still allocates \(1/N\) to that slot so N stays the model factor count (identical indicator set vs A/C).

---

## Fairness (only intentional difference = weighting)

Shared across A/B/C:

- Candles / timestamp sequence / 20-base universe  
- Relative BTC.D proxy (no present-day scale)  
- Factor scores (shared `factor_cache` in ABC runner)  
- Horizon 4h, threshold **0.60**, crossing SM, max 10, one-per-pair  
- Entry = next bar open; fee 0.10%/side; slip 0.05%/side; same-candle SL-first  
- Strategies S1 / S2 / S3  

**Not forced identical:** trade opportunities (signals may differ when S differs).

**Not in this experiment:** Adaptive±BTC.D A/B (deferred).

---

## Commands

```bash
# Unit tests
.venv/bin/python -m pytest tests/test_abc_arms.py -q

# Smoke (short history; shrunk init window)
.venv/bin/python -m btcc.main sim-abc --days 14

# Full 1-year — DO NOT RUN until authorized
# .venv/bin/python -m btcc.main sim-abc --days 365
```

---

## Safety

- `safety.allow_trading: false`  
- Protocol: **not** 4×90 folds — `init_days=90` then `daily_rolling_window_days=90`

---

## After smoke passes

Authorize the fixed 1-year A/B/C experiment. No post-hoc threshold/indicator/pair/SL-TP retuning on the first run.
