# Pre-backtest — Entry Policy experiment (awaiting 1-year authorization)

**Status:** Implementation + unit tests + 10d smoke **PASS**. Full **1-year backtest NOT run** (awaiting your explicit authorization).

**Hypothesis:** Does the Late Entry / extended filter remove profitable opportunities that the weighted signal `S ≥ 0.60` otherwise gets right?

---

## Smoke checklist

| Check | Result |
|-------|--------|
| `ENTRY_POLICY_NORMAL_OK` | **PASS** |
| `ENTRY_POLICY_LATE_ALLOWED_OK` | **PASS** |
| `RECOVERED_TRADE_TRACKING_OK` | **PASS** (45 recovered on Adaptive 10d) |
| `THREE_EXIT_STRATEGIES_OK` | **PASS** (S1/S2/S3 unchanged) |
| `STATIC_EQUAL_ADAPTIVE_OK` | **PASS** |
| `FAIR_COMPARISON_OK` | **PASS** (identical `S` across policies) |
| `NO_LOOKAHEAD_OK` | **PASS** (`none_no_present_day_scaling`) |
| `BTC_D_HEALTH_GATE_OK` | **PASS** |
| `SAFETY_ALLOW_TRADING_FALSE_OK` | **PASS** |

Smoke artifact: `results/abc_compare_10d_thr0p60_20260830_003649`  
Verdict: `SMOKE_ENTRY_POLICY_VERDICT.json`  
Unit tests: `tests/test_entry_policy.py` — 10 passed.

### Adaptive 10d classification sample (not for tuning)

| Class | Count |
|-------|------:|
| BELOW_THRESHOLD | 38016 |
| LATE_ENTRY_REJECTED | 55 |
| LATE_ENTRY_ACCEPTED | 45 |
| NORMAL_ENTRY | 7 |
| SIGNAL_CONTINUATION | 66 |
| SAME_PAIR_ALREADY_OPEN | 11 |
| BTC_D_UNAVAILABLE | 80 |

Normal opportunities ≈ 4 legs/strategy; Late-allowed ≈ 48; recovered ≈ 45.

---

## Architecture (do not collapse)

```
MODEL: Static | Equal | Adaptive
        │
        ▼
SIGNAL: S ≥ 0.60  (fixed; not tuned)
        │
        ▼
ENTRY POLICY
  ├─ NORMAL_FILTERED      (extended Late Entry rejection ON)
  └─ LATE_ENTRY_ALLOWED   (bypass ONLY Late Entry / extended reject)
        │
        ▼
EXIT STRATEGY
  ├─ S1  1% SL / 1.5% TP
  ├─ S2  2% SL / 3% TP
  └─ S3  1% SL + 1% trail activation + 0.5% trail
```

Full grid (18 cells, reported separately):

- Static × {Normal, LateAllowed} × {S1, S2, S3}
- Equal × {Normal, LateAllowed} × {S1, S2, S3}
- Adaptive × {Normal, LateAllowed} × {S1, S2, S3}

---

## What LATE_ENTRY_ALLOWED may bypass

**Only:** Late Entry / extended-filter rejection (`late_entry_score ≥ alert_threshold 0.75` or class `HIGH_LATE_ENTRY_RISK` / `VERY_HIGH_LATE_ENTRY_RISK`).

## What it must NOT bypass

- `S ≥ 0.60`
- Invalid / missing market data
- Unhealthy or unavailable BTC.D
- Invalid indicators
- Duplicate active opportunity (same pair)
- Max 10 simultaneous opportunities (per policy book)
- Closed-bar / next-bar execution timing
- Fee 0.10%/side + slip 0.05%/side
- Long-only ALT/BTC
- Any other hard safety restriction (`allow_trading` remains **false**)

---

## Final configuration (locked for 1y)

| Knob | Value |
|------|-------|
| Threshold | **0.60** (fixed) |
| Weight arms | Static / Equal / Adaptive |
| Entry policies | `NORMAL_FILTERED`, `LATE_ENTRY_ALLOWED` |
| Exit strategies | S1 / S2 / S3 (unchanged) |
| Late Entry alert threshold | **0.75** (`signal_config.late_entry.alert_threshold`) |
| Max open / policy book | **10** |
| Fee / slip | **0.10%** / **0.05%** per side |
| BTC.D | Relative top-N, **no present-day scale** |
| BTC.D health max age | **7200 s (`BTC_D_MAX_AGE`)** — stale if observation age > 2h |
| BTC.D historical (1y) | **Daily** relative proxy |
| BTC.D live/short | **~Hourly** relative proxy |
| Horizon | 4h |
| Adaptive learning | 90d init → daily rolling 90d @ 23:00 America/Sao_Paulo |
| `safety.allow_trading` | **false** |

### Classifications recorded

`NORMAL_ENTRY`, `LATE_ENTRY_ACCEPTED`, `LATE_ENTRY_REJECTED`, `BELOW_THRESHOLD`, `SAME_PAIR_ALREADY_OPEN`, `MAX_OPEN_TRADES`, `BTC_D_UNAVAILABLE`, `DATA_INVALID` (+ continuation / health / no-next-bar).

Recovered flag: `recovered_by_late_allowed` when Normal would reject for late and Late-Allowed accepts.

---

## Analytics dashboard (entry-policy)

- BTC accumulation: Normal vs LateAllowed (per S1/S2/S3)
- Win rate: Normal vs LateAllowed
- Recovered count over time
- Cumulative BTC of recovered trades
- Recovered wins vs losses
- Monthly BTC of recovered trades
- Score `S` distribution: normal / late-accepted / late-rejected
- Future 4h return: normal vs recovered late

---

## Rules for the 1-year run

1. Run **exactly once** after authorization.
2. **Do not** tune Late Entry filter, threshold, or exits from results.
3. **Do not** select the “better” policy before analysis.
4. Analyze recovered trades separately to answer: did Late Entry remove good or bad trades?

---

## Commands

```bash
# Unit tests
.venv/bin/python -m pytest tests/test_entry_policy.py -q

# Smoke (already passed)
.venv/bin/python -m btcc.main sim-abc --days 10

# Full 1-year — DO NOT RUN until authorized
# .venv/bin/python -m btcc.main sim-abc --days 365
```

---

## Awaiting

**Explicit authorization** to run the single fixed 1-year experiment.
