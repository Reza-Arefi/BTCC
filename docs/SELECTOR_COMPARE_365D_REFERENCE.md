# Selector Compare 365d — Reference

**Baseline ID:** `baseline-selector-compare-A-F-T1-T12-v1`  
**Frozen settings:** `docs/experiments/BASELINE_selector_compare_A_F_T1_T12_v1.json`  
**Experiment:** `selector_experiment_v1` / `selector_compare_365d`  
**Config:** `configs/selector_experiment_config.yaml`  
**Runner:** `scripts/run_selector_experiment_1y.py`  
**This run:** `results/selector_compare_365d_thr0p60_20260901_132655/`  
**Launch (resume):** `results/SELECTOR_EXPERIMENT_LAUNCH_20260902_150221.json`  
**Purpose:** Compare fixed trailing exits **T1–T12** vs dynamic selectors **A–F** over **365 days**, same entries for everyone.

---

## Big picture

Every opportunity is opened when the **entry signal S ≥ 0.60**. On that same signal:

1. **T1–T12** each run their own fixed stop/trailing geometry (12 independent paper accounts).
2. Those 12 outcomes are also recorded as **counterfactual history** (what each trail would have done).
3. **A–F** each pick **one** of T1–T12 for that trade, using different scoring rules on past counterfactual P/L (6 more independent accounts).

**18 arms total**, each with its own **$1,000** compounding account. Historical only (`allow_trading: false`).

This is **not** the E-memory lookback experiment (E-3…E-90). Here **E** is the original `rank_ewma` with **full history** and **7-day half-life**.

---

## Shared experiment settings

| Setting | Value | Meaning |
|--------|--------|---------|
| Eval window | 365 days | Out-of-sample length |
| Entry | `S >= 0.60` | No upper cap (`upper_threshold: null`) |
| Factor weights | **Static** | No live weight learning |
| Late-entry rejection | **OFF** | Late / extended entries are **allowed** |
| BTC.D filter | Disabled | Dominance does not block trades |
| Capital / arm | $1,000 | Independent equity curves |
| Notional | $100 (compounding) | Size scales with equity when compounding |
| Max open | 10 | Concurrent opportunities |
| One opp / pair | Yes | No stacking same pair |
| Telegram / live | OFF / false | Research only |

**Selector switching (A–F only):**

| Setting | Value | Meaning |
|--------|--------|---------|
| `minimum_selection_duration_hours` | 6 | Hold chosen trail at least 6h before switching |
| `switch_margin` | 0.0005 | New trail must beat current by this score margin |

---

## Fixed strategies T1–T12 (exit geometry)

Long-only trails. No fixed take-profit. Exit via **stop-loss** and/or **trailing stop** after activation.

| Arm | SL | Trail activate | Trail distance | Role / character |
|-----|-----|----------------|----------------|------------------|
| **T1** | −0.75% | +0.75% | 0.25% | Tight; quick lock-in |
| **T2** | −1.00% | +1.00% | 0.25% | Slightly wider SL, tight trail |
| **T3** | −0.75% | +0.75% | 0.50% | Tight SL, looser trail |
| **T4** | −1.00% | +1.00% | 0.50% | **Benchmark** mid geometry |
| **T5** | −1.50% | +1.00% | 0.50% | Wider SL than activate |
| **T6** | −1.50% | +1.50% | 0.50% | Symmetric mid |
| **T7** | −2.00% | +2.00% | 0.50% | Wide / patient |
| **T8** | −0.75% | +0.75% | 0.75% | Tight SL, wide trail gap |
| **T9** | −1.50% | +1.50% | 0.25% | Mid SL, very tight trail |
| **T10** | −2.00% | +1.50% | 0.50% | Wide SL, mid activate |
| **T11** | −2.00% | +2.00% | 1.50% | Wide everything |
| **T12** | −2.50% | +2.00% | 1.00% | Widest SL |

**How to read geometry:**

- **SL** = hard stop from entry (risk floor).
- **Activate** = unrealized gain needed before trailing engages.
- **Distance** = how far the trail sits below peak once active (smaller = tighter).

T1–T8 are the core trail-v3 family; T9–T12 extend the grid.

---

## Dynamic selectors A–F

Each selector scores every trail from **past closed counterfactual trades** (`exit_ts < decision time`, no lookahead), then picks the best under switching rules.

| Selector | Kind | Idea |
|----------|------|------|
| **A** | `ewma_7d` | EWMA of raw P/L per trail; half-life **7 days**. Simple recent performance. |
| **B** | `multi_horizon_ewma` | Blend of 3 EWMAs: **4h (15%) + 1d (30%) + 7d (55%)**. Short + medium memory. |
| **C** | `regime_conditional` | EWMA **inside current regime** (RANGE / trends / HIGH_VOL). Needs ≥5 regime obs; else blends toward global recent. Half-life 7d. |
| **D** | `recent_plus_regime` | **65%** recent global EWMA + **35%** regime EWMA. Half-life 7d, min 5 regime obs. |
| **E** | `rank_ewma` | **Original E-v1.** Per opportunity, rank trails by P/L (1 = best); EWMA of ranks (7d HL); invert so higher score = better. Full history (no lookback cap). Live paper uses this. |
| **F** | `downside_aware` | EWMA(return) − **0.5 ×** EWMA(\|negative returns\|). Penalizes trails that lose more often / harder. |

### Why E is special

Paper trading (`configs/selector_live_config.yaml`) freezes **E-v1** = this same `rank_ewma` / 7d / full history / late entry allowed.  
A–F in this backtest answer: *does a smarter chooser beat always-T1 / always-T4 / etc.?*

---

## Entry signal (shared)

- Score **S** from static factor weights (momentum, trend, volume, volatility, RSI, structure, btc_regime, etc. — fixed, not adaptive).
- Trade if **S ≥ 0.60**.
- Late-entry score/class may still be **logged**, but **not used to reject** (`disable_late_entry_rejection: true`).

---

## Regime labels (for C / D and analytics)

Classified from ADX / NATR-style rules:

| Regime | Rule sketch |
|--------|-------------|
| HIGH_VOLATILITY | NATR ≥ 0.035 |
| RANGE | ADX &lt; 20 |
| LOW_VOL_TREND | ADX &lt; 30 |
| NORMAL_TREND | ADX &lt; 40 |
| STRONG_TREND | otherwise |

S-bands for analytics: `0.60–0.65`, `0.65–0.70`, `0.70–0.80`, `0.80+`.

---

## What to look at in results

Directory: `results/selector_compare_365d_thr0p60_20260901_132655/`

| Artifact | Use |
|----------|-----|
| `analytics/plots/selector/cumulative_return_all.png` | Equity vs day for all 18 arms |
| `analytics/plots/selector/drawdown_all.png` | Risk |
| `analytics/selector_metrics.json` | Return, PF, trades, etc. |
| `selection_audit.csv` | Which trail A–F chose per opportunity |
| `strategy_legs.csv` | Every arm’s trade P/L |
| `daily_checkpoints/day_NNN/` | Progress / resume points |

**Fair comparison notes:**

- Same entries for all arms → differences are **exits** (T*) or **trail choice** (A–F).
- A–F can only pick among T1–T12; they cannot invent a new exit.
- If E ≈ T1 most of the time, E is concentrating on one trail, not diversifying.

---

## Explicitly out of scope for this run

- E-memory variants (E-3, E-7, … E-90 lookbacks)
- T1 late-filter baseline arms
- Adaptive V2 / live weight updates
- BTC.D gating
- Real trading (`allow_trading` must stay false)

---

## How to resume / re-run

```bash
# Resume this folder
BTCC_SELECTOR_OUT_DIR=results/selector_compare_365d_thr0p60_20260901_132655 \
BTCC_TELEGRAM_FORCE_OFF=1 PYTHONUNBUFFERED=1 PYTHONPATH=. \
.venv/bin/python scripts/run_selector_experiment_1y.py

# Fresh run (omit BTCC_SELECTOR_OUT_DIR)
```

Log: `results/selector_compare_365d.log`
