# E2–T7 Selection Report

**Experiment:** 90-day Binance BTC relative-strength backtest  
**Window:** 2026-06-12 → 2026-09-10 UTC (60d select + 30d holdout)  
**Signals:** 15m closed bars · **Exits:** 1s OHLC trail simulation  
**Universe:** live BTC pairs from `binance_btc_bot/config/binance_bot.yaml`  
**Costs in sim:** fees = 0, slippage = 0 (optimistic; live will be worse)

---

## Final candidate

| Knob | Choice | Notes |
|------|--------|--------|
| **Entry** | **E2** | Faster momentum factor profile |
| **Exit** | **T7** | 2.0% SL / 2.0% activation / 0.5% trail |
| **Selector** | **None** | A–F did not beat fixed T7 on select |
| **Allocation** | **25%** (aggressive) or **12.5%** (conservative baseline) | See Stage 4 caveat |
| **Max open** | **4** (with 25%) or **8** (with 12.5%) | Stream rarely >8 concurrent |

**Recommended research freeze for next live/paper work:**

```yaml
entry: E2
exit: T7
selector: null
# Conservative (matches current bot defaults for size):
allocation_per_trade: 0.125
max_simultaneous_trades: 8
# Aggressive Stage-4 pick (same edge, more size / DD):
# allocation_per_trade: 0.25
# max_simultaneous_trades: 4
long_threshold: 0.65
signal_interval: 15m
exit_sim_interval: 1s
```

Settings file: [`configs/e2_t7_candidate.yaml`](../configs/e2_t7_candidate.yaml)  
Bundled rankings / decisions / plots: [`reports/E2_T7/`](./E2_T7/)

---

## Method (do not reverse)

Entry and exit were treated as separate knobs:

1. **Stage 1 — Entry** (BASE / E1–E5) with counterfactual **T1–T10** on each entry  
2. **Stage 2 — Exit** (T1–T10) on frozen **E2** entry stream  
3. **Stage 3 — Selectors** (A–F) vs fixed **T7** on E2  
4. **Stage 4 — Sizing / capacity** on frozen **E2+T7**

Walk-forward rule: **pick on 60d select; confirm on 30d holdout**. Do not pick on full 90d alone.

Tape: Binance Vision/API · relative series = ALTUSDT / BTCUSDT · trail state machine on 1s bars.

---

## Stage 1 — Entry (BASE / E1–E5 × T1–T10)

**Primary rule:** mean Net BTC across T1–T10 on select.

| Rank | Entry | Select mean Net BTC | Select T1 Net BTC | Holdout mean Net BTC |
|------|-------|---------------------|-------------------|----------------------|
| 1 | **E2** | **+6.76e-6** | +1.28e-5 | **+2.38e-4** |
| 2 | BASE | −7.13e-5 | −4.38e-5 | (confirm in CSV) |
| 3–6 | E1, E4, E5, E3 | −4.7e-4 … −5.7e-4 | all − | mixed/− |

- **Winner:** **E2** (also wins T1-only ranking).  
- Artifacts: `results/stage1_entry_90d_1s_20260910_115513/`

---

## Stage 2 — Exit (T1–T10 on E2)

**Primary rule:** Net BTC on select for E2 legs only.

| Rank | Exit | Select Return % | Select Max DD | Holdout Return % |
|------|------|-----------------|---------------|------------------|
| 1 | **T7** | **+1.71%** | −3.08% | **+6.49%** |
| 2 | T10 | +0.51% | −3.14% | +4.89% |
| 3 | T3 | +0.44% | −1.86% | … |
| 4 | T1 | +0.16% | −1.59% | … |
| … | T2 last on select | −1.02% | … | … |

**T7 geometry** (from `binance_bot.yaml`):

- `arm_sl_activation_trail`: 0.02 (2.0%)  
- `activation`: 0.02 (2.0%)  
- `trail_distance`: 0.005 (0.5%)

- **Winner:** **T7** (holdout confirms #1).  
- Artifacts: `results/stage2_exit_E2_90d_1s_20260910_155823/`

---

## Stage 3 — Selectors A–F vs fixed T7

**Rule:** keep a selector only if it beats T7 on **both** select and holdout.

| Arm | Kind | Select Return % |
|-----|------|-----------------|
| **T7** | fixed | **+1.71%** (best) |
| B | selector | +0.55% |
| D | selector | +0.38% |
| C / F / E | selector | lower |
| A | selector | −0.83% |

- **Keep selector:** **No**  
- **Live exit:** fixed **T7**  
- Artifacts: `results/stage3_selector_E2_T7_90d_1s_20260910_160330/`

---

## Stage 4 — Sizing / capacity (E2+T7)

Swept allocation 5–25% and `max_open` 4–12.

| Config | Select Return | Select DD | Holdout Return | Holdout DD |
|--------|---------------|-----------|----------------|------------|
| Baseline 12.5% × 8 | +1.71% | −3.08% | +6.49% | −2.76% |
| Stage-4 pick 25% × 4 | +3.41% | −5.96% | +13.08% | −3.99% |

**Caveat:** on this stream, return and DD scale almost **linearly** with allocation (same profit factor ≈ 1.11; almost no capacity skips). Raising `max_open` above 8 does little (raw concurrency ≤ 8). Prefer **12.5% × 8** for paper/live unless intentionally accepting higher DD.

- Artifacts: `results/stage4_sizing_E2_T7_90d_1s_20260910_160717/`

---

## How to reproduce

```bash
cd /path/to/BTCC
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r binance_btc_bot/requirements.txt
export PYTHONPATH="$(pwd)"

# Stage 1 (downloads 15m + full-window 1s; long-running)
python -m binance_btc_bot.tools.run_stage1_entry_90d_1s

# Stages 2–4 reuse Stage 1 E2 counterfactual legs (fast)
python -m binance_btc_bot.tools.run_stage2_exit_90d_1s \
  --entry E2 --stage1-dir results/stage1_entry_90d_1s_<runid>

python -m binance_btc_bot.tools.run_stage3_selector_90d_1s \
  --stage1-dir results/stage1_entry_90d_1s_<runid> \
  --fixed-exit T7

python -m binance_btc_bot.tools.run_stage4_sizing_90d_1s \
  --stage1-dir results/stage1_entry_90d_1s_<runid>
```

---

## Limitations

- Exit sim is 1s OHLC (not ticks); residual same-bar path ambiguity remains.  
- Fees/slippage = 0 in this research pass.  
- Selectors replayed offline from counterfactual T legs (causal CF history only).  
- Live Binance uses **native OCO trailing**, not the local 1s sim — shadow-compare before sizing up.  
- Stage-1 live YAML still shows `strategy: T4` / `momentum_profile: e2`; this report proposes **T7** + **E2** as the research winner, not an auto-enable of live trading.

---

## Decision log (IDs)

| Stage | Result dir |
|-------|------------|
| 1 | `results/stage1_entry_90d_1s_20260910_115513` |
| 2 | `results/stage2_exit_E2_90d_1s_20260910_155823` |
| 3 | `results/stage3_selector_E2_T7_90d_1s_20260910_160330` |
| 4 | `results/stage4_sizing_E2_T7_90d_1s_20260910_160717` |

**Bottom line:** freeze **E2 entry + T7 exit**, no selector; size **12.5% × 8** for conservative deployment, or **25% × 4** only if higher drawdown is acceptable.
