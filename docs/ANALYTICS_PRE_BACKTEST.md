# Analytics layer — pre-1y report (STOP — awaiting authorization)

## Smoke verdict

```
ANALYTICS_OK
PLOTS_OK
DAILY_SNAPSHOT_OK
TELEGRAM_DISABLED_IN_BACKTEST_OK
LIVE_TELEGRAM_CONFIG_OK
NO_LOOKAHEAD_IN_ANALYTICS_OK
```

Smoke used existing 14d ABC run → `results/abc_compare_14d_thr0p60_20260829_232417/analytics/`  
(13 comparison plots, 56 arm plots, 44 metric CSVs, dated daily snapshots)

**Full 1-year backtest NOT run.**

---

## 1. Files created / modified

**Created**
- `btcc/analytics/__init__.py`
- `btcc/analytics/metrics.py` — shared metric definitions
- `btcc/analytics/plots.py` — matplotlib plotters
- `btcc/analytics/pipeline.py` — `build_arm_analytics` / `build_abc_analytics` / `update_live_analytics`
- `btcc/analytics/daily.py` — append-only daily snapshots
- `tests/test_analytics.py`
- `requirements-analytics.txt`
- `docs/ANALYTICS_PRE_BACKTEST.md` (this file)

**Modified**
- `btcc/sim/abc_compare.py` — runs analytics after ABC (Telegram forced off)
- `btcc/sim/engine.py` — daily live analytics refresh + summary hook
- `btcc/sim/alerts.py` — `daily_summary` / `capacity_saturation`
- `configs/signal_config.yaml` — `send_ranking_every_cycle: false`, `send_daily_summary: true`
- `configs/sim_config.yaml` — daily summary / capacity alert flags
- `requirements.txt` — matplotlib

---

## 2. Plot list

**Comparison (ABC)**  
- `btc_accumulation_strategy_{1,2,3}.png`  
- `rolling_win_rate_strategy_{1,2,3}.png`  
- `drawdown_strategy_{1,2,3}.png`  
- `adaptive_advantage_strategy_{1,2,3}.png`  
- `win_loss_net_pnl.png`

**Per arm**  
- adaptive weights, weight vs usefulness, learning progress  
- indicator predictive correlation  
- score vs return (scatter + binned)  
- accuracy by score bucket  
- simultaneous opportunities, rejected signals  
- monthly PnL, BTC.D regime accuracy  
- drawdown / accumulation / win-rate (arm-local)

---

## 3. Metrics list (CSV under `metrics/`)

cumulative BTC, rolling + cum win rate, weight timeseries, indicator corr (full + rolling), weight vs usefulness, score vs return, accuracy buckets, win/loss summary, drawdown series + stats, simultaneous open + stats JSON, rejection breakdown, monthly performance, BTC.D regimes, learning progress, adaptive advantage deltas

---

## 4. Data outputs layout

```
results/<abc_run>/analytics/
  metrics/  plots/  daily_snapshots/  reports/
  arms/{static,equal,adaptive}/...
  predictions/ opportunities/ weights/ data_summary/ config_snapshot/
```

Live: `data/live_analytics/` with the same logical structure; each refresh copies a timestamped snapshot under `daily_snapshots/live_refresh_YYYYMMDD_HHMMSS/` (never overwrites prior snapshots).

Every plot is backed by a CSV in `metrics/`.

---

## 5. Backtest Telegram

**Disabled.** `build_abc_analytics(..., telegram_enabled=False)`; raises if True. No Telegram from analytics or ABC backtest path.

---

## 6. Live Telegram

| Event | Sent? |
|-------|-------|
| Every 15m prediction / ranking | **No** (`send_ranking_every_cycle: false`) |
| Exhaustion spam | **No** |
| 3+ opens / 60m | Yes |
| BTC.D / data failure | Yes |
| Runtime error | Yes |
| Startup / shutdown | Yes |
| Capacity saturation | Yes (optional) |
| Daily ~23:00 summary | Yes |

---

## 7. How live plots update

At/after 23:00 America/Sao_Paulo (once per local day), or when a daily weight update runs, `AdaptiveSimEngine` calls `update_live_analytics()` which rebuilds metrics/plots from persistent sim CSVs.

---

## 8. Historical snapshot preservation

Daily files are uniquely named (`daily_{arm}_{UTC_stamp}.md/json`). Live refreshes copy metrics/plots into `live_refresh_{stamp}/`. No overwrite of prior snapshots.

---

## 9. Temporal validity

- Correlations / score–return / accuracy use `matured_predictions` (`pred_ts + 4h ≤ asof`)
- Weight plots use recorded update timestamps
- Adaptive advantage uses each arm’s own trade path (signals free to differ)

---

## Next

Authorize the fixed **365-day A/B/C** run when ready. Analytics will be generated automatically into the result directory with Telegram off.
