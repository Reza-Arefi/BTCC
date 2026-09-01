# Analytics dashboard layout (plot specification)

Trading logic is unchanged. This documents plot organization only.

## Plot directories

Under each arm `analytics/plots/` (and ABC `analytics/plots/`):

| Folder | Contents |
|--------|----------|
| `primary/` | $1,000 compounded portfolio (S1–S5), cumulative P/L %, portfolio drawdown %, cumulative BTC PnL from trades, total BTC-equivalent account value, rolling/cumulative win rate |
| `adaptive/` | Weight evolution (**BTC.D / btc_regime excluded**), learning progress |
| `prediction/` | Score vs return, buckets, indicator correlations (btc_regime labeled *Context only*) |
| `trades/` | Expectancy, profit factor, win/loss counts, recovered late trades |
| `risk/` | Concurrent opportunities (max=10), rejections (no BTC.D gate), funnel |
| `contextual/` | BTC.D over Day + regimes — **analysis only** |
| `comparisons/` | Normal vs Late, Static/Equal/Adaptive overlays |

Filenames encode `model` + `entry_policy` + metric (no extra nested arm folder).

## Daily summary

Each `daily_checkpoints/day_NNN/daily_summary.json` includes $1,000 account values, P/L %, trade/win stats, drawdown, adaptive weights, `weight_update_occurred_today`, today's realized USD P/L, open/rejected counts.

## Capital model (analytics)

- Trade size remains `notional_usd` ($100).
- Each account (`model × entry_policy × S1–S5`) tracks equity from **`$1,000`** using closed-trade `pnl_usd_equiv`.
- Compounds: `capital[t+1] = capital[t] + realized_PnL_USD[t]`.

## Day axis

- X-axis is **Day 1 … Day N** (`day_number`).
- **Day 90** marked as init → daily adaptation (or smoke `init_days` when shortened).

## Daily vs full emit

- Each simulated day (`build_arm_analytics_asof`, `lite=True`): primary dashboard plots + checkpoint snapshot.
- End of arm / ABC (`lite=False`): full research suite.
- Checkpoints remain append-only under `daily_checkpoints/day_NNN/`.

## BTC.D

- Contextual only (`require_for_new_trades: false`).
- Not shown as a trading rejection stage in the funnel.
