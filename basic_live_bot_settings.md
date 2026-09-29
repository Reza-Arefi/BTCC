# Basic live bot settings

Snapshot of the Binance BTC live bot as of **2026-09-29**. Use this to restore the same identity later. Authoritative runtime file is still `binance_btc_bot/config/binance_bot.yaml`. On-disk YAML stays `live.enabled=false` / `dry_run=true`; live writes only via `--live3-arm --authorize-live`.

## Identity

| Field | Value |
|---|---|
| Config version | `W2_E2_S65_N4_v1` |
| Stats session | `W2_E2_S65_N4` (T30 history is excluded from win rate / hourly) |
| Market | Binance **spot**, quote **BTC**, no leverage |
| Strategy | **W2** |
| Selector | NONE |
| Entry profile | **E2** signed S |
| Entry rule | **NEW_CROSS** into `S >= 0.65` on a **closed 15m** bar |
| Late-entry research filter | **OFF** |
| Max open trades | **4** |
| Allocation per trade | **25%** of trading equity |
| Total cap | **100%** (4 × 25%) |
| One position per symbol | yes |
| Planned loss at hard SL | **1.25%** equity (25% × 5% W2 SL) |

Validated fingerprint (research window 2026-03-25 → 2026-09-21): S≥0.65 + W2 Nmax=4 equal 25% — about **+21.8%** / **DD -27.4%** after fees.

## W2 exits (Binance native OCO)

| Leg | Value |
|---|---|
| Hard stop-loss | **5%** (`arm_sl_activation_trail = 0.05`) |
| Trail activation | **4%** |
| Trail distance | **2%** (`trailingDelta` 200 bips) |

Normal W2 geometry does not change when the safety manager is idle.

## Extra live rules (not part of W2 signal)

- **2-hour same-coin block:** after a close, do not reopen that symbol for 2 hours. Hard block, not late-entry. Persisted in `data/binance_btc_bot/post_close_block.json`.
- **Leftover inventory:** do not flatten leftover alts. They do not occupy the N=4 slots. Skip new W2 entries on leftover symbols.
- **Safety manager:** portfolio overlay only. Blocks **new entries**; does not change the W2 entry signal. State in `data/binance_btc_bot/safety_manager_state.json`. Only Telegram `/restart` (after health checks) clears halt / defensive mode. `/resume` cannot bypass a safety halt.

### Safety manager numbers

| Rule | Setting |
|---|---|
| Enabled | true |
| Consecutive exchange-confirmed SL closes → SAFETY_HALT | **2** |
| Win (trail/TP) resets SL counter | yes |
| DEFENSIVE_MODE | **4** open and **≥3** at **≤ -2%** (no auto-close of the book) |
| Defensive loss tighten | -2% → 1.0% trail; -2.5% → 0.5% trail; -3% market exit |
| Defensive profit lock | breakeven at +1.0% (stop entry+0.2%); lock +1.5% (stop +0.75%); trail +2.0% (peak−1.0%) |
| OCO replace | never looser; min improvement 0.25%; cooldown 60s |

## How to arm (do not flatten leftovers)

YAML kill defaults stay off. Live session:

```bash
cd /home/reza/Desktop/BTCC
source .venv/bin/activate
python -m binance_btc_bot --live3-arm --authorize-live
```

Requires `LIVE_TRADING_ENABLED=true` and `BINANCE_LIVE3_AUTHORIZED=true`. Run in a GNOME Terminal / tmux session that survives closing Cursor.

Telegram: `/status` `/risk` `/restart` `/close SYMBOL` `/close_all` (then `/confirm`).
