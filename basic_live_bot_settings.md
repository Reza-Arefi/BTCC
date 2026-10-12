# Basic live bot settings

Snapshot of the Binance BTC live bot as of **2026-10-12**. Use this to restore the same identity later. Authoritative runtime file is still `binance_btc_bot/config/binance_bot.yaml`. On-disk YAML stays `live.enabled=false` / `dry_run=true`; live writes only via `--live3-arm --authorize-live`.

## Identity

| Field | Value |
|---|---|
| Config version | `TRN4_E2_S65_N4_WKND_v1` (previous: `W2_E2_S65_N4_v1`) |
| Market | Binance **spot**, quote **BTC**, no leverage |
| Strategy | **TRN4** |
| Selector | NONE |
| Entry profile | **E2** signed S |
| Entry rule | **NEW_CROSS** into `S >= 0.65` on a **closed 15m** bar |
| Entry blackout | **no new entries FRI 12:00 → MON 00:00 UTC** (FRI 09:00 → SUN 21:00 BRT) |
| Late-entry research filter | **OFF** |
| Max open trades | **4** |
| Allocation per trade | **25%** of trading equity |
| Total cap | **100%** (4 × 25%) |
| One position per symbol | yes |
| Planned loss at hard SL | **1.25%** equity (25% × 5% TRN4 SL) |

Selection evidence (replay of all 149 live entries 2026-09-13 → 2026-10-12, native 1m klines, 0.1%/side fees, N=4 × 25% compounding): TRN4 + FRI 12:00→MON 00:00 block — 49 trades, 71% win, PF 2.32, about **+26.2%** / **DD -4.3%**. In-sample on ~1 month; live execution ran ~0.2 pp/trade worse than the replay. Tools: `binance_btc_bot/tools/run_three_candidates_20261012.py`, `binance_btc_bot/tools/run_weekend_window_grid_20261012.py`.

## TRN4 exits (Binance native OCO)

| Leg | Value |
|---|---|
| Hard stop-loss | **5%** (`arm_sl_activation_trail = 0.05`) |
| Trail activation | **4%** |
| Trail distance | **0.5%** (`trailingDelta` 50 bips) |

50 bips is inside `minTrailingAboveDelta`/`maxTrailingAboveDelta` for all 38 pairs (checked 2026-10-12). Normal TRN4 geometry does not change when the safety manager is idle. Positions opened before the switch keep the OCO they were opened with.

## Extra live rules (not part of the S signal)

- **Weekend entry blackout:** `entry.blackout` in the YAML. Only blocks **new** entries; open positions keep their OCO. A cross that happens inside the window is consumed (logged as `ENTRY_BLACKOUT`), not deferred to Monday — Monday needs a fresh cross.
- **2-hour same-coin block:** after a close, do not reopen that symbol for 2 hours. Hard block, not late-entry. Persisted in `data/binance_btc_bot/post_close_block.json`.
- **Leftover inventory:** do not flatten leftover alts. They do not occupy the N=4 slots. Skip new entries on leftover symbols.
- **Safety manager:** portfolio overlay only. Blocks **new entries**; does not change the entry signal. State in `data/binance_btc_bot/safety_manager_state.json`. Only Telegram `/restart` (after health checks) clears halt / defensive mode. `/resume` cannot bypass a safety halt.

### Safety manager numbers

| Rule | Setting |
|---|---|
| Enabled | true |
| Consecutive exchange-confirmed SL closes → SAFETY_HALT | **2** |
| Win (trail/TP) resets SL counter | yes |
| DEFENSIVE_MODE | **4** open and **≥3** at **≤ -2%** (no auto-close of the book) |
| Defensive loss tighten | -2% → 1.0% trail; -2.5% → 0.5% trail; -3% market exit |
| Defensive profit lock | breakeven at +1.0% (stop entry+0.2%); lock +1.5% (stop +0.75%); trail +2.0% (peak−1.0%) |
| OCO replace | never looser; min improvement 0.25%; cooldown 60s; keeps the trade's own trailingDelta |

## How to apply on the live machine

The live machine has local, uncommitted edits (fixed take-profit +4%, stats session). Save them before pulling:

```bash
cd /home/reza/Desktop/BTCC
source .venv/bin/activate
git status
git stash push -m "pre-TRN4 local live edits"     # or commit them on a branch
git pull origin main
python -m binance_btc_bot --check-config          # must print STATUS: OK ... TRN4_E2_S65_N4_WKND_v1
```

Then stop the running session and re-arm (preflight reseeds the runtime state to TRN4):

```bash
python -m binance_btc_bot --live3-arm --authorize-live
```

Requires `LIVE_TRADING_ENABLED=true` and `BINANCE_LIVE3_AUTHORIZED=true`. Run in a GNOME Terminal / tmux session that survives closing Cursor. Do not `git stash pop` the old fixed-TP edits back on top — they would override the TRN4 trail.

Telegram: `/status` `/risk` `/restart` `/close SYMBOL` `/close_all` (then `/confirm`).
