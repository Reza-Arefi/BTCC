# Basic live bot settings

Snapshot of the Binance BTC live bot as of **2026-10-14**. Use this to restore the same identity later. Authoritative runtime file is still `binance_btc_bot/config/binance_bot.yaml`. On-disk YAML stays `live.enabled=false` / `dry_run=true`; live writes only via `--live3-arm --authorize-live`.

## Identity

| Field | Value |
|---|---|
| Config version | `F4H_X16_S65_N6_v1` (previous: `TRN4_E2_S65_N4_WKND_v1`, before that `W2_E2_S65_N4_v1`) |
| Market | Binance **spot**, quote **BTC**, no leverage |
| Strategy | **F4H** (fixed TP / SL / time exit) |
| Selector | NONE |
| Entry profile | **E2** signed S |
| Signal bars | **4h** ALT/BTC bars built from **1m** ratio closes (UTC 00/04/08/12/16/20) |
| Factor scale | 15m factor lookbacks applied to 4h bars (x16 time scale), 400-bar window |
| Entry rule | **NEW_CROSS** into `S >= 0.65` between two consecutive **closed 4h** bars |
| Entry window | first **30 min** after each 4h close; each bar is decided once |
| Entry blackout | **OFF** (no weekend block) |
| Late-entry research filter | **OFF** |
| Max open trades | **6** |
| Allocation per trade | **16.66%** of trading equity |
| Total cap | **99.96%** (6 × 16.66%) |
| One position per symbol | yes |
| Planned loss at hard SL | **4.17%** equity (16.66% × 25% SL) |

Selection evidence (`htf_4h/step8_3y.py`, 36 months 2023-10 → 2026-09, 38 pairs, 0.1%/side fees, 6 × 1/6 compounding): **+750%**, max DD **-30.9%**, 867 trades, 47% win, PF 1.45, 23/36 positive months. With an extra 0.22% per trade live execution cost: +521%, DD -34.9%. Tuned on the last 7 months; the 17 months never used in tuning returned +108% (DD -28.9%). Long underwater stretches are normal (longest 331 days). Coin filtering was tested (`htf_4h/step9_coins.py`) and did not survive — all coins stay.

## F4H exits

| Leg | Value |
|---|---|
| Hard stop-loss | **25%** (Binance OCO STOP_LOSS leg) |
| Take-profit | **+30%** fixed (OCO TAKE_PROFIT leg, no `trailingDelta`) |
| Max hold | **120h (5 days)** → bot cancels the OCO and market-sells (`TIME_EXIT`) |

The time exit is run by the bot loop (`engine.time_exit_cycle`), so the bot must be running for it to fire. Positions opened before the switch keep the OCO they were opened with (TRN4 trail); the time exit only applies to F4H trades.

## How the live signal works

- Every 4h close the bot waits for the final 1m candle of that bar, then builds ALT/BTC bars from 1m ALTUSDT and BTCUSDT closes and aggregates them to 4h (`binance_btc_bot/strategy/htf_bars.py`). Building from 4h/15m klines distorts high/low and was rejected (recall only 68–85% vs research).
- 1m bars are cached on disk in `<sqlite dir>/rel4h_1m/{BASE}.csv`. **The first start backfills ~400 × 4h bars of 1m data for 38 pairs (~3.8k requests, several minutes).** Later starts only fetch missing bars.
- `S` is computed on the last 400 bars and `S_prev` on the 400 bars ending one bar earlier. A cross is consumed once per bar — if all slots are full, the coin is in cooldown, or the safety manager blocks, that cross is not retried later.

## Extra live rules (not part of the S signal)

- **2-hour same-coin block:** after a close, do not reopen that symbol for 120 minutes (`entry.same_symbol_cooldown_minutes`, uses the last CLOSED trade's exit time in the DB).
- **Leftover inventory:** do not flatten leftover alts. They do not occupy the N=6 slots. Skip new entries on leftover symbols.
- **Safety manager:** portfolio overlay only. Blocks **new entries**; does not change the entry signal. State in `data/binance_btc_bot/safety_manager_state.json`. Only Telegram `/restart` (after health checks) clears a halt. `/resume` cannot bypass a safety halt.

### Safety manager numbers

| Rule | Setting |
|---|---|
| Enabled | true |
| Consecutive exchange-confirmed SL closes → SAFETY_HALT | **2** |
| Win (TP or profitable TIME_EXIT) resets SL counter | yes |
| DEFENSIVE_MODE | **OFF** (`safety_manager.defensive_enabled: false`; its -2%/-3% rules would cut every F4H trade long before the 25% stop) |

## How to apply on the live machine

The live machine may have local, uncommitted edits (fixed take-profit +4%, stats session, same-coin block). Save them before pulling:

```bash
cd /home/reza/Desktop/BTCC
source .venv/bin/activate
git status
git stash push -m "pre-F4H local live edits"      # or commit them on a branch
git pull origin main
python -m binance_btc_bot --check-config          # must print STATUS: OK ... F4H_X16_S65_N6_v1
```

Then stop the running session and re-arm (preflight reseeds the runtime state to F4H):

```bash
python -m binance_btc_bot --live3-arm --authorize-live
```

Requires `LIVE_TRADING_ENABLED=true` and `BINANCE_LIVE3_AUTHORIZED=true`. Run in a GNOME Terminal / tmux session that survives closing Cursor. Do not `git stash pop` the old edits back on top — the 2h same-coin block is now built in, and the old fixed-TP edit would override the F4H exits.

Telegram: `/status` `/risk` `/restart` `/close SYMBOL` `/close_all` (then `/confirm`).
