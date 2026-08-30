# BTCC Operations — remote continuous run (independent of local PC)

The production environment is the **remote Linux server**.  
The bot must keep running when your laptop is off, offline, asleep, or disconnected from SSH/Cursor.

Service name: **`btcc`**

---

## 1. Install the service (once)

```bash
cd /path/to/BTCC          # e.g. /home/ubuntu/LCADAME/BTCC
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Secrets (never commit)
# .env must contain:
#   BTCC_TELEGRAM_BOT_TOKEN=...
#   BTCC_TELEGRAM_CHAT_ID=...

sudo bash scripts/install_btcc_service.sh
sudo systemctl start btcc
```

`install_btcc_service.sh` will:
- install `/etc/systemd/system/btcc.service`
- `systemctl enable btcc` (auto-start on boot)
- point `WorkingDirectory` + venv at this repo

---

## 2. Start

```bash
sudo systemctl start btcc
```

## 3. Stop

```bash
sudo systemctl stop btcc
```

## 4. Status

```bash
sudo systemctl status btcc
```

## 5. Logs

```bash
journalctl -u btcc -f          # live follow
journalctl -u btcc -n 200      # last 200 lines
journalctl -u btcc --since "1 hour ago"
```

## 6. Restart

```bash
sudo systemctl restart btcc
```

## 7. Verify SSH disconnect survival

1. Start the service: `sudo systemctl start btcc`
2. Confirm running: `systemctl is-active btcc` → `active`
3. Disconnect SSH / close Cursor
4. Reconnect later and run:

```bash
systemctl is-active btcc
journalctl -u btcc --since "10 min ago"
```

The process should still be `active` with new cycle logs.

## 8. Verify reboot survival

```bash
sudo reboot
# after reconnect:
systemctl is-active btcc
journalctl -u btcc -b          # logs since this boot
```

systemd should have started `btcc` automatically (`WantedBy=multi-user.target`).

---

## 9. Where persistent data lives

All under the BTCC project root (paths relative to repo):

| Path | Contents |
|------|----------|
| `data/candles/` | Live 15m OHLCV cache |
| `data/predictions/` | Legacy + adaptive prediction CSVs, BTC.D history |
| `data/sim/` | Adaptive V2 predictions, opportunities, legs, weight history, **`state.json`** |
| `data/sim/runtime_state.json` | Last decision candle / cycle success metadata |
| `data/models/` | Legacy Champion archive (if used) |
| `logs/` | Audits / research reports |

**90-day window controls learning only — research archives are not deleted.**

On restart the bot reloads:
- adaptive weights (+ pending 23:00 deferred weights)
- open virtual opportunities + crossing state machine
- last daily-update local date (idempotent)
- last processed decision candle (skip duplicate cycle)

## 10. Configuration / secrets

| Item | Location | In Git? |
|------|----------|---------|
| Strategy / universe | `configs/signal_config.yaml` | yes |
| Adaptive V2 / sim | `configs/sim_config.yaml` | yes |
| Telegram tokens | `.env` (`BTCC_TELEGRAM_*`) | **no** (gitignored) |
| Exchange trading keys | **none** — public MEXC REST only | n/a |

`safety.allow_trading: false` is enforced at startup. The service runs:

- live data collection  
- predictions  
- adaptive weight updates  
- virtual trade simulations  
- Telegram alerts  

It does **not** place real exchange orders.

---

## Network / API outages

- MEXC REST retries with backoff; failures mark symbols unavailable.
- Stale candles / unhealthy BTC.D → **no new simulated opportunities** (`DATA_HEALTH_BLOCK`).
- Cycle exceptions → logged, exponential backoff sleep, systemd `Restart=on-failure` if the process dies.
- When data is healthy again, the next **closed** 15m candle is processed (no duplicate of the last successful decision candle).

---

## Local PC independence checklist

| Dependency | Required? |
|------------|-----------|
| Your laptop powered on | **No** |
| Your home internet | **No** |
| Cursor / SSH session | **No** |
| Remote server up + network to MEXC/CoinGecko/Telegram | **Yes** |
| systemd `btcc` active | **Yes** |

Turn off your local computer; the remote service continues collecting data, predicting, simulating trades, saving state, updating weights at 23:00 America/Sao_Paulo, and sending Telegram alerts.
