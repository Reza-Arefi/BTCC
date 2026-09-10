#!/usr/bin/env bash
# Download 15m + full-window 1s Binance candles, then run 90d multiarm (T1–T10 + A–F).
set -euo pipefail
cd "$(dirname "$0")"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
echo "PYTHONPATH=$PYTHONPATH"
echo "Starting 90d multiarm with 1s exits…"
python3 -m binance_btc_bot.tools.run_multiarm_1m_backtest
