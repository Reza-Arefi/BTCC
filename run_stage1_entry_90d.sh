#!/usr/bin/env bash
# Stage 1: pick entry (BASE/E1–E5) with freeze exit=T1, 90d, 1s exits.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs results
echo "Starting Stage 1 entry sweep (BASE/E1–E5, freeze T1, 90d 1s)…"
exec python -u -m binance_btc_bot.tools.run_stage1_entry_90d_1s
