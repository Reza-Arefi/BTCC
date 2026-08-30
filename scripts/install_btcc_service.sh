#!/usr/bin/env bash
# Install / update the BTCC systemd service (requires sudo).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(readlink -f "$ROOT")"
UNIT_SRC="$ROOT/deploy/btcc.service"
UNIT_DST="/etc/systemd/system/btcc.service"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: venv python not found at $PY"
  echo "Create it first: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

if [[ ! -f "$ROOT/.env" ]]; then
  echo "WARN: $ROOT/.env missing — Telegram will be disabled until created."
fi

# Rewrite WorkingDirectory / paths in a temp unit if ROOT differs from template
TMP="$(mktemp)"
sed \
  -e "s|/opt/dlami/nvme/tmp_storage/LCADAME/BTCC|$ROOT|g" \
  "$UNIT_SRC" > "$TMP"

echo "Installing unit → $UNIT_DST"
sudo cp "$TMP" "$UNIT_DST"
rm -f "$TMP"
sudo systemctl daemon-reload
sudo systemctl enable btcc.service

echo
echo "Installed and enabled. Useful commands:"
echo "  sudo systemctl start btcc"
echo "  sudo systemctl status btcc"
echo "  sudo systemctl stop btcc"
echo "  sudo systemctl restart btcc"
echo "  journalctl -u btcc -f"
echo
echo "NOTE: service runs paper simulation only (allow_trading=false)."
echo "Start now?  sudo systemctl start btcc"
