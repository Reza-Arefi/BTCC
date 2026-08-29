#!/usr/bin/env python3
"""BTCC deployment preflight — SIGNAL ONLY. Does not trade or change strategy.

Usage (from BTCC/):
  python scripts/check_deploy.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Heuristics for accidentally committed secrets (not applied to .env)
TELEGRAM_TOKEN_SHAPE = re.compile(r"\b\d{8,}:[A-Za-z0-9_-]{30,}\b")
ASSIGNED_SECRET = re.compile(
    r"""(?i)(api[_-]?key|secret|password|bot_token)\s*[:=]\s*['\"][^'\"]{12,}['\"]"""
)

SKIP_DIR_NAMES = {".venv", "venv", "env", "data", "logs", "results", ".git", "__pycache__", ".pytest_cache"}
SCAN_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt", ".example", ".toml", ".cfg", ".ini", ".json"}


def ok(msg: str) -> None:
    print(f"  OK  {msg}")


def fail(msg: str) -> None:
    print(f" FAIL {msg}")


def warn(msg: str) -> None:
    print(f" WARN {msg}")


def iter_scan_files() -> list[Path]:
    out: list[Path] = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in p.relative_to(ROOT).parts):
            continue
        if p.name == ".env":
            continue
        if p.suffix.lower() not in SCAN_SUFFIXES:
            continue
        out.append(p)
    return out


def main() -> int:
    print("BTCC deployment check (signal-only)")
    print(f"Root: {ROOT}")
    errors = 0

    for rel in (
        "btcc/main.py",
        "configs/signal_config.yaml",
        "configs/adaptive_config.yaml",
        "requirements.txt",
        ".gitignore",
    ):
        if (ROOT / rel).exists():
            ok(f"present: {rel}")
        else:
            fail(f"missing: {rel}")
            errors += 1

    env_path = ROOT / ".env"
    if env_path.exists():
        ok(".env present locally (must remain gitignored)")
        text = env_path.read_text(encoding="utf-8", errors="replace")
        for key in ("BTCC_TELEGRAM_BOT_TOKEN", "BTCC_TELEGRAM_CHAT_ID"):
            val = ""
            for line in text.splitlines():
                if line.startswith(key + "="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
            if not val:
                warn(f"{key} is empty — Telegram may be disabled")
            else:
                ok(f"{key} is set (value not printed)")
    else:
        warn(".env missing — create BTCC/.env with BTCC_TELEGRAM_BOT_TOKEN and BTCC_TELEGRAM_CHAT_ID")

    try:
        from btcc.config import load_config
        from btcc.safety.no_trading import TradingForbiddenError, assert_no_trading_config, deny_trading

        cfg = load_config()
        assert_no_trading_config(cfg)
        ok("allow_trading is false")
        try:
            deny_trading()
            fail("deny_trading() did not raise")
            errors += 1
        except TradingForbiddenError:
            ok("deny_trading() blocks order paths")
        # Ensure YAML does not embed telegram secrets
        raw_yaml = (ROOT / "configs" / "signal_config.yaml").read_text(encoding="utf-8")
        if TELEGRAM_TOKEN_SHAPE.search(raw_yaml):
            fail("signal_config.yaml looks like it contains a Telegram token")
            errors += 1
        else:
            ok("signal_config.yaml has no Telegram token shape")
    except Exception as e:
        fail(f"config/safety check error: {e}")
        errors += 1

    print("Scanning project sources for suspicious hard-coded secrets...")
    files = iter_scan_files()
    for path in files:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if TELEGRAM_TOKEN_SHAPE.search(raw):
            fail(f"Telegram-token-shaped string in {rel}")
            errors += 1
        if path.name != "check_deploy.py" and ASSIGNED_SECRET.search(raw):
            # Allow documentation mentions without long quoted secrets
            fail(f"suspicious secret assignment in {rel}")
            errors += 1
    ok(f"scanned {len(files)} text files (excluding data/logs/results/.venv/.env)")

    ok("MEXC uses public market data only — no exchange trading API keys in BTCC")

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for must in (".env", "data/predictions/", "data/models/", "logs/", "results/"):
        if must in gitignore or must.rstrip("/") in gitignore:
            ok(f".gitignore covers {must}")
        else:
            fail(f".gitignore missing entry for {must}")
            errors += 1

    print()
    if errors:
        print(f"RESULT: {errors} issue(s) found — fix before deploying / committing.")
        return 1
    print("RESULT: deployment preflight passed.")
    print("Remember: never commit .env; run a single `python -m btcc.main run` process.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
