"""Structured logging setup with secret scrubbing."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from binance_btc_bot.secrets import ScrubbingFilter


def setup_logging(level: str = "INFO", log_dir: str | Path | None = None) -> None:
    root = logging.getLogger()
    scrub = ScrubbingFilter()
    if root.handlers:
        for h in root.handlers:
            h.addFilter(scrub)
        return
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.addFilter(scrub)
    root.addHandler(sh)
    if log_dir:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path / "binance_btc_bot.log")
        fh.setFormatter(fmt)
        fh.addFilter(scrub)
        root.addHandler(fh)
