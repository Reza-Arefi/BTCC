"""Configuration loader for the Binance BTC compounding bot."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "binance_bot.yaml"

FROZEN_STRATEGIES = {
    "T1": (0.0075, 0.0075, 0.0025),
    "T2": (0.01, 0.01, 0.0025),
    "T3": (0.0075, 0.0075, 0.005),
    "T4": (0.01, 0.01, 0.005),
    "T5": (0.015, 0.01, 0.005),
    "T6": (0.015, 0.015, 0.005),
    "T7": (0.02, 0.02, 0.005),
    "T8": (0.0075, 0.0075, 0.0075),
    "T9": (0.015, 0.015, 0.0025),
    "T10": (0.02, 0.015, 0.005),
}

EXPECTED_BTC_PAIRS = (
    "AAVEBTC",
    "ADABTC",
    "ARBBTC",
    "ATOMBTC",
    "AVAXBTC",
    "BCHBTC",
    "BNBBTC",
    "CAKEBTC",
    "DASHBTC",
    "DOGEBTC",
    "DOTBTC",
    "ETCBTC",
    "ETHBTC",
    "FETBTC",
    "FILBTC",
    "HBARBTC",
    "ICPBTC",
    "INJBTC",
    "LINKBTC",
    "LTCBTC",
    "NEARBTC",
    "NEXOBTC",
    "PAXGBTC",
    "PENDLEBTC",
    "RENDERBTC",
    "SOLBTC",
    "STXBTC",
    "SUIBTC",
    "TAOBTC",
    "THETABTC",
    "TRXBTC",
    "UNIBTC",
    "WBTCBTC",
    "WLDBTC",
    "XRPBTC",
    "XAUTBTC",
    "XTZBTC",
    "ZECBTC",
)


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"invalid config: {p}")
    cfg = deepcopy(raw)
    cfg["_config_path"] = str(p)
    cfg["_package_root"] = str(PACKAGE_ROOT)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    live = cfg.get("live") or {}
    strategy = str(live.get("strategy", "")).upper()
    allowed_live = frozenset(FROZEN_STRATEGIES.keys())
    if strategy not in allowed_live:
        raise ValueError(f"live.strategy must be one of {sorted(allowed_live)}")
    if live.get("selector") not in (None, "null", ""):
        raise ValueError("live.selector must be null (selectors not enabled live)")
    if bool((cfg.get("risk") or {}).get("no_leverage", True)) is not True:
        raise ValueError("risk.no_leverage must remain true (spot only)")
    if str((cfg.get("exchange") or {}).get("market", "")).lower() != "spot":
        raise ValueError("exchange.market must be spot")

    # Portfolio is authoritative for slots + per-trade allocation.
    from binance_btc_bot.portfolio.manager import validate_portfolio_config

    portfolio = validate_portfolio_config(cfg.get("portfolio"))
    cfg["_portfolio"] = {
        "max_simultaneous_trades": portfolio.max_simultaneous_trades,
        "allocation_per_trade": portfolio.allocation_per_trade,
        "max_total_allocation": portfolio.max_total_allocation,
        "one_position_per_symbol": portfolio.one_position_per_symbol,
        "total_planned_allocation": portfolio.total_planned_allocation,
    }

    risk = cfg.get("risk") or {}
    max_loss = float(risk.get("max_loss_per_trade", 0.005))
    if abs(max_loss - 0.005) > 1e-12:
        raise ValueError("risk.max_loss_per_trade must remain 0.005 (0.5% equity)")
    # Keep risk.max_allocation_pct aligned with portfolio (authoritative).
    risk = dict(risk)
    risk["max_allocation_pct"] = portfolio.allocation_per_trade
    risk["max_aggregate_exposure"] = portfolio.max_total_allocation
    cfg["risk"] = risk

    entry = cfg.get("entry") or {}
    thr = float(entry.get("long_threshold", 0.65))
    if abs(thr - 0.65) > 1e-12:
        raise ValueError("entry.long_threshold must be 0.65 for live entry layer")

    strategies = cfg.get("strategies") or {}
    for key, (sl, act, dist) in FROZEN_STRATEGIES.items():
        raw = strategies.get(key)
        if not raw:
            raise ValueError(f"missing frozen strategy {key}")
        if abs(float(raw["arm_sl_activation_trail"]) - sl) > 1e-12:
            raise ValueError(f"{key} arm_sl_activation_trail must stay {sl}")
        if abs(float(raw["activation"]) - act) > 1e-12:
            raise ValueError(f"{key} activation must stay {act}")
        if abs(float(raw["trail_distance"]) - dist) > 1e-12:
            raise ValueError(f"{key} trail_distance must stay {dist}")

    pairs = [str(x).upper() for x in (cfg.get("universe") or {}).get("btc_pairs") or []]
    if tuple(pairs) != EXPECTED_BTC_PAIRS:
        missing = sorted(set(EXPECTED_BTC_PAIRS) - set(pairs))
        extra = sorted(set(pairs) - set(EXPECTED_BTC_PAIRS))
        raise ValueError(f"universe.btc_pairs mismatch missing={missing} extra={extra}")

    if bool(live.get("enabled", False)):
        # Still allow loading, but live writes remain blocked elsewhere.
        pass
    if live.get("enabled") is not False and live.get("enabled") is not True:
        raise ValueError("live.enabled must be a boolean")
    if bool(live.get("enabled", False)) is True:
        # Config may be prepared, but operational default must stay false in this file.
        # Soft note only — hard refuse is in engine construction for writes.
        pass

def is_live_trading_enabled(cfg: dict[str, Any]) -> bool:
    """True only when live.enabled AND dry_run is false."""
    live = cfg.get("live") or {}
    return bool(live.get("enabled")) and not bool(live.get("dry_run", True))


def live_strategy_key(cfg: dict[str, Any]) -> str:
    return str((cfg.get("live") or {}).get("strategy") or "T4").upper()
