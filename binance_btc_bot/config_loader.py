"""Configuration loader for the Binance BTC compounding bot."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "binance_bot.yaml"
PRODUCTION_POINTER_PATH = REPO_ROOT / "configs" / "live_production.yaml"

# Frozen production identity from final T1–T30 selection.
PRODUCTION_CONFIG_VERSION = "T30_E2_T65_v1"
PRODUCTION_STRATEGY = "T30"
PRODUCTION_ENTRY_PROFILE = "e2"
PRODUCTION_THRESHOLD = 0.65
PRODUCTION_LATE_ENTRY = False

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
    # Live T21 = research wide-SL initial geometry (fixed OCO). Research adaptive
    # trail tightening (1.0%→0.50%→0.25% by peak) is NOT applied live — Binance
    # native OCO supports a single trailingDelta only.
    "T21": (0.05, 0.01, 0.01),
    # Live T30 = fixed OCO proxy from 21d study (SL -3% / act +1% / trail 0.25%).
    "T30": (0.03, 0.01, 0.0025),
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


def env_live_trading_enabled() -> bool:
    """Hard kill switch. Default false — unset/empty means disabled."""
    v = os.environ.get("LIVE_TRADING_ENABLED", "false")
    return str(v).strip().lower() in {"1", "true", "yes"}


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"invalid config: {p}")
    cfg = deepcopy(raw)
    cfg["_config_path"] = str(p)
    cfg["_package_root"] = str(PACKAGE_ROOT)
    _validate_config(cfg)
    cfg["_production_fingerprint"] = production_fingerprint(cfg)
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
    risk = dict(risk)
    risk["max_allocation_pct"] = portfolio.allocation_per_trade
    risk["max_aggregate_exposure"] = portfolio.max_total_allocation
    cfg["risk"] = risk

    entry = cfg.get("entry") or {}
    thr = float(entry.get("long_threshold", 0.65))
    if abs(thr - PRODUCTION_THRESHOLD) > 1e-12:
        raise ValueError(f"entry.long_threshold must be {PRODUCTION_THRESHOLD} for live entry layer")
    if bool(entry.get("late_entry_enabled", False)) is not False:
        raise ValueError("entry.late_entry_enabled must be false (late-entry OFF)")
    rule = str(entry.get("rule") or "NEW_CROSS").upper().replace("_", "")
    if rule not in {"NEWCROSS", ""}:
        raise ValueError("entry.rule must be NEW_CROSS")

    signal = cfg.get("signal") or {}
    mom = str(signal.get("momentum_profile") or "").lower()
    if mom != PRODUCTION_ENTRY_PROFILE:
        raise ValueError(f"signal.momentum_profile must be '{PRODUCTION_ENTRY_PROFILE}' for production")

    prod = cfg.get("production") or {}
    ver = str(prod.get("config_version") or "")
    if ver and ver != PRODUCTION_CONFIG_VERSION:
        raise ValueError(
            f"production.config_version must be '{PRODUCTION_CONFIG_VERSION}' (got {ver!r})"
        )
    if ver == PRODUCTION_CONFIG_VERSION or prod.get("validated_strategy"):
        if str(live.get("strategy") or "").upper() != PRODUCTION_STRATEGY:
            raise ValueError(f"live.strategy must be {PRODUCTION_STRATEGY} for production freeze")

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

    if live.get("enabled") is not False and live.get("enabled") is not True:
        raise ValueError("live.enabled must be a boolean")


def production_fingerprint(cfg: dict[str, Any]) -> dict[str, Any]:
    """Stable identity for every live trade / archive row."""
    live = cfg.get("live") or {}
    entry = cfg.get("entry") or {}
    signal = cfg.get("signal") or {}
    t30 = (cfg.get("strategies") or {}).get("T30") or {}
    prod = cfg.get("production") or {}
    return {
        "config_version": str(prod.get("config_version") or PRODUCTION_CONFIG_VERSION),
        "strategy": str(live.get("strategy") or "").upper(),
        "entry_profile": str(signal.get("momentum_profile") or "").lower(),
        "threshold": float(entry.get("long_threshold") or 0),
        "entry_rule": str(entry.get("rule") or "NEW_CROSS").upper(),
        "late_entry": bool(entry.get("late_entry_enabled", False)),
        "selector": live.get("selector"),
        "t30_sl": float(t30.get("arm_sl_activation_trail") or 0),
        "t30_activation": float(t30.get("activation") or 0),
        "t30_trail": float(t30.get("trail_distance") or 0),
        "live_enabled_yaml": bool(live.get("enabled", False)),
        "dry_run_yaml": bool(live.get("dry_run", True)),
        "LIVE_TRADING_ENABLED": env_live_trading_enabled(),
        "authoritative_config": str(cfg.get("_config_path") or DEFAULT_CONFIG_PATH),
    }


def validate_production_freeze(cfg: dict[str, Any]) -> list[str]:
    """Return list of mismatch messages (empty = OK)."""
    fp = production_fingerprint(cfg)
    errs: list[str] = []
    if fp["strategy"] != PRODUCTION_STRATEGY:
        errs.append(f"strategy={fp['strategy']} (expected {PRODUCTION_STRATEGY})")
    if fp["entry_profile"] != PRODUCTION_ENTRY_PROFILE:
        errs.append(f"entry_profile={fp['entry_profile']} (expected {PRODUCTION_ENTRY_PROFILE})")
    if abs(float(fp["threshold"]) - PRODUCTION_THRESHOLD) > 1e-12:
        errs.append(f"threshold={fp['threshold']} (expected {PRODUCTION_THRESHOLD})")
    if fp["late_entry"] is not False:
        errs.append("late_entry must be false")
    if fp["selector"] not in (None, "null", ""):
        errs.append("selector must be null")
    if abs(float(fp["t30_sl"]) - 0.03) > 1e-12:
        errs.append("T30 SL must be 0.03")
    if abs(float(fp["t30_activation"]) - 0.01) > 1e-12:
        errs.append("T30 activation must be 0.01")
    if abs(float(fp["t30_trail"]) - 0.0025) > 1e-12:
        errs.append("T30 trail must be 0.0025")
    if fp["config_version"] != PRODUCTION_CONFIG_VERSION:
        errs.append(f"config_version={fp['config_version']} (expected {PRODUCTION_CONFIG_VERSION})")
    return errs


def format_check_config_report(cfg: dict[str, Any]) -> str:
    fp = production_fingerprint(cfg)
    errs = validate_production_freeze(cfg)
    lines = [
        "=== PRODUCTION CONFIG CHECK ===",
        f"authoritative_config: {fp['authoritative_config']}",
        f"config_version:       {fp['config_version']}",
        f"strategy:             {fp['strategy']}",
        f"entry_profile:        {fp['entry_profile']}",
        f"threshold:            {fp['threshold']}",
        f"entry_rule:           {fp['entry_rule']}",
        f"late_entry:           {fp['late_entry']}",
        f"selector:             {fp['selector']}",
        f"T30 SL:               {fp['t30_sl']} (3%)",
        f"T30 activation:       {fp['t30_activation']} (1%)",
        f"T30 trail:            {fp['t30_trail']} (0.25%)",
        f"live.enabled (YAML):  {fp['live_enabled_yaml']}",
        f"live.dry_run (YAML):  {fp['dry_run_yaml']}",
        f"LIVE_TRADING_ENABLED: {fp['LIVE_TRADING_ENABLED']}",
        "",
    ]
    if errs:
        lines.append("STATUS: FAIL")
        lines.extend(f"  - {e}" for e in errs)
    else:
        lines.append("STATUS: OK — matches frozen T30_E2_T65_v1 fingerprint")
    if not fp["LIVE_TRADING_ENABLED"]:
        lines.append("")
        lines.append("LIVE TRADING DISABLED — NO ORDERS WILL BE SUBMITTED")
    return "\n".join(lines)


def is_live_trading_enabled(cfg: dict[str, Any]) -> bool:
    """True only when env kill-switch AND YAML live.enabled AND dry_run is false."""
    if not env_live_trading_enabled():
        return False
    live = cfg.get("live") or {}
    return bool(live.get("enabled")) and not bool(live.get("dry_run", True))


def live_strategy_key(cfg: dict[str, Any]) -> str:
    return str((cfg.get("live") or {}).get("strategy") or PRODUCTION_STRATEGY).upper()
