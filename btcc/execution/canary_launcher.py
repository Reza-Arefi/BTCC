"""Isolated REAL T1 single-shot canary launcher (NOT btcc.service).

Modes:
  preflight  — authenticated reads + sizing checks; ZERO writes
  run        — requires explicit arming env vars; at most one BUY/position/SELL

Arming (both required for writes; never inferred from config alone):
  BTCC_ALLOW_TRADING=true   (or ALLOW_TRADING=true)
  CANARY_WRITES_ARMED=true  (or BTCC_CANARY_WRITES_ARMED=true)

Default config keeps allow_trading: false. This module never mutates the
paper bot (E-v1-25pct-PAPER / btcc.service).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from btcc.execution.canary import CanaryPhase
from btcc.execution.factory import build_canary_stack
from btcc.execution.mexc.client import MexcReadOnlyClient
from btcc.execution.mexc.credentials import load_mexc_credentials, redact_secrets
from btcc.execution.mexc.write_client import MexcWriteClient
from btcc.execution.price_monitor import MexcPublicTickerSource, PriceMonitorState
from btcc.execution.risk import RiskContext
from btcc.execution.sizing import size_alt_btc_position
from btcc.execution.t1 import T1
from btcc.execution.write_gate import WriteGate
from btcc.safety.no_trading import TradingForbiddenError

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "selector_t1_production_config.yaml"
DEFAULT_STATE_DIR = ROOT / "data" / "canary_t1"

# Explicit canary asset classification (preflight / inventory invariant only).
# Sizing uses free BTC only — dust/account assets are never capital.
TRADING_CAPITAL_ASSETS = frozenset({"BTC"})
STABLE_ACCOUNT_ASSETS = frozenset({"USDT", "USDC"})
# Explicitly allowed non-trading account/dust assets. Not capital. Not positions.
# Do NOT auto-sell/convert. Still visible in account totals / reconciliation.
ALLOWED_ACCOUNT_DUST_ASSETS = frozenset({"MX"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_preflight_inventory(
    balances: list[Any],
    *,
    symbol_base: str | None = None,
    allow_base_inventory: bool = False,
) -> dict[str, Any]:
    """Classify balances for canary preflight.

    - BTC = trading capital (sizing input lives in available BTC path separately)
    - USDT/USDC = allowed stables (non-trading for ALT/BTC canary)
    - MX = explicitly allowed account/dust (non-trading; never sized; not a position)
    - symbol base (e.g. ETH) = unexpected before BUY unless allow_base_inventory
    - any other asset with total > 0 = UNEXPECTED_EXCHANGE_STATE
    """
    capital: list[str] = []
    stables: list[str] = []
    allowed_dust: list[dict[str, Any]] = []
    unexpected: list[str] = []
    base = (symbol_base or "").upper()

    for b in balances:
        asset = str(getattr(b, "asset", "")).upper()
        total = float(getattr(b, "total", 0) or 0)
        if total <= 0:
            continue
        if asset in TRADING_CAPITAL_ASSETS:
            capital.append(asset)
            continue
        if asset in STABLE_ACCOUNT_ASSETS:
            stables.append(asset)
            continue
        if asset in ALLOWED_ACCOUNT_DUST_ASSETS:
            allowed_dust.append(
                {
                    "asset": asset,
                    "total": total,
                    "free": float(getattr(b, "free", 0) or 0),
                    "locked": float(getattr(b, "locked", 0) or 0),
                    "role": "ALLOWED_ACCOUNT_DUST",
                    "trading_inventory": False,
                    "sized": False,
                }
            )
            continue
        if base and asset == base and allow_base_inventory:
            continue
        unexpected.append(asset)

    return {
        "capital": capital,
        "stables": stables,
        "allowed_dust": allowed_dust,
        "unexpected": unexpected,
    }


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def canary_writes_armed_from_env() -> bool:
    return _env_truthy("CANARY_WRITES_ARMED") or _env_truthy("BTCC_CANARY_WRITES_ARMED")


def allow_trading_from_env() -> bool:
    return _env_truthy("BTCC_ALLOW_TRADING") or _env_truthy("ALLOW_TRADING")


def assert_explicit_canary_write_arming() -> None:
    """Fail closed unless BOTH allow_trading and canary write arming are explicit."""
    if not allow_trading_from_env():
        raise TradingForbiddenError(
            "Canary writes blocked: set BTCC_ALLOW_TRADING=true (explicit). "
            "Config allow_trading remains false by default."
        )
    if not canary_writes_armed_from_env():
        raise TradingForbiddenError(
            "Canary writes blocked: set CANARY_WRITES_ARMED=true (or BTCC_CANARY_WRITES_ARMED=true)."
        )


def load_canary_config(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"invalid canary config: {p}")
    # Hard isolation: never load paper selector_live config here
    if "selector_live" in raw:
        raise TradingForbiddenError("refusing paper selector_live config in canary launcher")
    return raw


@dataclass
class PreflightCheck:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class PreflightReport:
    ok: bool
    ts: str
    symbol: str
    checks: list[PreflightCheck] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    available_btc: float | None = None
    price_alt_btc: float | None = None
    sized_quantity: float | None = None
    notional_btc: float | None = None
    writes_attempted: bool = False
    allowed_account_dust: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "ts": self.ts,
            "symbol": self.symbol,
            "checks": [asdict(c) for c in self.checks],
            "blockers": list(self.blockers),
            "available_btc": self.available_btc,
            "price_alt_btc": self.price_alt_btc,
            "sized_quantity": self.sized_quantity,
            "notional_btc": self.notional_btc,
            "writes_attempted": self.writes_attempted,
            "allowed_account_dust": list(self.allowed_account_dust),
            "asset_policy": {
                "trading_capital": sorted(TRADING_CAPITAL_ASSETS),
                "stable_account": sorted(STABLE_ACCOUNT_ASSETS),
                "allowed_account_dust": sorted(ALLOWED_ACCOUNT_DUST_ASSETS),
                "note": "Dust assets are visible in account totals but never sized or treated as positions.",
            },
            "t1": {
                "stop_loss_pct": T1.stop_loss_pct,
                "activation_pct": T1.activation_pct,
                "trailing_pct": T1.trailing_pct,
            },
        }


def _add(
    report: PreflightReport, name: str, ok: bool, detail: str = ""
) -> None:
    report.checks.append(PreflightCheck(name=name, ok=ok, detail=detail))
    if not ok:
        report.blockers.append(f"{name}: {detail}")


def run_preflight(
    *,
    symbol: str,
    config_path: Path | None = None,
    client: MexcReadOnlyClient | None = None,
    max_skew_ms: int = 5000,
) -> PreflightReport:
    """Authenticated read-only preflight. Never places orders."""
    symbol = str(symbol).upper().replace("/", "").replace("-", "").replace("_", "")
    report = PreflightReport(ok=False, ts=_utc_now(), symbol=symbol, writes_attempted=False)
    cfg = load_canary_config(config_path)

    # Config gates must remain closed in file
    if bool(cfg.get("allow_trading", False)):
        _add(report, "config_allow_trading", False, "committed/config allow_trading must be false")
    else:
        _add(report, "config_allow_trading", True, "false")

    try:
        creds = load_mexc_credentials(require=True)
    except Exception as e:  # noqa: BLE001
        _add(report, "credentials", False, redact_secrets(str(e)))
        report.ok = False
        return report
    assert creds is not None
    _add(report, "credentials", True, "loaded from environment")

    ro = client or MexcReadOnlyClient(creds)
    try:
        server_ms = ro.get_server_time()
        local_ms = int(time.time() * 1000)
        skew = abs(server_ms - local_ms)
        _add(report, "server_time", skew <= max_skew_ms, f"skew_ms={skew}")
    except Exception as e:  # noqa: BLE001
        _add(report, "server_time", False, redact_secrets(str(e), creds))

    try:
        meta = ro.get_symbol_metadata(symbol)
        ok_meta = bool(meta.is_trading and meta.quote_asset.upper() == "BTC")
        _add(
            report,
            "exchangeInfo_symbol",
            ok_meta,
            f"status={meta.status} base={meta.base_asset} quote={meta.quote_asset} "
            f"step={meta.quantity_step} min_qty={meta.min_quantity}",
        )
    except Exception as e:  # noqa: BLE001
        meta = None
        _add(report, "exchangeInfo_symbol", False, redact_secrets(str(e), creds))

    try:
        snap = ro.get_account_snapshot(quote_asset="BTC")
        report.available_btc = float(snap.available_balance)
        _add(
            report,
            "account_btc",
            report.available_btc is not None and report.available_btc >= 0,
            f"available={report.available_btc} locked={snap.locked_balance}",
        )
    except Exception as e:  # noqa: BLE001
        _add(report, "account_btc", False, redact_secrets(str(e), creds))

    try:
        opens = ro.get_open_orders(symbol=symbol)
        # Unexpected open orders for the canary symbol → block
        _add(
            report,
            "open_orders",
            len(opens) == 0,
            f"count={len(opens)}",
        )
    except Exception as e:  # noqa: BLE001
        _add(report, "open_orders", False, redact_secrets(str(e), creds))

    try:
        fills = ro.get_recent_fills(symbol, limit=5)
        _add(report, "fills_readable", True, f"recent_returned={len(fills)}")
    except Exception as e:  # noqa: BLE001
        _add(report, "fills_readable", False, redact_secrets(str(e), creds))

    # Public ticker for sizing (no write)
    try:
        tick = MexcPublicTickerSource(base_url=creds.base_url).fetch_price(symbol)
        report.price_alt_btc = float(tick.price)
        _add(report, "ticker_price", report.price_alt_btc > 0, f"price={report.price_alt_btc}")
    except Exception as e:  # noqa: BLE001
        _add(report, "ticker_price", False, redact_secrets(str(e), creds))

    if meta is not None and report.available_btc is not None and report.price_alt_btc:
        size = size_alt_btc_position(
            available_btc=float(report.available_btc),
            price_alt_btc=float(report.price_alt_btc),
            meta=meta,
            allocation_pct=float((cfg.get("sizing") or {}).get("position_allocation_pct", 0.25)),
        )
        report.sized_quantity = size.quantity if size.ok else None
        report.notional_btc = size.notional_btc if size.ok else None
        _add(
            report,
            "sizing_25pct",
            size.ok,
            (
                size.reason
                if not size.ok
                else (
                    f"qty={size.quantity} raw_qty={size.raw_quantity} "
                    f"serialized={size.quantity_serialized!r}"
                )
            ),
        )
        if size.ok and report.available_btc > 0:
            frac = float(size.notional_btc) / float(report.available_btc)
            _add(report, "sizing_not_over_25pct", frac <= 0.25 + 1e-9, f"frac={frac}")

    # Inventory invariant: trading capital vs allowed account/dust vs unexpected ALTs.
    # Pre-canary: symbol base (ETH for ETHBTC) with balance > 0 is unexpected.
    # MX is an explicitly allowed non-trading account asset — not capital, not a position.
    try:
        bals = ro.get_balances()
        base_asset = meta.base_asset if meta is not None else None
        classified = classify_preflight_inventory(
            bals,
            symbol_base=base_asset,
            allow_base_inventory=False,
        )
        report.allowed_account_dust = list(classified["allowed_dust"])
        if classified["allowed_dust"]:
            dust_summary = ",".join(
                f"{d['asset']}={d['total']}" for d in classified["allowed_dust"]
            )
            _add(
                report,
                "allowed_account_dust",
                True,
                f"{dust_summary} (non-trading; not sized; not a position; not auto-sold)",
            )
        else:
            _add(report, "allowed_account_dust", True, "none")
        _add(
            report,
            "no_unexpected_alt_inventory",
            len(classified["unexpected"]) == 0,
            (
                "none"
                if not classified["unexpected"]
                else f"UNEXPECTED_EXCHANGE_STATE assets={classified['unexpected'][:10]}"
            ),
        )
    except Exception as e:  # noqa: BLE001
        _add(report, "no_unexpected_alt_inventory", False, redact_secrets(str(e), creds))

    _add(report, "t1_params", True, f"sl={T1.stop_loss_pct} act={T1.activation_pct} trail={T1.trailing_pct}")
    _add(report, "writes_attempted", True, "false")

    report.ok = len(report.blockers) == 0
    return report


def build_isolated_canary_stack(
    *,
    state_dir: Path | None = None,
    config_path: Path | None = None,
    arm_writes: bool = False,
    write_client: Any | None = None,
    read_client: Any | None = None,
    credentials: Any | None = None,
    price_source: Any | None = None,
) -> dict[str, Any]:
    """Build canary stack isolated from paper. arm_writes requires env arming when True."""
    cfg = load_canary_config(config_path)
    root = Path(state_dir) if state_dir else DEFAULT_STATE_DIR
    root.mkdir(parents=True, exist_ok=True)

    allow = bool(arm_writes) and allow_trading_from_env() and canary_writes_armed_from_env()
    if arm_writes and not allow:
        assert_explicit_canary_write_arming()

    creds = credentials
    if creds is None and (arm_writes or read_client is None):
        creds = load_mexc_credentials(require=False)

    wc = write_client
    if allow and wc is None and creds is not None:
        gate = WriteGate(allow_trading=True, canary_writes_armed=True, protection_api_confirmed=False)
        wc = MexcWriteClient(creds, gate=gate)

    src = price_source
    if src is None and allow:
        src = MexcPublicTickerSource(base_url=(creds.base_url if creds else "https://api.mexc.com"))

    stack = build_canary_stack(
        root_dir=root,
        client=read_client,
        write_client=wc,
        credentials=creds,
        allow_trading=allow,
        arm_writes=allow,
        protection_api_confirmed=False,
        max_positions=1,
        allocation_pct=float((cfg.get("sizing") or {}).get("position_allocation_pct", 0.25)),
        strategy_version=str(cfg.get("version") or "T1-ONLY-CANARY"),
        price_source=src,
    )
    stack["config"] = cfg
    stack["state_dir"] = root
    stack["armed"] = allow
    stack["paper_service"] = "UNTOUCHED"
    return stack


def disarm_after_canary_complete(stack: dict[str, Any], *, reason: str = "CANARY_COMPLETE") -> None:
    """Make further writes unavailable in-process after the single shot."""
    rt = stack["runtime"]
    if rt.canary.phase != CanaryPhase.CANARY_COMPLETE and reason == "CANARY_COMPLETE":
        # still disarm on halt/complete paths
        pass
    rt.kill_switch.halt_submissions(reason)
    # Replace gate references with closed gate (WriteGate is frozen)
    closed = WriteGate(allow_trading=False, canary_writes_armed=False, protection_api_confirmed=False)
    rt.gate = closed
    stack["gate"] = closed
    broker = stack.get("broker")
    if broker is not None:
        broker.gate = closed
        if getattr(broker, "_write", None) is not None:
            broker._write.gate = closed
    stack["armed"] = False
    rt.audit.record("CANARY_WRITES_DISARMED", reason=reason)


@dataclass
class TerminalWaitResult:
    terminal: str  # CANARY_COMPLETE | HALTED | TIMEOUT | FATAL
    phase: str
    elapsed_s: float
    waited: bool = True
    halt_reason: str | None = None
    error: str | None = None


def _base_asset_from_symbol(symbol: str) -> str:
    sym = str(symbol).upper().replace("/", "").replace("-", "").replace("_", "")
    if sym.endswith("BTC") and len(sym) > 3:
        return sym[:-3]
    return sym


def read_base_inventory(broker: Any, *, base_asset: str) -> float | None:
    """Return free+locked base qty, or None if balances unavailable."""
    try:
        if hasattr(broker, "get_balances_detailed"):
            bals = broker.get_balances_detailed()
            for b in bals:
                if str(getattr(b, "asset", "")).upper() == base_asset.upper():
                    return float(getattr(b, "total", 0) or 0)
            return 0.0
        if hasattr(broker, "get_balances"):
            bals = broker.get_balances()
            if isinstance(bals, dict):
                return float(bals.get(base_asset.upper()) or bals.get(base_asset) or 0.0)
    except Exception as e:  # noqa: BLE001
        logger.warning("inventory_read_failed: %s", e)
        return None
    return None


def wait_for_canary_terminal(
    runtime: Any,
    *,
    poll_interval_s: float = 0.05,
    timeout_s: float | None = None,
    sleep_fn: Any = time.sleep,
    clock_fn: Any = time.monotonic,
) -> TerminalWaitResult:
    """Block until CANARY_COMPLETE / HALTED (or optional timeout).

    Stale/reconnect keeps phase=OPEN — this wait intentionally continues.
    Does not return merely because the monitor thread started.
    """
    t0 = float(clock_fn())
    try:
        while True:
            phase = runtime.canary.phase
            if phase == CanaryPhase.CANARY_COMPLETE:
                return TerminalWaitResult(
                    terminal="CANARY_COMPLETE",
                    phase=phase.value,
                    elapsed_s=float(clock_fn()) - t0,
                    waited=True,
                )
            if phase == CanaryPhase.HALTED or bool(getattr(runtime, "halted", False)):
                return TerminalWaitResult(
                    terminal="HALTED",
                    phase=CanaryPhase.HALTED.value,
                    elapsed_s=float(clock_fn()) - t0,
                    waited=True,
                    halt_reason=getattr(runtime, "halt_reason", None)
                    or getattr(runtime.canary, "halt_reason", None),
                )
            if timeout_s is not None and (float(clock_fn()) - t0) >= float(timeout_s):
                return TerminalWaitResult(
                    terminal="TIMEOUT",
                    phase=phase.value,
                    elapsed_s=float(clock_fn()) - t0,
                    waited=True,
                    error="wait_timeout_while_non_terminal",
                )
            # Observability only — hourly LIVE report; never places orders.
            try:
                maybe = getattr(runtime, "maybe_send_hourly_report", None)
                if callable(maybe):
                    maybe()
            except Exception:  # noqa: BLE001
                pass
            sleep_fn(float(poll_interval_s))
    except Exception as e:  # noqa: BLE001
        return TerminalWaitResult(
            terminal="FATAL",
            phase=getattr(getattr(runtime, "canary", None), "phase", CanaryPhase.HALTED).value
            if getattr(runtime, "canary", None) is not None
            else "HALTED",
            elapsed_s=float(clock_fn()) - t0,
            waited=True,
            error=str(e),
        )


def finalize_canary_lifecycle(
    stack: dict[str, Any],
    *,
    symbol: str,
    inventory_flat_eps: float = 1e-12,
) -> dict[str, Any]:
    """Stop monitor, verify inventory, disarm only when safe.

    Success: CANARY_COMPLETE + flat base inventory → disarm WriteGate.
    HALT with remaining inventory: attempt emergency sell once; do not claim
    CANARY_COMPLETE; do not open another entry. Disarm only if flat.
    """
    rt = stack["runtime"]
    broker = stack["broker"]
    base = _base_asset_from_symbol(symbol)
    out: dict[str, Any] = {
        "phase": rt.canary.phase.value,
        "halt_reason": getattr(rt, "halt_reason", None) or rt.canary.halt_reason,
        "base_asset": base,
        "inventory": None,
        "inventory_flat": None,
        "position_remaining": None,
        "disarmed": False,
        "canary_complete": rt.canary.phase == CanaryPhase.CANARY_COMPLETE,
        "emergency_sell_attempted": False,
    }

    # Always stop monitor before process exit (do not rely on daemon lifetime).
    try:
        rt.stop_price_monitor()
    except Exception as e:  # noqa: BLE001
        out["monitor_stop_error"] = str(e)

    inv = read_base_inventory(broker, base_asset=base)
    out["inventory"] = inv
    flat = inv is not None and inv <= inventory_flat_eps
    out["inventory_flat"] = flat if inv is not None else None
    out["position_remaining"] = (inv is not None and inv > inventory_flat_eps)

    if rt.canary.phase == CanaryPhase.CANARY_COMPLETE:
        if inv is None:
            # Fail closed: cannot confirm flat → do not claim clean success disarm? 
            # Still disarm entries (complete) but flag unknown inventory.
            out["inventory_unknown"] = True
            disarm_after_canary_complete(stack, reason="CANARY_COMPLETE_INVENTORY_UNREADABLE")
            out["disarmed"] = True
            out["ok"] = False
            out["reason"] = "CANARY_COMPLETE_BUT_INVENTORY_UNREADABLE"
            return out
        if not flat:
            rt.halt(f"CANARY_COMPLETE_BUT_INVENTORY_REMAINS:{base}={inv}")
            out["phase"] = rt.canary.phase.value
            out["canary_complete"] = False
            out["ok"] = False
            out["reason"] = rt.halt_reason
            # Keep WriteGate open for protective exit attempt
            try:
                pid = rt.canary.position_id
                if pid and inv > inventory_flat_eps:
                    out["emergency_sell_attempted"] = True
                    rt._emergency_market_sell(
                        symbol=symbol,
                        position_id=pid,
                        quantity=float(inv),
                        entry_coid=f"canary-{pid[:8]}",
                    )
                    inv2 = read_base_inventory(broker, base_asset=base)
                    out["inventory"] = inv2
                    flat2 = inv2 is not None and inv2 <= inventory_flat_eps
                    out["inventory_flat"] = flat2 if inv2 is not None else None
                    out["position_remaining"] = inv2 is not None and inv2 > inventory_flat_eps
                    if flat2:
                        disarm_after_canary_complete(stack, reason="HALTED_THEN_FLATTENED")
                        out["disarmed"] = True
            except Exception as e:  # noqa: BLE001
                out["emergency_sell_error"] = str(e)
            return out
        disarm_after_canary_complete(stack, reason="CANARY_COMPLETE")
        out["disarmed"] = True
        out["ok"] = True
        out["reason"] = "CANARY_COMPLETE"
        return out

    # HALTED / other terminal
    if out["position_remaining"]:
        try:
            pid = rt.canary.position_id or "unknown"
            out["emergency_sell_attempted"] = True
            rt._emergency_market_sell(
                symbol=symbol,
                position_id=str(pid),
                quantity=float(inv or 0.0),
                entry_coid=f"canary-{(pid or 'x')[:8]}",
            )
            inv2 = read_base_inventory(broker, base_asset=base)
            out["inventory"] = inv2
            flat2 = inv2 is not None and inv2 <= inventory_flat_eps
            out["inventory_flat"] = flat2 if inv2 is not None else None
            out["position_remaining"] = inv2 is not None and inv2 > inventory_flat_eps
            if flat2:
                disarm_after_canary_complete(stack, reason=out.get("halt_reason") or "HALTED_FLATTENED")
                out["disarmed"] = True
                out["ok"] = False
                out["reason"] = "HALTED_THEN_FLATTENED"
                return out
        except Exception as e:  # noqa: BLE001
            out["emergency_sell_error"] = str(e)
        # Still open — do NOT disarm (exits may still be needed by operator tools)
        # and do NOT claim CANARY_COMPLETE.
        out["ok"] = False
        out["reason"] = "HALTED_POSITION_REMAINING"
        out["armed"] = bool(stack.get("armed"))
        return out

    # Flat after halt (or inventory unread but halt with no known base)
    disarm_after_canary_complete(stack, reason=out.get("halt_reason") or "HALTED")
    out["disarmed"] = True
    out["ok"] = False
    out["reason"] = out.get("halt_reason") or "HALTED"
    return out


def run_single_shot(
    *,
    symbol: str,
    s_value: float,
    price_alt_btc: float | None = None,
    config_path: Path | None = None,
    state_dir: Path | None = None,
    require_preflight: bool = True,
    intent_nonce: str = "canary1",
    stack: dict[str, Any] | None = None,
    wait_poll_interval_s: float = 0.05,
    wait_timeout_s: float | None = None,
    sleep_fn: Any = time.sleep,
) -> dict[str, Any]:
    """Execute at most one canary entry and wait for terminal state.

    Remains alive through monitoring until CANARY_COMPLETE / HALTED / timeout.
    Never returns solely because monitoring started.
    """
    assert_explicit_canary_write_arming()
    symbol_n = str(symbol).upper().replace("/", "").replace("-", "").replace("_", "")

    if require_preflight:
        pf = run_preflight(symbol=symbol_n, config_path=config_path)
        if not pf.ok:
            raise TradingForbiddenError(f"preflight failed: {pf.blockers}")
    else:
        pf = None

    stack = stack or build_isolated_canary_stack(
        state_dir=state_dir,
        config_path=config_path,
        arm_writes=True,
    )
    if not stack.get("armed"):
        raise TradingForbiddenError("canary stack not armed")

    rt = stack["runtime"]
    broker = stack["broker"]
    if not stack["gate"].market_writes_allowed:
        raise TradingForbiddenError("WriteGate CLOSED — refusing canary run")

    # Fresh account + meta
    snap = broker.get_account_state() if hasattr(broker, "get_account_state") else None
    meta = broker.get_symbol_metadata(symbol_n)
    if price_alt_btc is None:
        if pf and pf.price_alt_btc:
            price_alt_btc = float(pf.price_alt_btc)
        else:
            price_alt_btc = float(MexcPublicTickerSource().fetch_price(symbol_n).price)
    available = float(getattr(snap, "available", None) or (pf.available_btc if pf else 0.0) or 0.0)
    if available <= 0 and pf and pf.available_btc is not None:
        available = float(pf.available_btc)

    ctx = RiskContext(
        account=None,
        symbol_meta=meta,
        open_position_count=0,
        current_exposure_pct=0.0,
        market_data_stale=False,
        account_data_stale=False,
    )
    # Prefer typed account snapshot when broker supports balances
    try:
        from btcc.execution.account import AccountSnapshot
        from btcc.execution.modes import ExecutionMode

        bals = broker.get_balances_detailed() if hasattr(broker, "get_balances_detailed") else []
        btc = next((b for b in bals if b.asset.upper() == "BTC"), None)
        if btc is not None:
            available = float(btc.free)
            ctx.account = AccountSnapshot(
                mode=ExecutionMode.REAL.value,
                available_balance=float(btc.free),
                locked_balance=float(btc.locked),
                total_balance=float(btc.total),
                currency="BTC",
                source="mexc",
            )
    except Exception:  # noqa: BLE001
        pass

    signal_ts = candle_ts = _utc_now()
    entry = rt.try_canary_entry(
        symbol=symbol_n,
        price_alt_btc=float(price_alt_btc),
        s_value=float(s_value),
        signal_ts=signal_ts,
        candle_ts=candle_ts,
        available_btc=float(available),
        symbol_meta=meta,
        account_ctx=ctx,
        intent_nonce=intent_nonce,
    )

    result: dict[str, Any] = {
        "entry": {
            "ok": entry.ok,
            "reason": entry.reason,
            "position_id": entry.position_id,
            "executed_quantity": entry.executed_quantity,
            "average_price": entry.average_price,
            "protection_id": entry.protection_id,
            "halt": entry.halt,
        },
        "phase": rt.canary.phase.value,
        "armed": stack.get("armed"),
        "monitoring": False,
        "waited_for_terminal": False,
    }

    if not entry.ok:
        # Entry failed / halted during entry — stop monitor if any, reconcile inventory, disarm if flat.
        fin = finalize_canary_lifecycle(stack, symbol=symbol_n)
        result.update({"finalize": fin, "terminal": "HALTED" if entry.halt else "ENTRY_FAILED"})
        result["armed"] = stack.get("armed")
        result["phase"] = rt.canary.phase.value
        return result

    # KEEP PROCESS ALIVE — do not return on monitoring=True alone.
    result["monitoring"] = True
    if rt.price_monitor is None or rt.price_monitor.state not in {
        PriceMonitorState.RUNNING,
        PriceMonitorState.RECONNECTING,
        PriceMonitorState.STALE,
    }:
        # Monitor must be running (or recovering) after successful entry.
        rt.halt("MONITOR_NOT_ALIVE_AFTER_ENTRY")
        fin = finalize_canary_lifecycle(stack, symbol=symbol_n)
        result.update({"finalize": fin, "terminal": "HALTED", "waited_for_terminal": False})
        result["phase"] = rt.canary.phase.value
        result["armed"] = stack.get("armed")
        return result

    wait_res = wait_for_canary_terminal(
        rt,
        poll_interval_s=wait_poll_interval_s,
        timeout_s=wait_timeout_s,
        sleep_fn=sleep_fn,
    )
    result["wait"] = {
        "terminal": wait_res.terminal,
        "phase": wait_res.phase,
        "elapsed_s": wait_res.elapsed_s,
        "waited": wait_res.waited,
        "halt_reason": wait_res.halt_reason,
        "error": wait_res.error,
    }
    result["waited_for_terminal"] = True

    if wait_res.terminal == "TIMEOUT":
        # Do not claim complete; leave protective path available.
        result["terminal"] = "TIMEOUT"
        result["phase"] = rt.canary.phase.value
        result["position_remaining"] = True
        result["ok"] = False
        result["reason"] = "WAIT_TIMEOUT_POSITION_MAY_REMAIN_OPEN"
        # Stop monitor for clean test teardown, but do not disarm if still OPEN.
        try:
            rt.stop_price_monitor()
        except Exception:  # noqa: BLE001
            pass
        return result

    fin = finalize_canary_lifecycle(stack, symbol=symbol_n)
    result["finalize"] = fin
    result["terminal"] = wait_res.terminal
    result["phase"] = rt.canary.phase.value
    result["armed"] = stack.get("armed")
    result["ok"] = bool(fin.get("ok"))
    result["reason"] = fin.get("reason")
    result["position_remaining"] = fin.get("position_remaining")
    result["disarmed"] = fin.get("disarmed")
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(
        prog="run_t1_real_canary",
        description="Isolated REAL T1 single-shot canary (NOT btcc.service / paper).",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    p.add_argument("--symbol", required=True, help="Spot ALT/BTC symbol, e.g. ETHBTC")
    p.add_argument("--preflight", action="store_true", help="Read-only preflight; no orders")
    p.add_argument("--run", action="store_true", help="Armed single-shot entry (requires env gates)")
    p.add_argument("--s-value", type=float, default=None, help="Signal S for --run (must be >= 0.60)")
    p.add_argument("--price", type=float, default=None, help="Optional ALT/BTC mid override for sizing/entry")
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument(
        "--wait-timeout-s",
        type=float,
        default=None,
        help="Optional max seconds to wait for CANARY_COMPLETE/HALTED (default: wait forever)",
    )
    args = p.parse_args(argv)

    if not args.preflight and not args.run:
        p.error("specify --preflight and/or --run")

    if args.preflight:
        report = run_preflight(symbol=args.symbol, config_path=args.config)
        payload = report.to_dict()
        print(json.dumps(payload, indent=2, default=str))
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        if not report.ok:
            logger.error("PREFLIGHT_NO_GO blockers=%s", report.blockers)
            return 2
        logger.info("PREFLIGHT_OK symbol=%s (no writes)", args.symbol)
        if not args.run:
            return 0

    if args.run:
        # Hard refuse unless explicitly armed — do not trade from this default task path
        try:
            assert_explicit_canary_write_arming()
        except TradingForbiddenError as e:
            logger.error("RUN_REFUSED: %s", e)
            return 3
        if args.s_value is None:
            p.error("--run requires --s-value")
        out = run_single_shot(
            symbol=args.symbol,
            s_value=float(args.s_value),
            price_alt_btc=args.price,
            config_path=args.config,
            state_dir=args.state_dir,
            require_preflight=True,
            wait_timeout_s=args.wait_timeout_s,
        )
        print(json.dumps(out, indent=2, default=str))
        if out.get("ok") and out.get("terminal") == "CANARY_COMPLETE":
            return 0
        if out.get("terminal") == "TIMEOUT" or out.get("position_remaining"):
            return 5
        if out.get("entry", {}).get("ok"):
            return 4
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
