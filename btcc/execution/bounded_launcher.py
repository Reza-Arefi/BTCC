"""Isolated REAL_BOUNDED_6H launcher (NOT btcc.service, NOT canary).

Modes:
  preflight  — authenticated reads + sizing checks; ZERO writes
  run        — requires BOTH env arms; bounded multi-position session

Arming (both required for writes; never inferred from config alone):
  BTCC_ALLOW_TRADING=true
  BTCC_REAL_6H_ARMED=true

Hard caps (CLI cannot raise):
  duration ≤ 6h, max_positions ≤ 4, allocation ≤ 25%, exposure ≤ 100%

Default config keeps allow_trading: false. Never mutates paper bot.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from btcc.execution.bounded_session import (
    MAX_ALLOCATION_HARD,
    MAX_EXPOSURE_HARD,
    MAX_POSITIONS_HARD,
    MAX_SESSION_DURATION_S,
    BoundedSession,
    BoundedSessionConfig,
    SessionPhase,
)
from btcc.execution.canary_launcher import (
    classify_preflight_inventory,
    allow_trading_from_env,
)
from btcc.execution.factory import build_bounded_6h_stack
from btcc.execution.mexc.client import MexcReadOnlyClient
from btcc.execution.mexc.credentials import load_mexc_credentials, redact_secrets
from btcc.execution.mexc.write_client import MexcWriteClient
from btcc.execution.price_monitor import MexcPublicTickerSource
from btcc.execution.production_t1_signal import ProductionT1SignalLoop
from btcc.execution.risk import RiskContext
from btcc.execution.write_gate import WriteGate
from btcc.safety.no_trading import TradingForbiddenError
# load_signal_config unused in launcher body — loop loads it

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "selector_t1_bounded_6h_config.yaml"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def bounded_6h_armed_from_env() -> bool:
    return _env_truthy("BTCC_REAL_6H_ARMED")


def assert_explicit_bounded_6h_arming() -> None:
    if not allow_trading_from_env():
        raise TradingForbiddenError(
            "Bounded 6h writes blocked: set BTCC_ALLOW_TRADING=true (explicit)."
        )
    if not bounded_6h_armed_from_env():
        raise TradingForbiddenError(
            "Bounded 6h writes blocked: set BTCC_REAL_6H_ARMED=true."
        )
    # Refuse accidental canary arming confusion — canary arm alone is not enough.
    if _env_truthy("CANARY_WRITES_ARMED") or _env_truthy("BTCC_CANARY_WRITES_ARMED"):
        if not bounded_6h_armed_from_env():
            raise TradingForbiddenError(
                "CANARY_WRITES_ARMED does not arm REAL_BOUNDED_6H; set BTCC_REAL_6H_ARMED=true"
            )


def load_bounded_6h_config(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"invalid bounded 6h config: {p}")
    if "selector_live" in raw:
        raise TradingForbiddenError("refusing paper selector_live config in bounded 6h launcher")
    return raw


def clamp_session_overrides(
    *,
    duration_s: float | None = None,
    max_positions: int | None = None,
    allocation_pct: float | None = None,
    max_total_exposure: float | None = None,
) -> dict[str, float | int]:
    """Apply requested values but never raise above hard safety caps."""
    dur = float(duration_s) if duration_s is not None else float(MAX_SESSION_DURATION_S)
    if dur > MAX_SESSION_DURATION_S + 1e-9:
        raise TradingForbiddenError(
            f"CLI/config duration_s={dur} exceeds hard limit {MAX_SESSION_DURATION_S}s"
        )
    if dur <= 0:
        raise TradingForbiddenError("duration_s must be > 0")
    mp = int(max_positions) if max_positions is not None else MAX_POSITIONS_HARD
    if mp > MAX_POSITIONS_HARD:
        raise TradingForbiddenError(f"max_positions={mp} exceeds hard limit {MAX_POSITIONS_HARD}")
    alloc = float(allocation_pct) if allocation_pct is not None else MAX_ALLOCATION_HARD
    if alloc > MAX_ALLOCATION_HARD + 1e-12:
        raise TradingForbiddenError(
            f"allocation_pct={alloc} exceeds hard limit {MAX_ALLOCATION_HARD}"
        )
    exp = float(max_total_exposure) if max_total_exposure is not None else MAX_EXPOSURE_HARD
    if exp > MAX_EXPOSURE_HARD + 1e-12:
        raise TradingForbiddenError(
            f"max_total_exposure={exp} exceeds hard limit {MAX_EXPOSURE_HARD}"
        )
    return {
        "duration_s": dur,
        "max_positions": mp,
        "allocation_pct": alloc,
        "max_total_exposure": exp,
    }


def make_session_dir(base: Path | None = None) -> Path:
    root = Path(base) if base else (ROOT / "data")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = root / f"live_6h_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_preflight(
    *,
    config_path: Path | None = None,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    """Read-only precheck. Never opens WriteGate."""
    cfg = load_bounded_6h_config(config_path)
    out_dir = Path(state_dir) if state_dir else make_session_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    creds = load_mexc_credentials(require=True)
    client = MexcReadOnlyClient(creds)
    bals = client.get_balances()
    inv = classify_preflight_inventory(bals)

    # Universe order-type capability audit (T1 requires MARKET exits).
    from btcc.execution.production_t1_signal import load_signal_config
    from btcc.execution.symbols import t1_market_exit_capability
    from btcc.universe import resolve_btc_market

    signal_cfg_path = None
    if isinstance(cfg.get("signal_config"), str):
        signal_cfg_path = ROOT / str(cfg["signal_config"])
    signal_cfg = load_signal_config(signal_cfg_path)
    universe_audit: list[dict[str, Any]] = []
    unsafe_symbols: list[str] = []
    meta_fetch_failures: list[str] = []
    for base in list((signal_cfg.get("universe") or {}).get("bases") or []):
        try:
            sym = resolve_btc_market(str(base), signal_cfg)
            # Fresh exchangeInfo — never rely on a hard-coded safe list.
            meta = client.get_symbol_metadata(sym, use_cache=False)
            ok, reason = t1_market_exit_capability(meta)
            row = {
                "base": str(base),
                "symbol": sym,
                "order_types": list(meta.order_types or ()),
                "MARKET": bool(meta.supports_market_orders is True),
                "LIMIT": "LIMIT" in (meta.order_types or ()),
                "LIMIT_MAKER": "LIMIT_MAKER" in (meta.order_types or ()),
                "t1_safe": ok,
                "reason": reason,
            }
            universe_audit.append(row)
            if not ok:
                unsafe_symbols.append(sym)
        except Exception as e:  # noqa: BLE001
            universe_audit.append(
                {"base": str(base), "t1_safe": False, "reason": f"META_ERROR:{e}"}
            )
            unsafe_symbols.append(str(base))
            meta_fetch_failures.append(str(base))

    report = {
        "ts": _utc_now(),
        "mode": "PRECHECK",
        "execution_mode": "REAL_BOUNDED_6H",
        "writes": False,
        "write_gate": "CLOSED",
        "inventory": inv,
        "credentials": redact_secrets({"api_key": creds.api_key}),
        "config_path": str(config_path or DEFAULT_CONFIG),
        "state_dir": str(out_dir),
        "hard_caps": {
            "duration_s": MAX_SESSION_DURATION_S,
            "max_positions": MAX_POSITIONS_HARD,
            "allocation_pct": MAX_ALLOCATION_HARD,
            "max_total_exposure": MAX_EXPOSURE_HARD,
        },
        "config_allow_trading": bool(cfg.get("allow_trading", False)),
        "universe_order_type_audit": universe_audit,
        "symbols_unsafe_for_t1": unsafe_symbols,
        "symbols_t1_safe": [
            r["symbol"] for r in universe_audit if r.get("t1_safe") and r.get("symbol")
        ],
        "entry_policy": "NO_MARKET_CAPABILITY_NO_ENTRY",
        "exchangeinfo_fetch_failures": meta_fetch_failures,
    }
    (out_dir / "precheck.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if inv.get("unexpected"):
        raise TradingForbiddenError(f"UNEXPECTED_EXCHANGE_STATE:{inv['unexpected']}")
    # Fail closed: cannot validate universe → do not authorize real trading start.
    if meta_fetch_failures:
        raise TradingForbiddenError(
            f"EXCHANGEINFO_UNAVAILABLE:{meta_fetch_failures}"
        )
    return report


def run_bounded_6h_session(
    *,
    config_path: Path | None = None,
    state_dir: Path | None = None,
    duration_s: float | None = None,
    max_positions: int | None = None,
    allocation_pct: float | None = None,
    dry_run_no_arm: bool = False,
    signal_loop: Any | None = None,
    poll_s: float = 5.0,
) -> dict[str, Any]:
    """Arm → run → cutoff → drain → flat → disarm → complete.

    When dry_run_no_arm=True, builds stack with gate CLOSED (tests / review).
    Live arming requires assert_explicit_bounded_6h_arming().
    """
    cfg = load_bounded_6h_config(config_path)
    sizing = cfg.get("sizing") or {}
    session_cfg = cfg.get("session") or {}
    overrides = clamp_session_overrides(
        duration_s=duration_s
        if duration_s is not None
        else float(session_cfg.get("duration_s", MAX_SESSION_DURATION_S)),
        max_positions=max_positions
        if max_positions is not None
        else int(sizing.get("max_open_positions", MAX_POSITIONS_HARD)),
        allocation_pct=allocation_pct
        if allocation_pct is not None
        else float(sizing.get("position_allocation_pct", MAX_ALLOCATION_HARD)),
        max_total_exposure=float(sizing.get("max_total_exposure", MAX_EXPOSURE_HARD)),
    )
    out_dir = Path(state_dir) if state_dir else make_session_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    armed = False
    if not dry_run_no_arm:
        assert_explicit_bounded_6h_arming()
        armed = True

    creds = None
    client = None
    write_client = None
    if armed:
        creds = load_mexc_credentials(require=True)
        client = MexcReadOnlyClient(creds)
        # Gate opened only after explicit arm check.
        gate_preview = WriteGate(
            allow_trading=True,
            canary_writes_armed=False,
            bounded_6h_writes_armed=True,
        )
        write_client = MexcWriteClient(creds, gate=gate_preview)

    stack = build_bounded_6h_stack(
        root_dir=out_dir,
        client=client,
        write_client=write_client,
        credentials=creds,
        allow_trading=armed,
        arm_writes=armed,
        max_positions=int(overrides["max_positions"]),
        allocation_pct=float(overrides["allocation_pct"]),
        max_total_exposure=float(overrides["max_total_exposure"]),
        duration_s=float(overrides["duration_s"]),
        strategy_version=str(cfg.get("version") or "T1-ONLY-BOUNDED-6H"),
        # Explicit live ticker when armed — never rely on factory test-stub default.
        price_source_factory=(
            (
                lambda: MexcPublicTickerSource(
                    base_url=str(getattr(creds, "base_url", None) or "https://api.mexc.com")
                )
            )
            if armed
            else None
        ),
        require_live_ticker=True if armed else False,
    )
    runtime = stack["runtime"]
    session: BoundedSession = stack["session"]
    runtime.stack_ref = stack

    meta = {
        "ts": _utc_now(),
        "execution_mode": "REAL_BOUNDED_6H",
        "armed": armed,
        "dry_run_no_arm": dry_run_no_arm,
        "overrides": overrides,
        "lifecycle": [
            "PRECHECK",
            "ARM",
            "RUNNING",
            "ENTRY_CUTOFF",
            "DRAIN",
            "FLAT_RECONCILE",
            "DISARM",
            "COMPLETE",
        ],
        "safety_gates": {
            "BTCC_ALLOW_TRADING": allow_trading_from_env() if armed else False,
            "BTCC_REAL_6H_ARMED": bounded_6h_armed_from_env() if armed else False,
            "canary_writes_armed": False,
            "write_gate_market_writes_allowed": bool(stack["gate"].market_writes_allowed),
        },
    }
    (out_dir / "session_meta.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8"
    )

    # Startup reconcile — fail closed on ambiguity.
    if armed and client is not None:
        bals = client.get_balances()
        inv = classify_preflight_inventory(bals)
        if inv.get("unexpected"):
            runtime.halt(f"UNEXPECTED_EXCHANGE_STATE:{inv['unexpected']}")
            outcome = {"ok": False, "reason": runtime.halt_reason, "phase": session.phase.value}
            (out_dir / "final_outcome.json").write_text(
                json.dumps(outcome, indent=2) + "\n", encoding="utf-8"
            )
            return outcome
        open_orders: list[dict[str, Any]] = []
        try:
            if hasattr(client, "get_open_orders"):
                open_orders = list(client.get_open_orders() or [])
        except Exception as e:  # noqa: BLE001
            runtime.halt(f"OPEN_ORDERS_UNAVAILABLE:{e}")
            return {"ok": False, "reason": runtime.halt_reason, "phase": session.phase.value}
        status = runtime.reconcile_startup(
            local_positions=[],
            exchange_inventory=[
                {"asset": getattr(b, "asset", ""), "total": getattr(b, "total", 0)}
                for b in bals
            ],
            open_protections={},
            open_orders=open_orders,
        )
        if status != "NORMAL":
            outcome = {"ok": False, "reason": runtime.halt_reason, "phase": session.phase.value}
            (out_dir / "final_outcome.json").write_text(
                json.dumps(outcome, indent=2) + "\n", encoding="utf-8"
            )
            return outcome

        # Fresh exchangeInfo capability gate before any entries (fail closed).
        from btcc.execution.production_t1_signal import load_signal_config
        from btcc.execution.symbols import t1_market_exit_capability
        from btcc.universe import resolve_btc_market

        signal_cfg_path = None
        if isinstance(cfg.get("signal_config"), str):
            signal_cfg_path = ROOT / str(cfg["signal_config"])
        signal_cfg = load_signal_config(signal_cfg_path)
        safe_syms: list[str] = []
        unsafe_syms: list[str] = []
        meta_fail: list[str] = []
        broker = stack.get("broker") or runtime.broker
        for base in list((signal_cfg.get("universe") or {}).get("bases") or []):
            try:
                sym = resolve_btc_market(str(base), signal_cfg)
                try:
                    m = broker.get_symbol_metadata(sym, use_cache=False)
                except TypeError:
                    m = broker.get_symbol_metadata(sym)
                ok, _reason = t1_market_exit_capability(m)
                if ok:
                    safe_syms.append(sym)
                else:
                    unsafe_syms.append(sym)
            except Exception as e:  # noqa: BLE001
                meta_fail.append(f"{base}:{e}")
        runtime.universe_bases = {
            str(b).upper() for b in (signal_cfg.get("universe") or {}).get("bases") or []
        }
        runtime.audit.record(
            "STARTUP_MARKET_CAPABILITY",
            t1_safe=safe_syms,
            t1_unsafe=unsafe_syms,
            meta_failures=meta_fail,
        )
        if meta_fail:
            runtime.halt(f"EXCHANGEINFO_UNAVAILABLE:{meta_fail}")
            outcome = {
                "ok": False,
                "reason": runtime.halt_reason,
                "phase": session.phase.value,
            }
            (out_dir / "final_outcome.json").write_text(
                json.dumps(outcome, indent=2) + "\n", encoding="utf-8"
            )
            return outcome
        if not safe_syms:
            runtime.halt("NO_T1_SAFE_MARKETS")
            outcome = {
                "ok": False,
                "reason": runtime.halt_reason,
                "phase": session.phase.value,
                "t1_unsafe": unsafe_syms,
            }
            (out_dir / "final_outcome.json").write_text(
                json.dumps(outcome, indent=2) + "\n", encoding="utf-8"
            )
            return outcome
        (out_dir / "startup_market_capability.json").write_text(
            json.dumps(
                {
                    "t1_safe": safe_syms,
                    "t1_unsafe": unsafe_syms,
                    "entry_policy": "NO_MARKET_CAPABILITY_NO_ENTRY",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    session.arm()
    session.save(out_dir / "session_state.json")
    session.start_running()
    runtime.audit.record("SESSION_RUNNING", deadline=session.deadline_utc)
    session.save(out_dir / "session_state.json")

    if dry_run_no_arm:
        # Construction-only path for review — do not loop / trade.
        outcome = {
            "ok": True,
            "dry_run_no_arm": True,
            "phase": session.phase.value,
            "deadline_utc": session.deadline_utc,
            "writes": False,
            "note": "stack built with WriteGate CLOSED; no orders placed",
            "signal_loop": "not_started_dry_build",
        }
        (out_dir / "final_outcome.json").write_text(
            json.dumps(outcome, indent=2) + "\n", encoding="utf-8"
        )
        return outcome

    # Production continuous signal path (required for --run).
    # Uses SignalEngine market data + CrossingStateMachine + evaluate_trail_entry.
    # Never loads Selector E paper engine.
    if signal_loop is None:
        signal_cfg_path = None
        if isinstance(cfg.get("signal_config"), str):
            signal_cfg_path = ROOT / str(cfg["signal_config"])
        prod_loop = ProductionT1SignalLoop(
            runtime,
            signal_config_path=signal_cfg_path,
            require_s_min=float((cfg.get("strategy") or {}).get("s_min", 0.60)),
            max_positions=int(overrides["max_positions"]),
        )
        prod_loop.bootstrap()
        signal_loop = prod_loop.as_callback()
        stack["signal_loop"] = prod_loop
        # Wire BTC-universe bases for exchange-inventory emergency flatten.
        cfg_u = getattr(prod_loop, "cfg", None) or {}
        bases = list((cfg_u.get("universe") or {}).get("bases") or [])
        runtime.universe_bases = {str(b).upper() for b in bases}
        runtime.audit.record(
            "PRODUCTION_SIGNAL_LOOP_WIRED",
            class_name="ProductionT1SignalLoop",
            continuous=True,
            universe_bases=sorted(runtime.universe_bases),
        )

    try:
        while not runtime.halted and session.phase not in (
            SessionPhase.COMPLETE,
            SessionPhase.HALTED,
        ):
            runtime.advance_cutoff_if_due()
            runtime.maybe_send_hourly_report()
            # Continuous production evaluation every poll; engine de-dupes 15m candles.
            if signal_loop is not None:
                signal_loop(runtime)
            if runtime.try_complete_if_flat(stack=stack):
                break
            if session.phase in (SessionPhase.DRAIN, SessionPhase.ENTRY_CUTOFF):
                if session.open_count == 0:
                    runtime.try_complete_if_flat(stack=stack)
                    break
            session.save(out_dir / "session_state.json")
            rem = session.seconds_until_cutoff()
            if rem is not None and rem > 0 and session.phase == SessionPhase.RUNNING:
                time.sleep(min(float(poll_s), max(0.1, rem)))
            else:
                time.sleep(float(poll_s))
            if session.phase == SessionPhase.COMPLETE:
                break
    finally:
        session.save(out_dir / "session_state.json")

    # Critical path: exchange-inventory flatten before process exit when not COMPLETE.
    flatten_res = None
    if session.phase != SessionPhase.COMPLETE:
        # Always attempt inventory flatten on HALT / unclean end (even if local open=0
        # but exchange may still hold ALT — e.g. after local meta lost).
        if runtime.halted or session.open_count > 0 or session.phase == SessionPhase.HALTED:
            runtime.audit.record(
                "POST_LOOP_FLATTEN",
                open_count=session.open_count,
                halted=runtime.halted,
                halt_reason=runtime.halt_reason,
                phase=session.phase.value,
            )
            flatten_res = runtime.emergency_flatten_all_open(
                reason=f"POST_LOOP:{runtime.halt_reason or session.phase.value}"
            )
            session.save(out_dir / "session_state.json")

    if session.phase != SessionPhase.COMPLETE and not runtime.halted:
        if session.open_count > 0 or (flatten_res is not None and flatten_res.incomplete):
            runtime.halt("SESSION_END_NOT_FLAT")
        else:
            runtime.try_complete_if_flat(stack=stack)

    # Terminal WriteGate: always CLOSED on exit (COMPLETE or HALTED/incomplete).
    if stack["gate"].market_writes_allowed:
        close_reason = (
            "COMPLETE_GATE_FORCE_CLOSE"
            if session.phase == SessionPhase.COMPLETE and not runtime.flatten_incomplete
            else "TERMINAL_HALT_GATE_CLOSE"
        )
        if runtime.flatten_incomplete:
            close_reason = "EMERGENCY_FLATTEN_INCOMPLETE_GATE_CLOSE"
        runtime.close_market_write_gate(reason=close_reason, stack=stack)

    outcome = {
        "ok": session.phase == SessionPhase.COMPLETE
        and not runtime.halted
        and not runtime.flatten_incomplete
        and not bool(stack["gate"].market_writes_allowed),
        "phase": session.phase.value,
        "halt_reason": runtime.halt_reason,
        "entries": session.entries_count,
        "closes": session.closes_count,
        "open_remaining": session.open_count,
        "deadline_utc": session.deadline_utc,
        "completed_at_utc": session.completed_at_utc,
        "state_dir": str(out_dir),
        "write_gate_market_writes_allowed": bool(stack["gate"].market_writes_allowed),
        "signal_loop": type(stack.get("signal_loop") or signal_loop).__name__,
        "emergency_flatten_incomplete": bool(runtime.flatten_incomplete),
        "emergency_flatten_remaining": (
            None
            if runtime._last_flatten_result is None
            else runtime._last_flatten_result.remaining
        ),
    }
    if runtime.flatten_incomplete:
        outcome["status_flag"] = "EMERGENCY_FLATTEN_INCOMPLETE"
    (out_dir / "final_outcome.json").write_text(
        json.dumps(outcome, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return outcome
