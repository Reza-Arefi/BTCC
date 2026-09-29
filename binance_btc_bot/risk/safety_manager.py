"""Portfolio-level Safety Manager — persisted circuit breaker + defensive mode.

Independent from strategy logic. It never produces entries; it only:

* blocks NEW entries (SAFETY_HALT after N exchange-confirmed stop-loss closures,
  DEFENSIVE_MODE when >= K of M open positions are <= loss threshold,
  or when monitoring cannot verify prices),
* while DEFENSIVE_MODE is active, plans risk-reducing stop moves for open
  positions (never looser, never backward),
* clears only via an explicit ``restart(checks)`` where every check passed.

State survives process restarts (atomic JSON). A corrupt/unreadable state file
fails closed into SAFETY_HALT.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

STATE_VERSION = 1

HALT_REASON_CONSECUTIVE_SL = "CONSECUTIVE_STOP_LOSSES"
HALT_REASON_STATE_CORRUPT = "SAFETY_STATE_CORRUPT"

# close_reason values produced by OrderLifecycle._map_exit_close_reason.
STOP_LOSS_REASONS = frozenset({"HARD_SL", "STOP_LOSS", "SAFETY_STOP"})
WIN_REASONS = frozenset({"TRAILING_EXIT", "TAKE_PROFIT"})

STAGE_NONE = "NONE"
STAGE_LOSS_TIGHTEN = "LOSS_TIGHTEN"
STAGE_LOSS_STRONG = "LOSS_STRONG"
STAGE_BREAKEVEN = "PROFIT_BREAKEVEN"
STAGE_LOCK = "PROFIT_LOCK"
STAGE_TRAIL = "PROFIT_TRAIL"

ACTION_REPLACE = "REPLACE_STOP"
ACTION_EXIT = "EXIT_NOW"

NotifyFn = Callable[[str, str, str], Any]  # (severity, event, message)


@dataclass(frozen=True)
class SafetyManagerConfig:
    enabled: bool = True
    # --- circuit breaker ---
    sl_halt_count: int = 2
    reset_counter_on_win: bool = True
    sl_cooldown_minutes: float = 0.0
    # --- defensive mode trigger ---
    defensive_min_open: int = 4
    defensive_loss_count: int = 3
    defensive_loss_pct: float = -0.020
    # --- losing-position protection (DEFENSIVE MODE ONLY) ---
    loss_tighten_pct: float = -0.020
    loss_tighten_trail_pct: float = 0.010
    loss_strong_pct: float = -0.025
    loss_strong_trail_pct: float = 0.005
    loss_hard_exit_pct: float = -0.030
    # --- profit protection (DEFENSIVE MODE ONLY) ---
    profit_breakeven_pct: float = 0.010
    breakeven_offset_pct: float = 0.002
    profit_lock_pct: float = 0.015
    profit_lock_stop_pct: float = 0.0075
    profit_trail_pct: float = 0.020
    profit_trail_distance_pct: float = 0.010
    # --- order hygiene ---
    min_stop_improvement_pct: float = 0.0025
    replace_cooldown_sec: float = 60.0
    # --- monitoring / fail-closed ---
    monitor_interval_sec: float = 10.0
    monitor_stale_after_sec: float = 120.0
    state_filename: str = "safety_manager_state.json"

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "SafetyManagerConfig":
        raw = dict((cfg or {}).get("safety_manager") or {})
        kwargs: dict[str, Any] = {}
        for name, fld in cls.__dataclass_fields__.items():  # type: ignore[attr-defined]
            if name not in raw:
                continue
            val = raw[name]
            default = fld.default
            if isinstance(default, bool):
                kwargs[name] = bool(val)
            elif isinstance(default, int):
                kwargs[name] = int(val)
            elif isinstance(default, float):
                kwargs[name] = float(val)
            else:
                kwargs[name] = str(val)
        out = cls(**kwargs)
        out.validate()
        return out

    def validate(self) -> None:
        if self.sl_halt_count < 1:
            raise ValueError("safety_manager.sl_halt_count must be >= 1")
        if not (1 <= self.defensive_loss_count <= self.defensive_min_open):
            raise ValueError("safety_manager.defensive_loss_count must be in [1, defensive_min_open]")
        if not (self.loss_hard_exit_pct <= self.loss_strong_pct <= self.loss_tighten_pct < 0):
            raise ValueError("safety_manager loss levels must satisfy hard <= strong <= tighten < 0")
        if not (0 < self.loss_strong_trail_pct <= self.loss_tighten_trail_pct < 1):
            raise ValueError("safety_manager loss trails must satisfy 0 < strong <= tighten < 1")
        if not (0 < self.profit_breakeven_pct <= self.profit_lock_pct <= self.profit_trail_pct):
            raise ValueError("safety_manager profit levels must satisfy 0 < breakeven <= lock <= trail")
        if not (0 <= self.breakeven_offset_pct < self.profit_breakeven_pct):
            raise ValueError("safety_manager.breakeven_offset_pct must be in [0, profit_breakeven_pct)")
        if not (0 < self.profit_lock_stop_pct < self.profit_lock_pct):
            raise ValueError("safety_manager.profit_lock_stop_pct must be in (0, profit_lock_pct)")
        if not (0 < self.profit_trail_distance_pct < self.profit_trail_pct):
            raise ValueError("safety_manager.profit_trail_distance_pct must be in (0, profit_trail_pct)")


@dataclass(frozen=True)
class PositionView:
    trade_id: str
    symbol: str
    entry_price: float
    current_price: float
    current_stop: float | None = None

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price


@dataclass(frozen=True)
class ProtectionAction:
    trade_id: str
    symbol: str
    action: str
    new_stop: float | None
    stage: str
    pnl_pct: float
    reason: str


@dataclass
class TradeProtection:
    symbol: str
    entry_price: float
    stop: float | None = None
    stage: str = STAGE_NONE
    loss_stage: str = STAGE_NONE
    peak_price: float = 0.0
    last_replace_at: float = 0.0
    replacements: int = 0


@dataclass
class SafetyManagerState:
    version: int = STATE_VERSION
    safety_halt: bool = False
    halt_reason: str | None = None
    halt_at: float | None = None
    defensive_mode: bool = False
    defensive_reason: str | None = None
    defensive_at: float | None = None
    defensive_positions: list[dict[str, Any]] = field(default_factory=list)
    consecutive_stop_losses: int = 0
    last_sl_at: float | None = None
    last_sl_symbol: str | None = None
    last_sl_trade_id: str | None = None
    counted_trade_ids: list[str] = field(default_factory=list)
    protection: dict[str, dict[str, Any]] = field(default_factory=dict)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    last_restart_at: float | None = None
    updated_at: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SafetyManagerState":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}  # type: ignore[attr-defined]
        st = cls(**known)
        if not isinstance(st.safety_halt, bool) or not isinstance(st.defensive_mode, bool):
            raise ValueError("safety flags must be booleans")
        st.consecutive_stop_losses = int(st.consecutive_stop_losses)
        return st


class SafetyStateStore:
    """Atomic JSON persistence (tmp + replace). Corrupt ⇒ caller fails closed."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> tuple[SafetyManagerState, str | None]:
        if not self.path.exists():
            return SafetyManagerState(), None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("state root must be an object")
            return SafetyManagerState.from_dict(data), None
        except Exception as e:  # noqa: BLE001
            return SafetyManagerState(), f"{type(e).__name__}: {e}"

    def save(self, state: SafetyManagerState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state.updated_at = time.time()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(state), indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(tmp, self.path)


def default_safety_state_path(db_path: str | Path, filename: str = "safety_manager_state.json") -> Path:
    return Path(db_path).resolve().parent / filename


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


class SafetyManager:
    def __init__(
        self,
        config: SafetyManagerConfig,
        store: SafetyStateStore,
        *,
        notify: NotifyFn | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.store = store
        self.notify = notify
        self.clock = clock
        self._lock = threading.RLock()
        self.monitor_ok: bool = True
        self.monitor_error: str | None = None
        self.last_monitor_ok_at: float | None = None
        self.state, err = store.load()
        if err:
            self.state = SafetyManagerState()
            self._enter_halt(f"{HALT_REASON_STATE_CORRUPT}: {err}", severity="CRITICAL")

    # ------------------------------------------------------------------ gate
    def allow_new_entries(self) -> bool:
        return self.blocking_reason() is None

    def blocks_new_entries(self) -> bool:
        return not self.allow_new_entries()

    def blocking_reason(self) -> str | None:
        if not self.config.enabled:
            return None
        with self._lock:
            if self.state.safety_halt:
                return f"SAFETY_HALT: {self.state.halt_reason}"
            if self.state.defensive_mode:
                return f"DEFENSIVE_MODE: {self.state.defensive_reason}"
            if self._in_sl_cooldown():
                return "SL_COOLDOWN"
            if not self.monitor_ok:
                return f"SAFETY_CHECKS_FAILED: {self.monitor_error or 'monitoring failed'}"
            if self.last_monitor_ok_at is not None:
                age = self.clock() - self.last_monitor_ok_at
                if age > self.config.monitor_stale_after_sec:
                    return f"SAFETY_CHECKS_STALE: last verified {age:.0f}s ago"
        return None

    def mode(self) -> str:
        with self._lock:
            if self.state.safety_halt:
                return "SAFETY_HALT"
            if self.state.defensive_mode:
                return "DEFENSIVE_MODE"
        return "NORMAL"

    def _in_sl_cooldown(self) -> bool:
        cd = float(self.config.sl_cooldown_minutes or 0) * 60.0
        if cd <= 0 or self.state.consecutive_stop_losses <= 0 or not self.state.last_sl_at:
            return False
        return (self.clock() - float(self.state.last_sl_at)) < cd

    # ------------------------------------------------------------ monitoring
    def mark_monitor_ok(self) -> None:
        with self._lock:
            self.monitor_ok = True
            self.monitor_error = None
            self.last_monitor_ok_at = self.clock()

    def mark_monitor_failed(self, error: str) -> None:
        with self._lock:
            was_ok = self.monitor_ok
            self.monitor_ok = False
            self.monitor_error = str(error)[:300]
        if was_ok:
            self._log_transition("MONITOR_FAILED", {"error": self.monitor_error})
            self._send("WARNING", "SAFETY_MONITOR_FAILED",
                       f"⚠️ SAFETY: cannot verify positions/market data.\n{self.monitor_error}\nNew entries: BLOCKED")

    # -------------------------------------------------------- SL breaker
    def record_trade_closed(
        self,
        *,
        trade_id: str,
        symbol: str,
        close_reason: str | None,
        pnl_pct: float | None = None,
    ) -> str:
        """Record an exchange-confirmed closure. Idempotent per trade_id.

        Returns one of: IGNORED_DUPLICATE, SL_COUNTED, HALTED, COUNTER_RESET, NO_CHANGE.
        """
        reason = str(close_reason or "UNKNOWN").upper()
        with self._lock:
            self.state.protection.pop(str(trade_id), None)
            if str(trade_id) in self.state.counted_trade_ids:
                self._save()
                return "IGNORED_DUPLICATE"
            self.state.counted_trade_ids.append(str(trade_id))
            self.state.counted_trade_ids = self.state.counted_trade_ids[-500:]
            if reason in STOP_LOSS_REASONS:
                self.state.consecutive_stop_losses += 1
                self.state.last_sl_at = self.clock()
                self.state.last_sl_symbol = str(symbol)
                self.state.last_sl_trade_id = str(trade_id)
                n = self.state.consecutive_stop_losses
                limit = self.config.sl_halt_count
                self._log_transition("STOP_LOSS", {"trade_id": trade_id, "symbol": symbol, "count": n})
                if n >= limit and not self.state.safety_halt:
                    self._save()
                    self._enter_halt(f"{n}_{HALT_REASON_CONSECUTIVE_SL}", severity="CRITICAL",
                                     message=(f"🛑 SAFETY:\nTrade {symbol} stopped out.\n"
                                              f"Consecutive SL count = {n}.\nNEW ENTRIES HALTED.\n"
                                              f"Existing positions: STILL MANAGED\nUse /restart after checks."))
                    return "HALTED"
                self._save()
                self._send("WARNING", "SAFETY_STOP_LOSS",
                           f"⚠️ SAFETY:\nTrade {symbol} stopped out.\nConsecutive SL count = {n}/{limit}.")
                return "SL_COUNTED"
            if reason in WIN_REASONS and self.config.reset_counter_on_win:
                if self.state.consecutive_stop_losses > 0 and not self.state.safety_halt:
                    old = self.state.consecutive_stop_losses
                    self.state.consecutive_stop_losses = 0
                    self._log_transition("SL_COUNTER_RESET", {"trade_id": trade_id, "symbol": symbol, "old": old})
                    self._save()
                    self._send("INFO", "SAFETY_SL_RESET",
                               f"SAFETY: {symbol} closed by {reason}. Consecutive SL count reset {old} → 0.")
                    return "COUNTER_RESET"
            self._save()
            return "NO_CHANGE"

    # ------------------------------------------------------ defensive mode
    def evaluate_portfolio(self, positions: Iterable[PositionView]) -> bool:
        """Activate DEFENSIVE_MODE when the portfolio condition holds. Returns True if newly activated."""
        pos = list(positions)
        if not self.config.enabled:
            return False
        losers = [p for p in pos if p.pnl_pct <= self.config.defensive_loss_pct + 1e-12]
        condition = len(pos) >= self.config.defensive_min_open and len(losers) >= self.config.defensive_loss_count
        with self._lock:
            if not condition or self.state.defensive_mode:
                return False
            reason = (f"{len(losers)} of {len(pos)} open positions are <= "
                      f"{self.config.defensive_loss_pct * 100:.1f}%")
            self.state.defensive_mode = True
            self.state.defensive_reason = reason
            self.state.defensive_at = self.clock()
            self.state.defensive_positions = [
                {"trade_id": p.trade_id, "symbol": p.symbol, "pnl_pct": round(p.pnl_pct * 100, 3),
                 "triggering": p.pnl_pct <= self.config.defensive_loss_pct + 1e-12}
                for p in pos
            ]
            self._log_transition("DEFENSIVE_MODE", {"reason": reason, "positions": self.state.defensive_positions})
            self._save()
        lines = "\n".join(f"{p.symbol}: {_pct(p.pnl_pct)}" for p in pos)
        self._send("CRITICAL", "SAFETY_DEFENSIVE_MODE",
                   f"🚨 DEFENSIVE MODE\n\nReason:\n{reason}\n\nPositions:\n{lines}\n\n"
                   f"New entries: BLOCKED\nExisting positions: STILL MANAGED")
        return True

    def would_trigger_defensive(self, positions: Iterable[PositionView]) -> bool:
        pos = list(positions)
        losers = [p for p in pos if p.pnl_pct <= self.config.defensive_loss_pct + 1e-12]
        return len(pos) >= self.config.defensive_min_open and len(losers) >= self.config.defensive_loss_count

    # -------------------------------------------------- stop planning
    def register_position(self, trade_id: str, symbol: str, entry_price: float, stop: float | None) -> None:
        with self._lock:
            if str(trade_id) not in self.state.protection:
                self.state.protection[str(trade_id)] = asdict(
                    TradeProtection(symbol=str(symbol), entry_price=float(entry_price),
                                    stop=float(stop) if stop else None, peak_price=float(entry_price))
                )
                self._save()

    def protection_for(self, trade_id: str) -> TradeProtection | None:
        with self._lock:
            raw = self.state.protection.get(str(trade_id))
            return TradeProtection(**raw) if raw else None

    def plan_protection(self, positions: Iterable[PositionView]) -> list[ProtectionAction]:
        """Risk-reducing stop moves. Empty unless DEFENSIVE_MODE is active."""
        if not self.config.enabled:
            return []
        with self._lock:
            if not self.state.defensive_mode:
                return []
            now = self.clock()
            actions: list[ProtectionAction] = []
            for p in positions:
                tp_raw = self.state.protection.get(p.trade_id)
                tp = TradeProtection(**tp_raw) if tp_raw else TradeProtection(
                    symbol=p.symbol, entry_price=p.entry_price, stop=p.current_stop, peak_price=p.entry_price)
                if tp.stop is None and p.current_stop:
                    tp.stop = float(p.current_stop)
                tp.peak_price = max(float(tp.peak_price or 0), float(p.current_price))
                act = self._plan_one(p, tp, now)
                self.state.protection[p.trade_id] = asdict(tp)
                if act is not None:
                    actions.append(act)
            self._save()
            return actions

    def _plan_one(self, p: PositionView, tp: TradeProtection, now: float) -> ProtectionAction | None:
        c = self.config
        entry = p.entry_price
        price = p.current_price
        pnl = p.pnl_pct
        peak_pnl = (tp.peak_price - entry) / entry if entry > 0 else 0.0

        if pnl <= c.loss_hard_exit_pct + 1e-12:
            return ProtectionAction(p.trade_id, p.symbol, ACTION_EXIT, None, "LOSS_HARD_LIMIT", pnl,
                                    f"{p.symbol} {_pct(pnl)} <= hard limit {_pct(c.loss_hard_exit_pct)}")

        # Loss stages are sticky: once reached, the tighter trail distance is kept.
        if pnl <= c.loss_strong_pct + 1e-12:
            tp.loss_stage = STAGE_LOSS_STRONG
        elif pnl <= c.loss_tighten_pct + 1e-12 and tp.loss_stage == STAGE_NONE:
            tp.loss_stage = STAGE_LOSS_TIGHTEN

        candidates: list[tuple[float, str]] = []
        hard_floor = entry * (1.0 + c.loss_hard_exit_pct)
        if tp.loss_stage == STAGE_LOSS_STRONG:
            candidates.append((max(price * (1.0 - c.loss_strong_trail_pct), hard_floor), STAGE_LOSS_STRONG))
        elif tp.loss_stage == STAGE_LOSS_TIGHTEN:
            candidates.append((max(price * (1.0 - c.loss_tighten_trail_pct), hard_floor), STAGE_LOSS_TIGHTEN))

        if peak_pnl >= c.profit_breakeven_pct - 1e-12:
            candidates.append((entry * (1.0 + c.breakeven_offset_pct), STAGE_BREAKEVEN))
        if peak_pnl >= c.profit_lock_pct - 1e-12:
            candidates.append((entry * (1.0 + c.profit_lock_stop_pct), STAGE_LOCK))
        if peak_pnl >= c.profit_trail_pct - 1e-12:
            candidates.append((tp.peak_price * (1.0 - c.profit_trail_distance_pct), STAGE_TRAIL))

        if not candidates:
            return None
        desired, stage = max(candidates, key=lambda x: x[0])
        current = float(tp.stop) if tp.stop else 0.0
        if desired <= current:
            return None  # never loosen / never move backward
        if desired >= price:
            # Protected level already crossed — the stop would trigger immediately.
            return ProtectionAction(p.trade_id, p.symbol, ACTION_EXIT, None, stage, pnl,
                                    f"{p.symbol} price {price:.10g} <= protected level {desired:.10g}")
        if current > 0 and desired < current * (1.0 + c.min_stop_improvement_pct):
            return None
        if tp.last_replace_at and (now - tp.last_replace_at) < c.replace_cooldown_sec:
            return None
        return ProtectionAction(p.trade_id, p.symbol, ACTION_REPLACE, desired, stage, pnl,
                                f"{p.symbol} {_pct(pnl)} stage={stage} stop {current:.10g} → {desired:.10g}")

    def confirm_stop(self, trade_id: str, new_stop: float, stage: str) -> None:
        """Record an exchange-accepted tighter stop. Rejects any loosening."""
        with self._lock:
            raw = self.state.protection.get(str(trade_id))
            if raw is None:
                return
            tp = TradeProtection(**raw)
            if tp.stop is not None and float(new_stop) < float(tp.stop):
                raise ValueError(f"refusing to loosen stop {tp.stop} → {new_stop}")
            prev_stage = tp.stage
            tp.stop = float(new_stop)
            tp.stage = stage
            tp.last_replace_at = self.clock()
            tp.replacements += 1
            self.state.protection[str(trade_id)] = asdict(tp)
            self._log_transition("STOP_TIGHTENED", {"trade_id": trade_id, "symbol": tp.symbol,
                                                    "stop": new_stop, "stage": stage})
            self._save()
        if stage != prev_stage:
            pnl_note = {
                STAGE_BREAKEVEN: f"reached {_pct(self.config.profit_breakeven_pct)}.\nStop tightened toward breakeven.",
                STAGE_LOCK: f"reached {_pct(self.config.profit_lock_pct)}.\nPart of the profit is now protected.",
                STAGE_TRAIL: f"reached {_pct(self.config.profit_trail_pct)}.\nTighter trailing protection activated.",
                STAGE_LOSS_TIGHTEN: f"<= {_pct(self.config.loss_tighten_pct)}.\nDefensive stop tightened.",
                STAGE_LOSS_STRONG: f"<= {_pct(self.config.loss_strong_pct)}.\nStronger defensive stop applied.",
            }.get(stage, f"stop tightened ({stage}).")
            self._send("INFO", "SAFETY_PROTECTION", f"🛡 PROTECTION:\n{tp.symbol} {pnl_note}\nNew stop: {new_stop:.10g}")

    def enter_halt(self, reason: str) -> None:
        self._enter_halt(reason, severity="CRITICAL")

    # ------------------------------------------------------------ restart
    def restart(self, checks: Mapping[str, tuple[bool, str]]) -> tuple[bool, list[str]]:
        """Clear SAFETY_HALT / DEFENSIVE_MODE only if every check passed."""
        failed = [f"{name}: {detail}" if detail else name for name, (ok, detail) in checks.items() if not ok]
        if not checks:
            failed = ["no safety checks were executed"]
        if failed:
            self._log_transition("RESTART_BLOCKED", {"failed": failed})
            return False, failed
        with self._lock:
            old = {"safety_halt": self.state.safety_halt, "halt_reason": self.state.halt_reason,
                   "defensive_mode": self.state.defensive_mode, "consecutive_sl": self.state.consecutive_stop_losses}
            self.state.safety_halt = False
            self.state.halt_reason = None
            self.state.halt_at = None
            self.state.defensive_mode = False
            self.state.defensive_reason = None
            self.state.defensive_at = None
            self.state.defensive_positions = []
            self.state.consecutive_stop_losses = 0
            self.state.last_restart_at = self.clock()
            self._log_transition("RESTARTED", {"previous": old})
            self._save()
        return True, []

    # ------------------------------------------------------------- status
    def status(self) -> dict[str, Any]:
        with self._lock:
            st = self.state
            return {
                "enabled": self.config.enabled,
                "mode": self.mode(),
                "safety_halt": st.safety_halt,
                "halt_reason": st.halt_reason,
                "halt_at": st.halt_at,
                "defensive_mode": st.defensive_mode,
                "defensive_reason": st.defensive_reason,
                "defensive_at": st.defensive_at,
                "defensive_positions": list(st.defensive_positions),
                "consecutive_stop_losses": st.consecutive_stop_losses,
                "sl_halt_count": self.config.sl_halt_count,
                "last_sl_at": st.last_sl_at,
                "last_sl_symbol": st.last_sl_symbol,
                "monitor_ok": self.monitor_ok,
                "monitor_error": self.monitor_error,
                "last_monitor_ok_at": self.last_monitor_ok_at,
                "blocking_reason": self.blocking_reason(),
                "entries_allowed_by_safety": self.allow_new_entries(),
                "protection": {k: dict(v) for k, v in st.protection.items()},
            }

    # ----------------------------------------------------------- internals
    def _enter_halt(self, reason: str, *, severity: str = "CRITICAL", message: str | None = None) -> None:
        with self._lock:
            self.state.safety_halt = True
            self.state.halt_reason = str(reason)
            self.state.halt_at = self.clock()
            self._log_transition("SAFETY_HALT", {"reason": reason})
            try:
                self._save()
            except Exception as e:  # noqa: BLE001 — halt stays active in memory
                logger.error("safety state save failed during halt: %s", e)
        self._send(severity, "SAFETY_HALT", message or f"🛑 SAFETY_HALT\nReason: {reason}\nNew entries: BLOCKED\n"
                                                        f"Existing positions: STILL MANAGED\nUse /restart after checks.")

    def _log_transition(self, event: str, details: dict[str, Any]) -> None:
        entry = {"ts": self.clock(), "event": event, **details}
        logger.warning("SAFETY_MANAGER %s %s", event, json.dumps(details, default=str)[:800])
        with self._lock:
            self.state.transitions.append(entry)
            self.state.transitions = self.state.transitions[-200:]

    def _save(self) -> None:
        self.store.save(self.state)

    def _send(self, severity: str, event: str, message: str) -> None:
        if not self.notify:
            return
        try:
            self.notify(severity, event, message)
        except Exception:  # noqa: BLE001
            logger.warning("safety manager notify failed (ignored)")
