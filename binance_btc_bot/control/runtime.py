"""Telegram runtime control plane — isolated from frozen strategy/risk YAML.

Affects NEW entries only. Never mutates T1–T10/T21/T30 definitions, research selectors'
algorithms, risk ceilings, or Binance protection geometry.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from binance_btc_bot.config_loader import FROZEN_STRATEGIES
from binance_btc_bot.secrets import scrub_exception, scrub_text
from binance_btc_bot.strategy.selectors import SELECTOR_KINDS, select_strategy_for_research
from binance_btc_bot.strategy.trails import TrailStrategy, get_strategy
from binance_btc_bot.strategy.provider import StrategyProvider

logger = logging.getLogger(__name__)

ALLOWED_STRATEGIES = frozenset(FROZEN_STRATEGIES.keys())
ALLOWED_SELECTORS = frozenset({"NONE", *SELECTOR_KINDS.keys()})
ALLOWED_MAX_TRADES = frozenset({3, 4, 5, 6, 8, 10})
RUNTIME_STATE_VERSION = 1
CONFIRM_TTL_SEC = 120.0
DEFAULT_LIVE_STRATEGY = "T30"


class OperatorMode(str, Enum):
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    HALTED = "HALTED"


@dataclass
class PendingConfirmation:
    action: str
    payload: dict[str, Any]
    created_at: float
    chat_id: str

    def expired(self, now: float | None = None) -> bool:
        t = time.time() if now is None else now
        return (t - self.created_at) > CONFIRM_TTL_SEC


@dataclass
class RuntimeControlState:
    version: int = RUNTIME_STATE_VERSION
    mode: str = OperatorMode.RUNNING.value
    strategy: str = "T30"
    selector: str = "NONE"
    max_simultaneous_trades: int = 8
    corrupt: bool = False
    corrupt_reason: str = ""
    pending: dict[str, Any] | None = None
    last_update_ids: list[int] = field(default_factory=list)
    processed_update_ids: list[int] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)
    last_hourly_report_at: float | None = None
    hour_counters: dict[str, int] = field(default_factory=dict)
    hour_events: list[dict[str, Any]] = field(default_factory=list)
    hour_window_start: float | None = None
    equity_btc_at_hour_start: float | None = None
    btc_free_at_hour_start: float | None = None
    btc_locked_at_hour_start: float | None = None
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "strategy": self.strategy,
            "selector": self.selector,
            "max_simultaneous_trades": self.max_simultaneous_trades,
            "corrupt": self.corrupt,
            "corrupt_reason": self.corrupt_reason,
            "pending": self.pending,
            "last_update_ids": list(self.last_update_ids)[-50:],
            "processed_update_ids": list(self.processed_update_ids)[-200:],
            "audit": list(self.audit)[-200:],
            "last_hourly_report_at": self.last_hourly_report_at,
            "hour_counters": dict(self.hour_counters),
            "hour_events": list(self.hour_events)[-100:],
            "hour_window_start": self.hour_window_start,
            "equity_btc_at_hour_start": self.equity_btc_at_hour_start,
            "btc_free_at_hour_start": self.btc_free_at_hour_start,
            "btc_locked_at_hour_start": self.btc_locked_at_hour_start,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> RuntimeControlState:
        if not isinstance(raw, dict):
            raise ValueError("runtime state must be a dict")
        ver = int(raw.get("version") or 0)
        if ver != RUNTIME_STATE_VERSION:
            raise ValueError(f"unsupported runtime state version {ver}")
        strategy = str(raw.get("strategy") or DEFAULT_LIVE_STRATEGY).upper()
        selector = str(raw.get("selector") or "NONE").upper()
        if selector in ("", "NULL", "NONE"):
            selector = "NONE"
        max_n = int(raw.get("max_simultaneous_trades") or 8)
        mode = str(raw.get("mode") or OperatorMode.RUNNING.value).upper()
        if strategy not in ALLOWED_STRATEGIES:
            raise ValueError(f"invalid strategy in state: {strategy}")
        if selector not in ALLOWED_SELECTORS:
            raise ValueError(f"invalid selector in state: {selector}")
        if max_n not in ALLOWED_MAX_TRADES:
            raise ValueError(f"invalid max in state: {max_n}")
        if mode not in {m.value for m in OperatorMode}:
            raise ValueError(f"invalid mode in state: {mode}")
        return cls(
            version=ver,
            mode=mode,
            strategy=strategy,
            selector=selector,
            max_simultaneous_trades=max_n,
            corrupt=bool(raw.get("corrupt")),
            corrupt_reason=str(raw.get("corrupt_reason") or ""),
            pending=raw.get("pending") if isinstance(raw.get("pending"), dict) else None,
            last_update_ids=[int(x) for x in (raw.get("last_update_ids") or [])][-50:],
            processed_update_ids=[int(x) for x in (raw.get("processed_update_ids") or [])][-200:],
            audit=[x for x in (raw.get("audit") or []) if isinstance(x, dict)][-200:],
            last_hourly_report_at=raw.get("last_hourly_report_at"),
            hour_counters={str(k): int(v) for k, v in (raw.get("hour_counters") or {}).items()},
            hour_events=[x for x in (raw.get("hour_events") or []) if isinstance(x, dict)][-100:],
            hour_window_start=raw.get("hour_window_start"),
            equity_btc_at_hour_start=raw.get("equity_btc_at_hour_start"),
            btc_free_at_hour_start=raw.get("btc_free_at_hour_start"),
            btc_locked_at_hour_start=raw.get("btc_locked_at_hour_start"),
            updated_at=float(raw.get("updated_at") or time.time()),
        )


class RuntimeStateStore:
    """Persist runtime control state beside the bot DB (never secrets)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def load_or_default(self) -> RuntimeControlState:
        with self._lock:
            if not self.path.exists():
                state = RuntimeControlState()
                self.save(state)
                return state
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                return RuntimeControlState.from_dict(raw)
            except Exception as e:  # noqa: BLE001
                # Fail closed — block new entries until recovered.
                bad = RuntimeControlState(
                    mode=OperatorMode.HALTED.value,
                    corrupt=True,
                    corrupt_reason=scrub_exception(e),
                )
                try:
                    bak = self.path.with_suffix(self.path.suffix + ".corrupt")
                    bak.write_text(self.path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
                except Exception:  # noqa: BLE001
                    pass
                self.save(bad)
                return bad

    def save(self, state: RuntimeControlState) -> None:
        with self._lock:
            state.updated_at = time.time()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            payload = scrub_text(json.dumps(state.to_dict(), indent=2, default=str))
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self.path)


class RuntimeStrategyProvider(StrategyProvider):
    """Mutable strategy/selector for NEW entries only (open trades keep their freeze)."""

    def __init__(
        self,
        strategy_key: str = DEFAULT_LIVE_STRATEGY,
        *,
        selector_key: str | None = None,
        strategies_cfg: Any = None,
    ) -> None:
        self._strategies_cfg = strategies_cfg
        self._lock = threading.RLock()
        self._strategy = DEFAULT_LIVE_STRATEGY
        self._selector: str | None = None
        self.set_strategy(strategy_key)
        self.set_selector(selector_key)

    def set_strategy(self, key: str) -> str:
        k = str(key).upper()
        if k not in ALLOWED_STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(ALLOWED_STRATEGIES)}")
        # Resolve against frozen definitions — never mutate them.
        get_strategy(k, self._strategies_cfg)
        with self._lock:
            self._strategy = k
        return k

    def set_selector(self, key: str | None) -> str | None:
        if key in (None, "", "null", "NULL", "NONE"):
            with self._lock:
                self._selector = None
            return None
        k = str(key).upper()
        if k not in SELECTOR_KINDS:
            raise ValueError(f"selector must be NONE or one of {sorted(SELECTOR_KINDS)}")
        with self._lock:
            self._selector = k
        return k

    def get_strategy(self, context: Any = None) -> TrailStrategy:
        with self._lock:
            sel = self._selector
            default = self._strategy
            cfg = self._strategies_cfg
        if sel and isinstance(context, dict) and context.get("candidate_scores"):
            # Existing research helper — algorithms unchanged.
            picked = select_strategy_for_research(
                sel,
                candidate_scores=context["candidate_scores"],
                default=default,
            )
            return get_strategy(picked, cfg)
        return get_strategy(default, cfg)

    def strategy_key(self) -> str:
        with self._lock:
            return self._strategy

    def selector_key(self) -> str | None:
        with self._lock:
            return self._selector


@dataclass
class ControlResult:
    ok: bool
    message: str
    need_confirm: bool = False
    data: dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        return self.message


class RuntimeController:
    """Authoritative runtime control (Telegram + persistence)."""

    def __init__(
        self,
        store: RuntimeStateStore,
        *,
        strategies_cfg: Any = None,
        authorized_chat_id: str,
        on_apply: Callable[["RuntimeController"], None] | None = None,
    ) -> None:
        self.store = store
        self.strategies_cfg = strategies_cfg
        self.authorized_chat_id = str(authorized_chat_id).strip()
        self.on_apply = on_apply
        self._lock = threading.RLock()
        self.state = store.load_or_default()
        self.provider = RuntimeStrategyProvider(
            self.state.strategy,
            selector_key=None if self.state.selector == "NONE" else self.state.selector,
            strategies_cfg=strategies_cfg,
        )
        self._seen_update_ids: set[int] = set(self.state.processed_update_ids)

    # --- gates ---
    def fail_closed_halt(self, reason: str) -> None:
        """Block new entries and persist HALTED (fail-closed). Does not close positions."""
        with self._lock:
            self.state.mode = OperatorMode.HALTED.value
            self.state.pending = None
            if reason:
                self.state.corrupt_reason = scrub_text(reason)[:500]
            self._persist()
        self._audit(
            chat_id="system",
            command="FAIL_CLOSED",
            old="",
            new="HALTED",
            result=scrub_text(reason)[:200],
            authorized=True,
        )

    def blocks_new_entries(self) -> bool:
        with self._lock:
            if self.state.corrupt:
                return True
            return self.state.mode != OperatorMode.RUNNING.value

    def display_mode(self, *, safety_halted: bool = False) -> str:
        with self._lock:
            if self.state.corrupt:
                return OperatorMode.HALTED.value
            if safety_halted and self.state.mode == OperatorMode.RUNNING.value:
                return OperatorMode.HALTED.value
            return self.state.mode

    def _persist(self) -> None:
        self.store.save(self.state)
        if self.on_apply:
            try:
                self.on_apply(self)
            except Exception as e:  # noqa: BLE001
                logger.warning("runtime on_apply failed: %s", scrub_exception(e))

    def _audit(
        self,
        *,
        chat_id: str,
        command: str,
        old: Any,
        new: Any,
        result: str,
        authorized: bool,
    ) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "authorized_chat": bool(authorized),
            "chat_id_match": scrub_text(str(chat_id))[:32] if authorized else "REDACTED_UNAUTH",
            "command": scrub_text(str(command))[:200],
            "old_value": scrub_text(str(old))[:200],
            "new_value": scrub_text(str(new))[:200],
            "result": scrub_text(str(result))[:200],
        }
        with self._lock:
            self.state.audit.append(entry)
            self.state.audit = self.state.audit[-200:]
            self._persist()

    def is_authorized(self, chat_id: str | int | None) -> bool:
        if not self.authorized_chat_id:
            return False
        return str(chat_id).strip() == self.authorized_chat_id

    def mark_update_seen(self, update_id: int) -> bool:
        """Return False if duplicate (already processed)."""
        uid = int(update_id)
        with self._lock:
            if uid in self._seen_update_ids:
                return False
            self._seen_update_ids.add(uid)
            self.state.processed_update_ids.append(uid)
            self.state.processed_update_ids = self.state.processed_update_ids[-200:]
            self._persist()
            return True

    def bump_counter(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.state.hour_counters[key] = int(self.state.hour_counters.get(key) or 0) + n
            self._persist()

    def append_hour_event(self, timestamp: Any, text: str) -> None:
        with self._lock:
            self.state.hour_events.append({"timestamp": timestamp, "text": scrub_text(str(text))[:200]})
            self.state.hour_events = self.state.hour_events[-100:]
            self._persist()

    def reset_hour_counters(
        self,
        *,
        equity_btc: float | None = None,
        btc_free: float | None = None,
        btc_locked: float | None = None,
    ) -> None:
        with self._lock:
            self.state.hour_counters = {}
            self.state.hour_events = []
            self.state.hour_window_start = time.time()
            self.state.equity_btc_at_hour_start = equity_btc
            self.state.btc_free_at_hour_start = btc_free
            self.state.btc_locked_at_hour_start = btc_locked
            self.state.last_hourly_report_at = time.time()
            self._persist()

    # --- command handlers ---
    def handle_text(self, *, chat_id: str, text: str, update_id: int | None = None) -> ControlResult:
        raw = (text or "").strip()
        if update_id is not None:
            if not self.mark_update_seen(int(update_id)):
                return ControlResult(True, "Duplicate update ignored.", data={"duplicate": True})

        authorized = self.is_authorized(chat_id)
        if not authorized:
            self._audit(
                chat_id=str(chat_id),
                command=raw,
                old=None,
                new=None,
                result="REJECTED_UNAUTHORIZED",
                authorized=False,
            )
            return ControlResult(False, "Unauthorized.")

        if not raw.startswith("/"):
            return ControlResult(False, "Commands must start with /.")

        parts = raw.split()
        cmd = parts[0].lower().split("@")[0]
        args = parts[1:]

        try:
            if cmd == "/strategy":
                return self._cmd_strategy(args, chat_id=str(chat_id), raw=raw)
            if cmd == "/selector":
                return self._cmd_selector(args, chat_id=str(chat_id), raw=raw)
            if cmd == "/max":
                return self._cmd_max(args, chat_id=str(chat_id), raw=raw)
            if cmd in {"/start", "/resume"}:
                return self._cmd_start_resume(cmd, chat_id=str(chat_id))
            if cmd == "/pause":
                return self._set_mode(OperatorMode.PAUSED, chat_id=str(chat_id), command=cmd)
            if cmd == "/stop":
                return self._set_mode(OperatorMode.STOPPED, chat_id=str(chat_id), command=cmd)
            if cmd == "/emergency":
                return self._cmd_emergency_request(chat_id=str(chat_id))
            if cmd == "/confirm_emergency":
                return self._cmd_confirm_emergency(chat_id=str(chat_id))
            if cmd == "/confirm":
                return self._cmd_confirm(chat_id=str(chat_id))
            if cmd == "/cancel":
                return self._cmd_cancel(chat_id=str(chat_id))
            if cmd in {
                "/status",
                "/config",
                "/positions",
                "/balance",
                "/health",
                "/reconcile",
                "/help",
                "/performance",
            }:
                # Read-only / reconcile dispatched by TelegramControlPlane with engine.
                return ControlResult(True, cmd, data={"dispatch": cmd[1:], "args": args})
            self._audit(
                chat_id=str(chat_id),
                command=raw,
                old=None,
                new=None,
                result="REJECTED_UNKNOWN",
                authorized=True,
            )
            return ControlResult(False, "Unknown command. Send /help.")
        except Exception as e:  # noqa: BLE001
            err = scrub_exception(e)
            self._audit(
                chat_id=str(chat_id),
                command=raw,
                old=None,
                new=None,
                result=f"ERROR:{err}",
                authorized=True,
            )
            return ControlResult(False, f"Command failed: {err}")

    def _set_mode(self, mode: OperatorMode, *, chat_id: str, command: str) -> ControlResult:
        with self._lock:
            old = self.state.mode
            self.state.mode = mode.value
            self.state.pending = None
            self._persist()
        self._audit(
            chat_id=chat_id,
            command=command,
            old=old,
            new=mode.value,
            result="OK",
            authorized=True,
        )
        msg = {
            OperatorMode.PAUSED: "PAUSED — new entries blocked. Existing positions & Binance protection continue.",
            OperatorMode.STOPPED: "STOPPED — new entries blocked. Existing positions & Binance protection continue.",
            OperatorMode.RUNNING: "RUNNING — new entries allowed (subject to safety).",
            OperatorMode.HALTED: "HALTED — new entries blocked.",
        }[mode]
        return ControlResult(True, msg, data={"mode": mode.value})

    def _cmd_start_resume(self, cmd: str, *, chat_id: str) -> ControlResult:
        with self._lock:
            if self.state.corrupt:
                return ControlResult(
                    False,
                    "FAIL CLOSED: runtime state corrupt — recover/replace runtime_control_state.json before resume.",
                )
            old = self.state.mode
            self.state.mode = OperatorMode.RUNNING.value
            self.state.pending = None
            self._persist()
        self._audit(chat_id=chat_id, command=cmd, old=old, new="RUNNING", result="OK", authorized=True)
        return ControlResult(
            True,
            "RUNNING — new entries allowed after safety checks. Existing positions unchanged.",
            data={"mode": "RUNNING", "startup_checks": True},
        )

    def _cmd_strategy(self, args: list[str], *, chat_id: str, raw: str) -> ControlResult:
        if len(args) != 1:
            return ControlResult(False, "Usage: /strategy T1…T10|T21|T30")
        key = args[0].upper()
        if key not in ALLOWED_STRATEGIES:
            return ControlResult(False, f"Invalid strategy. Allowed: {', '.join(sorted(ALLOWED_STRATEGIES))}")
        with self._lock:
            old = self.state.strategy
            if key == old:
                return ControlResult(True, f"Strategy already {key}.")
            self.state.pending = {
                "action": "strategy",
                "payload": {"strategy": key},
                "created_at": time.time(),
                "chat_id": chat_id,
            }
            self._persist()
        self._audit(chat_id=chat_id, command=raw, old=old, new=key, result="NEED_CONFIRM", authorized=True)
        return ControlResult(
            True,
            f"⚠️ Confirm strategy change\n{old} → {key}\nAffects NEW entries only.\nExisting trades keep their protection.\n\n/confirm   or   /cancel",
            need_confirm=True,
            data={"old": old, "new": key},
        )

    def _cmd_selector(self, args: list[str], *, chat_id: str, raw: str) -> ControlResult:
        if len(args) != 1:
            return ControlResult(False, "Usage: /selector NONE|A|B|C|D|E|F")
        key = args[0].upper()
        if key not in ALLOWED_SELECTORS:
            return ControlResult(False, f"Invalid selector. Allowed: {', '.join(sorted(ALLOWED_SELECTORS))}")
        with self._lock:
            old = self.state.selector
            if key == old:
                return ControlResult(True, f"Selector already {key}.")
            self.state.pending = {
                "action": "selector",
                "payload": {"selector": key},
                "created_at": time.time(),
                "chat_id": chat_id,
            }
            self._persist()
        self._audit(chat_id=chat_id, command=raw, old=old, new=key, result="NEED_CONFIRM", authorized=True)
        return ControlResult(
            True,
            f"⚠️ Confirm selector change\n{old} → {key}\nAffects NEW entries only (existing research definitions, unchanged algorithms).\n\n/confirm   or   /cancel",
            need_confirm=True,
            data={"old": old, "new": key},
        )

    def _cmd_max(self, args: list[str], *, chat_id: str, raw: str) -> ControlResult:
        if len(args) != 1:
            return ControlResult(False, "Usage: /max 3|4|5|6|8|10")
        try:
            n = int(args[0])
        except ValueError:
            return ControlResult(False, "Invalid max. Allowed: 3, 4, 5, 6, 8, 10")
        if n not in ALLOWED_MAX_TRADES:
            return ControlResult(False, "Invalid max. Allowed: 3, 4, 5, 6, 8, 10")
        with self._lock:
            old = self.state.max_simultaneous_trades
            if n == old:
                return ControlResult(True, f"Max already {n}.")
            self.state.pending = {
                "action": "max",
                "payload": {"max": n},
                "created_at": time.time(),
                "chat_id": chat_id,
            }
            self._persist()
        self._audit(chat_id=chat_id, command=raw, old=old, new=n, result="NEED_CONFIRM", authorized=True)
        return ControlResult(
            True,
            f"⚠️ Confirm\nMax simultaneous trades:\n{old} → {n}\n\nExisting positions are unaffected.\nNew entries will be limited to {n}.\n\n/confirm   or   /cancel",
            need_confirm=True,
            data={"old": old, "new": n},
        )

    def _cmd_emergency_request(self, *, chat_id: str) -> ControlResult:
        with self._lock:
            self.state.pending = {
                "action": "emergency",
                "payload": {},
                "created_at": time.time(),
                "chat_id": chat_id,
            }
            self._persist()
        self._audit(
            chat_id=chat_id,
            command="/emergency",
            old=self.state.mode,
            new="HALTED",
            result="NEED_CONFIRM_EMERGENCY",
            authorized=True,
        )
        return ControlResult(
            True,
            "⚠️ EMERGENCY STOP\nThis will:\n"
            "- block new entries\n"
            "- cancel eligible pending entry orders\n"
            "- preserve existing protective orders\n"
            "- reconcile Binance\n"
            "- enter HALTED\n\n"
            "Reply:\n/confirm_emergency",
            need_confirm=True,
        )

    def _cmd_confirm_emergency(self, *, chat_id: str) -> ControlResult:
        with self._lock:
            pend = self.state.pending
            if not pend or pend.get("action") != "emergency":
                return ControlResult(False, "No pending emergency confirmation.")
            if pend.get("chat_id") != chat_id:
                return ControlResult(False, "Confirmation chat mismatch.")
            if PendingConfirmation(
                action="emergency",
                payload={},
                created_at=float(pend.get("created_at") or 0),
                chat_id=chat_id,
            ).expired():
                self.state.pending = None
                self._persist()
                return ControlResult(False, "Emergency confirmation expired. Send /emergency again.")
            old = self.state.mode
            self.state.mode = OperatorMode.HALTED.value
            self.state.pending = None
            self._persist()
        self._audit(
            chat_id=chat_id,
            command="/confirm_emergency",
            old=old,
            new="HALTED",
            result="OK",
            authorized=True,
        )
        return ControlResult(
            True,
            "HALTED (emergency confirmed). New entries blocked. Protective orders preserved. Reconcile requested.",
            data={"emergency": True, "mode": "HALTED"},
        )

    def _cmd_confirm(self, *, chat_id: str) -> ControlResult:
        with self._lock:
            pend = self.state.pending
            if not pend:
                return ControlResult(False, "Nothing to confirm.")
            if pend.get("action") == "emergency":
                return ControlResult(False, "Use /confirm_emergency for emergency.")
            if pend.get("chat_id") != chat_id:
                return ControlResult(False, "Confirmation chat mismatch.")
            if PendingConfirmation(
                action=str(pend.get("action")),
                payload=dict(pend.get("payload") or {}),
                created_at=float(pend.get("created_at") or 0),
                chat_id=chat_id,
            ).expired():
                self.state.pending = None
                self._persist()
                return ControlResult(False, "Confirmation expired. Re-issue the command.")
            action = str(pend.get("action"))
            payload = dict(pend.get("payload") or {})
            old_s = self.state.strategy
            old_sel = self.state.selector
            old_max = self.state.max_simultaneous_trades
            if action == "strategy":
                key = str(payload["strategy"]).upper()
                self.provider.set_strategy(key)
                self.state.strategy = key
                new = key
                old = old_s
            elif action == "selector":
                key = str(payload["selector"]).upper()
                self.provider.set_selector(None if key == "NONE" else key)
                self.state.selector = key
                new = key
                old = old_sel
            elif action == "max":
                n = int(payload["max"])
                if n not in ALLOWED_MAX_TRADES:
                    return ControlResult(False, "Invalid max in pending payload.")
                self.state.max_simultaneous_trades = n
                new = n
                old = old_max
            else:
                return ControlResult(False, f"Unknown pending action {action}")
            self.state.pending = None
            self._persist()
        self._audit(chat_id=chat_id, command="/confirm", old=old, new=new, result="APPLIED", authorized=True)
        return ControlResult(True, f"Applied {action}: {old} → {new}", data={"action": action, "old": old, "new": new})

    def _cmd_cancel(self, *, chat_id: str) -> ControlResult:
        with self._lock:
            had = self.state.pending is not None
            self.state.pending = None
            self._persist()
        self._audit(
            chat_id=chat_id,
            command="/cancel",
            old=had,
            new=None,
            result="CANCELLED" if had else "NOTHING",
            authorized=True,
        )
        return ControlResult(True, "Cancelled." if had else "Nothing pending.")

    def apply_to_engine(self, engine: Any) -> None:
        """Push runtime settings into a live engine (NEW entries only)."""
        with self._lock:
            strategy = self.state.strategy
            selector = None if self.state.selector == "NONE" else self.state.selector
            max_n = self.state.max_simultaneous_trades
        if hasattr(engine, "strategy_provider") and isinstance(engine.strategy_provider, RuntimeStrategyProvider):
            engine.strategy_provider.set_strategy(strategy)
            engine.strategy_provider.set_selector(selector)
        elif hasattr(engine, "strategy_provider"):
            # Replace fixed provider with runtime provider preserving strategies cfg.
            engine.strategy_provider = self.provider
        if hasattr(engine, "entry_engine"):
            engine.entry_engine.strategy_key = strategy
            engine.entry_engine.selector_key = selector
            engine.entry_engine.max_open = max_n
        if hasattr(engine, "portfolio") and hasattr(engine.portfolio, "set_max_simultaneous_trades"):
            engine.portfolio.set_max_simultaneous_trades(max_n)
        # Wire entry block into safety without changing HALT semantics for incidents.
        if hasattr(engine, "safety"):
            engine.safety.set_entries_blocker(self.blocks_new_entries)


def default_runtime_state_path(db_path: str | Path) -> Path:
    return Path(db_path).resolve().parent / "runtime_control_state.json"


HELP_TEXT = """BTCC Telegram controls

Runtime (confirm when prompted):
/strategy T1…T10|T21|T30
/selector NONE|A|B|C|D|E|F
/max 3|4|5|6|8|10
/confirm  /cancel

State:
/start  /resume  /pause  /stop
/emergency → /confirm_emergency

Read-only:
/status  /config  /positions  /balance  /health
/performance  /reconcile

Frozen YAML research/risk/T1–T10/T21/T30 definitions cannot be changed here.
"""
