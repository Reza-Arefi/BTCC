"""Factory for execution stack. Defaults to PAPER. REAL never silently becomes PAPER."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from btcc.execution.intent import IntentJournal
from btcc.execution.kill_switch import ExecutionKillSwitch
from btcc.execution.modes import ExecutionMode
from btcc.execution.oms import OrderManagementSystem
from btcc.execution.order_state import OrderJournal
from btcc.execution.paper_broker import PaperBroker
from btcc.execution.real_broker import RealBroker
from btcc.execution.risk import ExecutionRiskGate
from btcc.safety.no_trading import TradingForbiddenError
from btcc.sim.accounting import CostModel
from btcc.sim.paper_equity import PaperEquityTracker

ROOT = Path(__file__).resolve().parents[2]


def resolve_execution_mode(sim_cfg: dict[str, Any] | None = None) -> ExecutionMode:
    """Resolve mode from config. Defaults PAPER. Forces PAPER if allow_trading is false."""
    sim = sim_cfg or {}
    sl = sim.get("selector_live_raw") or {}
    raw = sl.get("execution_mode", sim.get("execution_mode", "PAPER"))
    mode = ExecutionMode.parse(raw)
    allow_trading = bool(sim.get("allow_trading", False))
    paper_mode = bool(sim.get("paper_mode", True))
    if allow_trading:
        raise TradingForbiddenError("allow_trading must remain false — refusing REAL stack")
    if mode == ExecutionMode.REAL:
        raise TradingForbiddenError(
            "execution_mode=REAL is not enabled; staying fail-closed "
            "(use PaperBroker for live paper; RealBroker only in explicit tests)"
        )
    if mode == ExecutionMode.REAL_CANARY_SINGLE_SHOT:
        raise TradingForbiddenError(
            "execution_mode=REAL_CANARY_SINGLE_SHOT is not allowed in live resolve — "
            "use build_canary_stack (writes remain gated; never wired as paper broker)"
        )
    if mode == ExecutionMode.REAL_BOUNDED_6H:
        raise TradingForbiddenError(
            "execution_mode=REAL_BOUNDED_6H is not allowed in live resolve — "
            "use build_bounded_6h_stack (writes remain gated; never wired as paper broker)"
        )
    if mode == ExecutionMode.REAL_SHADOW:
        raise TradingForbiddenError(
            "execution_mode=REAL_SHADOW is not allowed in live resolve — "
            "use build_shadow_stack (read-only observation; never wired as paper broker)"
        )
    if mode == ExecutionMode.TEST:
        raise TradingForbiddenError(
            "execution_mode=TEST is not allowed in live resolve — use build_test_execution_stack"
        )
    if not paper_mode and mode == ExecutionMode.PAPER:
        return ExecutionMode.PAPER
    return mode


def _execution_dir(sim_cfg: dict[str, Any]) -> Path:
    storage = sim_cfg.get("storage") or {}
    root = storage.get("root_dir") or "data/selector_live"
    p = Path(root)
    if not p.is_absolute():
        p = ROOT / p
    return p / "execution"


def build_execution_stack(
    sim_cfg: dict[str, Any],
    *,
    costs: CostModel,
    equity: PaperEquityTracker | None = None,
    open_opps_provider: Any | None = None,
    force_mode: ExecutionMode | None = None,
    enable_intent_journal: bool = True,
) -> OrderManagementSystem:
    """Build OMS+broker for the live engine. Live path: always PAPER."""
    if bool(sim_cfg.get("allow_trading", False)):
        raise TradingForbiddenError("allow_trading must remain false")
    mode = force_mode if force_mode is not None else resolve_execution_mode(sim_cfg)
    if mode == ExecutionMode.REAL:
        raise TradingForbiddenError("build_execution_stack refuses REAL — no PAPER fallback")
    if mode == ExecutionMode.REAL_CANARY_SINGLE_SHOT:
        raise TradingForbiddenError(
            "build_execution_stack refuses REAL_CANARY_SINGLE_SHOT — use build_canary_stack"
        )
    if mode == ExecutionMode.REAL_BOUNDED_6H:
        raise TradingForbiddenError(
            "build_execution_stack refuses REAL_BOUNDED_6H — use build_bounded_6h_stack"
        )
    if mode == ExecutionMode.REAL_SHADOW:
        raise TradingForbiddenError(
            "build_execution_stack refuses REAL_SHADOW — use build_shadow_stack"
        )
    if mode == ExecutionMode.TEST:
        raise TradingForbiddenError(
            "build_execution_stack refuses TEST — use build_test_execution_stack"
        )

    sl = sim_cfg.get("selector_live_raw") or {}
    max_positions = int(
        sim_cfg.get("max_open_opportunities", sl.get("max_open_opportunities", 4))
    )
    max_exposure = float(
        sl.get("max_total_exposure", sim_cfg.get("max_total_exposure", 1.0))
    )
    allocation = float(sim_cfg.get("position_allocation_pct", 0.25))
    strategy_version = str(
        sim_cfg.get("selector_version", sl.get("version", "E-v1-25pct-PAPER"))
    )

    exec_dir = _execution_dir(sim_cfg)
    kill = ExecutionKillSwitch(exec_dir / "kill_switch.json")
    order_journal = OrderJournal(exec_dir / "orders.jsonl") if enable_intent_journal else None
    intent_journal = IntentJournal(exec_dir / "intents.jsonl") if enable_intent_journal else None

    broker = PaperBroker(costs, equity=equity, open_opps_provider=open_opps_provider)
    risk = ExecutionRiskGate(
        mode=ExecutionMode.PAPER,
        allow_trading=False,
        kill_switch=kill,
        order_journal=order_journal,
        max_positions=max_positions,
        max_total_exposure=max_exposure,
        allocation_pct=allocation,
    )
    return OrderManagementSystem(
        broker=broker,
        risk=risk,
        intent_journal=intent_journal,
        order_journal=order_journal,
        kill_switch=kill,
        strategy_version=strategy_version,
    )


def build_test_execution_stack(
    *,
    root_dir: Path,
    broker: Any | None = None,
    max_positions: int = 4,
    max_total_exposure: float = 1.0,
    allocation_pct: float = 0.25,
    strategy_version: str = "E-v1-25pct-PAPER",
) -> OrderManagementSystem:
    """TEST-only OMS stack with SimulatedBroker. Never used by btcc.service."""
    from btcc.execution.sim import SimulatedBroker

    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    kill = ExecutionKillSwitch(root / "kill_switch.json")
    order_journal = OrderJournal(root / "orders.jsonl")
    intent_journal = IntentJournal(root / "intents.jsonl")
    sim_broker = broker if broker is not None else SimulatedBroker()
    if getattr(sim_broker, "mode", None) != ExecutionMode.TEST:
        raise TradingForbiddenError("build_test_execution_stack requires SimulatedBroker (TEST)")
    risk = ExecutionRiskGate(
        mode=ExecutionMode.TEST,
        allow_trading=False,
        kill_switch=kill,
        order_journal=order_journal,
        max_positions=max_positions,
        max_total_exposure=max_total_exposure,
        allocation_pct=allocation_pct,
    )
    return OrderManagementSystem(
        broker=sim_broker,
        risk=risk,
        intent_journal=intent_journal,
        order_journal=order_journal,
        kill_switch=kill,
        strategy_version=strategy_version,
    )


def build_shadow_stack(
    *,
    root_dir: Path,
    client: Any | None = None,
    credentials: Any | None = None,
    load_env_credentials: bool = False,
    local_view: Any | None = None,
    symbols: list[str] | None = None,
    fill_symbols: list[str] | None = None,
    max_skew_ms: int = 5000,
    max_age_ms: int = 30000,
) -> dict[str, Any]:
    """Build REAL_SHADOW observation stack. Never used as live paper OMS.

    Returns dict with broker, service, safety, kill_switch.
    Does not mutate paper equity/strategy. Writes remain impossible.
    """
    from btcc.execution.mexc.reconcile import LocalExecutionView
    from btcc.execution.shadow import ShadowBroker, ShadowReconciliationService
    from btcc.execution.safety_state import ExecutionSafetyState, REAL_TRADING_DISABLED

    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    safety = ExecutionSafetyState(
        reason_code=REAL_TRADING_DISABLED,
        detail="REAL_SHADOW stack initialized; trading impossible",
        real_entries_blocked=True,
    )
    broker = ShadowBroker(
        client=client,
        credentials=credentials,
        load_env_credentials=load_env_credentials,
        safety=safety,
    )
    kill = ExecutionKillSwitch(root / "shadow_kill_switch.json")
    service = ShadowReconciliationService(
        broker,
        safety=safety,
        local_view=local_view if local_view is not None else LocalExecutionView(),
        symbols=symbols,
        fill_symbols=fill_symbols,
        max_skew_ms=max_skew_ms,
        max_age_ms=max_age_ms,
        state_dir=root / "shadow",
    )
    return {
        "mode": ExecutionMode.REAL_SHADOW,
        "broker": broker,
        "service": service,
        "safety": safety,
        "kill_switch": kill,
        "allow_trading": False,
        "paper_touch": False,
    }


def build_real_broker_for_tests_only() -> RealBroker:
    """Test helper: construct RealBroker without credentials (reads fail closed).

    Must not be used by the live paper engine. Trading methods always deny by default.
    """
    return RealBroker()


def build_canary_stack(
    *,
    root_dir: Path,
    client: Any | None = None,
    write_client: Any | None = None,
    protection_transport: Any | None = None,  # unused — BOT_MANAGED (kept for API compat)
    credentials: Any | None = None,
    allow_trading: bool = False,
    arm_writes: bool = False,
    protection_api_confirmed: bool = False,  # unused for market writes / BOT_MANAGED
    max_positions: int = 1,
    allocation_pct: float = 0.25,
    strategy_version: str = "T1-ONLY-CANARY",
    price_source: Any | None = None,
) -> dict[str, Any]:
    """Build REAL_CANARY_SINGLE_SHOT stack. Never used by btcc.service / paper path.

    Default: allow_trading=False, WriteGate CLOSED — no real HTTP writes.
    Tests may arm writes with mocks. Production enablement is a separate explicit step.

    BOT-MANAGED T1: no MexcSpotProtectionAdapter; market writes gated by
    allow_trading + arm_writes only (protection_api_confirmed not required).
    """
    from btcc.execution.canary import CanaryState
    from btcc.execution.canary_runtime import CanaryRuntime, ExecutionAuditLog
    from btcc.execution.price_monitor import SequencePriceSource, T1PriceMonitor
    from btcc.execution.protection import ProtectionMode, ProtectiveExitManager, T1StateJournal
    from btcc.execution.safety_state import ExecutionSafetyState, REAL_TRADING_DISABLED
    from btcc.execution.t1 import T1
    from btcc.execution.write_gate import WriteGate

    _ = protection_transport  # intentionally unused (no native MEXC protection)
    _ = protection_api_confirmed  # not required for canary market writes

    if allocation_pct > 0.25 + 1e-12:
        raise TradingForbiddenError("canary allocation cannot exceed 25%")
    if max_positions > 1:
        raise TradingForbiddenError("canary max_positions must be 1")

    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Market writes from allow_trading + arm_writes ONLY.
    gate = WriteGate(
        allow_trading=bool(allow_trading),
        canary_writes_armed=bool(arm_writes) and bool(allow_trading),
        protection_api_confirmed=False,
    )
    safety = ExecutionSafetyState(
        reason_code=REAL_TRADING_DISABLED if not gate.market_writes_allowed else "CANARY_ARMED",
        detail="canary stack; writes gated; BOT_MANAGED T1",
        real_entries_blocked=not gate.market_writes_allowed,
    )

    # RealBroker without protection adapter (protection=None).
    broker = RealBroker(
        client=client,
        write_client=write_client,
        protection=None,
        credentials=credentials,
        safety=safety,
        write_gate=gate,
        mode=ExecutionMode.REAL_CANARY_SINGLE_SHOT,
    )
    kill = ExecutionKillSwitch(root / "canary_kill_switch.json")
    order_journal = OrderJournal(root / "canary_orders.jsonl")
    intent_journal = IntentJournal(root / "canary_intents.jsonl")
    t1_journal = T1StateJournal(root / "canary_t1_state.jsonl")
    risk = ExecutionRiskGate(
        mode=ExecutionMode.REAL_CANARY_SINGLE_SHOT,
        allow_trading=bool(allow_trading),
        kill_switch=kill,
        order_journal=order_journal,
        max_positions=max_positions,
        max_total_exposure=float(allocation_pct) * max_positions,
        allocation_pct=allocation_pct,
    )

    # BOT_MANAGED — no MexcSpotProtectionAdapter required.
    prot_mgr = ProtectiveExitManager(
        broker=None,
        geometry=T1,
        journal=t1_journal,
        mode=ProtectionMode.BOT_MANAGED,
    )
    audit = ExecutionAuditLog(root / "canary_audit.jsonl")

    # Telegram observability (same BTCC_TELEGRAM_* as paper). Never gates writes.
    from btcc.telegram.real_trades import (
        RealTradeNotifier,
        TelegramSentLedger,
        build_real_telegram_send_from_env,
    )

    trade_notifier = RealTradeNotifier(
        build_real_telegram_send_from_env(),
        enabled=True,
        ledger=TelegramSentLedger(root / "telegram_sent.json"),
    )

    runtime = CanaryRuntime(
        broker=broker,
        risk=risk,
        protection_manager=prot_mgr,
        canary=CanaryState(max_entries=1, max_positions=1),
        write_gate=gate,
        kill_switch=kill,
        intent_journal=intent_journal,
        order_journal=order_journal,
        audit=audit,
        strategy_version=strategy_version,
        max_positions=max_positions,
        allocation_pct=allocation_pct,
        trade_notifier=trade_notifier,
    )

    # Injectable price source; default SequencePriceSource for closed/test stacks.
    src = price_source if price_source is not None else SequencePriceSource(lambda: 0.05)
    monitor = T1PriceMonitor(
        source=src,
        symbol="ETHBTC",  # placeholder; updated on entry
        on_tick=lambda _tick: None,
    )
    runtime.attach_price_monitor(monitor)

    return {
        "mode": ExecutionMode.REAL_CANARY_SINGLE_SHOT,
        "broker": broker,
        "runtime": runtime,
        "risk": risk,
        "gate": gate,
        "kill_switch": kill,
        "protection": prot_mgr,
        "t1_journal": t1_journal,
        "allow_trading": bool(allow_trading),
        "paper_touch": False,
        "strategy_version": strategy_version,
        "exit": "T1_ONLY",
        "t1": T1,
        "protection_mode": "BOT_MANAGED",
        "price_monitor": monitor,
        "signal_evaluation_interval_s": 900.0,
        "price_monitor_default_poll_s": 1.0,
        "trade_notifier": trade_notifier,
    }


def build_bounded_6h_stack(
    *,
    root_dir: Path,
    client: Any | None = None,
    write_client: Any | None = None,
    credentials: Any | None = None,
    allow_trading: bool = False,
    arm_writes: bool = False,
    max_positions: int = 4,
    allocation_pct: float = 0.25,
    max_total_exposure: float = 1.0,
    duration_s: float = 6 * 3600,
    strategy_version: str = "T1-ONLY-BOUNDED-6H",
    price_source_factory: Any | None = None,
    session: Any | None = None,
    require_live_ticker: bool | None = None,
) -> dict[str, Any]:
    """Build REAL_BOUNDED_6H stack. Never used by btcc.service / paper / canary.

    Default: allow_trading=False, WriteGate CLOSED.
    Arming requires allow_trading + arm_writes → bounded_6h_writes_armed.
    Hard caps: duration≤6h, positions≤4, allocation≤25%, exposure≤100%.

    When armed, require_live_ticker defaults True and SequencePriceSource is rejected.
    """
    from btcc.execution.bounded_runtime import BoundedSessionRuntime, ExecutionAuditLog
    from btcc.execution.bounded_session import BoundedSession, BoundedSessionConfig
    from btcc.execution.multi_monitor import MultiPositionMonitorRegistry
    from btcc.execution.price_monitor import MexcPublicTickerSource, SequencePriceSource
    from btcc.execution.protection import ProtectionMode, ProtectiveExitManager, T1StateJournal
    from btcc.execution.safety_state import ExecutionSafetyState, REAL_TRADING_DISABLED
    from btcc.execution.t1 import T1
    from btcc.execution.write_gate import WriteGate

    armed = bool(arm_writes) and bool(allow_trading)
    live_required = bool(require_live_ticker) if require_live_ticker is not None else armed

    cfg = BoundedSessionConfig(
        duration_s=float(duration_s),
        max_positions=int(max_positions),
        allocation_pct=float(allocation_pct),
        max_total_exposure=float(max_total_exposure),
        strategy_version=strategy_version,
    )
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    if bool(arm_writes) and bool(allow_trading):
        # Distinct from canary: never set canary_writes_armed here.
        gate = WriteGate(
            allow_trading=True,
            canary_writes_armed=False,
            protection_api_confirmed=False,
            bounded_6h_writes_armed=True,
        )
    else:
        gate = WriteGate(
            allow_trading=False,
            canary_writes_armed=False,
            protection_api_confirmed=False,
            bounded_6h_writes_armed=False,
        )

    safety = ExecutionSafetyState(
        reason_code=REAL_TRADING_DISABLED if not gate.market_writes_allowed else "BOUNDED_6H_ARMED",
        detail="bounded 6h stack; writes gated; BOT_MANAGED T1",
        real_entries_blocked=not gate.market_writes_allowed,
    )
    broker = RealBroker(
        client=client,
        write_client=write_client,
        protection=None,
        credentials=credentials,
        safety=safety,
        write_gate=gate,
        mode=ExecutionMode.REAL_BOUNDED_6H,
    )
    kill = ExecutionKillSwitch(root / "bounded_kill_switch.json")
    order_journal = OrderJournal(root / "bounded_orders.jsonl")
    intent_journal = IntentJournal(root / "bounded_intents.jsonl")
    t1_journal = T1StateJournal(root / "bounded_t1_state.jsonl")
    risk = ExecutionRiskGate(
        mode=ExecutionMode.REAL_BOUNDED_6H,
        allow_trading=bool(allow_trading) and bool(arm_writes),
        kill_switch=kill,
        order_journal=order_journal,
        max_positions=cfg.max_positions,
        max_total_exposure=cfg.max_total_exposure,
        allocation_pct=cfg.allocation_pct,
    )
    prot_mgr = ProtectiveExitManager(
        broker=None,
        geometry=T1,
        journal=t1_journal,
        mode=ProtectionMode.BOT_MANAGED,
    )
    audit = ExecutionAuditLog(root / "bounded_audit.jsonl")

    from btcc.telegram.real_trades import (
        RealTradeNotifier,
        TelegramSentLedger,
        build_real_telegram_send_from_env,
    )

    trade_notifier = RealTradeNotifier(
        build_real_telegram_send_from_env(),
        enabled=True,
        ledger=TelegramSentLedger(root / "telegram_sent.json"),
    )

    sess = session or BoundedSession(config=cfg)

    def _src_factory():
        if price_source_factory is not None:
            return price_source_factory()
        # Armed live sessions MUST use the public ticker — never the test stub.
        # SequencePriceSource(lambda: 0.05) falsely activates T1 vs ~0.00x BTC pairs.
        if live_required:
            base = getattr(credentials, "base_url", None) or "https://api.mexc.com"
            return MexcPublicTickerSource(base_url=str(base))
        return SequencePriceSource(lambda: 0.05)

    # Registry callbacks rebound inside BoundedSessionRuntime.__init__.
    registry = MultiPositionMonitorRegistry(
        source_factory=_src_factory,
        on_position_tick=lambda *_a, **_k: None,
        require_live_ticker=live_required,
    )

    runtime = BoundedSessionRuntime(
        broker=broker,
        risk=risk,
        protection_manager=prot_mgr,
        session=sess,
        write_gate=gate,
        kill_switch=kill,
        intent_journal=intent_journal,
        order_journal=order_journal,
        audit=audit,
        monitor_registry=registry,
        trade_notifier=trade_notifier,
        strategy_version=strategy_version,
    )

    return {
        "mode": ExecutionMode.REAL_BOUNDED_6H,
        "broker": broker,
        "runtime": runtime,
        "session": sess,
        "risk": risk,
        "gate": gate,
        "kill_switch": kill,
        "protection": prot_mgr,
        "t1_journal": t1_journal,
        "allow_trading": bool(gate.market_writes_allowed),
        "paper_touch": False,
        "strategy_version": strategy_version,
        "exit": "T1_ONLY",
        "t1": T1,
        "protection_mode": "BOT_MANAGED",
        "monitors": registry,
        "signal_evaluation_interval_s": 900.0,
        "price_monitor_default_poll_s": 1.0,
        "trade_notifier": trade_notifier,
        "config": cfg,
        "root_dir": root,
    }
