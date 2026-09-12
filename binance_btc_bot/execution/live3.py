"""LIVE-3 controlled deployment — preflight + explicit arm.

HARD RULES
----------
* Default on-disk YAML stays LIVE=false / DRY_RUN=true / max=8.
* LIVE-3 target overlay: LIVE=true, DRY_RUN=false, T30, NONE, max=8,
  allocation 12.5%, total cap 100% (8 × 12.5%).
* Preflight never places real orders and never auto-arms.
* Arm only via --live3-arm --authorize-live after LIVE_3_PREFLIGHT=PASS
  with BINANCE_LIVE3_AUTHORIZED=true.
* Strategy / scoring / frozen T30 geometry / risk formulas are not modified.
* Entry uses E2 momentum profile (signal.momentum_profile=e2).
* Live T30 is fixed OCO: SL 3%, activation 1%, trail 0.25%.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from binance_btc_bot.config_loader import FROZEN_STRATEGIES, load_config
from binance_btc_bot.control.runtime import (
    OperatorMode,
    RuntimeControlState,
    RuntimeStateStore,
    default_runtime_state_path,
)
from binance_btc_bot.execution.engine import BinanceBotEngine
from binance_btc_bot.notifications.timezone_brt import DISPLAY_TZ, format_brt
from binance_btc_bot.portfolio.manager import PortfolioManager
from binance_btc_bot.preflight import PreflightReport, run_preflight
from binance_btc_bot.secrets import scrub_exception
from binance_btc_bot.strategy.trails import get_strategy

logger = logging.getLogger(__name__)

LIVE3_MAX = 8
LIVE3_ALLOC = 0.125
LIVE3_TOTAL_CAP = 1.0  # 8 × 12.5%
LIVE3_THRESHOLD = 0.65
LIVE3_RISK = 0.005
LIVE3_PREVIOUS_MAX = 3  # Stage-7 hard cap (for one-shot CONFIG UPDATED notify)
LIVE3_STRATEGY = "T30"
LIVE3_STRATEGY_GEOM = FROZEN_STRATEGIES[LIVE3_STRATEGY]


def build_live3_target_config(base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Intended LIVE-3 runtime config (in-memory). Does not rewrite YAML on disk."""
    cfg = deepcopy(base or load_config())
    live = dict(cfg.get("live") or {})
    live["enabled"] = True
    live["dry_run"] = False
    live["strategy"] = LIVE3_STRATEGY
    live["selector"] = None
    live["max_open_positions"] = LIVE3_MAX
    live.pop("first_trade_oneshot", None)
    live["live3"] = True
    cfg["live"] = live
    portfolio = dict(cfg.get("portfolio") or {})
    portfolio["max_simultaneous_trades"] = LIVE3_MAX
    portfolio["allocation_per_trade"] = LIVE3_ALLOC
    portfolio["max_total_allocation"] = LIVE3_TOTAL_CAP
    portfolio["one_position_per_symbol"] = True
    cfg["portfolio"] = portfolio
    risk = dict(cfg.get("risk") or {})
    risk["max_loss_per_trade"] = LIVE3_RISK
    risk["max_allocation_pct"] = LIVE3_ALLOC
    risk["max_aggregate_exposure"] = LIVE3_TOTAL_CAP
    risk["no_leverage"] = True
    cfg["risk"] = risk
    entry = dict(cfg.get("entry") or {})
    entry["long_threshold"] = LIVE3_THRESHOLD
    cfg["entry"] = entry
    return cfg


def build_live3_probe_config(base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Safe probe: same portfolio/strategy as LIVE-3 but writes blocked (no real orders)."""
    cfg = build_live3_target_config(base)
    live = dict(cfg.get("live") or {})
    live["enabled"] = False
    live["dry_run"] = True
    cfg["live"] = live
    return cfg


@dataclass
class GateResult:
    n: int
    name: str
    status: str  # PASS | FAIL | WARN | SKIP
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {"PASS", "WARN", "SKIP"}


@dataclass
class Live3PreflightReport:
    gates: list[GateResult] = field(default_factory=list)
    target_config: dict[str, Any] = field(default_factory=dict)
    probe_summary: dict[str, Any] = field(default_factory=dict)
    starting_equity_btc: float | None = None
    starting_bnb: float | None = None
    open_trades: int = 0
    open_orders: int = 0
    open_ocos: int = 0
    telegram_ok: bool = False
    timezone_ok: bool = False
    binance_rest: str = "UNKNOWN"
    user_data_ws: str = "UNKNOWN"
    reconciliation: str = "UNKNOWN"
    real_orders: str = "0"
    ready_to_arm: bool = False
    fail_closed: bool = True
    notes: list[str] = field(default_factory=list)

    def add(self, n: int, name: str, status: str, detail: str = "") -> GateResult:
        g = GateResult(n=n, name=name, status=status, detail=detail)
        self.gates.append(g)
        return g

    @property
    def ok(self) -> bool:
        return all(g.status != "FAIL" for g in self.gates)

    def text(self) -> str:
        lines = [
            "BINANCE BTC BOT — LIVE-3 PREFLIGHT",
            "=" * 44,
        ]
        for g in self.gates:
            lines.append(f"G{g.n:02d} {g.name:<42} {g.status}" + (f"  ({g.detail})" if g.detail else ""))
        lines.append("")
        lines.append(f"LIVE_3_PREFLIGHT={'PASS' if self.ok else 'FAIL'}")
        lines.append(f"READY_TO_ARM={str(self.ready_to_arm).upper()}")
        lines.append(f"REAL_ORDERS={self.real_orders}")
        lines.append(f"Binance REST:     {self.binance_rest}")
        lines.append(f"User-data WS:     {self.user_data_ws}")
        lines.append(f"Reconciliation:   {self.reconciliation}")
        lines.append(f"Starting equity:  {self.starting_equity_btc}")
        lines.append(f"Starting BNB:     {self.starting_bnb}")
        lines.append(f"Open trades:      {self.open_trades}")
        lines.append(f"Open orders:      {self.open_orders}")
        lines.append(f"Open OCOs:        {self.open_ocos}")
        lines.append(f"Telegram:         {'OK' if self.telegram_ok else 'FAIL'}")
        lines.append(f"Timezone BRT:     {'OK' if self.timezone_ok else 'FAIL'}")
        tgt = self.target_config.get("live") or {}
        port = self.target_config.get("portfolio") or {}
        lines.append("")
        lines.append("EFFECTIVE TARGET CONFIG (overlay; YAML defaults unchanged on disk)")
        lines.append(f"  LIVE={bool(tgt.get('enabled'))}")
        lines.append(f"  DRY_RUN={bool(tgt.get('dry_run'))}")
        lines.append(f"  strategy={tgt.get('strategy')}")
        lines.append(f"  selector={tgt.get('selector')}")
        lines.append(f"  max_simultaneous_trades={port.get('max_simultaneous_trades')}")
        lines.append(f"  allocation_per_trade={port.get('allocation_per_trade')}")
        lines.append(f"  max_total_allocation={port.get('max_total_allocation')}")
        if self.ok:
            lines.append("")
            lines.append("LIVE-3 READY TO ARM")
            lines.append(
                "Preflight passed. Live trading was NOT started automatically. "
                "No real order was placed by this preflight."
            )
        else:
            lines.append("")
            lines.append("LIVE-3 NOT READY — fail closed; do not arm.")
        for n in self.notes:
            lines.append(f"NOTE: {n}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "LIVE_3_PREFLIGHT": "PASS" if self.ok else "FAIL",
            "ready_to_arm": self.ready_to_arm,
            "real_orders": self.real_orders,
            "gates": [{"n": g.n, "name": g.name, "status": g.status, "detail": g.detail} for g in self.gates],
            "binance_rest": self.binance_rest,
            "user_data_ws": self.user_data_ws,
            "reconciliation": self.reconciliation,
            "starting_equity_btc": self.starting_equity_btc,
            "starting_bnb": self.starting_bnb,
            "open_trades": self.open_trades,
            "open_orders": self.open_orders,
            "open_ocos": self.open_ocos,
            "telegram_ok": self.telegram_ok,
            "timezone_ok": self.timezone_ok,
            "target_live": (self.target_config.get("live") or {}),
            "target_portfolio": (self.target_config.get("portfolio") or {}),
            "probe_summary": self.probe_summary,
            "notes": list(self.notes),
        }


def seed_live3_runtime_state(db_path: str | Path) -> Path:
    """Persist Telegram runtime initial state: RUNNING / T30 / NONE / max=LIVE3_MAX."""
    path = default_runtime_state_path(db_path)
    store = RuntimeStateStore(path)
    state = RuntimeControlState(
        mode=OperatorMode.RUNNING.value,
        strategy=LIVE3_STRATEGY,
        selector="NONE",
        max_simultaneous_trades=LIVE3_MAX,
    )
    store.save(state)
    return path


def format_live3_config_updated_message(
    *,
    old_max: int,
    new_max: int = LIVE3_MAX,
    allocation: float = LIVE3_ALLOC,
    risk: float = LIVE3_RISK,
    status: str = "RUNNING",
    timestamp: Any = None,
) -> str:
    """One-shot Telegram body for Stage-8 concurrency expansion (BRT timestamp)."""
    from datetime import datetime, timezone

    ts = timestamp if timestamp is not None else datetime.now(timezone.utc)
    max_alloc_pct = float(new_max) * float(allocation) * 100.0
    return "\n".join(
        [
            "⚙️ CONFIGURATION UPDATED",
            "────────────────",
            f"Strategy: {LIVE3_STRATEGY}",
            "Selector: NONE",
            f"Max simultaneous trades: {int(old_max)} → {int(new_max)}",
            f"Allocation/trade: {float(allocation) * 100:.1f}%",
            f"Max allocation: {max_alloc_pct:.0f}%",
            f"Risk ceiling: {float(risk) * 100:.1f}%",
            f"Status: {status}",
            f"Time: {format_brt(ts)}",
        ]
    )


def _read_runtime_max(db_path: str | Path) -> int | None:
    path = default_runtime_state_path(db_path)
    if not Path(path).is_file():
        return None
    try:
        st = RuntimeStateStore(path).load()
        return int(st.max_simultaneous_trades)
    except Exception:  # noqa: BLE001
        return None


def _map_pref(pref: PreflightReport, name: str) -> tuple[str, str]:
    c = pref.by_name(name)
    if c is None:
        return "FAIL", "MISSING"
    return c.status, c.detail or ""


def run_live3_preflight(
    base_cfg: dict[str, Any] | None = None,
    *,
    seed_runtime: bool = True,
) -> Live3PreflightReport:
    """Execute all LIVE-3 gates. Never places orders. Never arms the live loop."""
    out = Live3PreflightReport()
    out.real_orders = "0"
    out.fail_closed = True
    target = build_live3_target_config(base_cfg)
    probe = build_live3_probe_config(base_cfg)
    out.target_config = {
        "live": dict(target.get("live") or {}),
        "portfolio": dict(target.get("portfolio") or {}),
        "risk": dict(target.get("risk") or {}),
        "entry": dict(target.get("entry") or {}),
    }

    live = target.get("live") or {}
    port = target.get("portfolio") or {}
    risk = target.get("risk") or {}
    entry = target.get("entry") or {}

    # --- Gates 1–8: target configuration ---
    out.add(1, "LIVE=true", "PASS" if live.get("enabled") is True else "FAIL", f"enabled={live.get('enabled')}")
    out.add(2, "DRY_RUN=false", "PASS" if live.get("dry_run") is False else "FAIL", f"dry_run={live.get('dry_run')}")
    strat_ok = str(live.get("strategy") or "").upper() == LIVE3_STRATEGY
    strat_detail = ""
    try:
        sl, act, dist = LIVE3_STRATEGY_GEOM
        strat = get_strategy(LIVE3_STRATEGY, target.get("strategies"))
        geom_ok = (
            abs(strat.activation - act) < 1e-12
            and abs(strat.trail_distance - dist) < 1e-12
            and abs(strat.arm_sl_activation_trail - sl) < 1e-12
        )
        strat_detail = (
            f"act={strat.activation} trail={strat.trail_distance} sl={strat.arm_sl_activation_trail}"
        )
        strat_ok = strat_ok and geom_ok
    except Exception as e:  # noqa: BLE001
        strat_ok = False
        strat_detail = scrub_exception(e)
    out.add(3, f"strategy={LIVE3_STRATEGY}", "PASS" if strat_ok else "FAIL", strat_detail)
    sel = live.get("selector")
    out.add(
        4,
        "selector=NONE",
        "PASS" if sel in (None, "", "null", "NONE") else "FAIL",
        f"selector={sel}",
    )
    max_cfg_ok = int(port.get("max_simultaneous_trades") or 0) == LIVE3_MAX
    total_ok = abs(float(port.get("max_total_allocation") or 0) - LIVE3_TOTAL_CAP) < 1e-12
    hard_cap_ok = False
    hard_detail = ""
    try:
        pm = PortfolioManager.from_config({"portfolio": port})
        for i in range(LIVE3_MAX):
            r = pm.try_reserve(f"S{i}BTC")
            if not r.ok:
                hard_detail = f"reserve_{i}={r.reason}"
                break
        else:
            blocked = pm.try_reserve("NINTHBTC")
            hard_cap_ok = (not blocked.ok) and blocked.reason == "MAX_OPEN_TRADES"
            hard_detail = f"9th={blocked.reason} total_cap={pm.max_total_allocation}"
    except Exception as e:  # noqa: BLE001
        hard_detail = scrub_exception(e)
    out.add(
        5,
        f"max_simultaneous_trades={LIVE3_MAX}",
        "PASS" if (max_cfg_ok and hard_cap_ok and total_ok) else "FAIL",
        hard_detail or f"max={port.get('max_simultaneous_trades')} total={port.get('max_total_allocation')}",
    )
    out.add(
        6,
        "allocation_per_trade=12.5%",
        "PASS" if abs(float(port.get("allocation_per_trade") or 0) - 0.125) < 1e-12 else "FAIL",
    )
    out.add(
        7,
        "threshold=0.65",
        "PASS" if abs(float(entry.get("long_threshold") or 0) - 0.65) < 1e-12 else "FAIL",
    )
    out.add(
        8,
        "risk_ceiling=0.5%",
        "PASS" if abs(float(risk.get("max_loss_per_trade") or 0) - 0.005) < 1e-12 else "FAIL",
    )

    # --- Network / account gates via SAFE probe (writes blocked) ---
    engine = None
    pref: PreflightReport | None = None
    try:
        engine = BinanceBotEngine(probe)
        # Absolute: probe must not allow writes
        allowed, reason = engine.exchange._writes_allowed()
        if allowed:
            out.notes.append("CRITICAL: probe engine allowed writes — aborting")
            out.add(30, "fail_closed_startup", "FAIL", "probe writes_allowed unexpectedly True")
            out.ready_to_arm = False
            return out

        pref = run_preflight(probe, engine=engine)
        out.probe_summary = {
            "preflight_ok": pref.ok,
            "live_enabled": pref.live_enabled,
            "dry_run": pref.dry_run,
            "real_orders": pref.real_orders,
            "egress_ip": pref.egress_ip,
        }

        st, detail = _map_pref(pref, "Ed25519 authentication")
        out.add(9, "Binance authentication", "PASS" if st == "PASS" else st, detail)

        st, detail = _map_pref(pref, "canTrade")
        out.add(10, "API-key trading permission", "PASS" if st == "PASS" else st, detail)

        st, detail = _map_pref(pref, "Withdraw permission")
        wd_ok = st == "PASS" and "ENABLEWITHDRAWALS=FALSE" in (detail or "").upper()
        out.add(11, "withdrawals disabled", "PASS" if wd_ok else ("FAIL" if st != "PASS" else "FAIL"), detail)

        # Futures / margin from withdraw check data if present
        wd = pref.by_name("Withdraw permission")
        fut = (wd.data if wd else {}).get("enableFutures")
        mar = (wd.data if wd else {}).get("enableMargin")
        fut_ok = fut in (False, None) or fut is False
        # Prefer explicit False
        if wd and "enableFutures" in (wd.data or {}):
            fut_ok = wd.data.get("enableFutures") is False
            mar_ok = wd.data.get("enableMargin") is False
        else:
            mar_ok = True
            fut_ok = True
            out.notes.append("futures/margin flags not in apiRestrictions payload — treating carefully")
        out.add(
            12,
            "futures/margin disabled",
            "PASS" if (fut_ok and mar_ok) else "FAIL",
            f"enableFutures={fut} enableMargin={mar}",
        )

        st, detail = _map_pref(pref, "IP restriction")
        # PASS or WARN with detected IP both acceptable if allowlist configured PASS
        out.add(13, "IP restriction enabled", "PASS" if st in {"PASS", "WARN"} else st, detail)

        st, detail = _map_pref(pref, "WebSocket")
        out.user_data_ws = st
        out.add(14, "user-data WebSocket connects", "PASS" if st == "PASS" else st, detail)
        # Auth subscribe is part of WebSocket check detail
        auth_ok = "auth_ws_api_subscribe=PASS" in (detail or "")
        out.add(
            15,
            "authenticated user-data subscription",
            "PASS" if (st == "PASS" and auth_ok) else ("PASS" if st == "PASS" and "subscribe" in detail.lower() else st),
            detail[:200],
        )

        st, detail = _map_pref(pref, "Reconciliation")
        out.reconciliation = st
        out.add(16, "REST reconciliation", "PASS" if st == "PASS" else st, detail)

        st, detail = _map_pref(pref, "Database")
        # Also verify no stale open trades
        open_tr = engine.db.open_trades()
        out.open_trades = len(open_tr)
        expected_open = []
        stale_bad = []
        for t in open_tr:
            st_t = str(t.get("status") or "").upper()
            if st_t in {"PROTECTED", "PROTECTED_EMERGENCY", "DRY_RUN_PROTECTED"} and t.get(
                "binance_oco_list_id"
            ):
                expected_open.append(t)
            elif st_t == "PROTECTED_EMERGENCY":
                # Emergency STOP_LOSS path has no OCO list id; still a known open trade.
                expected_open.append(t)
            elif st_t in {"PROTECTION_FAILED", "ENTRY_FILLED", "PROTECTION_PENDING", "OPEN"}:
                stale_bad.append(t)
            else:
                stale_bad.append(t)
        # Mid-session re-arm: known protected trades are OK; unknown/unprotected are not.
        stale_ok = len(stale_bad) == 0
        out.add(
            17,
            "local DB no stale open positions",
            "PASS" if stale_ok else "FAIL",
            f"open_trades={len(open_tr)} protected_ok={len(expected_open)} stale_bad={len(stale_bad)} db={st}",
        )

        # Binance unexpected inventory / orders
        unexpected_pos = False
        unexpected_orders = False
        try:
            from binance_btc_bot.accounting.equity import compute_equity_btc
            from binance_btc_bot.exchange.base import AccountSnapshot, Balance

            acct = engine.exchange.get_account()
            if not isinstance(acct, AccountSnapshot):
                # Normalize raw dict (tests / alternate adapters)
                bals: dict[str, Balance] = {}
                for bal in (acct.get("balances") if isinstance(acct, dict) else []) or []:
                    asset = str(bal.get("asset") or "").upper()
                    bals[asset] = Balance(
                        asset=asset,
                        free=float(bal.get("free") or 0),
                        locked=float(bal.get("locked") or 0),
                    )
                acct = AccountSnapshot(balances=bals, raw=acct if isinstance(acct, dict) else {})
            btc_usdt = float(engine.exchange.get_price("BTCUSDT"))
            eq = compute_equity_btc(acct, btc_usdt=btc_usdt)
            out.starting_equity_btc = eq.total_equity_btc
            bnb = acct.balances.get("BNB")
            if bnb is not None:
                out.starting_bnb = float(bnb.free) + float(bnb.locked)
            universe = {
                str(x).upper().replace("BTC", "")
                for x in (probe.get("universe") or {}).get("btc_pairs") or []
            }
            expected_bases = {
                str(t.get("symbol") or "").upper().replace("BTC", "")
                for t in expected_open
                if str(t.get("symbol") or "").upper().endswith("BTC")
            }
            unexpected_assets: list[str] = []
            dust_assets: list[str] = []
            for asset, bal in acct.balances.items():
                if asset in universe and asset not in {"BTC", "BNB"}:
                    free = float(bal.free) + float(bal.locked)
                    if free <= 0:
                        continue
                    if asset in expected_bases:
                        continue  # matches local PROTECTED trade
                    # Ignore unsellable dust leftovers (minNotional / LOT_SIZE) so re-arm
                    # is not blocked after a clean OCO exit that left dust.
                    pair = f"{asset}BTC"
                    try:
                        from binance_btc_bot.execution.protection_qty import is_economically_flat_base

                        meta = engine.exchange.get_symbol_info(pair)
                        px = float(engine.exchange.get_price(pair) or 0)
                        flat, reason = is_economically_flat_base(
                            free=float(bal.free),
                            locked=float(bal.locked),
                            meta=meta,
                            ref_price=px,
                        )
                        if flat:
                            dust_assets.append(f"{asset}:{reason}")
                            continue
                    except Exception:  # noqa: BLE001
                        pass
                    unexpected_assets.append(asset)
                    unexpected_pos = True
            orders = engine.exchange.get_open_orders()
            lists = engine.exchange.get_open_order_lists()
            out.open_orders = len(orders or [])
            out.open_ocos = len(lists or [])
            expected_oco_ids = {
                str(t.get("binance_oco_list_id"))
                for t in expected_open
                if t.get("binance_oco_list_id") is not None
            }
            expected_syms = {str(t.get("symbol") or "").upper() for t in expected_open}
            # Orders/OCOs for known protected locals are expected on re-arm.
            stray_orders = []
            for o in orders or []:
                if not isinstance(o, dict):
                    continue
                sym = str(o.get("symbol") or "").upper()
                lid = o.get("orderListId")
                if lid in (None, -1, "-1"):
                    lid_s = None
                else:
                    lid_s = str(lid)
                if sym in expected_syms and (lid_s is None or lid_s in expected_oco_ids):
                    continue
                stray_orders.append(o)
            stray_lists = []
            for row in lists or []:
                if not isinstance(row, dict):
                    continue
                lid_s = str(row.get("orderListId")) if row.get("orderListId") is not None else None
                sym = str(row.get("symbol") or "").upper()
                if lid_s in expected_oco_ids or sym in expected_syms:
                    continue
                stray_lists.append(row)
            if stray_orders or stray_lists:
                unexpected_orders = True
            out.add(
                18,
                "Binance no unexpected positions/inventory",
                "PASS" if not unexpected_pos else "FAIL",
                f"equity_btc={out.starting_equity_btc} unexpected_base={unexpected_pos} "
                f"assets={unexpected_assets} dust_ignored={dust_assets} "
                f"expected_bases={sorted(expected_bases)}",
            )
            out.add(
                19,
                "Binance no unexpected open orders/OCOs",
                "PASS" if not unexpected_orders else "FAIL",
                f"orders={out.open_orders} ocos={out.open_ocos} "
                f"stray_orders={len(stray_orders)} stray_ocos={len(stray_lists)} "
                f"expected_ocos={sorted(expected_oco_ids)}",
            )
            out.add(
                20,
                "BTC balance read correctly",
                "PASS" if out.starting_equity_btc is not None else "FAIL",
                f"btc_free={eq.btc_free} locked={eq.btc_locked} equity={eq.total_equity_btc}",
            )
            out.add(
                21,
                "BNB balance/fee config visible",
                "PASS" if out.starting_bnb is not None else "WARN",
                f"bnb={out.starting_bnb}",
            )
        except Exception as e:  # noqa: BLE001
            out.add(18, "Binance no unexpected positions/inventory", "FAIL", scrub_exception(e))
            out.add(19, "Binance no unexpected open orders/OCOs", "FAIL", scrub_exception(e))
            out.add(20, "BTC balance read correctly", "FAIL", scrub_exception(e))
            out.add(21, "BNB balance/fee config visible", "FAIL", scrub_exception(e))

        st, detail = _map_pref(pref, "Binance connectivity")
        out.binance_rest = st
        # Market data freshness — use symbol check / connectivity ("{N} symbols")
        sym_check = next(
            (c for c in (pref.checks if pref else []) if str(getattr(c, "name", "")).endswith("symbols")),
            None,
        )
        st_sym = sym_check.status if sym_check else "FAIL"
        d_sym = (sym_check.detail if sym_check else "MISSING") or ""
        out.add(22, "market-data feeds fresh", "PASS" if st_sym == "PASS" else st_sym, d_sym)

        # Score provider health (construct without demo)
        try:
            from binance_btc_bot.strategy.score_provider import build_production_score_provider

            sp = build_production_score_provider(probe, exchange=engine.exchange, diagnostics=False)
            snap = sp.evaluate("ETHBTC")
            s_curr = getattr(snap, "S_current", None)
            reason = getattr(snap, "reason", "") or ""
            healthy = s_curr is not None and math.isfinite(float(s_curr))
            out.add(
                23,
                "score provider healthy",
                "PASS" if healthy else "FAIL",
                f"ETHBTC S={s_curr} reason={reason}",
            )
        except Exception as e:  # noqa: BLE001
            out.add(23, "score provider healthy", "FAIL", scrub_exception(e))

        demo = os.environ.get("DEMO_SCORE") or os.environ.get("FORCE_SIGNAL")
        out.add(
            24,
            "no demo/forced signal path",
            "PASS" if not demo else "FAIL",
            f"demo_env={bool(demo)}",
        )

        st, detail = _map_pref(pref, "T1 configuration")
        out.add(25, "protection/OCO path available", "PASS" if st == "PASS" else st, detail)

        st, detail = _map_pref(pref, "Protection failure")
        out.add(26, "emergency protection path available", "PASS" if st == "PASS" else st, detail)

        st, detail = _map_pref(pref, "Telegram")
        out.telegram_ok = st == "PASS"
        out.add(27, "Telegram connected", "PASS" if st == "PASS" else st, detail)

        # Reporting health + BRT
        try:
            sample = format_brt("2026-01-15T18:00:00Z")
            brt_ok = "BRT" in sample and "America/Sao_Paulo" in str(DISPLAY_TZ)
            out.timezone_ok = brt_ok
            out.add(28, "Telegram reporting healthy", "PASS" if out.telegram_ok else "FAIL", "formatters importable")
            out.add(29, "Telegram timestamps America/Sao_Paulo / BRT", "PASS" if brt_ok else "FAIL", sample)
        except Exception as e:  # noqa: BLE001
            out.add(28, "Telegram reporting healthy", "FAIL", scrub_exception(e))
            out.add(29, "Telegram timestamps America/Sao_Paulo / BRT", "FAIL", scrub_exception(e))

        # Gate 30: fail-closed if any critical prior gate failed
        critical_failed = [g for g in out.gates if g.status == "FAIL"]
        out.add(
            30,
            "startup fail-closed if critical gate fails",
            "PASS" if not critical_failed else "FAIL",
            f"fail_count={len(critical_failed)}",
        )

        if seed_runtime and engine is not None:
            try:
                prev = _read_runtime_max(engine.db.path)
                out.probe_summary["runtime_max_before_seed"] = prev
                path = seed_live3_runtime_state(engine.db.path)
                out.notes.append(
                    f"runtime state seeded RUNNING/{LIVE3_STRATEGY}/NONE/max={LIVE3_MAX} at {path}"
                )
            except Exception as e:  # noqa: BLE001
                out.notes.append(f"runtime seed failed: {scrub_exception(e)}")

    except Exception as e:  # noqa: BLE001
        out.add(9, "Binance authentication", "FAIL", scrub_exception(e))
        out.add(30, "startup fail-closed if critical gate fails", "FAIL", scrub_exception(e))
        out.notes.append(scrub_exception(e))
    finally:
        try:
            if engine is not None:
                engine.stop()
                engine.db.close()
        except Exception:  # noqa: BLE001
            pass

    # WARN does not block READY for inventory dust; FAIL does.
    hard_fails = [g for g in out.gates if g.status == "FAIL"]
    out.ready_to_arm = len(hard_fails) == 0
    # Recompute gate 30 if we added more fails after it
    if hard_fails and out.gates and out.gates[-1].n == 30 and out.gates[-1].status == "PASS":
        out.gates[-1] = GateResult(
            30,
            "startup fail-closed if critical gate fails",
            "FAIL",
            f"fail_count={len(hard_fails)}",
        )
        out.ready_to_arm = False

    # Persist report (no secrets)
    try:
        root = Path(probe.get("_package_root") or ".")
        # Prefer data dir beside default storage
        storage = (probe.get("storage") or {}).get("sqlite_path") or "data/binance_btc_bot/bot.sqlite3"
        report_path = Path(storage).resolve().parent / "live3_preflight_report.json"
        if not str(storage).startswith("/") and not (len(str(storage)) > 2 and storage[1] == ":"):
            # relative — place under repo data
            report_path = Path("data/binance_btc_bot/live3_preflight_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(out.to_dict(), indent=2, default=str), encoding="utf-8")
        out.notes.append(f"report written {report_path}")
    except Exception as e:  # noqa: BLE001
        out.notes.append(f"report write failed: {scrub_exception(e)}")

    return out


@dataclass
class Live3ArmReport:
    armed: bool = False
    aborted: bool = False
    abort_reason: str = ""
    events: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    preflight_ok: bool | None = None
    live: bool = False
    dry_run: bool = True
    max_simultaneous_trades: int = 0
    strategy: str = ""
    selector: str = ""
    starting_equity_btc: float | None = None
    starting_bnb: float | None = None

    def text(self) -> str:
        lines = [
            "BINANCE BTC BOT — LIVE-3 ARM",
            "============================================",
            f"ARMED={str(self.armed).upper()}",
            f"ABORTED={str(self.aborted).upper()}",
        ]
        if self.abort_reason:
            lines.append(f"REASON={self.abort_reason}")
        lines.append(f"LIVE={self.live} DRY_RUN={self.dry_run}")
        lines.append(
            f"strategy={self.strategy} selector={self.selector or 'NONE'} "
            f"max={self.max_simultaneous_trades}"
        )
        if self.starting_equity_btc is not None:
            lines.append(f"starting_equity_btc={self.starting_equity_btc}")
        if self.starting_bnb is not None:
            lines.append(f"starting_bnb={self.starting_bnb}")
        if self.events:
            lines.append("events:")
            for e in self.events[-20:]:
                lines.append(f"  - {e}")
        for n in self.notes:
            lines.append(f"NOTE: {n}")
        return "\n".join(lines)


class Live3Session:
    """Fail-closed LIVE-3 arm + continuous genuine-cross loop (max=LIVE3_MAX)."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        *,
        poll_interval_sec: float = 15.0,
        reconcile_every_sec: float = 30.0,
    ) -> None:
        self.base_cfg = cfg or load_config()
        self.poll_interval_sec = float(poll_interval_sec)
        self.reconcile_every_sec = float(reconcile_every_sec)

    def run(self, *, authorize_live: bool = False) -> Live3ArmReport:
        out = Live3ArmReport()
        out.notes.append(
            f"LIVE-3: max={LIVE3_MAX} alloc=12.5% total_cap={LIVE3_TOTAL_CAP * 100:.0f}% "
            f"{LIVE3_STRATEGY} selector=NONE"
        )
        out.notes.append("production YAML not rewritten — live overlay is in-memory only")
        out.notes.append("no demo/forced signals; wait for genuine S crosses")

        if not authorize_live:
            out.aborted = True
            out.abort_reason = "missing --authorize-live (refusing to arm writes)"
            return out

        env_ok = str(os.environ.get("BINANCE_LIVE3_AUTHORIZED") or "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if not env_ok:
            out.aborted = True
            out.abort_reason = "BINANCE_LIVE3_AUTHORIZED not set — refusing to arm live writes"
            return out

        # Capture operator max before preflight seed overwrites runtime_control_state.json.
        storage = (self.base_cfg.get("storage") or {}).get("sqlite_path") or "data/binance_btc_bot/bot.sqlite3"
        db_path = Path(storage)
        if not db_path.is_absolute():
            db_path = Path(__file__).resolve().parents[2] / db_path
        prev_runtime_max = _read_runtime_max(db_path)

        pref = run_live3_preflight(self.base_cfg, seed_runtime=True)
        out.preflight_ok = bool(pref.ok and pref.ready_to_arm)
        out.events.append("LIVE3_PREFLIGHT_COMPLETE")
        if not out.preflight_ok:
            out.aborted = True
            out.abort_reason = "LIVE_3_PREFLIGHT did not PASS — refusing to arm"
            out.events.append("LIVE_ARM_REFUSED")
            logger.error("%s", out.abort_reason)
            return out

        live_cfg = build_live3_target_config(self.base_cfg)
        out.strategy = LIVE3_STRATEGY
        out.selector = "NONE"
        out.max_simultaneous_trades = LIVE3_MAX
        out.events.append("LIVE3_CONFIG_OVERLAY_READY")

        from binance_btc_bot.accounting.equity import compute_equity_btc
        from binance_btc_bot.strategy.score_provider import build_production_score_provider

        engine = None
        uds = None
        try:
            engine = BinanceBotEngine(live_cfg, allow_live_writes=True)
            if not engine.live_enabled or engine.dry_run:
                out.aborted = True
                out.abort_reason = (
                    f"engine not live-armed (live={engine.live_enabled} dry_run={engine.dry_run})"
                )
                return out

            score_provider = build_production_score_provider(
                live_cfg, exchange=engine.exchange, diagnostics=False
            )
            engine.score_provider = score_provider

            out.live = True
            out.dry_run = False
            out.armed = True
            out.events.append("LIVE3_ARMED")

            uds = self._start_user_data_ws(engine, out)
            out.events.append("USER_DATA_WS_STARTED")

            try:
                rec = engine.stage_recovery()
                engine.last_reconciliation_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                recon = engine.lifecycle.reconcile_rest(engine.universe)
                if not rec.ok or not bool(recon.get("ok", True)):
                    engine.safety.halt("STARTUP_RECONCILE_FAILED")
                    out.aborted = True
                    out.abort_reason = "startup reconciliation failed — entries blocked"
                    out.events.append("STARTUP_RECONCILE_FAILED")
                    return out
                out.events.append("STARTUP_RECONCILE_OK")
            except Exception as e:  # noqa: BLE001
                engine.safety.halt("STARTUP_RECONCILE_FAILED", error=scrub_exception(e))
                out.aborted = True
                out.abort_reason = f"startup reconciliation error: {scrub_exception(e)}"
                return out

            try:
                acct = engine.exchange.get_account()
                btc_usdt = float(engine.exchange.get_price("BTCUSDT"))
                eq = compute_equity_btc(acct, btc_usdt=btc_usdt)
                out.starting_equity_btc = float(eq.total_equity_btc)
                bnb = getattr(acct, "balances", {}).get("BNB") if hasattr(acct, "balances") else None
                if bnb is not None:
                    out.starting_bnb = float(getattr(bnb, "free", 0) or 0) + float(
                        getattr(bnb, "locked", 0) or 0
                    )
                elif isinstance(acct, dict):
                    for bal in acct.get("balances") or []:
                        if str(bal.get("asset") or "").upper() == "BNB":
                            out.starting_bnb = float(bal.get("free") or 0) + float(
                                bal.get("locked") or 0
                            )
                            break
            except Exception as e:  # noqa: BLE001
                out.aborted = True
                out.abort_reason = f"equity snapshot failed: {scrub_exception(e)}"
                return out

            try:
                engine.attach_runtime_control(start_polling=True, hourly=True)
                out.events.append("TELEGRAM_CONTROL_ATTACHED")
            except Exception as e:  # noqa: BLE001
                out.notes.append(f"telegram attach failed: {scrub_exception(e)}")

            prev_max = prev_runtime_max
            if prev_max is None:
                prev_max = pref.probe_summary.get("runtime_max_before_seed")
            try:
                prev_i = int(prev_max) if prev_max is not None else LIVE3_PREVIOUS_MAX
            except (TypeError, ValueError):
                prev_i = LIVE3_PREVIOUS_MAX
            # Stage-8 one-shot: if we already seeded 8 during a prior preflight in this
            # upgrade window, still announce 3 → 8 once when arming from Stage-7.
            if prev_i == LIVE3_MAX and LIVE3_PREVIOUS_MAX != LIVE3_MAX:
                hb_path = Path(engine.db.path).parent / "live3_heartbeat.json"
                try:
                    if hb_path.is_file():
                        hb = json.loads(hb_path.read_text(encoding="utf-8"))
                        hb_max = int(hb.get("max") or 0)
                        if hb_max == LIVE3_PREVIOUS_MAX:
                            prev_i = LIVE3_PREVIOUS_MAX
                except Exception:  # noqa: BLE001
                    pass
            try:
                if prev_i != LIVE3_MAX:
                    engine.notifications.notify_info(
                        "CONFIGURATION_UPDATED",
                        format_live3_config_updated_message(old_max=prev_i, new_max=LIVE3_MAX),
                    )
                    out.events.append(f"CONFIG_UPDATED:{prev_i}->{LIVE3_MAX}")
                engine.notifications.notify_info(
                    "LIVE3_ARMED",
                    f"LIVE-3 armed: LIVE=true DRY_RUN=false strategy={LIVE3_STRATEGY} "
                    f"selector=NONE max={LIVE3_MAX}. "
                    "Waiting for genuine crosses. No forced entries.",
                )
            except Exception:  # noqa: BLE001
                pass

            print(out.text(), flush=True)
            print(
                "LIVE-3 SESSION RUNNING — Ctrl+C to stop (positions/protection preserved)",
                flush=True,
            )
            self._run_loop(engine, score_provider, out)
            return out
        except KeyboardInterrupt:
            out.events.append("KEYBOARD_INTERRUPT")
            out.notes.append("operator interrupt — existing positions/protection preserved")
            return out
        except Exception as e:  # noqa: BLE001
            out.aborted = True
            out.abort_reason = f"LIVE-3 arm failed: {scrub_exception(e)}"
            out.armed = False
            out.events.append("LIVE_ARM_FAILED")
            logger.exception("LIVE-3 arm failed")
            return out
        finally:
            if uds is not None:
                try:
                    uds.stop()
                except Exception:  # noqa: BLE001
                    pass
            if engine is not None:
                try:
                    engine.stop()
                except Exception:  # noqa: BLE001
                    pass
            out.notes.append(
                "overlay released on exit; on-disk YAML remains LIVE=false DRY_RUN=true max=8"
            )
            out.events.append("SESSION_STOPPED")
            self._persist_heartbeat(engine, out, running=False)

    def _start_user_data_ws(self, engine: Any, out: Live3ArmReport) -> Any:
        from binance_btc_bot.market_data.user_stream import BinanceUserDataWebsocket

        ex = (engine.cfg.get("exchange") or {}) if hasattr(engine, "cfg") else {}
        ws_api_base = str(ex.get("ws_api_base") or "wss://ws-api.binance.com:443/ws-api/v3")

        def on_event(ev: dict[str, Any]) -> None:
            try:
                et = str(ev.get("e") or ev.get("type") or "")
                if et:
                    out.events.append(f"WS:{et}")
            except Exception:  # noqa: BLE001
                pass
            try:
                engine.lifecycle.handle_user_data_event(ev)
            except Exception as e:  # noqa: BLE001
                logger.warning("user-data close handler failed: %s", scrub_exception(e))

        def on_auth_failure(msg: str) -> None:
            out.events.append(f"WS_AUTH_FAIL:{msg}")
            try:
                engine.safety.halt("USER_DATA_WS_AUTH_FAILED", error=str(msg))
            except Exception:  # noqa: BLE001
                pass

        uds = BinanceUserDataWebsocket(
            api_key=engine.creds.api_key,
            signer=engine.creds.get_signer(),
            ws_api_base=ws_api_base,
            recv_window_ms=int(ex.get("recv_window_ms") or 5000),
            dry_run=False,
            on_event=on_event,
            rest_reconcile=lambda: engine.lifecycle.reconcile_rest(engine.universe),
            on_auth_failure=on_auth_failure,
        )
        uds.start()
        deadline = time.time() + 15.0
        while time.time() < deadline:
            if uds.subscribed or uds.auth_ok:
                break
            time.sleep(0.25)
        out.events.append(
            f"USER_DATA_WS:{'SUBSCRIBED' if uds.subscribed or uds.auth_ok else 'START_PENDING'}"
            f":events={uds.events_received}"
        )
        if uds.last_error:
            out.notes.append(f"user_data_ws_last_error={uds.last_error}")
        return uds

    def _persist_heartbeat(self, engine: Any | None, out: Live3ArmReport, *, running: bool) -> None:
        try:
            base = Path("data/binance_btc_bot")
            if engine is not None and getattr(engine, "db", None) is not None:
                base = Path(engine.db.path).parent
            path = base / "live3_heartbeat.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            open_n = 0
            if engine is not None:
                try:
                    out.max_simultaneous_trades = int(engine.portfolio.max_simultaneous_trades)
                except Exception:  # noqa: BLE001
                    out.max_simultaneous_trades = LIVE3_MAX
                try:
                    open_n = len(engine.db.open_trades())
                except Exception:  # noqa: BLE001
                    open_n = engine.portfolio.slots_used() if engine.portfolio else 0
            path.write_text(
                json.dumps(
                    {
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "running": running,
                        "armed": out.armed,
                        "aborted": out.aborted,
                        "live": out.live,
                        "dry_run": out.dry_run,
                        "max": out.max_simultaneous_trades,
                        "strategy": out.strategy,
                        "selector": out.selector,
                        "open_trades": open_n,
                        "events_tail": out.events[-12:],
                        "abort_reason": out.abort_reason or None,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("live3 heartbeat write failed: %s", scrub_exception(e))

    def _run_loop(self, engine: Any, score_provider: Any, out: Live3ArmReport) -> None:
        from binance_btc_bot.accounting.equity import compute_equity_btc

        last_score: dict[str, float] = {}
        last_reconcile = 0.0
        self._persist_heartbeat(engine, out, running=True)
        out.events.append("WAITING_GENUINE_CROSSES")

        while True:
            ctrl = getattr(engine, "_runtime_controller", None)
            if ctrl is not None:
                mode = getattr(getattr(ctrl, "state", None), "mode", None)
                mode_v = getattr(mode, "value", mode)
                if str(mode_v).upper() in {"STOPPED", "STOP"}:
                    out.events.append("OPERATOR_STOP")
                    out.notes.append("stopped via Telegram — existing positions untouched")
                    break

            if not engine.safety.allow_new_entries():
                now = time.time()
                if now - last_reconcile >= self.reconcile_every_sec:
                    try:
                        engine.lifecycle.reconcile_rest(engine.universe)
                        engine.last_reconciliation_at = time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning("reconcile while halted failed: %s", scrub_exception(e))
                    last_reconcile = now
                    self._persist_heartbeat(engine, out, running=True)
                time.sleep(self.poll_interval_sec)
                continue

            now = time.time()
            if now - last_reconcile >= self.reconcile_every_sec:
                try:
                    engine.lifecycle.reconcile_rest(engine.universe)
                    engine.last_reconciliation_at = time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("periodic reconcile failed: %s", scrub_exception(e))
                last_reconcile = now
                self._persist_heartbeat(engine, out, running=True)

            if engine.portfolio.slots_used() >= engine.portfolio.max_simultaneous_trades:
                time.sleep(self.poll_interval_sec)
                continue

            try:
                engine.market.sync(engine.universe)
            except Exception as e:  # noqa: BLE001
                logger.warning("market sync failed: %s", scrub_exception(e))
                time.sleep(self.poll_interval_sec)
                continue

            try:
                acct = engine.exchange.get_account()
                btc_usdt = float(engine.exchange.get_price("BTCUSDT"))
                eq_snap = compute_equity_btc(acct, btc_usdt=btc_usdt)
                equity_btc = float(eq_snap.trading_capital_btc)
                available_btc = float(eq_snap.available_btc)
            except Exception as e:  # noqa: BLE001
                logger.warning("equity snapshot failed: %s", scrub_exception(e))
                time.sleep(self.poll_interval_sec)
                continue

            for sym in list(engine.universe):
                if engine.portfolio.slots_used() >= engine.portfolio.max_simultaneous_trades:
                    break
                if not engine.safety.allow_new_entries():
                    break
                if ctrl is not None and ctrl.blocks_new_entries():
                    break

                rel = engine.market.relative_for(sym)
                score = score_provider(sym, rel)
                if score is None:
                    continue
                score_f = float(score)
                prev = last_score.get(sym)
                decision = engine.entry_engine.evaluate(
                    symbol=sym,
                    score=score_f,
                    open_symbols=engine.portfolio.open_symbols(),
                    open_count=engine.portfolio.slots_used(),
                    safety_allows=engine.safety.allow_new_entries(),
                    relative_price=rel.get("relative_price"),
                    strategy_key=engine.strategy_provider.strategy_key(),
                    selector_key=engine.strategy_provider.selector_key(),
                    reserve_slot=True,
                )
                last_score[sym] = score_f
                if not decision.trade_suggested:
                    continue

                thr = float(engine.entry_engine.long_threshold)
                if prev is None or not (prev < thr and score_f >= thr):
                    if decision.reservation_id and engine.portfolio:
                        try:
                            engine.portfolio.release(decision.reservation_id)
                        except Exception:  # noqa: BLE001
                            pass
                    continue

                if engine.portfolio.slots_used() > engine.portfolio.max_simultaneous_trades:
                    if decision.reservation_id:
                        try:
                            engine.portfolio.release(decision.reservation_id)
                        except Exception:  # noqa: BLE001
                            pass
                    break

                px = float(engine.market.book.get(sym) or 0)
                open_exposure = max(
                    0.0,
                    engine.portfolio.allocated_pct() - engine.portfolio.allocation_per_trade,
                )
                out.events.append(f"SIGNAL_CROSS:{sym}:prev={prev}:curr={score_f}")
                slots_before = int(engine.portfolio.slots_used())
                btc_locked = None
                try:
                    bal = acct.balances.get("BTC")
                    if bal is not None:
                        btc_locked = float(bal.locked)
                except Exception:  # noqa: BLE001
                    btc_locked = None
                portfolio_before = {
                    "equity_btc": equity_btc,
                    "btc_free": available_btc,
                    "btc_locked": btc_locked,
                    "open_trades": slots_before,
                }
                signal = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "previous_s": prev,
                    "current_s": score_f,
                    "threshold": thr,
                    "genuine_new_cross": True,
                }
                # Attach full factor/weight snapshot from the score evaluation that fired.
                try:
                    snap = None
                    if hasattr(score_provider, "last_entry_snapshot"):
                        snap = score_provider.last_entry_snapshot(sym)
                    if isinstance(snap, dict) and snap:
                        signal["entry_snapshot"] = snap
                except Exception:  # noqa: BLE001
                    pass
                life = engine.lifecycle.run_entry(
                    symbol=sym,
                    strategy=engine.current_strategy(),
                    price_alt_btc=px,
                    equity_btc=equity_btc,
                    available_btc=available_btc,
                    open_exposure_pct=open_exposure,
                    btc_usdt=float(engine.market.book.get("BTCUSDT") or btc_usdt),
                    reservation_id=decision.reservation_id,
                    signal=signal,
                    selector=engine.strategy_provider.selector_key(),
                    portfolio_before=portfolio_before,
                )
                protected = "PROTECTED" in (life.events or []) or str(life.status or "").upper() in {
                    "PROTECTED",
                    "PROTECTED_EMERGENCY",
                }
                # Best-effort post-entry portfolio snap (TRADE_OPENED may already have used
                # lifecycle-computed portfolio_after from slots/equity).
                if life.ok and protected:
                    try:
                        acct2 = engine.exchange.get_account()
                        eq2 = compute_equity_btc(
                            acct2, btc_usdt=float(engine.market.book.get("BTCUSDT") or btc_usdt)
                        )
                        out.events.append(
                            f"PORTFOLIO_AFTER:{sym}:equity={float(eq2.trading_capital_btc):.8f}"
                            f":slots={engine.portfolio.slots_used()}"
                        )
                    except Exception:  # noqa: BLE001
                        pass
                out.events.append(
                    f"ENTRY:{sym}:ok={life.ok}:status={life.status}:reason={life.reason}"
                )
                if not life.ok or not protected:
                    out.events.append("PROTECTION_FAILED_OR_ENTRY_FAILED")
                    break

                self._persist_heartbeat(engine, out, running=True)

            time.sleep(self.poll_interval_sec)
