"""Production T1 continuous signal evaluation for REAL_BOUNDED_6H.

Reuses the live SignalEngine market-data → factors → ranking path (same universe
and relative ALT/BTC construction as production), then applies the frozen trail
entry gate:

  CrossingStateMachine (S ≥ 0.60 cross-into) + evaluate_trail_entry

Explicitly does **not** apply the removed late-entry rejection filter.
Does **not** load Selector E paper engine / research backtests.

Execution symbols are native MEXC ``*BTC`` markets from ``universe.btc_markets``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from btcc.execution.account import AccountSnapshot
from btcc.execution.modes import ExecutionMode
from btcc.execution.price_monitor import MexcPublicTickerSource
from btcc.execution.risk import RiskContext
from btcc.scheduler.cycle import SignalEngine
from btcc.sim.score import combined_score, extract_factor_scores, static_factor_weights
from btcc.sim.state_machine import CrossingStateMachine
from btcc.sim.trail_entry import evaluate_trail_entry
from btcc.universe import resolve_btc_market

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIGNAL_CONFIG = ROOT / "configs" / "signal_config.yaml"


def load_signal_config(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_SIGNAL_CONFIG
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"invalid signal config: {p}")
    raw["_root"] = str(ROOT)
    # Never enable paper Telegram ranking spam on the REAL path.
    tg = dict(raw.get("telegram") or {})
    tg["send_ranking_every_cycle"] = False
    tg["send_exhaustion_alerts"] = False
    tg["enabled"] = False
    raw["telegram"] = tg
    # Do not enable adaptive sim / paper selector via nested sim flags.
    sim = dict(raw.get("sim") or {})
    sim["enabled"] = False
    raw["sim"] = sim
    return raw


def detach_paper_engines(engine: SignalEngine) -> None:
    """Guarantee SignalEngine never drives paper Selector E / Adaptive sim."""
    engine.selector_live_engine = None
    engine.sim_engine = None


@dataclass
class SignalEvalResult:
    ranked_n: int = 0
    evaluated: int = 0
    entries_attempted: int = 0
    entries_ok: int = 0
    rejections: list[str] = field(default_factory=list)
    skipped_duplicate_candle: bool = False
    decision_candle_ts: str | None = None


class ProductionT1SignalLoop:
    """Continuous production entry loop bound to a BoundedSessionRuntime."""

    def __init__(
        self,
        runtime: Any,
        *,
        signal_cfg: dict[str, Any] | None = None,
        signal_config_path: Path | None = None,
        engine: SignalEngine | None = None,
        state_machine: CrossingStateMachine | None = None,
        ticker_source: Any | None = None,
        require_s_min: float = 0.60,
        max_positions: int = 4,
    ) -> None:
        self.runtime = runtime
        self.cfg = signal_cfg if signal_cfg is not None else load_signal_config(signal_config_path)
        self.weights = static_factor_weights(self.cfg)
        self.require_s_min = float(require_s_min)
        self.engine = engine if engine is not None else SignalEngine(self.cfg)
        detach_paper_engines(self.engine)
        self.sm = state_machine or CrossingStateMachine(
            long_threshold=float(self.require_s_min),
            upper_threshold=None,
            max_open=int(max_positions),
            one_per_pair=True,
            threshold_strict=False,
        )
        self.sm.max_open = int(max_positions)
        self.sm.long_threshold = float(self.require_s_min)
        self.ticker = ticker_source or MexcPublicTickerSource(
            base_url=str((self.cfg.get("data") or {}).get("mexc_rest") or "https://api.mexc.com")
        )
        self._bootstrapped = False
        self._pair_by_position: dict[str, str] = {}
        self.last_result: SignalEvalResult | None = None
        self.cycles_run: int = 0
        self.entries_attempted: int = 0

    def bootstrap(self) -> None:
        if self._bootstrapped:
            return
        detach_paper_engines(self.engine)
        self.engine.bootstrap()
        detach_paper_engines(self.engine)
        self._bootstrapped = True
        if hasattr(self.runtime, "audit"):
            self.runtime.audit.record(
                "PRODUCTION_SIGNAL_LOOP_BOOTSTRAP",
                universe_bases=list((self.cfg.get("universe") or {}).get("bases") or []),
                long_threshold=self.sm.long_threshold,
                max_open=self.sm.max_open,
                paper_engines_detached=True,
            )

    def _sync_closed_positions(self) -> None:
        open_ids = set(self.runtime.session.open_position_ids)
        for pid, pair in list(self._pair_by_position.items()):
            if pid not in open_ids:
                try:
                    self.sm.register_close(pid, pair)
                except Exception:  # noqa: BLE001
                    pass
                self._pair_by_position.pop(pid, None)

    def _free_btc_and_account(self) -> tuple[float, RiskContext]:
        broker = self.runtime.broker
        available = 0.0
        locked = 0.0
        account = None
        try:
            bals = broker.get_balances_detailed()
            btc = next((b for b in bals if str(b.asset).upper() == "BTC"), None)
            if btc is not None:
                available = float(btc.free)
                locked = float(btc.locked)
                account = AccountSnapshot(
                    mode=ExecutionMode.REAL.value,
                    available_balance=available,
                    locked_balance=locked,
                    total_balance=float(btc.total),
                    currency="BTC",
                    source="mexc",
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("balance read failed: %s", e)
            if hasattr(self.runtime, "audit"):
                self.runtime.audit.record("SIGNAL_BALANCE_READ_FAILED", error=str(e))
        ctx = RiskContext(
            account=account,
            open_position_count=self.runtime.session.open_count,
            current_exposure_pct=self.runtime.session.exposure_pct,
            market_data_stale=False,
            account_data_stale=account is None,
        )
        return available, ctx

    def _liquidity_ok(self, symbol_meta: Any) -> tuple[bool, str | None]:
        if symbol_meta is None:
            return False, "SYMBOL_META_MISSING"
        if not getattr(symbol_meta, "is_trading", False):
            return False, "SYMBOL_NOT_TRADING"
        if float(getattr(symbol_meta, "quantity_step", 0) or 0) <= 0:
            return False, "INVALID_QUANTITY_STEP"
        if float(getattr(symbol_meta, "min_quantity", 0) or 0) < 0:
            return False, "INVALID_MIN_QUANTITY"
        # T1 bot-managed exits are MARKET-only — no MARKET → no entry.
        from btcc.execution.symbols import t1_market_exit_capability

        ok_mkt, reason = t1_market_exit_capability(symbol_meta)
        if not ok_mkt:
            return False, reason
        return True, None

    def evaluate_once(self, runtime: Any | None = None) -> SignalEvalResult:
        """One continuous evaluation tick (idempotent per closed 15m candle)."""
        rt = runtime or self.runtime
        self.runtime = rt
        self.bootstrap()
        detach_paper_engines(self.engine)
        self._sync_closed_positions()
        self.cycles_run += 1

        result = SignalEvalResult()
        ranked = self.engine.run_cycle()
        meta = getattr(self.engine, "last_data_meta", {}) or {}
        result.decision_candle_ts = str(meta.get("decision_candle_ts") or "") or None
        if meta.get("skipped_duplicate"):
            result.skipped_duplicate_candle = True
            self.last_result = result
            return result

        result.ranked_n = len(ranked or [])
        if not ranked:
            self.last_result = result
            return result

        available, base_ctx = self._free_btc_and_account()
        if available <= 0 or base_ctx.account is None:
            result.rejections.append("NO_AVAILABLE_BTC")
            self.last_result = result
            if hasattr(rt, "audit"):
                rt.audit.record("SIGNAL_SKIP_NO_BTC", available=available)
            return result

        for row in ranked:
            result.evaluated += 1
            base = str(row.get("base") or "")
            if not base:
                continue
            exec_symbol = resolve_btc_market(base, self.cfg)
            pair_key = exec_symbol  # one opportunity per native BTC market

            factor_scores = extract_factor_scores(row.get("factors") or {})
            scored = combined_score(factor_scores, self.weights)
            s_val = float(scored["S"])

            decision = self.sm.evaluate(pair_key, s_val)
            # Production trail entry — NO late-entry rejection (filter removed).
            ep = evaluate_trail_entry(sm_decision=decision, health_allow_new_trades=True)
            if not ep.get("trade_suggested"):
                result.rejections.append(
                    f"{exec_symbol}:{ep.get('rejection_reason') or 'NOT_OPENED'}"
                )
                continue

            # Exchange tradability — always fresh exchangeInfo for MARKET capability.
            try:
                try:
                    symbol_meta = rt.broker.get_symbol_metadata(exec_symbol, use_cache=False)
                except TypeError:
                    symbol_meta = rt.broker.get_symbol_metadata(exec_symbol)
            except Exception as e:  # noqa: BLE001
                result.rejections.append(f"{exec_symbol}:META_UNAVAILABLE:{e}")
                continue
            ok_liq, liq_reason = self._liquidity_ok(symbol_meta)
            if not ok_liq:
                result.rejections.append(f"{exec_symbol}:{liq_reason}")
                continue

            # Execution price from live ticker (native book), not synthetic mid.
            try:
                tick = self.ticker.fetch_price(exec_symbol)
                price = float(tick.price)
            except Exception as e:  # noqa: BLE001
                result.rejections.append(f"{exec_symbol}:TICKER_FAILED:{e}")
                continue
            if price <= 0:
                result.rejections.append(f"{exec_symbol}:INVALID_PRICE")
                continue

            # Refresh free BTC under capacity (locked never sized).
            available, ctx = self._free_btc_and_account()
            ctx.symbol_meta = symbol_meta
            ctx.open_position_count = rt.session.open_count
            ctx.current_exposure_pct = rt.session.exposure_pct
            if available <= 0 or ctx.account is None:
                result.rejections.append(f"{exec_symbol}:NO_AVAILABLE_BTC")
                break

            candle_ts = str(row.get("timestamp") or result.decision_candle_ts or "")
            signal_ts = candle_ts
            result.entries_attempted += 1
            self.entries_attempted += 1
            if hasattr(rt, "audit"):
                rt.audit.record(
                    "SIGNAL_ENTRY_ATTEMPT",
                    symbol=exec_symbol,
                    S=s_val,
                    available_btc=available,
                    locked_btc=float(ctx.account.locked_balance),
                    open_positions=rt.session.open_count,
                )
            entry = rt.try_entry(
                symbol=exec_symbol,
                price_alt_btc=price,
                s_value=s_val,
                signal_ts=signal_ts,
                candle_ts=candle_ts,
                available_btc=available,
                locked_btc=float(ctx.account.locked_balance),
                symbol_meta=symbol_meta,
                account_ctx=ctx,
                intent_nonce=f"sig-{exec_symbol}-{candle_ts}",
            )
            if entry.ok and entry.position_id:
                result.entries_ok += 1
                self.sm.register_open(pair_key, entry.position_id)
                self._pair_by_position[entry.position_id] = pair_key
                if hasattr(rt, "audit"):
                    rt.audit.record(
                        "SIGNAL_ENTRY_FILLED",
                        symbol=exec_symbol,
                        position_id=entry.position_id,
                        S=s_val,
                    )
            else:
                result.rejections.append(f"{exec_symbol}:{entry.reason}")
                if entry.halt:
                    break

        self.last_result = result
        return result

    def as_callback(self) -> Callable[[Any], SignalEvalResult]:
        """Launcher-compatible ``signal_loop(runtime)`` callback."""
        return lambda runtime: self.evaluate_once(runtime)
