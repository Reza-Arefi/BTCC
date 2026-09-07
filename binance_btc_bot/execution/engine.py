"""Live/dry-run orchestration — strategy via StrategyProvider; Binance-native trailing."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from binance_btc_bot.config_loader import is_live_trading_enabled, load_config
from binance_btc_bot.credentials import (
    credentials_required_for_mode,
    load_binance_credentials_from_env,
    validate_binance_credentials,
)
from binance_btc_bot.envfile import load_dotenv
from binance_btc_bot.exchange.binance import BinanceExchange
from binance_btc_bot.execution.dry_broker import DryRunBroker
from binance_btc_bot.execution.entry import EntryExecutor
from binance_btc_bot.execution.lifecycle import OrderLifecycle
from binance_btc_bot.execution.recovery import RecoveryManager, RecoveryReport
from binance_btc_bot.execution.trailing import TrailingExecutor
from binance_btc_bot.market_data.rest import RestMarketData
from binance_btc_bot.market_data.symbols import validate_universe, validate_usdt_legs
from binance_btc_bot.market_data.user_stream import UserDataStream
from binance_btc_bot.notifications.manager import NotificationManager
from binance_btc_bot.portfolio.manager import PortfolioManager
from binance_btc_bot.risk.safety import SafetySystem
from binance_btc_bot.secrets import scrub_exception, scrub_text
from binance_btc_bot.storage.database import BotDatabase
from binance_btc_bot.strategy.entries import LiveEntryEngine
from binance_btc_bot.strategy.provider import StrategyProvider, build_strategy_provider

logger = logging.getLogger(__name__)

ScoreProvider = Callable[[str, dict[str, Any]], float | None]


@dataclass
class EngineReport:
    stage: str
    ok: bool
    details: dict[str, Any] = field(default_factory=dict)


class BinanceBotEngine:
    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        *,
        score_provider: ScoreProvider | None = None,
        strategy_provider: StrategyProvider | None = None,
        notifications: NotificationManager | None = None,
        allow_live_writes: bool = False,
    ) -> None:
        load_dotenv()
        self.cfg = cfg or load_config()
        live = self.cfg.get("live") or {}
        risk = self.cfg.get("risk") or {}
        ex = self.cfg.get("exchange") or {}
        storage = self.cfg.get("storage") or {}

        env_dry = os.environ.get("DRY_RUN")
        live_enabled_cfg = bool(live.get("enabled", False))
        oneshot = bool(live.get("first_trade_oneshot", False))
        live3 = bool(live.get("live3", False))
        authorized_live_mode = oneshot or live3
        if allow_live_writes and authorized_live_mode and live_enabled_cfg:
            # Explicit oneshot / LIVE-3 arming overrides ambient DRY_RUN=true from .env.
            dry_run = False
        elif env_dry is not None:
            dry_run = str(env_dry).strip().lower() in {"1", "true", "yes"}
        else:
            dry_run = bool(live.get("dry_run", True))

        writes = live_enabled_cfg and not dry_run
        if writes and not (allow_live_writes and authorized_live_mode):
            raise RuntimeError(
                "Refusing to construct engine with live writes enabled. "
                "Use --first-trade --authorize-live or --live3-arm --authorize-live "
                "only after the matching preflight PASS "
                "(or keep live.enabled=false / dry_run=true)."
            )
        if writes and allow_live_writes and oneshot:
            # Controlled first-trade oneshot — still require env authorization.
            if str(os.environ.get("BINANCE_FIRST_TRADE_AUTHORIZED") or "").strip().lower() not in {
                "1",
                "true",
                "yes",
            }:
                raise RuntimeError(
                    "Refusing live oneshot without BINANCE_FIRST_TRADE_AUTHORIZED=true"
                )
            max_n = int((self.cfg.get("portfolio") or {}).get("max_simultaneous_trades") or 0)
            if max_n != 1:
                raise RuntimeError("first-trade oneshot requires portfolio.max_simultaneous_trades=1")
            if live3:
                raise RuntimeError("first-trade oneshot cannot combine with live3")
            self.dry_run = False
            self.live_enabled = True
        elif writes and allow_live_writes and live3:
            if str(os.environ.get("BINANCE_LIVE3_AUTHORIZED") or "").strip().lower() not in {
                "1",
                "true",
                "yes",
            }:
                raise RuntimeError("Refusing LIVE-3 without BINANCE_LIVE3_AUTHORIZED=true")
            max_n = int((self.cfg.get("portfolio") or {}).get("max_simultaneous_trades") or 0)
            if max_n != 8:
                raise RuntimeError("LIVE-3 requires portfolio.max_simultaneous_trades=8")
            if oneshot:
                raise RuntimeError("LIVE-3 cannot combine with first_trade_oneshot")
            self.dry_run = False
            self.live_enabled = True
        else:
            # Absolute safety for normal construction: never allow real writes.
            self.dry_run = True
            self.live_enabled = False
        self.last_reconciliation_at: str | None = None
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        self.creds = load_binance_credentials_from_env()
        # Validate presence only when signed ops would be required; never print secrets.
        need_creds = credentials_required_for_mode(
            live_enabled=self.live_enabled,
            dry_run=self.dry_run,
            need_account=False,
        )
        validate_binance_credentials(self.creds, required=need_creds)

        self.notifications = notifications or NotificationManager.from_config(self.cfg)
        self.safety = SafetySystem(on_notify=self._safety_notify)

        root = self.cfg.get("_package_root") or "."
        db_path = storage.get("sqlite_path") or "data/binance_btc_bot/bot.sqlite3"
        if not str(db_path).startswith("/"):
            repo = Path(root).resolve().parent
            db_path = str(repo / db_path)
        self.db = BotDatabase(db_path)

        # Preload signer when path is configured (dry-run still OK without it).
        signer = self.creds.get_signer() if self.creds.private_key_path else None
        self.exchange = BinanceExchange(
            api_key=self.creds.api_key or None,
            signer=signer,
            public_rest_base=str(ex.get("public_rest_base") or "https://data-api.binance.vision"),
            private_rest_base=str(ex.get("private_rest_base") or "https://api.binance.com"),
            recv_window_ms=int(ex.get("recv_window_ms") or 5000),
            dry_run=self.dry_run,
            live_enabled=self.live_enabled,
        )
        self.market = RestMarketData(
            self.exchange,
            stale_after_sec=float(risk.get("stale_data_max_age_sec") or 30),
        )

        self.strategy_provider = strategy_provider or build_strategy_provider(self.cfg)
        # Execution obtains strategy only via provider — not hard-coded T1.
        self.strategy = self.strategy_provider.get_strategy()

        self.portfolio = PortfolioManager.from_config(self.cfg)
        entry_cfg = self.cfg.get("entry") or {}
        risk = self.cfg.get("risk") or {}
        self.entry_engine = LiveEntryEngine(
            long_threshold=float(entry_cfg.get("long_threshold") or 0.65),
            upper_threshold=entry_cfg.get("upper_threshold"),
            strategy_key=self.strategy_provider.strategy_key(),
            selector_key=self.strategy_provider.selector_key(),
            max_open=self.portfolio.max_simultaneous_trades,
            one_per_symbol=self.portfolio.config.one_position_per_symbol,
            portfolio=self.portfolio,
        )
        self.entry_exec = EntryExecutor(
            self.exchange,
            self.db,
            self.safety,
            portfolio=self.portfolio,
            max_loss_per_trade=float(risk.get("max_loss_per_trade") or 0.005),
            max_allocation_pct=float(self.portfolio.allocation_per_trade),
            max_aggregate_exposure=float(self.portfolio.max_total_allocation),
            fee_buffer_pct=float(risk.get("fee_buffer_pct") or 0.002),
            notifications=self.notifications,
            live_enabled=self.live_enabled,
            dry_run=self.dry_run,
        )
        self.trail_exec = TrailingExecutor(self.exchange, self.db, notifications=self.notifications)
        self.dry_broker = DryRunBroker()
        self.lifecycle = OrderLifecycle(
            self.exchange,
            self.db,
            self.safety,
            portfolio=self.portfolio,
            trailing=self.trail_exec,
            notifications=self.notifications,
            dry_broker=self.dry_broker,
            live_enabled=self.live_enabled,
            dry_run=self.dry_run,
            max_loss_per_trade=float(risk.get("max_loss_per_trade") or 0.005),
            max_allocation_pct=float(self.portfolio.allocation_per_trade),
            max_aggregate_exposure=float(self.portfolio.max_total_allocation),
            fee_buffer_pct=float(risk.get("fee_buffer_pct") or 0.002),
        )
        self.user_stream = UserDataStream(
            rest_reconcile=lambda: self.lifecycle.reconcile_rest(self.universe),
        )
        self.user_stream.connect()
        self.recovery_mgr = RecoveryManager(
            self.exchange,
            self.db,
            self.safety,
            notifications=self.notifications,
            dry_broker=self.dry_broker,
            portfolio=self.portfolio,
        )
        self.score_provider = score_provider
        self.universe = [str(x).upper() for x in (self.cfg.get("universe") or {}).get("btc_pairs") or []]

        try:
            self.notifications.notify_info(
                "BOT_STARTED",
                f"Binance BTC bot started (dry_run={self.dry_run}, live={self.live_enabled}, "
                f"strategy={self.strategy_provider.strategy_key()})",
            )
        except Exception:  # noqa: BLE001
            logger.warning("BOT_STARTED notification failed (ignored)")

    def _safety_notify(self, event: str, message: str, **kwargs: Any) -> None:
        severity = str(kwargs.pop("severity", "WARNING")).upper()
        try:
            if severity == "CRITICAL":
                self.notifications.notify_critical(event, message, **kwargs)
            elif severity == "ERROR":
                self.notifications.notify_error(event, message, **kwargs)
            elif severity == "WARNING":
                self.notifications.notify_warning(event, message, **kwargs)
            else:
                self.notifications.notify_info(event, message, **kwargs)
        except Exception:  # noqa: BLE001
            logger.warning("safety notify failed (ignored)")

    def current_strategy(self):
        return self.strategy_provider.get_strategy()

    def stage1_connectivity(self) -> EngineReport:
        try:
            ok = self.exchange.ping()
            btc = self.exchange.get_price("BTCUSDT")
            eth = self.exchange.get_symbol_info("ETHBTC")
            return EngineReport(
                "stage1_connectivity",
                ok and btc > 0 and eth.is_trading,
                {
                    "ping": ok,
                    "BTCUSDT": btc,
                    "ETHBTC_status": eth.status,
                    "live_enabled": self.live_enabled,
                    "credentials": self.creds.masked_summary(),
                },
            )
        except Exception as e:  # noqa: BLE001
            self.notifications.notify_error("API_ERROR", scrub_exception(e))
            return EngineReport("stage1_connectivity", False, {"error": scrub_exception(e)})

    def stage2_market_data(self) -> EngineReport:
        try:
            strategy = self.current_strategy()
            book = self.market.sync(self.universe)
            usdt = validate_usdt_legs(self.exchange, self.universe)
            vals = validate_universe(self.exchange, self.universe, strategy=strategy)
            bad = [v for v in vals if not v.ok]
            return EngineReport(
                "stage2_market_data",
                len(bad) == 0 and not usdt["bad"],
                {
                    "n_prices": len(book.prices),
                    "validated_symbols": len(vals),
                    "failed_symbols": [(v.symbol, v.reasons) for v in bad],
                    "usdt_bad": usdt["bad"],
                    "strategy": strategy.key,
                },
            )
        except Exception as e:  # noqa: BLE001
            return EngineReport("stage2_market_data", False, {"error": scrub_exception(e)})

    def stage3_relative_signal_sample(self, symbol: str = "ETHBTC") -> EngineReport:
        strategy = self.current_strategy()
        self.market.sync(self.universe)
        rel = self.market.relative_for(symbol)
        score = None
        if self.score_provider:
            score = self.score_provider(symbol, rel)
        decision = self.entry_engine.evaluate(
            symbol=symbol,
            score=float(score if score is not None else 0.0),
            open_symbols=self.portfolio.open_symbols(),
            open_count=self.portfolio.slots_used(),
            safety_allows=self.safety.allow_new_entries(),
            relative_price=rel["relative_price"],
            strategy_key=self.strategy_provider.strategy_key(),
            selector_key=self.strategy_provider.selector_key(),
            reserve_slot=False,
        )
        self.db.insert_signal(
            symbol=symbol,
            score=decision.score,
            relative_price=rel["relative_price"],
            strategy=strategy.key,
            classification=decision.classification,
            payload={"rel": rel, "decision": decision.__dict__},
        )
        if decision.trade_suggested:
            self.notifications.notify_info(
                "SIGNAL_DETECTED",
                f"Signal on {symbol} score={decision.score}",
                symbol=symbol,
            )
        return EngineReport(
            "stage3_signal",
            True,
            {
                "relative": rel,
                "decision": decision.__dict__,
                "strategy": strategy.key,
                "score_provider": bool(self.score_provider),
            },
        )

    def stage4_risk_sizing(self, symbol: str = "ETHBTC", equity_btc: float = 1.0) -> EngineReport:
        strategy = self.current_strategy()
        self.market.sync([symbol])
        px = self.market.book.get(symbol)
        meta = self.exchange.get_symbol_info(symbol)
        from binance_btc_bot.risk.sizing import size_position

        size = size_position(
            equity_btc=equity_btc,
            available_btc=equity_btc,
            price_alt_btc=float(px or 0),
            strategy=strategy,
            meta=meta,
            max_loss_per_trade=float((self.cfg.get("risk") or {}).get("max_loss_per_trade") or 0.005),
            max_allocation_pct=float(self.portfolio.allocation_per_trade),
            open_exposure_pct=self.portfolio.allocated_pct(),
            max_aggregate_exposure=float(self.portfolio.max_total_allocation),
            fee_buffer_pct=float((self.cfg.get("risk") or {}).get("fee_buffer_pct") or 0.002),
        )
        return EngineReport("stage4_risk", size.ok, {**size.__dict__, "strategy": strategy.key})

    def stage5_order_validation_dry(self, symbol: str = "ETHBTC") -> EngineReport:
        """Full dry entry lifecycle (BUY → fill → OCO verify → PROTECTED) — no real orders."""
        strategy = self.current_strategy()
        self.market.sync(self.universe)
        px = float(self.market.book.get(symbol) or 0)
        btc_usdt = float(self.market.book.get("BTCUSDT") or 0)
        result = self.lifecycle.run_entry(
            symbol=symbol,
            strategy=strategy,
            price_alt_btc=px,
            equity_btc=1.0,
            available_btc=1.0,
            open_exposure_pct=0.0,
            btc_usdt=btc_usdt,
        )
        return EngineReport(
            "stage5_order_validation_dry",
            bool(result.ok),
            {
                "lifecycle": {
                    "ok": result.ok,
                    "reason": result.reason,
                    "status": result.status,
                    "dry_run": result.dry_run,
                    "events": list(result.events),
                    "fill_avg": result.fill.avg_price if result.fill else None,
                    "fill_qty": result.fill.executed_qty if result.fill else None,
                    "size": result.size.__dict__ if result.size else None,
                },
                "trailing": {
                    "ok": result.oco.ok if result.oco else False,
                    "reason": result.oco.reason if result.oco else None,
                    "activation": result.oco.activation_price if result.oco else None,
                    "trail_bips": result.oco.trail_bips if result.oco else None,
                    "initial_stop": result.oco.initial_stop if result.oco else None,
                },
                "live_enabled": is_live_trading_enabled(self.cfg),
                "strategy": strategy.key,
                "selector": self.strategy_provider.selector_key(),
            },
        )

    def stage_recovery(self) -> RecoveryReport:
        report = self.recovery_mgr.recover(self.universe, dry_run=self.dry_run or not self.live_enabled)
        self.last_reconciliation_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        return report

    def run_dry_pipeline(self, demo_scores: dict[str, float] | None = None) -> EngineReport:
        demo_scores = demo_scores or {}
        strategy = self.current_strategy()
        s1 = self.stage1_connectivity()
        s2 = self.stage2_market_data()
        if not (s1.ok and s2.ok):
            return EngineReport("dry_pipeline", False, {"stage1": s1.details, "stage2": s2.details})

        self.market.sync(self.universe)
        signals = []
        for sym in self.universe:
            rel = self.market.relative_for(sym)
            score = demo_scores.get(sym)
            if score is None and self.score_provider:
                score = self.score_provider(sym, rel)
            if score is None:
                continue
            decision = self.entry_engine.evaluate(
                symbol=sym,
                score=float(score),
                open_symbols=self.portfolio.open_symbols(),
                open_count=self.portfolio.slots_used(),
                safety_allows=self.safety.allow_new_entries(),
                relative_price=rel["relative_price"],
                strategy_key=self.strategy_provider.strategy_key(),
                selector_key=self.strategy_provider.selector_key(),
                reserve_slot=True,
            )
            self.db.insert_signal(
                symbol=sym,
                score=score,
                relative_price=rel["relative_price"],
                strategy=strategy.key,
                classification=decision.classification,
                payload={"rel": rel, "reservation_id": decision.reservation_id},
            )
            logger.info(
                "SIGNAL symbol=%s relative_strength=%s strategy=%s classification=%s score=%s",
                sym,
                rel["relative_price"],
                strategy.key,
                decision.classification,
                score,
            )
            if decision.trade_suggested:
                self.notifications.notify_info(
                    "SIGNAL_DETECTED",
                    f"Signal on {sym} score={score} strategy={strategy.key}",
                    symbol=sym,
                )
            signals.append(decision.__dict__)
            if not decision.trade_suggested:
                continue
            px = float(self.market.book.get(sym) or 0)
            # Full Layer 2 lifecycle (simulated fills/OCO). Never real orders.
            life = self.lifecycle.run_entry(
                symbol=sym,
                strategy=strategy,
                price_alt_btc=px,
                equity_btc=1.0,
                available_btc=1.0,
                open_exposure_pct=max(
                    0.0,
                    self.portfolio.allocated_pct() - self.portfolio.allocation_per_trade,
                ),
                btc_usdt=float(self.market.book.get("BTCUSDT") or 0),
                reservation_id=decision.reservation_id,
            )
            if life.ok and life.fill:
                self.user_stream.publish(
                    {
                        "event_id": f"entry:{life.trade_id}",
                        "type": "TRADE_PROTECTED",
                        "trade_id": life.trade_id,
                        "symbol": sym,
                    }
                )

        recovery = self.stage_recovery()
        return EngineReport(
            "dry_pipeline",
            True,
            {
                "signals": len(signals),
                "safety": self.safety.state.value,
                "recovery_notes": recovery.notes,
                "live_enabled": False,
                "dry_run": True,
                "strategy": strategy.key,
                "selector": self.strategy_provider.selector_key(),
            },
        )

    def stop(self) -> None:
        try:
            if getattr(self, "_hourly_report", None):
                self._hourly_report.stop()
            if getattr(self, "_telegram_control", None):
                self._telegram_control.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.notifications.notify_info("BOT_STOPPED", "Binance BTC bot stopped")
        except Exception:  # noqa: BLE001
            pass

    def control_view(self) -> dict[str, Any]:
        """Safe operational snapshot for Telegram status/hourly report (no secrets)."""
        from binance_btc_bot.accounting.equity import compute_equity_btc
        from binance_btc_bot.risk.safety import SafetyState

        open_trades = self.db.open_trades()
        equity_btc = None
        btc_free = None
        btc_locked = None
        stables = None
        alts = None
        bnb_free = None
        try:
            if self.creds.present:
                acct = self.exchange.get_account()
                btc_usdt = float(self.exchange.get_price("BTCUSDT"))
                eq = compute_equity_btc(acct, btc_usdt=btc_usdt)
                equity_btc = eq.total_equity_btc
                btc_free = eq.btc_free
                btc_locked = eq.btc_locked
                stables = float(eq.usdt_as_btc or 0) + float(eq.other_stables_as_btc or 0)
                alts = float(eq.alt_balances_btc_value or 0)
                for bal in acct.get("balances") or []:
                    if str(bal.get("asset") or "").upper() == "BNB":
                        bnb_free = float(bal.get("free") or 0)
                        break
        except Exception:  # noqa: BLE001
            pass

        last_signal = "none"
        try:
            cur = self.db._conn.execute(
                "SELECT symbol, score, classification, created_at FROM signals ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                last_signal = f"{row['symbol']} S={row['score']} {row['classification']}"
        except Exception:  # noqa: BLE001
            pass

        last_entry = "none"
        last_exit = "none"
        realized = 0.0
        fees = 0.0
        try:
            cur = self.db._conn.execute(
                "SELECT symbol, status, entry_time, exit_time, realized_pnl_btc, fees_btc "
                "FROM trades ORDER BY created_at DESC LIMIT 50"
            )
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                if last_entry == "none" and r.get("entry_time"):
                    last_entry = f"{r.get('symbol')} {r.get('entry_time')} {r.get('status')}"
                if last_exit == "none" and r.get("exit_time"):
                    last_exit = f"{r.get('symbol')} {r.get('exit_time')}"
                if r.get("realized_pnl_btc") is not None:
                    realized += float(r["realized_pnl_btc"] or 0)
                if r.get("fees_btc") is not None:
                    fees += float(r["fees_btc"] or 0)
        except Exception:  # noqa: BLE001
            pass

        binance_rest = "UNKNOWN"
        try:
            self.exchange.ping()
            binance_rest = "OK"
        except Exception as e:  # noqa: BLE001
            binance_rest = f"FAIL:{scrub_exception(e)}"

        positions = []
        unrealized_total = 0.0
        unrealized_any = False
        for t in open_trades:
            sym = t.get("symbol")
            entry_px = t.get("entry_price")
            qty = t.get("quantity")
            cur_px = None
            unreal = None
            try:
                if sym:
                    cur_px = float(self.exchange.get_price(str(sym)))
                    if entry_px is not None and qty is not None and cur_px is not None:
                        unreal = (cur_px - float(entry_px)) * float(qty)
                        unrealized_total += unreal
                        unrealized_any = True
            except Exception:  # noqa: BLE001
                pass
            status = str(t.get("status") or "")
            oco_id = t.get("binance_oco_list_id")
            if status in {"PROTECTED", "DRY_RUN_PROTECTED", "PROTECTED_EMERGENCY"}:
                prot = f"ACTIVE({status})"
            elif status in {"PROTECTION_PENDING", "ENTRY_FILLED"}:
                prot = f"PENDING({status})"
            elif status == "PROTECTION_FAILED":
                prot = "FAILED"
            else:
                prot = status or "UNKNOWN"
            if oco_id:
                prot = f"{prot} oco={oco_id}"
            unreal_pct = None
            if unreal is not None and entry_px is not None and float(entry_px) != 0 and qty is not None:
                try:
                    unreal_pct = 100.0 * float(unreal) / (float(entry_px) * float(qty))
                except Exception:  # noqa: BLE001
                    unreal_pct = None
            positions.append(
                {
                    "symbol": sym,
                    "status": status,
                    "strategy": t.get("strategy"),
                    "selector": t.get("selector") or "NONE",
                    "entry_time": t.get("entry_time"),
                    "quantity": qty,
                    "entry_price": entry_px,
                    "current_price": cur_px if cur_px is not None else "n/a",
                    "unrealized_pnl_btc": unreal if unreal is not None else "n/a",
                    "unrealized_pnl_pct": unreal_pct if unreal_pct is not None else "n/a",
                    "protection_state": prot,
                    "binance_oco_list_id": oco_id,
                }
            )

        # Performance from authoritative trade rows (no invented P/L).
        performance: dict[str, Any] = {
            "total_trades": "N/A",
            "open_trades": len(open_trades),
            "wins": "N/A",
            "losses": "N/A",
            "win_rate": "N/A",
            "realized_pnl_btc": realized,
            "unrealized_pnl_btc": unrealized_total if unrealized_any else "n/a",
            "best_trade": "N/A",
            "worst_trade": "N/A",
            "average_trade_btc": "N/A",
            "total_fees_btc": fees,
            "equity_btc": equity_btc,
            "cumulative_return_pct": "N/A",
            "closed_trades": 0,
        }
        hour_closed = hour_wins = hour_losses = 0
        hour_gross = hour_fees = hour_net = 0.0
        hour_events: list[dict[str, Any]] = []
        try:
            cur = self.db._conn.execute(
                "SELECT symbol, status, realized_pnl_btc, fees_btc, entry_time, exit_time, created_at "
                "FROM trades"
            )
            all_trades = [dict(r) for r in cur.fetchall()]
            closed_rows = [r for r in all_trades if str(r.get("status") or "").upper() == "CLOSED"]
            pnls = [float(r["realized_pnl_btc"]) for r in closed_rows if r.get("realized_pnl_btc") is not None]
            wins = sum(1 for p in pnls if p > 0)
            losses = sum(1 for p in pnls if p < 0)
            performance["total_trades"] = len(all_trades)
            performance["closed_trades"] = len(closed_rows)
            performance["wins"] = wins
            performance["losses"] = losses
            if len(pnls) == 0:
                performance["win_rate"] = "N/A"
                performance["average_trade_btc"] = "N/A"
            else:
                performance["win_rate"] = f"{(100.0 * wins / len(pnls)):.1f}%"
                performance["average_trade_btc"] = sum(pnls) / len(pnls)
            if pnls:
                best_i = max(range(len(closed_rows)), key=lambda i: float(closed_rows[i].get("realized_pnl_btc") or 0))
                worst_i = min(range(len(closed_rows)), key=lambda i: float(closed_rows[i].get("realized_pnl_btc") or 0))
                performance["best_trade"] = (
                    f"{closed_rows[best_i].get('symbol')} {float(closed_rows[best_i].get('realized_pnl_btc') or 0):.8f}"
                )
                performance["worst_trade"] = (
                    f"{closed_rows[worst_i].get('symbol')} {float(closed_rows[worst_i].get('realized_pnl_btc') or 0):.8f}"
                )
            cutoff = time.time() - 3600.0
            for r in closed_rows:
                # Prefer exit_time ISO; fall back to created_at epoch.
                et = r.get("exit_time")
                in_hour = False
                if et:
                    try:
                        from binance_btc_bot.notifications.timezone_brt import parse_utc

                        dt = parse_utc(et)
                        if dt and dt.timestamp() >= cutoff:
                            in_hour = True
                    except Exception:  # noqa: BLE001
                        in_hour = False
                if in_hour:
                    hour_closed += 1
                    pnl = float(r.get("realized_pnl_btc") or 0)
                    fee = float(r.get("fees_btc") or 0)
                    hour_net += pnl
                    hour_fees += fee
                    hour_gross += pnl + fee
                    if pnl > 0:
                        hour_wins += 1
                    elif pnl < 0:
                        hour_losses += 1
            ev_cur = self.db._conn.execute(
                "SELECT event, symbol, created_at, reason FROM bot_events WHERE created_at >= ? ORDER BY created_at ASC LIMIT 80",
                (cutoff,),
            )
            for er in ev_cur.fetchall():
                ev = str(er["event"] or "")
                sym = er["symbol"] or ""
                ts = float(er["created_at"] or 0)
                text = ev
                if ev in {"ENTRY_FILLED", "TRADE_OPENED"}:
                    text = f"🟢 {sym} opened"
                elif ev in {"TRADE_PROTECTED", "OCO_ACCEPTED"}:
                    text = f"🛡 {sym} protected"
                elif ev in {"EXIT_FILLED", "TRADE_CLOSED"}:
                    text = f"🔴 {sym} closed"
                elif "RETRY" in ev.upper():
                    text = f"⚠️ {sym} protection retry"
                elif "PROTECTION_FAILED" in ev.upper() or "UNPROTECTED" in ev.upper():
                    text = f"🔴 {sym} protection failure"
                elif "EMERGENCY" in ev.upper():
                    text = f"🚨 {sym} emergency"
                hour_events.append({"timestamp": ts, "text": text})
        except Exception:  # noqa: BLE001
            pass

        prot_open = sum(
            1
            for t in open_trades
            if str(t.get("status") or "") in {"PROTECTED", "DRY_RUN_PROTECTED", "PROTECTED_EMERGENCY"}
        )
        protection_status = (
            f"OK ({prot_open}/{len(open_trades)} protected)"
            if open_trades
            else "OK (flat)"
        )

        return {
            "live_enabled": self.live_enabled,
            "dry_run": self.dry_run,
            "safety_halted": self.safety.state == SafetyState.HALT,
            "safety_state": self.safety.state.value,
            "open_count": len(open_trades),
            "slots_remaining": self.portfolio.slots_remaining(),
            "equity_btc": equity_btc,
            "btc_free": btc_free,
            "btc_locked": btc_locked,
            "bnb_free": bnb_free,
            "stables_as_btc": stables,
            "alts_as_btc": alts,
            "stables_alts_as_btc": (None if stables is None and alts is None else float(stables or 0) + float(alts or 0)),
            "last_signal": last_signal,
            "last_entry": last_entry,
            "last_exit": last_exit,
            "last_trade": last_entry if last_entry != "none" else last_exit,
            "last_reconciliation": self.last_reconciliation_at,
            "critical_errors": list(self.safety.reasons)[-5:] if self.safety.state == SafetyState.HALT else [],
            "positions": positions,
            "binance_rest": binance_rest,
            "user_data_ws": "LOCAL_BUS" if self.dry_run else "CHECK",
            "market_data": "REST",
            "reconciliation": "OK" if self.last_reconciliation_at else "PENDING",
            "protection_status": protection_status,
            "telegram": "ATTACHED" if getattr(self, "_telegram_control", None) else "n/a",
            "execution_state": "DRY_RUN" if self.dry_run else ("LIVE" if self.live_enabled else "DISABLED"),
            "last_heartbeat": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "errors_warnings": ",".join(self.safety.reasons[-5:]) if self.safety.reasons else "none",
            "errors_count": sum(1 for r in self.safety.reasons if "ERROR" in str(r).upper() or "FAIL" in str(r).upper()),
            "warnings_count": sum(1 for r in self.safety.reasons if "WARN" in str(r).upper()),
            "protection_failures": sum(1 for r in self.safety.reasons if "PROTECTION" in str(r).upper()),
            "unknown_order_states": 0,
            "reconciliation_failures": sum(1 for r in self.safety.reasons if "RECONCIL" in str(r).upper()),
            "halt_events": 1 if self.safety.state == SafetyState.HALT else 0,
            "pending_orders": 0,
            "realized_pnl_btc": realized,
            "unrealized_pnl_btc": unrealized_total if unrealized_any else "n/a",
            "fees_btc": fees,
            "performance": performance,
            "hour_closed_trades": hour_closed,
            "hour_wins": hour_wins,
            "hour_losses": hour_losses,
            "hour_gross_pnl_btc": hour_gross,
            "hour_fees_btc": hour_fees,
            "hour_net_realized_pnl_btc": hour_net,
            "hour_events": hour_events,
        }

    def attach_runtime_control(
        self,
        *,
        start_polling: bool = True,
        hourly: bool = True,
        hourly_interval_sec: float = 3600.0,
    ) -> Any:
        """Attach Telegram runtime control + optional hourly report (dry-run safe)."""
        import os

        from binance_btc_bot.control.hourly_report import HourlyReportScheduler
        from binance_btc_bot.control.runtime import (
            RuntimeController,
            RuntimeStateStore,
            default_runtime_state_path,
        )
        from binance_btc_bot.control.telegram_control import TelegramControlPlane
        from binance_btc_bot.notifications.base import NotificationEvent, Severity

        chat_id = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
        store = RuntimeStateStore(default_runtime_state_path(self.db.path))
        ctrl = RuntimeController(
            store,
            strategies_cfg=self.cfg.get("strategies"),
            authorized_chat_id=chat_id,
            on_apply=lambda c: c.apply_to_engine(self),
        )
        ctrl.apply_to_engine(self)
        # Startup reconciliation before new entries (always safe).
        try:
            rec = self.stage_recovery()
            self.last_reconciliation_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if not rec.ok:
                self.safety.halt("STARTUP_RECONCILE_FAILED")
        except Exception as e:  # noqa: BLE001
            self.safety.halt("STARTUP_RECONCILE_FAILED", error=scrub_exception(e))

        def emergency() -> dict[str, Any]:
            self.safety.halt("OPERATOR_EMERGENCY")
            # Cancel eligible pending ENTRY orders only; never market-sell positions.
            cancelled: list[str] = []
            try:
                if not self.dry_run and self.live_enabled:
                    for o in self.exchange.get_open_orders():
                        # Only cancel unprotected entry-looking BUY orders; skip OCO sells.
                        if str(o.get("side") or "").upper() != "BUY":
                            continue
                        if o.get("orderListId") not in (None, -1, "-1"):
                            continue
                        try:
                            self.exchange.cancel_order(str(o.get("symbol")), order_id=str(o.get("orderId")))
                            cancelled.append(str(o.get("orderId")))
                        except Exception:  # noqa: BLE001
                            pass
            except Exception as e:  # noqa: BLE001
                return {"halted": True, "cancelled": cancelled, "error": scrub_exception(e)}
            try:
                recon = self.lifecycle.reconcile_rest(self.universe)
            except Exception as e:  # noqa: BLE001
                recon = {"ok": False, "error": scrub_exception(e)}
            try:
                self.notifications.notify_critical(
                    "OPERATOR_EMERGENCY",
                    "Emergency halt confirmed via Telegram",
                )
            except Exception:  # noqa: BLE001
                pass
            return {"halted": True, "cancelled": cancelled, "reconcile": recon, "protection_preserved": True}

        plane = TelegramControlPlane(
            ctrl,
            engine_view=self.control_view,
            reconcile_fn=lambda: self.lifecycle.reconcile_rest(self.universe),
            emergency_fn=emergency,
        )
        self._runtime_controller = ctrl
        self._telegram_control = plane

        def _send_hourly(text: str) -> None:
            try:
                self.notifications.telegram.send(
                    NotificationEvent(
                        event="HOURLY_REPORT",
                        message=text,
                        severity=Severity.INFO,
                    )
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("hourly telegram send failed: %s", scrub_exception(e))

        if hourly:
            self._hourly_report = HourlyReportScheduler(
                ctrl,
                view_fn=self.control_view,
                send_fn=_send_hourly,
                interval_sec=hourly_interval_sec,
            )
            if start_polling:
                self._hourly_report.start()
        if start_polling and plane.configured():
            plane.start()
        return ctrl

    def status_text(self) -> str:
        """Human-readable operational status for --status."""
        strategy = self.current_strategy()
        nstat = self.notifications.status()
        tg = nstat.get("telegram") or {}
        sms = nstat.get("sms") or {}

        conn = "UNKNOWN"
        try:
            self.exchange.ping()
            conn = "OK"
        except Exception as e:  # noqa: BLE001
            conn = f"FAIL ({scrub_exception(e)})"

        account = "SKIPPED (dry / live disabled)"
        if self.creds.present and not self.dry_run and self.live_enabled:
            try:
                self.exchange.get_account()
                account = "OK"
            except Exception as e:  # noqa: BLE001
                account = f"FAIL ({scrub_exception(e)})"
        elif self.creds.present:
            account = "CREDENTIALS_SET (not queried; live disabled)"
        else:
            account = "NO_CREDENTIALS"

        open_trades = self.db.open_trades()
        exposure = sum(float(t.get("btc_value") or 0) for t in open_trades if t.get("status") == "OPEN")

        last_signal = None
        try:
            cur = self.db._conn.execute(
                "SELECT symbol, score, classification, created_at FROM signals ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                last_signal = dict(row)
        except Exception:  # noqa: BLE001
            last_signal = None

        last_trade = None
        try:
            cur = self.db._conn.execute(
                "SELECT trade_id, symbol, strategy, status, entry_time FROM trades ORDER BY created_at DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                last_trade = dict(row)
        except Exception:  # noqa: BLE001
            last_trade = None

        open_orders_n = 0
        if not self.dry_run and self.live_enabled and self.creds.present:
            try:
                open_orders_n = len(self.exchange.get_open_orders())
            except Exception:  # noqa: BLE001
                open_orders_n = -1

        tg_status = "OK" if tg.get("configured") else ("DISABLED" if not tg.get("enabled", True) else "NOT_CONFIGURED")
        sms_status = "OK" if sms.get("configured") else ("DISABLED" if not sms.get("enabled", True) else "NOT_CONFIGURED")

        safe = (
            not self.live_enabled
            and self.dry_run
            and self.safety.state.value != "HALT"
            and self.strategy_provider.selector_key() is None
        )
        lines = [
            "BINANCE BTC BOT",
            "-------------------------",
            f"Connection:       {conn}",
            f"Account:          {account}",
            f"Live trading:     {'ENABLED' if self.live_enabled else 'DISABLED'}",
            f"Dry run:          {str(self.dry_run).upper()}",
            f"Strategy:         {strategy.key}",
            f"Selector:         {self.strategy_provider.selector_key() or 'NONE'}",
            f"Provider:         {type(self.strategy_provider).__name__}",
            "",
            f"Open orders:      {open_orders_n}",
            f"Open trades:      {len(open_trades)}",
            f"Portfolio slots:  {self.portfolio.slots_used()}/{self.portfolio.max_simultaneous_trades}",
            f"Alloc / trade:    {100.0 * self.portfolio.allocation_per_trade:.2f}%",
            f"BTC exposure:     {exposure:.8f}",
            f"Entry threshold:  S >= {float((self.cfg.get('entry') or {}).get('long_threshold') or 0.65):.2f} (cross-into)",
            "",
            f"Telegram:         {tg_status}",
            f"SMS:              {sms_status}",
            f"Credentials:      key={self.creds.masked_summary()['api_key']} "
            f"ed25519={self.creds.masked_summary()['ed25519_private_key']} "
            f"signer={self.creds.masked_summary()['signer']} "
            f"auth={self.creds.masked_summary()['auth_mode']}",
            "",
            f"Last signal:      {last_signal}",
            f"Last trade:       {last_trade}",
            f"Last reconciliation:",
            f"{self.last_reconciliation_at or 'N/A'}",
            "",
            f"Safety:           {self.safety.state.value}",
            f"STATUS:           {'SAFE' if safe else 'CHECK'}",
        ]
        return "\n".join(lines)
