"""Layer 3 — Production preflight / Stage-6 readiness (no real orders).

Verifies everything possible while keeping:
  LIVE=false  DRY_RUN=true  STRATEGY=T1  SELECTOR=NONE  MAX=8  ALLOC=12.5%
"""

from __future__ import annotations

import json
import logging
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from binance_btc_bot.accounting.equity import compute_equity_btc, worked_risk_example
from binance_btc_bot.config_loader import is_live_trading_enabled, load_config
from binance_btc_bot.credentials import load_binance_credentials_from_env
from binance_btc_bot.exchange.binance import BinanceAPIError, BinanceExchange
from binance_btc_bot.exchange.base import AccountSnapshot, Balance
from binance_btc_bot.execution.lifecycle import OrderLifecycle
from binance_btc_bot.execution.dry_broker import DryRunBroker
from binance_btc_bot.execution.trailing import TrailingExecutor
from binance_btc_bot.notifications.manager import NotificationManager
from binance_btc_bot.portfolio.manager import PortfolioManager
from binance_btc_bot.risk.safety import SafetyState, SafetySystem
from binance_btc_bot.secrets import scrub_exception
from binance_btc_bot.storage.database import BotDatabase
from binance_btc_bot.strategy.trails import get_strategy, map_trail_to_binance_oco

logger = logging.getLogger(__name__)


# Official Binance Spot semantics for T1 LONG exit (SELL OCO) — from
# binance-spot-api-docs rest-api.md + faqs/trailing-stop-faq.md (fetched Layer 3).
T1_BINANCE_SEMANTICS = {
    "endpoint": "POST /api/v3/orderList/oco",
    "side": "SELL",
    "aboveType": "TAKE_PROFIT",
    "aboveStopPrice": "entry × (1 + activation) = entry × 1.0075",
    "aboveTrailingDelta": "25 BIPS (= 0.25%)",
    "belowType": "STOP_LOSS",
    "belowStopPrice": "entry × (1 − arm_sl) = entry × 0.9925",
    "quantity": "actual filled qty from MARKET BUY (both OCO legs)",
    "activation": (
        "TAKE_PROFIT SELL with stopPrice: trailing tracking starts only after "
        "market price >= aboveStopPrice (activation threshold)."
    ),
    "trailing_movement": (
        "After activation, Binance tracks the maximum price; a decrease of "
        "aboveTrailingDelta BIPS from that maximum triggers a MARKET SELL."
    ),
    "hard_stop": (
        "STOP_LOSS SELL triggers when market price <= belowStopPrice "
        "(crash / pre-activation safety)."
    ),
    "research_vs_binance": (
        "Research simulator ratchets the stop after activation and effectively "
        "replaces the initial SL with the trail. Binance OCO keeps BOTH legs "
        "until one fills: after activation the trail is the economic exit while "
        "the hard SL remains a crash safety net. Research config is NOT modified."
    ),
    "verified_against": "https://github.com/binance/binance-spot-api-docs (rest-api.md + trailing-stop-faq.md)",
    "live_order_submit": False,
}


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | FAIL | WARN | SKIP | RESTRICTED | OPTIONAL
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {"PASS", "WARN", "SKIP", "RESTRICTED", "OPTIONAL"}


@dataclass
class PreflightReport:
    checks: list[CheckResult] = field(default_factory=list)
    live_enabled: bool = False
    dry_run: bool = True
    egress_ip: str | None = None
    t1_semantics: dict[str, Any] = field(default_factory=lambda: dict(T1_BINANCE_SEMANTICS))
    equity_formula: str = ""
    risk_example: dict[str, Any] = field(default_factory=dict)
    real_orders: str = "DISABLED"

    def add(self, name: str, status: str, detail: str = "", **data: Any) -> CheckResult:
        c = CheckResult(name=name, status=status, detail=detail, data=data)
        self.checks.append(c)
        return c

    def by_name(self, name: str) -> CheckResult | None:
        for c in self.checks:
            if c.name == name:
                return c
        return None

    @property
    def ok(self) -> bool:
        # RESTRICTED private-API checks do not fail the preflight on a
        # geo-blocked Stage-prep host, but Stage-6 authorization still requires PASS.
        hard = [c for c in self.checks if c.status == "FAIL"]
        return len(hard) == 0

    def stage6_ready(self) -> bool:
        """True only when private Binance checks also PASS (eligible network)."""
        return self.stage6_live_authorize_gate()["ok"]

    def stage6_live_authorize_gate(self) -> dict[str, Any]:
        """Hard gate for first real trade / Stage-6 live arming.

        ALL listed items must PASS (Withdraw must be FALSE/OFF).
        Any miss → DO NOT ENABLE LIVE.
        """
        required_pass = [
            "Binance connectivity",
            "Ed25519 authentication",
            "Account access",
            "canTrade",
            "Spot permission",
            "IP restriction",
            "WebSocket",
            "Telegram",
            "T1 configuration",
            "Portfolio",
            "Risk",
            "Recovery",
            "Reconciliation",
        ]
        failures: list[str] = []
        results: dict[str, str] = {}
        for name in required_pass:
            c = self.by_name(name)
            st = c.status if c else "MISSING"
            results[name] = st
            if c is None or c.status != "PASS":
                failures.append(f"{name}={st}")
        # Universe size gate is named "{N} symbols" (dynamic).
        sym_check = next(
            (c for c in self.checks if str(c.name).endswith("symbols")),
            None,
        )
        if sym_check is None:
            failures.append("symbols=MISSING")
            results["symbols"] = "MISSING"
        else:
            results[sym_check.name] = sym_check.status
            if sym_check.status != "PASS":
                failures.append(f"{sym_check.name}={sym_check.status}")
        wd = self.by_name("Withdraw permission")
        wd_st = wd.status if wd else "MISSING"
        results["Withdraw permission"] = wd_st
        # Authoritative: API-key enableWithdrawals must be FALSE (from apiRestrictions).
        # Account /api/v3/account canWithdraw is NOT used as a proxy and is insufficient.
        wd_ok = False
        if wd is not None:
            detail = (wd.detail or "").upper()
            if wd.status == "PASS" and "ENABLEWITHDRAWALS=FALSE" in detail:
                wd_ok = True
            if wd.status == "OFF":
                wd_ok = True
        if not wd_ok:
            failures.append(
                f"Withdraw permission={wd_st} (must be API-key enableWithdrawals=FALSE; "
                f"account.canWithdraw is not authoritative)"
            )

        # SMS is optional — never a Stage-6 trading-safety blocker.
        sms = self.by_name("SMS")
        results["SMS"] = (sms.status if sms else "MISSING") + " (optional)"

        ok = len(failures) == 0
        return {
            "ok": ok,
            "failures": failures,
            "results": results,
            "message": "STAGE6_LIVE_GATE_PASS" if ok else "DO NOT ENABLE LIVE: " + "; ".join(failures),
        }

    def text(self) -> str:
        lines = ["BINANCE BTC BOT — LAYER 3 PREFLIGHT", "=" * 40]
        width = max(len(c.name) for c in self.checks) if self.checks else 10
        for c in self.checks:
            lines.append(f"{c.name:<{width}}  {c.status}" + (f"  ({c.detail})" if c.detail else ""))
        lines.append("")
        lines.append(f"{'REAL ORDERS':<{width}}  {self.real_orders}")
        lines.append(f"{'LIVE':<{width}}  {str(self.live_enabled).upper()}")
        lines.append(f"{'DRY_RUN':<{width}}  {str(self.dry_run).upper()}")
        lines.append(f"{'EGRESS_IP':<{width}}  {self.egress_ip or 'UNKNOWN'}")
        lines.append(f"{'STAGE6_READY':<{width}}  {str(self.stage6_ready()).upper()}")
        lines.append("")
        lines.append("T1 Binance semantics (documented; no live submit):")
        lines.append(json.dumps(self.t1_semantics, indent=2))
        if self.risk_example:
            lines.append("")
            lines.append("Risk worked example:")
            lines.append(json.dumps(self.risk_example, indent=2))
        if self.equity_formula:
            lines.append("")
            lines.append("Equity formula:")
            lines.append(self.equity_formula)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stage6_ready": self.stage6_ready(),
            "live_enabled": self.live_enabled,
            "dry_run": self.dry_run,
            "real_orders": self.real_orders,
            "egress_ip": self.egress_ip,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail, "data": c.data} for c in self.checks
            ],
            "t1_semantics": self.t1_semantics,
            "risk_example": self.risk_example,
            "equity_formula": self.equity_formula,
        }


def detect_egress_ip(timeout: float = 5.0) -> str | None:
    for url in ("https://api.ipify.org", "https://ifconfig.me/ip"):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                ip = resp.read().decode("utf-8", errors="replace").strip()
                if ip and all(c.isdigit() or c == "." for c in ip):
                    return ip
        except Exception:  # noqa: BLE001
            continue
    return None


class PreflightRunner:
    def __init__(self, cfg: dict[str, Any] | None = None, *, engine: Any | None = None) -> None:
        self.cfg = cfg or load_config()
        self.engine = engine

    def run(self) -> PreflightReport:
        report = PreflightReport()
        live = self.cfg.get("live") or {}
        report.live_enabled = bool(live.get("enabled", False))
        report.dry_run = bool(live.get("dry_run", True))
        report.real_orders = "DISABLED"
        report.equity_formula = (
            "total_equity_btc = btc_free + btc_locked + stables_as_btc + mapped_alts_btc\n"
            "available_btc = btc_free\n"
            "trading_capital_btc = total_equity_btc\n"
            "FORBIDDEN: using available USDT alone as equity"
        )
        report.risk_example = worked_risk_example()

        # Absolute: never allow real orders during preflight.
        if report.live_enabled or not report.dry_run:
            report.add("Dry-run", "FAIL", "live.enabled or dry_run misconfigured for preflight")
            return report
        report.add("Dry-run", "PASS", "live.enabled=false dry_run=true")

        engine = self.engine
        if engine is None:
            from binance_btc_bot.execution.engine import BinanceBotEngine

            engine = BinanceBotEngine(self.cfg)
            self.engine = engine

        report.egress_ip = detect_egress_ip()
        self._check_ip(report)
        self._check_connectivity(report, engine)
        self._check_time_sync(report, engine)
        self._check_ed25519(report, engine)
        self._check_account(report, engine)
        self._check_symbols(report, engine)
        self._check_t1_oco(report, engine)
        self._check_portfolio_config(report, engine)
        self._check_risk(report, engine)
        self._check_notifications(report, engine)
        self._check_websocket(report, engine)
        self._check_database(report, engine)
        self._check_recovery_slots(report, engine)
        self._check_protection_failure(report, engine)
        self._check_fill_handling(report)
        self._check_emergency_uncertainty(report, engine)

        return report

    def _check_ip(self, report: PreflightReport) -> None:
        ip = report.egress_ip
        expected = (self.cfg.get("preflight") or {}).get("expected_egress_ips") or []
        if not ip:
            report.add("IP restriction", "WARN", "could not detect egress IP")
            return
        if expected:
            if ip in expected:
                report.add("IP restriction", "PASS", f"egress={ip} allowlisted")
            else:
                report.add(
                    "IP restriction",
                    "FAIL",
                    f"egress={ip} not in configured allowlist — update Binance API key IP restriction",
                    detected=ip,
                    expected=list(expected),
                )
        else:
            report.add(
                "IP restriction",
                "WARN",
                f"egress={ip} — confirm Binance API key allowlist matches this Stage-6 host "
                f"(do not assume 100.27.80.205)",
                detected=ip,
            )

    def _check_connectivity(self, report: PreflightReport, engine: Any) -> None:
        try:
            ok = engine.exchange.ping()
            # Also probe private host without auth to detect geo restriction.
            private_ok, private_detail = self._probe_private_host(engine)
            if ok and private_ok:
                report.add("Binance connectivity", "PASS", "public+private reachable")
            elif ok and not private_ok:
                report.add(
                    "Binance connectivity",
                    "RESTRICTED",
                    f"public OK; private api.binance.com: {private_detail}",
                )
            else:
                report.add("Binance connectivity", "FAIL", "public ping failed")
        except Exception as e:  # noqa: BLE001
            report.add("Binance connectivity", "FAIL", scrub_exception(e))

    def _probe_private_host(self, engine: Any) -> tuple[bool, str]:
        try:
            url = f"{engine.exchange.private_rest_base}/api/v3/ping"
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = resp.read().decode("utf-8")
                if raw.strip() in {"{}", ""}:
                    return True, "OK"
                data = json.loads(raw) if raw else {}
                if data.get("code") == 0 and "restricted" in str(data.get("msg", "")).lower():
                    return False, "HTTP 200 geo-restricted body"
                return True, "OK"
        except urllib.error.HTTPError as e:
            if e.code == 451:
                return False, "HTTP 451 restricted location"
            return False, f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001
            return False, scrub_exception(e)

    def _check_time_sync(self, report: PreflightReport, engine: Any) -> None:
        try:
            data = engine.exchange._request("GET", "/api/v3/time")
            server = int(data["serverTime"])
            skew = abs(int(time.time() * 1000) - server)
            if skew > 3000:
                report.add("Server time sync", "FAIL", f"skew_ms={skew} (>3000)")
            else:
                report.add("Server time sync", "PASS", f"skew_ms={skew}")
        except Exception as e:  # noqa: BLE001
            report.add("Server time sync", "FAIL", scrub_exception(e))

    def _check_ed25519(self, report: PreflightReport, engine: Any) -> None:
        creds = engine.creds
        summary = creds.masked_summary()
        if summary.get("api_key") != "SET" or summary.get("ed25519_private_key") != "SET":
            report.add("Ed25519 authentication", "FAIL", "credentials incomplete", **summary)
            return
        if not creds.signer_ready:
            report.add("Ed25519 authentication", "FAIL", creds._load_error or "signer not ready", **summary)
            return
        # Prove signing works locally without hitting network.
        try:
            sig = creds.get_signer().sign_payload("symbol=ETHBTC&timestamp=1")
            if not sig:
                report.add("Ed25519 authentication", "FAIL", "empty signature")
                return
        except Exception as e:  # noqa: BLE001
            report.add("Ed25519 authentication", "FAIL", scrub_exception(e))
            return
        # Network proof via signed account (may be geo-restricted).
        try:
            engine.exchange.get_account()
            report.add("Ed25519 authentication", "PASS", "signed account accepted", **summary)
        except BinanceAPIError as e:
            if e.status == 451 or "restricted" in str(e).lower():
                report.add(
                    "Ed25519 authentication",
                    "RESTRICTED",
                    "local Ed25519 OK; signed REST blocked by geo/eligibility — re-run on Stage-6 eligible network",
                    **summary,
                )
            elif e.status in {401, 403}:
                report.add("Ed25519 authentication", "FAIL", f"auth rejected HTTP {e.status}", **summary)
            else:
                report.add("Ed25519 authentication", "WARN", scrub_exception(e), **summary)
        except Exception as e:  # noqa: BLE001
            report.add("Ed25519 authentication", "WARN", scrub_exception(e), **summary)

    def _check_account(self, report: PreflightReport, engine: Any) -> None:
        try:
            acct = engine.exchange.get_account()
            raw = acct.raw or {}
            can_trade = bool(raw.get("canTrade", True))
            can_withdraw = bool(raw.get("canWithdraw", False))
            perms = raw.get("permissions") or []
            report.add(
                "Account access",
                "PASS",
                f"balances={len(acct.balances)} "
                f"(note: account.canWithdraw={can_withdraw} is account-capability only; "
                f"not API-key Enable Withdrawals)",
                canTrade=can_trade,
                account_canWithdraw_informational=can_withdraw,
                permissions=perms,
            )
            if can_trade:
                report.add("canTrade", "PASS", "TRUE")
            else:
                report.add("canTrade", "FAIL", "FALSE — Spot trading disabled on account/key")
            if can_trade or "SPOT" in [str(p).upper() for p in perms]:
                report.add("Spot permission", "PASS", f"canTrade={can_trade} permissions={perms}")
            else:
                report.add("Spot permission", "FAIL", "Spot trading not enabled on key/account")

            # Withdraw least-privilege: API-KEY flag enableWithdrawals (NOT account.canWithdraw).
            # /api/v3/account canWithdraw is account capability and is often true even when the
            # API key has Enable Withdrawals disabled in Binance API Management.
            self._check_api_key_withdraw_permission(
                report,
                engine,
                account_can_withdraw=can_withdraw,
            )

            # Equity calculation demo (real balances if present).
            try:
                btc_usdt = float(engine.exchange.get_price("BTCUSDT"))
            except Exception:  # noqa: BLE001
                btc_usdt = 0.0
            eq = compute_equity_btc(acct, btc_usdt=btc_usdt, alt_prices_btc={})
            report.add(
                "Account balances / equity",
                "PASS",
                f"total_equity_btc={eq.total_equity_btc:.8f} available_btc={eq.available_btc:.8f}",
                equity=eq.to_dict(),
            )
        except BinanceAPIError as e:
            if e.status == 451 or "restricted" in str(e).lower():
                report.add("Account access", "RESTRICTED", "private account endpoint geo-blocked")
                report.add("canTrade", "RESTRICTED", "cannot verify until eligible network")
                report.add("Spot permission", "RESTRICTED", "cannot verify until eligible network")
                report.add("Withdraw permission", "RESTRICTED", "cannot verify until eligible network")
                # Synthetic equity demo (never uses USDT alone).
                demo = AccountSnapshot(
                    balances={
                        "BTC": Balance("BTC", 0.8, 0.1),
                        "USDT": Balance("USDT", 5000.0, 0.0),
                        "ETH": Balance("ETH", 2.0, 0.0),
                    }
                )
                eq = compute_equity_btc(demo, btc_usdt=100_000.0, alt_prices_btc={"ETH": 0.05})
                report.add(
                    "Account balances / equity",
                    "PASS",
                    "formula verified with synthetic balances (live account unreachable)",
                    equity=eq.to_dict(),
                )
            else:
                report.add("Account access", "FAIL", scrub_exception(e))
                report.add("canTrade", "FAIL", "account unavailable")
                report.add("Spot permission", "FAIL", "account unavailable")
                report.add("Withdraw permission", "FAIL", "account unavailable")
        except Exception as e:  # noqa: BLE001
            report.add("Account access", "FAIL", scrub_exception(e))
            report.add("canTrade", "FAIL", scrub_exception(e))

    def _check_api_key_withdraw_permission(
        self,
        report: PreflightReport,
        engine: Any,
        *,
        account_can_withdraw: bool,
    ) -> None:
        """Verify API-key Enable Withdrawals is OFF via apiRestrictions.

        Security model: the trading key must not be able to withdraw. Account-level
        ``canWithdraw`` from ``/api/v3/account`` is NOT sufficient / not authoritative.
        """
        try:
            restrictions = engine.exchange.get_api_key_restrictions()
        except BinanceAPIError as e:
            if e.status == 451 or "restricted" in str(e).lower():
                report.add(
                    "Withdraw permission",
                    "RESTRICTED",
                    "apiRestrictions geo-blocked; cannot verify enableWithdrawals",
                )
                return
            report.add(
                "Withdraw permission",
                "FAIL",
                f"apiRestrictions unavailable — fail closed ({scrub_exception(e)})",
            )
            return
        except Exception as e:  # noqa: BLE001
            report.add(
                "Withdraw permission",
                "FAIL",
                f"apiRestrictions error — fail closed ({scrub_exception(e)})",
            )
            return

        if not isinstance(restrictions, dict) or "enableWithdrawals" not in restrictions:
            report.add(
                "Withdraw permission",
                "FAIL",
                "apiRestrictions missing enableWithdrawals — fail closed",
            )
            return

        enable_withdrawals = bool(restrictions.get("enableWithdrawals"))
        key_bits = {
            "enableWithdrawals": enable_withdrawals,
            "enableReading": restrictions.get("enableReading"),
            "enableSpotAndMarginTrading": restrictions.get("enableSpotAndMarginTrading"),
            "enableFutures": restrictions.get("enableFutures"),
            "enableMargin": restrictions.get("enableMargin"),
            "enableInternalTransfer": restrictions.get("enableInternalTransfer"),
            "permitsUniversalTransfer": restrictions.get("permitsUniversalTransfer"),
            "ipRestrict": restrictions.get("ipRestrict"),
            "account_canWithdraw_informational": account_can_withdraw,
        }
        if enable_withdrawals:
            report.add(
                "Withdraw permission",
                "FAIL",
                "enableWithdrawals=TRUE — disable Enable Withdrawals on this API key",
                **key_bits,
            )
            return
        # PASS: key cannot withdraw. Note account.canWithdraw separately (not authoritative).
        report.add(
            "Withdraw permission",
            "PASS",
            "enableWithdrawals=FALSE (API key); account.canWithdraw is account-capability only "
            f"(informational={account_can_withdraw})",
            **key_bits,
        )
    def _check_symbols(self, report: PreflightReport, engine: Any) -> None:
        universe = [str(x).upper() for x in (self.cfg.get("universe") or {}).get("btc_pairs") or []]
        try:
            infos = engine.exchange.get_symbol_infos(universe)
            bad = [s for s, i in infos.items() if not i.is_trading or not i.oco_allowed]
            if bad:
                report.add(
                    f"{len(universe)} symbols",
                    "FAIL",
                    f"non-trading or oco-disabled: {bad[:5]}",
                    n=len(universe),
                    bad=bad,
                )
            else:
                # Spot-check ETHBTC trailing filter admits 25 BIPS.
                eth = infos.get("ETHBTC")
                ok_trail = True
                detail = f"n={len(universe)} all TRADING+ocoAllowed"
                if eth and eth.min_trailing_above_delta is not None:
                    ok_trail = eth.min_trailing_above_delta <= 25 <= (eth.max_trailing_above_delta or 25)
                    detail += f" ETHBTC trailΔ∈[{eth.min_trailing_above_delta},{eth.max_trailing_above_delta}]"
                report.add(
                    f"{len(universe)} symbols",
                    "PASS" if ok_trail else "FAIL",
                    detail,
                    n=len(universe),
                )
            report.add("Symbol metadata / filters", "PASS" if not bad else "FAIL", "exchangeInfo cached")
        except Exception as e:  # noqa: BLE001
            report.add(f"{len(universe)} symbols", "FAIL", scrub_exception(e))
            report.add("Symbol metadata / filters", "FAIL", scrub_exception(e))

    def _check_t1_oco(self, report: PreflightReport, engine: Any) -> None:
        strategy = get_strategy("T1")
        if strategy.activation != 0.0075 or strategy.trail_distance != 0.0025 or strategy.arm_sl_activation_trail != 0.0075:
            report.add("T1 configuration", "FAIL", "T1 parameters altered")
            return
        try:
            meta = engine.exchange.get_symbol_info("ETHBTC")
            entry = 0.05
            qty = 1.0
            mapping = map_trail_to_binance_oco(
                strategy=strategy,
                symbol="ETHBTC",
                entry_price=entry,
                quantity=qty,
                symbol_info=meta,
            )
            req = mapping.request
            if not mapping.allowed or req is None:
                report.add("T1 configuration", "FAIL", ";".join(mapping.constraint_notes))
                return
            checks = {
                "side": req.side == "SELL",
                "aboveType": req.above_type == "TAKE_PROFIT",
                "aboveTrailingDelta": req.above_trailing_delta == 25,
                "belowType": req.below_type == "STOP_LOSS",
                "aboveStopApprox": abs(req.above_stop_price / entry - 1.0075) < 1e-4,
                "belowStopApprox": abs(req.below_stop_price / entry - 0.9925) < 1e-4,
                "quantity": req.quantity == qty,
            }
            # Build the exact params the adapter would send (without submitting).
            params = {
                "symbol": req.symbol,
                "side": req.side,
                "quantity": req.quantity,
                "aboveType": req.above_type,
                "aboveStopPrice": req.above_stop_price,
                "aboveTrailingDelta": req.above_trailing_delta,
                "belowType": req.below_type,
                "belowStopPrice": req.below_stop_price,
            }
            blocked = engine.exchange.place_trailing_exit(req)
            if not blocked.dry_run:
                report.add("T1 OCO construction", "FAIL", "dry-run gate failed — order not blocked")
                return
            if all(checks.values()):
                report.add(
                    "T1 configuration",
                    "PASS",
                    "activation=0.75% trail=0.25% SL=0.75%; OCO params match Spot API",
                    params=params,
                    notes=list(mapping.constraint_notes),
                    semantics=T1_BINANCE_SEMANTICS,
                )
            else:
                report.add("T1 configuration", "FAIL", f"param mismatch {checks}", params=params)
        except Exception as e:  # noqa: BLE001
            report.add("T1 configuration", "FAIL", scrub_exception(e))

    def _check_portfolio_config(self, report: PreflightReport, engine: Any) -> None:
        pm: PortfolioManager = engine.portfolio
        max_n = pm.max_simultaneous_trades
        alloc = float(pm.allocation_per_trade)
        total = float(pm.max_total_allocation)
        if max_n == 8:
            report.add("8-slot configuration", "PASS", f"max_simultaneous_trades={max_n}")
        elif max_n == 3:
            report.add("3-slot configuration", "PASS", f"max_simultaneous_trades={max_n}")
        elif max_n == 1:
            report.add("1-slot configuration", "PASS", f"max_simultaneous_trades={max_n}")
        else:
            report.add(
                "Portfolio slots",
                "WARN",
                f"max={max_n} (expected 1, 3, or 8)",
            )
        if abs(alloc - 0.125) < 1e-12:
            report.add("12.5% allocation", "PASS", f"{max_n} × 12.5% planned")
        else:
            report.add("12.5% allocation", "FAIL", f"alloc={alloc}")

        prod8_ok = max_n == 8 and abs(alloc - 0.125) < 1e-12 and abs(total - 1.0) < 1e-12
        oneshot_ok = max_n == 1 and abs(alloc - 0.125) < 1e-12
        legacy3_ok = max_n == 3 and abs(alloc - 0.125) < 1e-12 and abs(total - 0.375) < 1e-12
        if prod8_ok:
            report.add("Portfolio", "PASS", "max=8 alloc=12.5% total_cap=100%")
        elif oneshot_ok:
            report.add("Portfolio", "PASS", "max=1 alloc=12.5%")
        elif legacy3_ok:
            report.add("Portfolio", "PASS", "max=3 alloc=12.5% total_cap=37.5%")
        else:
            report.add(
                "Portfolio",
                "FAIL",
                f"max={max_n} alloc={alloc} total={total}",
            )

    def _check_risk(self, report: PreflightReport, engine: Any) -> None:
        risk = (self.cfg.get("risk") or {}).get("max_loss_per_trade")
        if abs(float(risk) - 0.005) < 1e-12:
            report.add(
                "0.5% risk ceiling",
                "PASS",
                "max_loss_per_trade=0.005 = loss budget at hard SL, NOT stop distance",
                example=report.risk_example,
            )
            report.add("Risk", "PASS", "0.5% loss ceiling; no leverage")
        else:
            report.add("0.5% risk ceiling", "FAIL", f"got {risk}")
            report.add("Risk", "FAIL", f"got {risk}")

    def _check_notifications(self, report: PreflightReport, engine: Any) -> None:
        n = engine.notifications
        st = n.status()
        tg = st.get("telegram") or {}
        sms = st.get("sms") or {}

        # Telegram is the required ops channel for Stage-6 (no SMS dependency).
        if tg.get("configured"):
            report.add("Telegram", "PASS", "configured")
        else:
            report.add("Telegram", "FAIL", "not configured — required for Stage-6 ops alerts")

        # SMS is optional and never a trading-safety / Stage-6 blocker.
        # Do not send SMS; do not treat mock routing as evidence of readiness.
        if sms.get("configured"):
            report.add("SMS", "OPTIONAL", "CONFIGURED (not required for Stage-6; no SMS sent)")
        else:
            report.add("SMS", "OPTIONAL", "NOT CONFIGURED")

    def _check_websocket(self, report: PreflightReport, engine: Any) -> None:
        """Local user-data stack + public Vision WS + modern WS API subscribe probe."""
        details: list[str] = []
        try:
            import websockets  # noqa: F401
        except ImportError:
            report.add("WebSocket", "FAIL", "websockets package not installed")
            return
        details.append("package_import=OK")

        # --- Local authenticated user-data stack (no orders) ---
        try:
            from binance_btc_bot.market_data.user_stream import (
                DEFAULT_WS_API_BASE,
                SUBSCRIBE_METHOD,
                BinanceUserDataWebsocket,
                build_subscribe_signature_request,
                parse_user_stream_event,
            )

            reconciled: list[dict[str, Any]] = []
            received: list[dict[str, Any]] = []

            def _on_event(ev: dict[str, Any]) -> None:
                received.append(ev)

            uds = BinanceUserDataWebsocket(
                dry_run=True,
                on_event=_on_event,
                rest_reconcile=lambda: reconciled.append({"ok": True}) or {"ok": True},
            )
            uds.start()

            if engine.creds.signer_ready and engine.creds.api_key:
                req = build_subscribe_signature_request(
                    api_key=engine.creds.api_key,
                    signer=engine.creds.get_signer(),
                    timestamp_ms=1_700_000_000_000,
                    recv_window_ms=5000,
                    request_id="preflight-sign-check",
                )
                if req.get("method") != SUBSCRIBE_METHOD or "signature" not in (req.get("params") or {}):
                    report.add("WebSocket", "FAIL", "Ed25519 subscribe.signature request build failed")
                    uds.stop()
                    return
                # Ensure private key material never appears in request id / method logs.
                details.append("subscribe_signature_signing=OK")
            else:
                details.append("subscribe_signature_signing=SKIP(no_signer)")

            wrapped = {
                "subscriptionId": 0,
                "event": {
                    "e": "executionReport",
                    "E": 100,
                    "s": "ETHBTC",
                    "i": 42,
                    "X": "FILLED",
                    "t": 7,
                },
            }
            ev1 = uds.inject_message(wrapped)
            ev2 = uds.inject_message(wrapped)  # dedupe
            if ev1 is None or ev1.get("type") != "executionReport" or len(received) != 1:
                report.add("WebSocket", "FAIL", "event unwrap/parse/dedupe failed", received=len(received))
                uds.stop()
                return
            details.append("event_unwrap=OK")
            details.append("event_parse=OK")
            details.append("dedupe=OK")

            uds.stop()  # disconnect → REST reconcile
            if not reconciled:
                report.add("WebSocket", "FAIL", "REST reconcile not called on disconnect")
                return
            details.append("disconnect_reconcile=OK")

            uds.start()
            uds.bus.reconnect()
            details.append("reconnect=OK")
            uds.stop()

            shape = BinanceUserDataWebsocket(dry_run=True, ws_api_base=DEFAULT_WS_API_BASE)
            if not shape.stream_url or "ws-api" not in shape.stream_url:
                report.add("WebSocket", "FAIL", "ws_api_base URL missing")
                return
            details.append("ws_api_url=OK")
            if "listenKey" in (shape.stream_url or "") or "/ws/" in (shape.stream_url or ""):
                report.add("WebSocket", "FAIL", "legacy listenKey URL still in use")
                return
            parsed = parse_user_stream_event(wrapped)
            if not parsed or parsed["type"] != "executionReport":
                report.add("WebSocket", "FAIL", "parse_user_stream_event failed")
                return
        except Exception as e:  # noqa: BLE001
            report.add("WebSocket", "FAIL", f"local user-data stack: {scrub_exception(e)}")
            return

        # --- Public market WS via Vision (no private endpoint / no orders) ---
        live_public = "UNKNOWN"
        try:
            from binance_btc_bot.market_data.websocket import (
                DEFAULT_PUBLIC_WS_BASE,
                OFFICIAL_WS_BASE,
                BinanceMarketWebsocket,
            )

            ex_cfg = self.cfg.get("exchange") or {}
            public_base = str(ex_cfg.get("public_ws_base") or DEFAULT_PUBLIC_WS_BASE)
            ws = BinanceMarketWebsocket(["btcusdt"], ws_base=public_base)
            ws.start()
            deadline = time.time() + 10.0
            while time.time() < deadline:
                if ws.last_prices or (ws.last_error and "451" in ws.last_error):
                    break
                time.sleep(0.2)
            ws.stop()
            if ws.last_prices:
                live_public = "PASS"
                details.append(f"public_vision_ws=PASS prices={list(ws.last_prices)}")
            elif ws.last_error and ("451" in ws.last_error or "restricted" in ws.last_error.lower()):
                live_public = "RESTRICTED"
                details.append(f"public_vision_ws=RESTRICTED ({ws.last_error[:80]})")
            else:
                live_public = "WARN"
                details.append(f"public_vision_ws=WARN ({ws.last_error or 'timeout'})")

            official = BinanceMarketWebsocket(["btcusdt"], ws_base=OFFICIAL_WS_BASE)
            official.start()
            deadline = time.time() + 6.0
            while time.time() < deadline:
                if official.last_prices or official.last_error:
                    break
                time.sleep(0.2)
            official.stop()
            if official.last_error and ("451" in official.last_error or "rejected" in official.last_error.lower()):
                details.append("official_stream.binance.com=RESTRICTED(451)")
            elif official.last_prices:
                details.append("official_stream.binance.com=PASS")
            else:
                details.append(f"official_stream.binance.com={official.last_error or 'UNKNOWN'}")
        except Exception as e:  # noqa: BLE001
            live_public = "WARN"
            details.append(f"public_ws_probe={scrub_exception(e)}")

        # --- Modern authenticated user-data subscribe (no orders) ---
        auth_ud = "SKIP"
        try:
            from binance_btc_bot.market_data.user_stream import (
                DEFAULT_WS_API_BASE,
                BinanceUserDataWebsocket,
            )

            ex_cfg = self.cfg.get("exchange") or {}
            ws_api = str(ex_cfg.get("ws_api_base") or DEFAULT_WS_API_BASE)
            if engine.creds.signer_ready and engine.creds.api_key:
                probe = BinanceUserDataWebsocket(
                    api_key=engine.creds.api_key,
                    signer=engine.creds.get_signer(),
                    ws_api_base=ws_api,
                    recv_window_ms=int(ex_cfg.get("recv_window_ms") or 5000),
                    dry_run=False,
                )
                result = probe.probe_subscribe(timeout_sec=20.0, unsubscribe=True)
                if result.get("ok") and result.get("subscribed"):
                    auth_ud = "PASS"
                    details.append(
                        f"auth_ws_api_subscribe=PASS method=userDataStream.subscribe.signature "
                        f"subscriptionId={result.get('subscription_id')}"
                    )
                else:
                    auth_ud = "FAIL"
                    details.append(
                        f"auth_ws_api_subscribe=FAIL ({result.get('error') or 'unknown'})"
                    )
            else:
                details.append("auth_ws_api_subscribe=SKIP(no_signer)")
        except Exception as e:  # noqa: BLE001
            auth_ud = "FAIL"
            details.append(f"auth_ws_api_subscribe=FAIL ({scrub_exception(e)})")

        if auth_ud == "FAIL":
            report.add("WebSocket", "FAIL", "; ".join(details))
        elif live_public == "RESTRICTED" and auth_ud != "PASS":
            report.add("WebSocket", "RESTRICTED", "; ".join(details))
        else:
            report.add("WebSocket", "PASS", "; ".join(details))

    def _check_database(self, report: PreflightReport, engine: Any) -> None:
        try:
            engine.db.insert_event("PREFLIGHT", reason="DB_OK")
            report.add("Database", "PASS", f"path={engine.db.path}")
        except Exception as e:  # noqa: BLE001
            report.add("Database", "FAIL", scrub_exception(e))

    def _check_recovery_slots(self, report: PreflightReport, engine: Any) -> None:
        """Simulate restart recovery at the configured max-slot capacity (no real writes)."""
        try:
            pm = PortfolioManager.from_config(self.cfg)
            max_n = int(pm.max_simultaneous_trades)
            trades = [
                {"trade_id": f"t{i}", "symbol": f"S{i}BTC", "status": "DRY_RUN_PROTECTED"}
                for i in range(max_n)
            ]
            rh = pm.rehydrate_from_trades(trades)
            if rh["slots_used"] != max_n or pm.slots_remaining() != 0:
                report.add("Recovery", "FAIL", f"expected {max_n} slots, got {rh}")
                report.add("Reconciliation", "FAIL", f"rehydrate capacity mismatch max={max_n}")
                return
            extra = pm.try_reserve("EXTRABTC")
            if extra.ok:
                report.add("Recovery", "FAIL", f"{max_n}+1 slot allowed after full rehydrate")
                report.add("Reconciliation", "FAIL", "over-capacity after rehydrate")
                return
            # max_n-1 active + 1 free
            if max_n >= 2:
                pm2 = PortfolioManager.from_config(self.cfg)
                rh2 = pm2.rehydrate_from_trades(trades[: max_n - 1])
                if rh2["slots_used"] != max_n - 1:
                    report.add("Recovery", "FAIL", f"expected {max_n - 1}, got {rh2}")
                    report.add("Reconciliation", "FAIL", "partial rehydrate mismatch")
                    return
                ok = pm2.try_reserve("NEWBTC")
                if not ok.ok or pm2.slots_used() != max_n:
                    report.add("Recovery", "FAIL", "failed to take single free slot after restart")
                    report.add("Reconciliation", "FAIL", "free-slot accept failed")
                    return
                seven = rh2
            else:
                seven = rh
            report.add(
                "Recovery",
                "PASS",
                f"rehydrate {max_n}→block next; rehydrate {max_n - 1}→accept 1",
                full=rh,
                partial=seven,
            )
            report.add("Reconciliation", "PASS", "restart rehydrate + slot capacity consistent")
        except Exception as e:  # noqa: BLE001
            report.add("Recovery", "FAIL", scrub_exception(e))
            report.add("Reconciliation", "FAIL", scrub_exception(e))

    def _check_protection_failure(self, report: PreflightReport, engine: Any) -> None:
        """Dry-run BUY filled → OCO rejected → PROTECTION_FAILED path.

        Isolated temp DB + silent notifier — must NEVER use engine.notifications
        (that would spam live Telegram with a synthetic ETHBTC UNPROTECTED_POSITION).
        """
        _ = engine  # intentionally unused; keep signature for PreflightRunner callers
        try:
            import tempfile
            from pathlib import Path
            from unittest.mock import MagicMock

            from binance_btc_bot.exchange.base import OrderResult, SymbolInfo
            from binance_btc_bot.notifications.base import NotificationResult

            class _SilentChannel:
                name = "silent"

                def configured(self) -> bool:
                    return False

                def send(self, event) -> NotificationResult:  # noqa: ANN001
                    return NotificationResult(ok=True, channel=self.name, skipped=True)

                def status(self) -> dict:
                    return {"configured": False, "enabled": False}

            tmp = tempfile.mkdtemp()
            db = BotDatabase(Path(tmp) / "pf.sqlite3")
            safety = SafetySystem()  # no on_notify → no Telegram via safety either
            pm = PortfolioManager.from_config(self.cfg)
            broker = DryRunBroker()
            broker.oco_behavior = "REJECT"
            broker.allow_emergency_protect = False
            silent = NotificationManager(telegram=_SilentChannel(), sms=_SilentChannel())
            ex = MagicMock()
            ex.get_symbol_info.side_effect = lambda s: SymbolInfo(
                symbol=str(s).upper(),
                status="TRADING",
                base_asset="ETH",
                quote_asset="BTC",
                quantity_step=0.0001,
                min_quantity=0.0001,
                max_quantity=1000.0,
                price_tick=0.000001,
                min_notional=0.0001,
                order_types=("LIMIT", "MARKET", "STOP_LOSS", "TAKE_PROFIT"),
                oco_allowed=True,
                min_trailing_above_delta=10,
                max_trailing_above_delta=2000,
                min_trailing_below_delta=10,
                max_trailing_below_delta=2000,
            )
            ex.get_account.return_value = AccountSnapshot(
                balances={
                    "ETH": Balance("ETH", 1.0, 0.0),
                    "BTC": Balance("BTC", 1.0, 0.0),
                    "BNB": Balance("BNB", 0.1, 0.0),
                }
            )
            ex.get_price.return_value = 0.05
            ex.get_open_orders.return_value = []
            ex.get_open_order_lists.return_value = []
            ex.place_trailing_exit.return_value = OrderResult(ok=True, status="DRY_RUN", dry_run=True)
            life = OrderLifecycle(
                ex,
                db,
                safety,
                portfolio=pm,
                trailing=TrailingExecutor(ex, db, notifications=silent),
                notifications=silent,
                dry_broker=broker,
                live_enabled=False,
                dry_run=True,
            )
            slots_before = pm.slots_used()
            r = life.run_entry(
                symbol="ETHBTC",
                strategy=get_strategy("T1"),
                price_alt_btc=0.05,
                equity_btc=1.0,
                available_btc=1.0,
            )
            if r.status != "PROTECTION_FAILED":
                report.add("Protection failure", "FAIL", f"status={r.status} reason={r.reason}")
                return
            if safety.state != SafetyState.HALT or safety.allow_new_entries():
                report.add("Protection failure", "FAIL", "HALT not engaged")
                return
            if pm.slots_used() < slots_before + 1:
                report.add("Protection failure", "FAIL", "slot incorrectly released")
                return
            report.add(
                "Protection failure",
                "PASS",
                "PROTECTION_FAILED→HALT; slot retained; emergency attempted (silent notify)",
                events=r.events,
            )
        except Exception as e:  # noqa: BLE001
            report.add("Protection failure", "FAIL", scrub_exception(e))

    def _check_fill_handling(self, report: PreflightReport) -> None:
        from binance_btc_bot.execution.fills import aggregate_fills_from_order

        agg = aggregate_fills_from_order(
            symbol="ETHBTC",
            side="BUY",
            order_id="1",
            client_order_id="c",
            status="FILLED",
            executed_qty=None,
            cumulative_quote_qty=None,
            fills_raw=[
                {"price": "0.0500", "qty": "1", "commission": "0.00001", "commissionAsset": "BTC"},
                {"price": "0.0502", "qty": "1", "commission": "0.00001", "commissionAsset": "BTC"},
            ],
        )
        if abs(agg.avg_price - 0.0501) < 1e-12 and abs(agg.commission_btc - 0.00002) < 1e-12:
            report.add(
                "Actual fill handling",
                "PASS",
                "weighted avg + commission used (never request price)",
                avg=agg.avg_price,
                qty=agg.executed_qty,
                commission_btc=agg.commission_btc,
            )
        else:
            report.add("Actual fill handling", "FAIL", f"agg={agg}")

    def _check_emergency_uncertainty(self, report: PreflightReport, engine: Any) -> None:
        """API uncertainty → no new risk (never blind retry BUY)."""
        safety = SafetySystem()
        safety.halt("API_FAILURE")
        if safety.allow_new_entries():
            report.add("Emergency / uncertainty", "FAIL", "HALT still allows entries")
            return
        # Idempotent lookup must precede any resubmit (lifecycle contract).
        broker = DryRunBroker()
        o1 = broker.place_market_buy(symbol="ETHBTC", quantity=1.0, client_order_id="idem_1", ref_price=0.05)
        o2 = broker.place_market_buy(symbol="ETHBTC", quantity=1.0, client_order_id="idem_1", ref_price=0.05)
        if o1.order_id != o2.order_id:
            report.add("Emergency / uncertainty", "FAIL", "duplicate BUY on retry")
            return
        report.add(
            "Emergency / uncertainty",
            "PASS",
            "uncertainty→HALT/no new risk; clientOrderId idempotent",
        )


def run_preflight(cfg: dict[str, Any] | None = None, *, engine: Any | None = None) -> PreflightReport:
    return PreflightRunner(cfg, engine=engine).run()
