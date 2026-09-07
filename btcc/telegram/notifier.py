"""Telegram notifier — ranking + exhaustion. Labels baseline model probabilities."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

EMOJI = {
    "NORMAL": "🟢",
    "EXTENDED": "🟡",
    "HIGH_LATE_ENTRY_RISK": "🟠",
    "VERY_HIGH_LATE_ENTRY_RISK": "🔴⭐",
}
MEDALS = ["🥇", "🥈", "🥉", "4.", "5."]
SEP = "──────────────────"
COIN_SEP = "━━━━━━━━━━━━━━━━━━"


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, enabled: bool = True):
        self.token = token
        self.chat_id = chat_id
        self.enabled = enabled and bool(token) and bool(chat_id)
        self._last_exhaustion: dict[str, datetime] = {}

    def send(self, text: str) -> bool:
        if not self.enabled:
            logger.info("Telegram disabled/missing creds — full message:\n%s", text)
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            r = requests.post(url, json={"chat_id": self.chat_id, "text": text}, timeout=30)
            r.raise_for_status()
            logger.info("Telegram message sent OK (%d chars)", len(text))
            return True
        except Exception as e:
            logger.error("Telegram send failed: %s", e)
            return False

    def format_ranking(
        self,
        ranked: list[dict[str, Any]],
        dominance: dict[str, Any],
        top_n: int = 5,
        detail_n: int = 3,
        ts: datetime | None = None,
        data_meta: dict[str, Any] | None = None,
    ) -> str:
        ts = ts or datetime.now(timezone.utc)
        kind = ranked[0].get("probability_kind", "baseline_model_probability") if ranked else "baseline_model_probability"
        lines = [
            "🧠 BTC RELATIVE STRENGTH SIGNAL",
            "",
            f"Time: {ts.strftime('%Y-%m-%d %H:%M')} UTC",
            f"Probability type: {kind}",
            "(NOT a validated hit-rate until calibrated)",
            f"Model: {ranked[0].get('model_version', 'config') if ranked else 'n/a'}",
            "",
            f"BTC Dominance: {_fmt_dom(dominance.get('btc_dominance_pct'))}",
            f"BTC.D source: {dominance.get('source', 'n/a')} [slow macro]",
            f"BTC.D 4h: {_fmt_dom_chg(dominance.get('change_4h_pp'), dominance.get('change_4h_status'))}",
            f"BTC.D 24h: {_fmt_dom_chg(dominance.get('change_24h_pp'), dominance.get('change_24h_status'))}",
            "",
            f"Top {top_n} — ranked by 4h baseline model probability",
            "",
        ]

        for i, row in enumerate(ranked[:top_n]):
            medal = MEDALS[i] if i < len(MEDALS) else f"{i+1}."
            sc = row.get("signal_class", {})
            lines.append(f"{medal} {row['base']}/BTC")
            lines.append(f"Class: {sc.get('label', 'n/a')} {sc.get('emoji', '')}")
            lines.append(f"Signal score: {row.get('signal_score', 0):.3f}")
            lines.append(f"1h   {100*row['p_1h']:.0f}%")
            lines.append(f"4h   {100*row['p_4h']:.0f}% ⭐")
            lines.append(f"8h   {100*row['p_8h']:.0f}%")
            lines.append(f"12h  {100*row['p_12h']:.0f}%")
            lines.append(f"24h  {100*row['p_24h']:.0f}%")
            if i < detail_n:
                f = row["factors"]
                lines.append("")
                lines.append(f"Momentum    {f['momentum']['score']:.2f}")
                lines.append(f"Trend       {f['trend']['score']:.2f}")
                lines.append(f"BTC Regime  {f['btc_regime']['score']:.2f}")
                lines.append(f"Volume      {f['volume']['score']:.2f}")
                lines.append(f"Volatility  {f['volatility']['score']:.2f}")
                lines.append(f"RSI         {f['rsi']['score']:.2f}")
                lines.append(f"Structure   {f['structure']['score']:.2f}")
                le = row["late_entry"]
                em = EMOJI.get(le["classification"], "")
                lines.append("")
                lines.append(f"Late Entry: {le['late_entry_score']:.2f} {em} ({le['classification']})")
                lines.append("Late Entry top reasons:")
                for reason in le.get("top_reasons", [])[:3]:
                    lines.append(
                        f"  - {reason['component']}: {reason['value']:.2f} "
                        f"(w={reason['weight']:.0%}, contrib={reason['contribution']:.3f})"
                    )
                if row.get("data_warnings"):
                    lines.append("Data warnings: " + "; ".join(row["data_warnings"]))
            lines.append("")
            lines.append("---")
            lines.append("")

        if data_meta:
            lines.append("Data timestamps:")
            lines.append(f"  Candle close used: {data_meta.get('decision_candle_ts', 'n/a')}")
            lines.append(f"  BTC candle age: {data_meta.get('btc_age', 'n/a')}")
            lines.append(f"  Unavailable symbols: {data_meta.get('unavailable', [])}")
            if data_meta.get("model_version"):
                lines.append(f"  Champion model: {data_meta.get('model_version')}")
        lines += ["", "SIGNAL ONLY", "NO TRADE EXECUTION"]
        return "\n".join(lines)

    def format_indicator_breakdown(
        self,
        ranked: list[dict[str, Any]],
        top_n: int = 5,
    ) -> str:
        """Audit/diagnostic: individual indicator scores for the same Top-N ranking."""
        lines = [
            "📊 INDICATOR BREAKDOWN — TOP 5",
            "",
        ]
        for i, row in enumerate(ranked[:top_n]):
            medal = MEDALS[i] if i < len(MEDALS) else f"{i+1}."
            f = row["factors"]
            mom = f["momentum"]
            trend = f["trend"]
            regime = f["btc_regime"]
            vol = f["volume"]
            volat = f["volatility"]
            rsi = f["rsi"]
            struct = f["structure"]
            le = row["late_entry"]
            le_em = EMOJI.get(le["classification"], "")

            lines.append(f"{medal} {row['base']}/BTC")
            lines.append(COIN_SEP)
            lines.append("")
            lines.append("MOMENTUM")
            lines.append(f"Momentum        {_fmt_indicator(mom.get('score'))}")
            lines.append(SEP)
            lines.append("")
            lines.append("TREND")
            lines.append(f"EMA             {_fmt_indicator(trend.get('ema_score'))}")
            lines.append(f"MACD            {_fmt_indicator(trend.get('macd_score'))}")
            lines.append(f"Ichimoku        {_fmt_indicator(trend.get('ichimoku_score'))}")
            lines.append(f"ADX             {_fmt_indicator(trend.get('adx_score'))}")
            lines.append(SEP)
            lines.append("")
            lines.append("BTC REGIME")
            lines.append(
                f"BTC Regime      {_fmt_indicator(regime.get('regime_indicator_score', regime.get('score')))}"
            )
            lines.append(
                f"BTC Dominance   {_fmt_indicator(regime.get('dominance_indicator_score'))}"
            )
            lines.append(SEP)
            lines.append("")
            lines.append("VOLUME")
            lines.append(f"RVOL            {_fmt_indicator(vol.get('rvol_score'))}")
            lines.append(SEP)
            lines.append("")
            lines.append("VOLATILITY")
            lines.append(f"Bollinger       {_fmt_indicator(volat.get('bollinger_score'))}")
            lines.append(f"ATR             {_fmt_indicator(volat.get('atr_score'))}")
            lines.append(f"NATR            {_fmt_indicator(volat.get('natr_score'))}")
            lines.append(SEP)
            lines.append("")
            lines.append("RSI")
            lines.append(f"RSI             {_fmt_indicator(rsi.get('score'))}")
            lines.append(SEP)
            lines.append("")
            lines.append("STRUCTURE")
            lines.append(f"Price/Market    {_fmt_indicator(struct.get('score'))}")
            lines.append(SEP)
            lines.append("")
            lines.append(f"4h Probability ⭐ {100 * row['p_4h']:.0f}%")
            lines.append(f"Late Entry      {le['late_entry_score']:.2f} {le_em}")
            lines.append(COIN_SEP)
            lines.append("")

        lines += ["SIGNAL ONLY", "NO TRADE EXECUTION"]
        return "\n".join(lines)

    def send_ranking_and_breakdown(
        self,
        ranked: list[dict[str, Any]],
        dominance: dict[str, Any],
        top_n: int = 5,
        detail_n: int = 3,
        ts: datetime | None = None,
        data_meta: dict[str, Any] | None = None,
    ) -> tuple[bool, bool]:
        """Send main Top-N message then indicator breakdown (same Top-N order)."""
        ok1 = self.send(
            self.format_ranking(ranked, dominance, top_n, detail_n, ts, data_meta)
        )
        ok2 = self.send(self.format_indicator_breakdown(ranked, top_n))
        return ok1, ok2

    def format_exhaustion(self, row: dict[str, Any]) -> str:
        le = row["late_entry"]
        mom = row["factors"]["momentum"]
        lines = [
            "🚨 BTC RELATIVE-STRENGTH EXHAUSTION",
            "",
            f"{row['base']}/BTC",
            f"Class: {row.get('signal_class', {}).get('label', 'n/a')}",
            "",
            f"4h baseline model probability: {100*row['p_4h']:.0f}%",
            "",
            "Late Entry Score:",
            f"{le['late_entry_score']:.2f} 🔴⭐ VERY HIGH",
            "",
            "Reasons (by contribution):",
        ]
        for reason in le.get("top_reasons", []):
            lines.append(
                f"  {reason['component']}: {reason['value']:.2f} "
                f"(contrib {reason['contribution']:.3f})"
            )
        lines += [
            "",
            "Current relative momentum:",
            f"1h   {_pct(mom.get('return_1h'))}",
            f"4h   {_pct(mom.get('return_4h'))}",
            f"8h   {_pct(mom.get('return_8h'))}",
            f"12h  {_pct(mom.get('return_12h'))}",
            f"24h  {_pct(mom.get('return_24h'))}",
            "",
            "Strong relative-strength signal,",
            "but the move is highly extended.",
            "",
            "SIGNAL ONLY",
        ]
        return "\n".join(lines)

    def maybe_send_exhaustion(self, ranked, threshold: float, cooldown_minutes: int) -> None:
        now = datetime.now(timezone.utc)
        for row in ranked:
            le = row["late_entry"]["late_entry_score"]
            if le < threshold:
                continue
            sym = row["symbol"]
            last = self._last_exhaustion.get(sym)
            if last and (now - last).total_seconds() < cooldown_minutes * 60:
                continue
            self.send(self.format_exhaustion(row))
            self._last_exhaustion[sym] = now


def _pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{100*x:+.1f}%"


def _fmt_dom(x: float | None) -> str:
    return f"{x:.2f}%" if x is not None else "INSUFFICIENT_DATA"


def _fmt_dom_chg(x: float | None, status: str | None) -> str:
    if status != "OK" or x is None:
        return "INSUFFICIENT_DATA"
    return f"{x:+.2f} pp"


def _fmt_indicator(x: float | None) -> str:
    if x is None:
        return "INSUFFICIENT_DATA"
    return f"{x:.2f}"
