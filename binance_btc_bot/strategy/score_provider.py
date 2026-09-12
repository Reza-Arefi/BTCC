"""Production score provider — identical S formula to Binance T1 backtests.

Path (must match selector_backtest / predict_coin_at_bar):

  BASEUSDT + BTCUSDT 15m closed candles
  → build_alt_btc (BASEUSDT / BTCUSDT relative OHLCV)
  → compute_all_factors (btcc.factors.combine)
  → extract_factor_scores + combined_score / static_factor_weights
  → signed S ∈ [-1, +1]

Entry gate (LiveEntryEngine) still requires NEW CROSS:

  previous S < 0.65  AND  current S >= 0.65

This module never logs API credentials. Missing / stale / insufficient /
invalid scores return None so the entry loop cannot open on garbage.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import yaml

from btcc.factors.combine import compute_all_factors
from btcc.late_entry.score import late_entry_score
from btcc.series.relative import build_alt_btc
from btcc.sim.score import (
    ACTIVE_SIGNAL_KEYS,
    ALL_FACTOR_KEYS,
    combined_score,
    extract_factor_scores,
    static_factor_weights,
)

from binance_btc_bot.market_data.klines import (
    closed_candle_age_sec,
    drop_incomplete_candle,
    klines_to_dataframe,
)
from binance_btc_bot.strategy.relative_price import base_from_btc_pair, usdt_pair_for_base

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIGNAL_CONFIG = REPO_ROOT / "configs" / "signal_config.yaml"

MIN_HISTORY_BARS = 100
DEFAULT_LOOKBACK = 1000
DEFAULT_INTERVAL = "15m"
DEFAULT_THRESHOLD = 0.65
# Allow up to two full intervals after close before declaring stale.
DEFAULT_MAX_CLOSED_AGE_SEC = 1800.0

KlineFetcher = Callable[[str, str, int], list]


@dataclass(frozen=True)
class ScoreSnapshot:
    """Diagnostic record for one symbol evaluation (no secrets)."""

    timestamp: str
    symbol: str
    relative_price: float | None
    S_previous: float | None
    S_current: float | None
    cross_detected: bool
    decision_candle_ts: str | None = None
    reason: str | None = None
    interval: str = DEFAULT_INTERVAL
    long_threshold: float = DEFAULT_THRESHOLD
    history_bars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def log_line(self) -> str:
        return (
            f"SCORE_DIAG timestamp={self.timestamp} symbol={self.symbol} "
            f"relative_price={self.relative_price} S_previous={self.S_previous} "
            f"S_current={self.S_current} cross_detected={self.cross_detected} "
            f"decision_candle_ts={self.decision_candle_ts} reason={self.reason}"
        )


@dataclass
class ProductionScoreProvider:
    """Callable score provider compatible with BinanceBotEngine / FirstTradeController."""

    signal_cfg: dict[str, Any]
    kline_fetcher: KlineFetcher
    interval: str = DEFAULT_INTERVAL
    lookback_bars: int = DEFAULT_LOOKBACK
    min_history_bars: int = MIN_HISTORY_BARS
    long_threshold: float = DEFAULT_THRESHOLD
    max_closed_candle_age_sec: float = DEFAULT_MAX_CLOSED_AGE_SEC
    diagnostics: bool = False
    now_fn: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        self.weights = static_factor_weights(self.signal_cfg)
        self._btc_cache: tuple[str | None, pd.DataFrame | None] = (None, None)
        self._last: dict[str, ScoreSnapshot] = {}
        # Full factor dump from the latest successful evaluate() per symbol.
        self._last_entry_snapshot: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ API
    def __call__(self, symbol: str, rel: Mapping[str, Any] | None = None) -> float | None:
        snap = self.evaluate(symbol, rel=rel)
        if snap.S_current is None or not math.isfinite(float(snap.S_current)):
            return None
        return float(snap.S_current)

    def last_entry_snapshot(self, symbol: str) -> dict[str, Any] | None:
        """Return the factor/weight snapshot from the latest successful evaluate()."""
        return self._last_entry_snapshot.get(symbol.upper())

    def evaluate(
        self,
        symbol: str,
        *,
        rel: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> ScoreSnapshot:
        sym = symbol.upper()
        now_utc = now or self.now_fn()
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        ts_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            frames = self._load_frames(sym, now=now_utc)
        except Exception as e:  # noqa: BLE001
            snap = ScoreSnapshot(
                timestamp=ts_iso,
                symbol=sym,
                relative_price=_rel_price(rel),
                S_previous=None,
                S_current=None,
                cross_detected=False,
                reason=f"MARKET_DATA_ERROR:{type(e).__name__}",
                interval=self.interval,
                long_threshold=self.long_threshold,
            )
            return self._finish(snap)

        if frames is None:
            snap = ScoreSnapshot(
                timestamp=ts_iso,
                symbol=sym,
                relative_price=_rel_price(rel),
                S_previous=None,
                S_current=None,
                cross_detected=False,
                reason="MISSING_MARKET_DATA",
                interval=self.interval,
                long_threshold=self.long_threshold,
            )
            return self._finish(snap)

        rel_df, alt_vol, btc_df, age = frames
        if age > float(self.max_closed_candle_age_sec):
            snap = ScoreSnapshot(
                timestamp=ts_iso,
                symbol=sym,
                relative_price=float(rel_df["close"].iloc[-1]) if len(rel_df) else _rel_price(rel),
                S_previous=None,
                S_current=None,
                cross_detected=False,
                decision_candle_ts=str(rel_df["timestamp"].iloc[-1]) if len(rel_df) else None,
                reason=f"STALE_MARKET_DATA:age_sec={age:.1f}",
                interval=self.interval,
                long_threshold=self.long_threshold,
                history_bars=len(rel_df),
            )
            return self._finish(snap)

        if len(rel_df) < int(self.min_history_bars):
            snap = ScoreSnapshot(
                timestamp=ts_iso,
                symbol=sym,
                relative_price=float(rel_df["close"].iloc[-1]) if len(rel_df) else None,
                S_previous=None,
                S_current=None,
                cross_detected=False,
                decision_candle_ts=str(rel_df["timestamp"].iloc[-1]) if len(rel_df) else None,
                reason=f"INSUFFICIENT_HISTORY:bars={len(rel_df)}",
                interval=self.interval,
                long_threshold=self.long_threshold,
                history_bars=len(rel_df),
            )
            return self._finish(snap)

        detail_curr = detail_from_frames(
            rel_df,
            alt_vol,
            btc_df,
            self.signal_cfg,
            interval=self.interval,
            weights=self.weights,
        )
        s_curr = None if detail_curr is None else detail_curr.get("S")
        s_prev = None
        if len(rel_df) >= int(self.min_history_bars) + 1:
            detail_prev = detail_from_frames(
                rel_df.iloc[:-1],
                alt_vol.iloc[:-1],
                btc_df.iloc[:-1],
                self.signal_cfg,
                interval=self.interval,
                weights=self.weights,
            )
            s_prev = None if detail_prev is None else detail_prev.get("S")

        if s_curr is None:
            self._last_entry_snapshot.pop(sym, None)
            snap = ScoreSnapshot(
                timestamp=ts_iso,
                symbol=sym,
                relative_price=float(rel_df["close"].iloc[-1]),
                S_previous=s_prev,
                S_current=None,
                cross_detected=False,
                decision_candle_ts=str(rel_df["timestamp"].iloc[-1]),
                reason="INVALID_SCORE",
                interval=self.interval,
                long_threshold=self.long_threshold,
                history_bars=len(rel_df),
            )
            return self._finish(snap)

        cross = new_cross_into(s_prev, float(s_curr), self.long_threshold)
        # Persist full factor snapshot for this evaluation (used at entry).
        entry_snap = dict(detail_curr or {})
        entry_snap.update(
            {
                "timestamp": ts_iso,
                "symbol": sym,
                "S_previous": s_prev,
                "S_current": float(s_curr),
                "cross_detected": bool(cross),
                "long_threshold": float(self.long_threshold),
                "decision_candle_ts": str(rel_df["timestamp"].iloc[-1]),
                "relative_price": float(rel_df["close"].iloc[-1]),
                "history_bars": len(rel_df),
                "interval": self.interval,
                "momentum_profile": str(
                    (self.signal_cfg.get("factors") or {}).get("momentum_profile") or "base"
                ),
            }
        )
        self._last_entry_snapshot[sym] = json_safe(entry_snap)
        snap = ScoreSnapshot(
            timestamp=ts_iso,
            symbol=sym,
            relative_price=float(rel_df["close"].iloc[-1]),
            S_previous=s_prev,
            S_current=float(s_curr),
            cross_detected=cross,
            decision_candle_ts=str(rel_df["timestamp"].iloc[-1]),
            reason=None,
            interval=self.interval,
            long_threshold=self.long_threshold,
            history_bars=len(rel_df),
        )
        return self._finish(snap)

    def last_snapshot(self, symbol: str) -> ScoreSnapshot | None:
        return self._last.get(symbol.upper())

    # -------------------------------------------------------------- internals
    def _finish(self, snap: ScoreSnapshot) -> ScoreSnapshot:
        self._last[snap.symbol] = snap
        if self.diagnostics:
            logger.info(snap.log_line())
        return snap

    def _load_frames(
        self,
        btc_pair: str,
        *,
        now: datetime,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float] | None:
        base = base_from_btc_pair(btc_pair)
        usdt = usdt_pair_for_base(base)
        limit = int(self.lookback_bars) + 5  # headroom for incomplete tip

        btc_df = self._fetch_closed("BTCUSDT", limit=limit, now=now)
        alt_df = self._fetch_closed(usdt, limit=limit, now=now)
        if btc_df is None or alt_df is None or btc_df.empty or alt_df.empty:
            return None

        # Align on intersection via build_alt_btc (inner join on timestamp).
        rel = build_alt_btc(alt_df, btc_df)
        if rel is None or rel.empty:
            return None

        # Trim BTC / ALT volume frames to the same closed tip as relative.
        tip = pd.to_datetime(rel["timestamp"].iloc[-1], utc=True)
        btc_aligned = btc_df[pd.to_datetime(btc_df["timestamp"], utc=True) <= tip].copy()
        alt_aligned = alt_df[pd.to_datetime(alt_df["timestamp"], utc=True) <= tip].copy()
        if btc_aligned.empty or alt_aligned.empty:
            return None

        age = closed_candle_age_sec(rel, interval=self.interval, now=now)
        # Keep lookback window
        if len(rel) > int(self.lookback_bars):
            rel = rel.iloc[-int(self.lookback_bars) :].reset_index(drop=True)
            tip = pd.to_datetime(rel["timestamp"].iloc[-1], utc=True)
            btc_aligned = btc_aligned[pd.to_datetime(btc_aligned["timestamp"], utc=True) <= tip]
            alt_aligned = alt_aligned[pd.to_datetime(alt_aligned["timestamp"], utc=True) <= tip]
            btc_aligned = btc_aligned.iloc[-len(rel) :].reset_index(drop=True)
            alt_aligned = alt_aligned.iloc[-len(rel) :].reset_index(drop=True)
            # Prefer exact timestamp intersection lengths
            rel = rel.reset_index(drop=True)

        return rel.reset_index(drop=True), alt_aligned.reset_index(drop=True), btc_aligned.reset_index(drop=True), age

    def _fetch_closed(self, market: str, *, limit: int, now: datetime) -> pd.DataFrame | None:
        sym = market.upper()
        if sym == "BTCUSDT" and self._btc_cache[0] is not None:
            # Invalidate BTC cache when wall-clock crosses a new interval tip.
            cached_key, cached_df = self._btc_cache
            if cached_df is not None and cached_key == self._cache_key(now):
                return cached_df.copy()

        raw = self.kline_fetcher(sym, self.interval, int(limit))
        if not raw:
            return None
        df = drop_incomplete_candle(klines_to_dataframe(raw), interval=self.interval, now=now)
        if df is None or df.empty:
            return None
        if sym == "BTCUSDT":
            self._btc_cache = (self._cache_key(now), df.copy())
        return df

    def _cache_key(self, now: datetime) -> str:
        # Bucket by closed-candle open time so intra-bar polls reuse BTC klines.
        ms = int(now.timestamp() * 1000)
        from binance_btc_bot.market_data.klines import interval_to_ms

        step = interval_to_ms(self.interval)
        open_ms = (ms // step) * step - step  # last fully closed open
        return f"{self.interval}:{open_ms}"


def new_cross_into(
    s_previous: float | None,
    s_current: float | None,
    long_threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """Bar-to-bar NEW CROSS: previous S < thr AND current S >= thr."""
    if s_previous is None or s_current is None:
        return False
    if not (math.isfinite(float(s_previous)) and math.isfinite(float(s_current))):
        return False
    return float(s_previous) < float(long_threshold) and float(s_current) >= float(long_threshold)


def json_safe(obj: Any) -> Any:
    """Convert nested factor dumps into JSON-serializable primitives."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return json_safe(obj.item())
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


def detail_from_frames(
    rel_hist: pd.DataFrame,
    alt_vol_hist: pd.DataFrame,
    btc_hist: pd.DataFrame,
    signal_cfg: dict[str, Any],
    *,
    interval: str = DEFAULT_INTERVAL,
    weights: dict[str, float] | None = None,
    dominance_pct: float | None = None,
    dom_changes: dict[int, float | None] | None = None,
) -> dict[str, Any] | None:
    """Full factor + weight + contribution + late_entry dump (JSON-safe).

    Same validity gates as ``score_from_frames``. Returns None when S cannot be formed.
    """
    if rel_hist is None or len(rel_hist) < MIN_HISTORY_BARS:
        return None
    if alt_vol_hist is None or btc_hist is None or len(alt_vol_hist) < 1 or len(btc_hist) < 1:
        return None
    for frame in (rel_hist, alt_vol_hist, btc_hist):
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in frame.columns]
        if cols and not np.isfinite(frame[cols].to_numpy(dtype=float)).all():
            return None

    factors = compute_all_factors(
        rel_hist,
        alt_vol_hist,
        btc_hist,
        dominance_pct,
        dom_changes or {},
        signal_cfg,
        interval,
    )
    factor_scores = extract_factor_scores(factors)
    for key in ACTIVE_SIGNAL_KEYS:
        val = factor_scores.get(key)
        if val is None or not np.isfinite(float(val)):
            return None

    w = weights if weights is not None else static_factor_weights(signal_cfg)
    scored = combined_score(factor_scores, w)
    s_val = scored.get("S")
    if s_val is None or not np.isfinite(float(s_val)):
        return None

    late = late_entry_score(rel_hist, factors, signal_cfg, interval)
    # Drop huge prose notes from btc_regime if present
    factors_out = dict(factors)
    br = factors_out.get("btc_regime")
    if isinstance(br, dict) and "dominance_source_note" in br:
        br = dict(br)
        note = br.get("dominance_source_note")
        br["dominance_source_note"] = (str(note)[:160] if note is not None else None)
        factors_out["btc_regime"] = br

    return json_safe(
        {
            "S": float(s_val),
            "factor_scores": factor_scores,
            "signed": scored.get("signed"),
            "weights": scored.get("weights"),
            "contributions": scored.get("contributions"),
            "active_signal_keys": list(ACTIVE_SIGNAL_KEYS),
            "all_factor_keys": list(ALL_FACTOR_KEYS),
            "factors": factors_out,
            "late_entry": late,
            "signal_score_unsigned": factors.get("signal_score"),
        }
    )


def score_from_frames(
    rel_hist: pd.DataFrame,
    alt_vol_hist: pd.DataFrame,
    btc_hist: pd.DataFrame,
    signal_cfg: dict[str, Any],
    *,
    interval: str = DEFAULT_INTERVAL,
    weights: dict[str, float] | None = None,
    dominance_pct: float | None = None,
    dom_changes: dict[int, float | None] | None = None,
) -> float | None:
    """Compute signed S using the exact backtest factor → combined_score path.

    Returns None when history is insufficient or S / active factors are invalid.
    ``btc_regime`` weight is forced to 0 (same as backtest), so dominance may be None.
    """
    if rel_hist is None or len(rel_hist) < MIN_HISTORY_BARS:
        return None
    if alt_vol_hist is None or btc_hist is None or len(alt_vol_hist) < 1 or len(btc_hist) < 1:
        return None
    # Refuse silently-corrupt series (NaN / inf in the decision window).
    for frame in (rel_hist, alt_vol_hist, btc_hist):
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in frame.columns]
        if cols and not np.isfinite(frame[cols].to_numpy(dtype=float)).all():
            return None

    factors = compute_all_factors(
        rel_hist,
        alt_vol_hist,
        btc_hist,
        dominance_pct,
        dom_changes or {},
        signal_cfg,
        interval,
    )
    factor_scores = extract_factor_scores(factors)
    for key in ACTIVE_SIGNAL_KEYS:
        val = factor_scores.get(key)
        if val is None or not np.isfinite(float(val)):
            return None

    w = weights if weights is not None else static_factor_weights(signal_cfg)
    scored = combined_score(factor_scores, w)
    s_val = scored.get("S")
    if s_val is None or not np.isfinite(float(s_val)):
        return None
    return float(s_val)


def backtest_signed_S(
    rel_hist: pd.DataFrame,
    alt_vol_hist: pd.DataFrame,
    btc_hist: pd.DataFrame,
    signal_cfg: dict[str, Any],
    *,
    interval: str = DEFAULT_INTERVAL,
    dominance_pct: float | None = None,
    dom_changes: dict[int, float | None] | None = None,
) -> float | None:
    """Reference backtest path (predict_coin_at_bar → extract → combined_score)."""
    from btcc.backtest.predict import predict_coin_at_bar

    row = predict_coin_at_bar(
        rel_hist,
        alt_vol_hist,
        btc_hist,
        dominance_pct,
        dom_changes or {},
        signal_cfg,
        interval,
    )
    if row is None:
        return None
    factor_scores = extract_factor_scores(row["factors"])
    for key in ACTIVE_SIGNAL_KEYS:
        val = factor_scores.get(key)
        if val is None or not np.isfinite(float(val)):
            return None
    scored = combined_score(factor_scores, static_factor_weights(signal_cfg))
    s_val = scored.get("S")
    if s_val is None or not np.isfinite(float(s_val)):
        return None
    return float(s_val)


def load_signal_config(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_SIGNAL_CONFIG
    if not p.is_absolute():
        p = REPO_ROOT / p
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"invalid signal config: {p}")
    return raw


def build_production_score_provider(
    cfg: dict[str, Any],
    *,
    exchange: Any,
    diagnostics: bool | None = None,
) -> ProductionScoreProvider:
    """Wire provider to exchange.get_klines (public Vision REST)."""
    signal = dict(cfg.get("signal") or {})
    path = signal.get("signal_config_path") or "configs/signal_config.yaml"
    signal_cfg = load_signal_config(path)
    # Live bot may override momentum profile (e.g. e2) without mutating research YAML.
    mom_profile = signal.get("momentum_profile")
    if mom_profile:
        factors = dict(signal_cfg.get("factors") or {})
        factors["momentum_profile"] = str(mom_profile).strip().lower()
        signal_cfg = {**signal_cfg, "factors": factors}
    entry = cfg.get("entry") or {}
    thr = float(entry.get("long_threshold") or signal.get("long_threshold") or DEFAULT_THRESHOLD)
    diag = bool(signal.get("diagnostics", False)) if diagnostics is None else bool(diagnostics)

    def _fetch(symbol: str, interval: str, limit: int) -> list:
        return list(exchange.get_klines(symbol, interval=interval, limit=limit))

    return ProductionScoreProvider(
        signal_cfg=signal_cfg,
        kline_fetcher=_fetch,
        interval=str(signal.get("interval") or DEFAULT_INTERVAL),
        lookback_bars=int(signal.get("lookback_bars") or DEFAULT_LOOKBACK),
        min_history_bars=int(signal.get("min_history_bars") or MIN_HISTORY_BARS),
        long_threshold=thr,
        max_closed_candle_age_sec=float(
            signal.get("max_closed_candle_age_sec") or DEFAULT_MAX_CLOSED_AGE_SEC
        ),
        diagnostics=diag,
    )


def _rel_price(rel: Mapping[str, Any] | None) -> float | None:
    if not rel:
        return None
    try:
        v = rel.get("relative_price")
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
