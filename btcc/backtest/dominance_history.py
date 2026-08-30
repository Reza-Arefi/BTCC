"""Historical BTC dominance — no look-ahead.

Relative proxy (default free-tier path), identical to live:
  BTC.D_relative = 100 * BTC_market_cap / sum(available top-N market_caps)

NO present-day /global level calibration.

Resolution for days≤90 is hourly — sufficient for a slow-moving macro factor
on 15m decisions (last-known observation only; never interpolated).

At decision time t only values with timestamp <= t are visible.

global/market_cap_chart is Pro-only; we do NOT scrape TradingView.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from btcc.data.relative_btc_d import TOP_COIN_IDS, compute_relative_btc_d_pct

logger = logging.getLogger(__name__)

# Back-compat alias — single source of truth is relative_btc_d.TOP_COIN_IDS
_TOP_COIN_IDS = TOP_COIN_IDS


class HistoricalDominanceSeries:
    """BTC dominance time series for backtest.

    At decision time t only values with timestamp <= t are visible.
    Never interpolates missing history.
    """

    def __init__(self, series: pd.DataFrame | None = None, meta: dict | None = None):
        # Expected columns: timestamp, btc_dominance_pct
        self.df = series if series is not None else pd.DataFrame(
            columns=["timestamp", "btc_dominance_pct"]
        )
        self.meta = meta or {}
        if not self.df.empty:
            self.df = self.df.sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def _session() -> requests.Session:
        session = requests.Session()
        session.headers.update({"User-Agent": "BTCC-backtest/0.1 (research)"})
        key = os.getenv("BTCC_COINGECKO_API_KEY") or os.getenv("COINGECKO_API_KEY")
        if key:
            session.headers["x-cg-demo-api-key"] = key
            session.headers["x-cg-pro-api-key"] = key
        return session

    @classmethod
    def _load_local_csv(cls, cache_dir: Path) -> pd.DataFrame | None:
        for name in ("btc_dominance_historical.csv", "btc_dominance.csv"):
            p = cache_dir / name
            if not p.exists():
                continue
            df = pd.read_csv(p)
            if "timestamp" not in df.columns or "btc_dominance_pct" not in df.columns:
                logger.warning("Local dominance CSV missing required columns: %s", p)
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            logger.info("Loaded local BTC.D CSV: %s (%d rows)", p, len(df))
            return df[["timestamp", "btc_dominance_pct"]].sort_values("timestamp")
        return None

    @classmethod
    def _get_json(cls, session: requests.Session, url: str, params: dict | None = None,
                  retries: int = 4) -> dict | None:
        for attempt in range(retries):
            try:
                r = session.get(url, params=params, timeout=90)
                if r.status_code == 429:
                    wait = 45 * (attempt + 1)
                    logger.warning("CoinGecko 429 — sleep %ds", wait)
                    time.sleep(wait)
                    continue
                if r.status_code in (401, 403):
                    logger.warning("CoinGecko auth %s for %s", r.status_code, url)
                    return None
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                logger.warning("CoinGecko request error (%s): %s", attempt, e)
                time.sleep(5 * (attempt + 1))
        return None

    @classmethod
    def _fetch_coin_market_caps(
        cls,
        session: requests.Session,
        coin_id: str,
        days: int,
        coin_cache_dir: Path,
        force: bool = False,
    ) -> pd.DataFrame | None:
        """Fetch/cache one coin's market_cap series (hourly for days≤90)."""
        coin_cache_dir.mkdir(parents=True, exist_ok=True)
        path = coin_cache_dir / f"{coin_id}_{days}d.parquet"
        if path.exists() and not force:
            df = pd.read_parquet(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df

        data = cls._get_json(
            session,
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
            {"vs_currency": "usd", "days": str(days)},
        )
        time.sleep(1.5)  # free-tier pacing
        if not data:
            return None
        caps = data.get("market_caps", [])
        if not caps:
            return None
        df = pd.DataFrame(caps, columns=["ts_ms", "market_cap"])
        df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df = (
            df[["timestamp", "market_cap"]]
            .dropna()
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        df.to_parquet(path, index=False)
        logger.info("Cached %s market caps: %d points", coin_id, len(df))
        return df

    @classmethod
    def _current_true_dominance(cls, session: requests.Session) -> float | None:
        data = cls._get_json(session, "https://api.coingecko.com/api/v3/global")
        if not data:
            return None
        try:
            return float(data["data"]["market_cap_percentage"]["btc"])
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def _reconstruct_from_top_coins(
        cls,
        session: requests.Session,
        days: int,
        cache_dir: Path,
        force: bool = False,
    ) -> tuple[pd.DataFrame, dict]:
        """Build hourly BTC.D from CoinGecko free market_chart endpoints."""
        coin_cache = cache_dir / "coin_caps"
        frames: dict[str, pd.DataFrame] = {}
        for coin_id in _TOP_COIN_IDS:
            df = cls._fetch_coin_market_caps(session, coin_id, days, coin_cache, force=force)
            if df is None or df.empty:
                logger.warning("Missing market caps for %s — skipping", coin_id)
                continue
            frames[coin_id] = df.rename(columns={"market_cap": coin_id})

        if "bitcoin" not in frames:
            return pd.DataFrame(columns=["timestamp", "btc_dominance_pct"]), {
                "status": "btc_market_chart_unavailable",
                "source": "coingecko_top_coins_reconstructed",
            }

        # Align all series onto BTC timestamps (last-known within 2h)
        base = frames["bitcoin"][["timestamp", "bitcoin"]].sort_values("timestamp")
        merged = base.copy()
        for coin_id, df in frames.items():
            if coin_id == "bitcoin":
                continue
            merged = pd.merge_asof(
                merged,
                df[["timestamp", coin_id]].sort_values("timestamp"),
                on="timestamp",
                direction="backward",
                tolerance=pd.Timedelta("2h"),
            )

        coin_cols = [c for c in merged.columns if c != "timestamp"]
        rows_out = []
        for _, row in merged.iterrows():
            caps = {c: row[c] for c in coin_cols if pd.notna(row[c]) and float(row[c]) > 0}
            pct, meta = compute_relative_btc_d_pct(caps)
            if pct is None:
                continue
            # Soft sanity on relative share (not absolute global BTC.D)
            if not (15.0 < pct < 95.0):
                continue
            rows_out.append({"timestamp": row["timestamp"], "btc_dominance_pct": pct})

        out = pd.DataFrame(rows_out)
        if out.empty:
            return out, {
                "source": "coingecko_top_coins_relative",
                "representation": "relative_btc_share_of_top_n",
                "status": "merge_empty",
                "days": days,
                "n_points": 0,
                "n_coins_used": len(frames),
                "coins_used": list(frames.keys()),
                "calibration": "none_no_present_day_scaling",
            }
        out = out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

        meta = {
            "source": "coingecko_top_coins_relative",
            "representation": "relative_btc_share_of_top_n",
            "status": "OK",
            "days": days,
            "n_points": len(out),
            "n_coins_used": len(frames),
            "coins_used": list(frames.keys()),
            "resolution": "hourly_if_days_le_90_else_daily",
            "calibration": "none_no_present_day_scaling",
            "note": (
                "RELATIVE BTC.D = BTC_mcap / sum(top_N_mcaps) from CoinGecko free "
                "market_chart. NOT absolute global BTC dominance. Present-day "
                "/global level calibration is intentionally DISABLED to avoid "
                "look-ahead contamination. Prefer Pro global/market_cap_chart or "
                "local_csv for absolute BTC.D. Live DominanceFeed uses the same "
                "formula + TOP_COIN_IDS via /coins/markets."
            ),
            "fetched_utc": datetime.now(timezone.utc).isoformat(),
        }
        return out, meta

    @classmethod
    def _try_pro_global_chart(
        cls, session: requests.Session, days: int, btc_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict] | None:
        data = cls._get_json(
            session,
            "https://api.coingecko.com/api/v3/global/market_cap_chart",
            {"days": str(days)},
        )
        if not data:
            return None
        tot_caps = data.get("market_cap_chart", [])
        if not tot_caps:
            return None
        tot_df = pd.DataFrame(tot_caps, columns=["ts_ms", "total_cap"])
        tot_df["timestamp"] = pd.to_datetime(tot_df["ts_ms"], unit="ms", utc=True)
        m = pd.merge_asof(
            btc_df.sort_values("timestamp"),
            tot_df.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta("3h"),
        )
        m = m.dropna(subset=["market_cap", "total_cap"])
        m = m[m["total_cap"] > 0]
        m["btc_dominance_pct"] = m["market_cap"] / m["total_cap"] * 100.0
        out = m[["timestamp", "btc_dominance_pct"]].drop_duplicates("timestamp").sort_values("timestamp")
        meta = {
            "source": "coingecko_global_market_cap_chart",
            "status": "OK",
            "days": days,
            "n_points": len(out),
            "resolution": "hourly_for_days_le_90",
            "calibration": "exact_btc_over_total",
            "fetched_utc": datetime.now(timezone.utc).isoformat(),
            "note": "BTC.D = BTC cap / total crypto cap from CoinGecko Pro/Demo chart.",
        }
        return out.reset_index(drop=True), meta

    @classmethod
    def fetch_coingecko(
        cls, days: int, cache_dir: str | Path, force: bool = False
    ) -> "HistoricalDominanceSeries":
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"btc_dominance_{days}d.parquet"
        meta_path = cache_dir / f"btc_dominance_{days}d_meta.json"

        if cache_path.exists() and not force:
            df = pd.read_parquet(cache_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            logger.info(
                "Loaded cached BTC.D history: %d points (source=%s)",
                len(df), meta.get("source", "cache"),
            )
            return cls(df, meta)

        # 1) User-provided CSV
        local = cls._load_local_csv(cache_dir)
        if local is not None and not local.empty:
            meta = {
                "source": "local_csv",
                "status": "OK",
                "days": days,
                "n_points": len(local),
                "fetched_utc": datetime.now(timezone.utc).isoformat(),
            }
            local.to_parquet(cache_path, index=False)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return cls(local.reset_index(drop=True), meta)

        session = cls._session()

        # 2) Prefer Pro/Demo global chart if key unlocks it
        btc = cls._fetch_coin_market_caps(
            session, "bitcoin", days, cache_dir / "coin_caps", force=force
        )
        if btc is not None and not btc.empty:
            pro = cls._try_pro_global_chart(session, days, btc)
            if pro is not None:
                out, meta = pro
                if not out.empty:
                    out.to_parquet(cache_path, index=False)
                    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                    logger.info("Fetched BTC.D via Pro global chart: %d points", len(out))
                    return cls(out, meta)

        # 3) Reconstruct from free CoinGecko market_chart of top coins (hourly)
        logger.info(
            "Reconstructing BTC.D from CoinGecko top-coin market_caps "
            "(global chart unavailable on free tier)..."
        )
        out, meta = cls._reconstruct_from_top_coins(session, days, cache_dir, force=force)
        if out.empty:
            logger.error("BTC.D reconstruction failed — dominance INSUFFICIENT_DATA")
            return cls.empty(meta.get("status", "INSUFFICIENT_DATA"))

        out.to_parquet(cache_path, index=False)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info(
            "Fetched BTC.D history: %d points (source=%s, coins=%d)",
            len(out), meta.get("source"), meta.get("n_coins_used"),
        )
        return cls(out, meta)

    @classmethod
    def empty(cls, reason: str) -> "HistoricalDominanceSeries":
        return cls(
            pd.DataFrame(columns=["timestamp", "btc_dominance_pct"]),
            {"status": reason, "n_points": 0, "source": "none"},
        )

    def observation_at(self, ts: pd.Timestamp) -> tuple[float | None, pd.Timestamp | None, str]:
        """Return (dominance_pct, observation_timestamp, status) using last-known ≤ t.

        Never uses future observations. Never interpolates.
        """
        if self.df.empty:
            return None, None, "INSUFFICIENT_DATA"
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        sub = self.df[self.df["timestamp"] <= t]
        if sub.empty:
            return None, None, "INSUFFICIENT_DATA"
        row = sub.iloc[-1]
        return float(row["btc_dominance_pct"]), pd.Timestamp(row["timestamp"]), "OK"

    def value_at(self, ts: pd.Timestamp) -> float | None:
        pct, _obs, _st = self.observation_at(ts)
        return pct

    def change_at(self, ts: pd.Timestamp, hours: float) -> tuple[float | None, str]:
        pct, obs_ts, status = self.observation_at(ts)
        if status != "OK" or pct is None or obs_ts is None:
            return None, "INSUFFICIENT_DATA"
        if self.df.empty:
            return None, "INSUFFICIENT_DATA"
        target = obs_ts - pd.Timedelta(hours=hours)
        prior = self.df[self.df["timestamp"] <= target]
        if prior.empty:
            return None, "INSUFFICIENT_DATA"
        prior_row = prior.iloc[-1]
        age_h = (obs_ts - prior_row["timestamp"]).total_seconds() / 3600.0
        if age_h < hours * 0.9:
            return None, "INSUFFICIENT_DATA"
        return float(pct - prior_row["btc_dominance_pct"]), "OK"

    def dom_changes_at(self, ts: pd.Timestamp) -> dict[int, float | None]:
        return {h: self.change_at(ts, h)[0] for h in (1, 4, 12, 24)}

    def coverage_report(self, start: pd.Timestamp, end: pd.Timestamp) -> dict:
        if self.df.empty:
            return {
                "n_points": 0,
                "n_in_window": 0,
                "status": self.meta.get("status", "INSUFFICIENT_DATA"),
                "source": self.meta.get("source", "none"),
            }
        in_range = self.df[(self.df["timestamp"] >= start) & (self.df["timestamp"] <= end)]
        # Median spacing between observations
        spacing_h = None
        if len(self.df) >= 2:
            deltas = self.df["timestamp"].diff().dropna().dt.total_seconds() / 3600.0
            spacing_h = float(deltas.median())
        return {
            "n_points": len(self.df),
            "n_in_window": len(in_range),
            "first": str(self.df["timestamp"].iloc[0]),
            "last": str(self.df["timestamp"].iloc[-1]),
            "median_spacing_hours": spacing_h,
            "source": self.meta.get("source", "coingecko"),
            "calibration": self.meta.get("calibration"),
            "n_coins_used": self.meta.get("n_coins_used"),
            "status": "OK",
        }
