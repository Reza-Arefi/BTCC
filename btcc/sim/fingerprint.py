"""Reproducibility fingerprint for every backtest output directory."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _file_sha256(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
            if f.tell() > max_bytes:
                h.update(b"|partial|")
                break
    return h.hexdigest()


def build_candle_data_fingerprint(candle_dir: Path, interval: str = "15m") -> dict[str, Any]:
    """Summarize cached OHLCV files for reproducibility (no full re-hash of multi-GB)."""
    d = candle_dir / interval
    files = sorted(d.glob("*.parquet"))
    rows = []
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["timestamp"])
            ts = pd.to_datetime(df["timestamp"], utc=True)
            rows.append({
                "symbol": f.stem,
                "n_candles": int(len(df)),
                "first_timestamp": str(ts.min()),
                "last_timestamp": str(ts.max()),
                "calendar_days": float((ts.max() - ts.min()).total_seconds() / 86400),
            })
        except Exception as e:
            rows.append({"symbol": f.stem, "error": str(e)})
    out: dict[str, Any] = {
        "candle_dir": str(candle_dir),
        "interval": interval,
        "n_files": len(files),
        "pairs": rows,
    }
    if rows and "calendar_days" in rows[0]:
        out["global_first"] = min(r["first_timestamp"] for r in rows if "first_timestamp" in r)
        out["global_last"] = max(r["last_timestamp"] for r in rows if "last_timestamp" in r)
        out["min_pair_calendar_days"] = min(r["calendar_days"] for r in rows if "calendar_days" in r)
        out["min_pair_candles"] = min(r["n_candles"] for r in rows if "n_candles" in r)
    return out


def build_btc_d_fingerprint(cache_dir: Path, days: int) -> dict[str, Any]:
    path = cache_dir / f"btc_dominance_{days}d.parquet"
    meta_path = cache_dir / f"btc_dominance_{days}d_meta.json"
    if not path.exists():
        return {"exists": False, "path": str(path)}
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    spacing_h = None
    if len(ts) >= 2:
        spacing_h = float(ts.diff().dropna().dt.total_seconds().median() / 3600.0)
    resolution = "daily" if spacing_h and spacing_h >= 20 else ("hourly" if spacing_h and spacing_h <= 2 else "mixed")
    return {
        "exists": True,
        "path": str(path),
        "n_points": len(df),
        "first_timestamp": str(ts.min()),
        "last_timestamp": str(ts.max()),
        "calendar_days": float((ts.max() - ts.min()).total_seconds() / 86400),
        "median_spacing_hours": spacing_h,
        "resolution": resolution,
        "source": meta.get("source"),
        "representation": meta.get("representation"),
        "calibration": meta.get("calibration"),
        "file_sha256_prefix": _file_sha256(path),
    }


def write_run_fingerprint(
    out_dir: Path,
    *,
    root: Path,
    sim_cfg: dict[str, Any],
    signal_cfg: dict[str, Any],
    meta: dict[str, Any],
    candle_dir: Path | None = None,
    dominance_cache_dir: Path | None = None,
) -> Path:
    """Write fingerprint.json + config snapshots into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bd_health = sim_cfg.get("btc_d_health") or {}
    ep = sim_cfg.get("entry_policies") or {}
    days = int(meta.get("days") or (sim_cfg.get("walk_forward") or {}).get("total_days") or 365)
    root = Path(root)
    if candle_dir is None:
        candle_dir = root / "data" / "backtest_candles"
    if dominance_cache_dir is None:
        dominance_cache_dir = root / "data" / "backtest_dominance"

    btc_d_meta = dict(meta.get("btc_d") or {})
    btc_d_fp = build_btc_d_fingerprint(dominance_cache_dir, days)
    if btc_d_fp.get("exists"):
        btc_d_meta.setdefault("resolution", btc_d_fp.get("resolution"))
        btc_d_meta.setdefault("median_spacing_hours", btc_d_fp.get("median_spacing_hours"))

    fp = {
        "backtest_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "git_branch": _git_branch(root),
        "bot_version": sim_cfg.get("bot_version"),
        "sim_version": sim_cfg.get("version"),
        "data_source_ohlcv": "mexc_public_rest_klines",
        "data_fingerprint": build_candle_data_fingerprint(candle_dir),
        "btc_d_fingerprint": btc_d_fp,
        "data_period": {
            "eval_start": meta.get("eval_start"),
            "eval_end": meta.get("eval_end"),
            "days": meta.get("days"),
            "init_start": meta.get("init_start"),
            "init_end": meta.get("init_end"),
            "daily_phase_start": meta.get("daily_phase_start"),
        },
        "universe": {
            "bases": list((signal_cfg.get("universe") or {}).get("bases") or []),
            "policy": (sim_cfg.get("universe_policy") or {}),
            "btc_symbol": (signal_cfg.get("universe") or {}).get("btc_symbol"),
        },
        "btc_d": {
            **btc_d_meta,
            "max_age_seconds": bd_health.get("max_age_seconds"),
            "BTC_D_MAX_AGE": bd_health.get("max_age_seconds"),
            "stale_definition": (
                f"observation_age > {bd_health.get('max_age_seconds')}s marks STALE "
                f"(require_for_new_trades={bd_health.get('require_for_new_trades')}; "
                f"{'BLOCKS' if bd_health.get('require_for_new_trades') else 'does NOT block'} new trades)"
            ),
            "historical_note": (
                "1-year historical BTC.D = daily relative proxy; "
                "live BTC.D = higher-frequency (~hourly) relative proxy. "
                "Both use last observation with timestamp <= decision_ts; no interpolation."
            ),
        },
        "entry_policies": list(ep.get("enabled") or ["NORMAL_FILTERED", "LATE_ENTRY_ALLOWED"]),
        "model_arms": ["static", "equal", "adaptive"],
        "exit_strategies": list((sim_cfg.get("strategies") or {}).keys()),
        "fee_rate_per_side": sim_cfg.get("fee_rate_per_side"),
        "slippage_rate_per_side": sim_cfg.get("slippage_rate_per_side"),
        "long_threshold": sim_cfg.get("long_threshold"),
        "primary_horizon_hours": sim_cfg.get("primary_horizon_hours"),
        "candle_interval": sim_cfg.get("candle_interval"),
        "same_candle_conflict": sim_cfg.get("same_candle_conflict"),
        "notional_usd": sim_cfg.get("notional_usd"),
        "max_open_opportunities": sim_cfg.get("max_open_opportunities"),
        "weight_update": sim_cfg.get("weight_update"),
        "walk_forward": sim_cfg.get("walk_forward"),
        "safety_allow_trading": (signal_cfg.get("safety") or {}).get("allow_trading"),
        "extra": {k: v for k, v in meta.items() if k not in ("eval_start", "eval_end", "days", "btc_d")},
    }
    path = out_dir / "fingerprint.json"
    path.write_text(json.dumps(fp, indent=2, default=str), encoding="utf-8")

    try:
        import yaml
        (out_dir / "sim_config.snapshot.yaml").write_text(
            yaml.safe_dump(sim_cfg, sort_keys=False), encoding="utf-8"
        )
        slim_signal = {
            "universe": signal_cfg.get("universe"),
            "factors": signal_cfg.get("factors"),
            "probability": signal_cfg.get("probability"),
            "late_entry": signal_cfg.get("late_entry"),
            "safety": signal_cfg.get("safety"),
            "experiment": signal_cfg.get("experiment"),
        }
        (out_dir / "signal_config.snapshot.yaml").write_text(
            yaml.safe_dump(slim_signal, sort_keys=False), encoding="utf-8"
        )
    except Exception:
        (out_dir / "sim_config.snapshot.json").write_text(
            json.dumps(sim_cfg, indent=2, default=str), encoding="utf-8"
        )
    return path


def _git_branch(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
