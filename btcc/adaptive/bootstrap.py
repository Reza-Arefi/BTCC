"""Bootstrap adaptive store + Champion from existing 90-day backtest."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from btcc.adaptive.model import create_champion_v1_from_config, get_champion
from btcc.adaptive.store import AdaptivePredictionStore

logger = logging.getLogger(__name__)


def bootstrap_from_backtest(
    cfg: dict[str, Any],
    adaptive_cfg: dict[str, Any],
    models_dir: Path,
    store: AdaptivePredictionStore,
    force: bool = False,
) -> dict[str, Any]:
    """Seed Champion v1 and import HISTORICAL_INITIAL_DATA from backtest Top-5 CSV.

    Note: the 90d backtest stored Top-5 rows (not all 20). These are tagged
    HISTORICAL_INITIAL_DATA and never overwritten. Live cycles append LIVE_OBSERVATION_DATA.
    """
    report: dict[str, Any] = {}
    champ = get_champion(models_dir)
    if champ is None or force:
        champ = create_champion_v1_from_config(
            {**cfg, "adaptive": adaptive_cfg}, models_dir
        )
        report["champion_created"] = champ.version
    else:
        report["champion_existing"] = champ.version

    # Import historical predictions once
    already = (
        not store.df.empty
        and "data_source" in store.df.columns
        and (store.df["data_source"] == adaptive_cfg.get("data_source_tag_historical", "HISTORICAL_INITIAL_DATA")).any()
    )
    if already and not force:
        report["historical_import"] = "skipped_already_present"
        report["store_rows"] = len(store.df)
        return report

    bt_rel = adaptive_cfg.get("initial_backtest", "results/backtest_90d_20260829_001035")
    root = Path(cfg.get("_root", "."))
    bt_dir = Path(bt_rel) if Path(bt_rel).is_absolute() else root / bt_rel
    csv_path = bt_dir / "top5_predictions.csv"
    if not csv_path.exists():
        report["historical_import"] = f"missing:{csv_path}"
        return report

    hist = pd.read_csv(csv_path)
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True)
    tag = adaptive_cfg.get("data_source_tag_historical", "HISTORICAL_INITIAL_DATA")
    rows = []
    for _, r in hist.iterrows():
        rows.append({
            "prediction_id": f"hist_{r.get('timestamp')}_{r.get('symbol')}_{r.get('rank')}",
            "timestamp": r["timestamp"],
            "symbol": r.get("symbol"),
            "base": r.get("base"),
            "rank": r.get("rank"),
            "model_version": "v1",
            "model_type": "champion",
            "data_source": tag,
            "probability_kind": "baseline_model_probability",
            "signal_score": r.get("signal_score"),
            "indicator_momentum": r.get("indicator_momentum"),
            "indicator_ema": r.get("indicator_ema"),
            "indicator_macd": r.get("indicator_macd"),
            "indicator_ichimoku": r.get("indicator_ichimoku"),
            "indicator_adx": r.get("indicator_adx"),
            "indicator_btc_regime": r.get("indicator_btc_regime"),
            "indicator_btc_dominance": r.get("indicator_btc_dominance"),
            "indicator_rvol": r.get("indicator_rvol"),
            "indicator_bollinger": r.get("indicator_bollinger"),
            "indicator_atr": r.get("indicator_atr"),
            "indicator_natr": r.get("indicator_natr"),
            "indicator_rsi": r.get("indicator_rsi"),
            "indicator_structure": r.get("indicator_structure"),
            "factor_momentum": r.get("factor_momentum"),
            "factor_trend": r.get("factor_trend"),
            "factor_btc_regime": r.get("factor_btc_regime"),
            "factor_volume": r.get("factor_volume"),
            "factor_volatility": r.get("factor_volatility"),
            "factor_rsi": r.get("factor_rsi"),
            "factor_structure": r.get("factor_structure"),
            "probability_1h": r.get("probability_1h"),
            "probability_4h": r.get("probability_4h"),
            "probability_8h": r.get("probability_8h"),
            "probability_12h": r.get("probability_12h"),
            "probability_24h": r.get("probability_24h"),
            "late_entry_score": r.get("late_entry_score"),
            "late_entry_class": r.get("late_entry_class"),
            "alt_btc_price": r.get("alt_btc_price"),
            "btc_price": r.get("btc_price"),
            "btc_dominance": r.get("btc_dominance"),
            "btc_dominance_obs_ts": r.get("btc_dominance_obs_ts"),
            "btc_dominance_age_hours": r.get("btc_dominance_age_hours"),
            "candle_timestamp": r.get("timestamp"),
            "data_age": None,
            "future_return_1h": r.get("future_return_1h"),
            "future_return_4h": r.get("future_return_4h"),
            "future_return_8h": r.get("future_return_8h"),
            "future_return_12h": r.get("future_return_12h"),
            "future_return_24h": r.get("future_return_24h"),
            "outperformed_1h": r.get("outperformed_1h"),
            "outperformed_4h": r.get("outperformed_4h"),
            "outperformed_8h": r.get("outperformed_8h"),
            "outperformed_12h": r.get("outperformed_12h"),
            "outperformed_24h": r.get("outperformed_24h"),
        })
    n = store.append_rows(rows)
    report["historical_import"] = f"imported_{n}_rows_from_{bt_dir.name}"
    report["store_rows"] = len(store.df)
    logger.info("Bootstrapped %d HISTORICAL_INITIAL_DATA rows from %s", n, csv_path)
    return report
