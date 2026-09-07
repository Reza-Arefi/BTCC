"""Research/backtest remains in the existing `btcc.sim` stack.

This package intentionally does not re-implement the selector backtest engine.
Use configs such as:

  configs/selector_experiment_binance_prior1y_alloc25_max4_config.yaml

Live Binance execution (this package) stays separate: T1 fixed, no A–F live.
"""
