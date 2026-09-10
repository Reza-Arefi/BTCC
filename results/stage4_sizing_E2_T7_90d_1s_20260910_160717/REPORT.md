# Stage 4 — Sizing / capacity (E2+T7)

- Source: `/home/reza/Desktop/BTC_backtest/results/stage1_entry_90d_1s_20260910_115513`
- Baseline live config: **12.5%** alloc × **max_open=8**

## Decision
- Winner: `A25_M4` → alloc **25%**, max_open **4**
- Change from baseline: `True`
- Select return: `3.4133988798715764` | DD: `-5.964960732357313`
- Holdout return: `13.077381856676018` | DD: `-3.9949107184620343`

## Live candidate (final)
```json
{
  "entry": "E2",
  "exit": "T7",
  "selector": null,
  "allocation_per_trade": 0.25,
  "max_simultaneous_trades": 4
}
```

## Alloc sweep @ max_open=8 (select)
```
     arm  fraction  trades_executed  skipped_max_open  return_pct  max_dd_pct  profit_factor
  A25_M8     0.250              128                 0    3.413399   -5.964961        1.11378
  A20_M8     0.200              128                 0    2.730719   -4.835707        1.11378
A17.5_M8     0.175              128                 0    2.389379   -4.259691        1.11378
  A15_M8     0.150              128                 0    2.048039   -3.675878        1.11378
A12.5_M8     0.125              128                 0    1.706699   -3.084107        1.11378
  A10_M8     0.100              128                 0    1.365360   -2.484216        1.11378
 A7.5_M8     0.075              128                 0    1.024020   -1.876035        1.11378
   A5_M8     0.050              128                 0    0.682680   -1.259391        1.11378
```

## Max-open sweep @ 12.5% (select)
```
      arm  max_open  trades_executed  skipped_max_open  return_pct  max_dd_pct  max_exposure
 A12.5_M4         4              128                 0    1.706699   -3.084107      0.493337
 A12.5_M6         6              128                 0    1.706699   -3.084107      0.493337
 A12.5_M8         8              128                 0    1.706699   -3.084107      0.493337
A12.5_M10        10              128                 0    1.706699   -3.084107      0.493337
A12.5_M12        12              128                 0    1.706699   -3.084107      0.493337
```

## Notes
- Raw E2+T7 stream never exceeded 8 concurrent trades, so raising max_open above 8 has little effect unless allocation changes entry density.
- Prefer baseline unless select+holdout clearly improve.