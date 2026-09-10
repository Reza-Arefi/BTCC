# Stage 3 — Selectors A–F vs fixed T7 (entry=E2)

- Source Stage 1: `/home/reza/Desktop/BTC_backtest/results/stage1_entry_90d_1s_20260910_115513`
- Window: `2026-06-12 11:45:00+00:00` → `2026-09-10 11:45:00+00:00`
- Rule: keep selector only if it beats **T7** on select AND holdout

## Decision
- Best selector on select: `B`
- Beats T7 on select: `False`
- Beats T7 on holdout: `True`
- **Keep selector:** `False`
- **Live candidate:** `{'entry': 'E2', 'exit': 'T7', 'exit_kind': 'fixed'}`

## Select ranking (60d)
```
Strategy     kind  Trades  Win %   Net BTC  Return %    Max DD  Expectancy  n_switches top_selected_T  top_selected_T_share
      T7    fixed     128  53.12  0.000134  1.706699 -0.030841    0.001067         NaN            NaN                   NaN
       B selector     128  53.12  0.000043  0.550384 -0.030311    0.000344        33.0             T7              0.578125
       D selector     128  52.34  0.000030  0.384699 -0.030961    0.000240        21.0             T7              0.585938
       C selector     128  52.34  0.000021  0.262043 -0.032837    0.000164        41.0             T7              0.406250
       F selector     128  51.56  0.000013  0.170945 -0.019481    0.000107         8.0             T8              0.445312
       E selector     128  52.34  0.000013  0.162937 -0.015937    0.000102         1.0             T1              1.000000
       A selector     128  51.56 -0.000065 -0.827549 -0.042741   -0.000517        15.0             T7              0.507812
```

## Holdout confirmation (30d)
```
Strategy     kind  Trades  Win %  Net BTC  Return %    Max DD  Expectancy  n_switches top_selected_T  top_selected_T_share
       B selector     202  59.90 0.000673  8.597423 -0.015419    0.003405        17.0             T7              0.628713
      T7    fixed     202  54.95 0.000508  6.489586 -0.027592    0.002570         NaN            NaN                   NaN
       D selector     202  57.43 0.000446  5.695489 -0.014744    0.002256        13.0             T7              0.594059
       A selector     202  56.93 0.000425  5.423661 -0.017214    0.002148         6.0             T7              0.603960
       C selector     202  58.91 0.000400  5.109877 -0.013221    0.002024        37.0             T7              0.366337
       E selector     202  53.96 0.000115  1.465055 -0.010558    0.000580         0.0             T1              1.000000
       F selector     202  53.47 0.000048  0.616563 -0.011895    0.000244         7.0             T1              0.490099
```

## Next
Stage 4: sizing / capacity on the frozen live candidate only.