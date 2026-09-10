# Stage 1 — Entry selection (BASE/E1–E5 × T1–T10)

- Window: `2026-06-12 11:45:00+00:00` → `2026-09-10 11:45:00+00:00` (90d)
- Select: first 60d | Holdout: last 30d
- Exits: T1–T10 on 1s tape

## Decision
- **Primary winner (mean T1–T10 on select):** `E2`
- Select mean Net BTC: `6.7586539672641444e-06`
- Select best T for winner: `T7`
- Select T1 Net BTC: `1.2763751038107735e-05`
- Holdout mean Net BTC: `0.00023780671731733427`
- T1-only select winner: `E2`

## Select ranking
```
Strategy  mean_net_btc_T1T10  median_net_btc_T1T10 best_T  best_T_net_btc  T1_net_btc  mean_return_pct  worst_max_dd  Trades_T1
      E2            0.000007              0.000006     T7        0.000134    0.000013         0.086278     -0.031406        128
    BASE           -0.000071             -0.000070     T7        0.000045   -0.000044        -0.910639     -0.058268        186
      E1           -0.000467             -0.000482     T3       -0.000148   -0.000166        -5.958857     -0.145602        379
      E4           -0.000475             -0.000484     T1       -0.000191   -0.000191        -6.057759     -0.135801        307
      E5           -0.000505             -0.000412     T3       -0.000016   -0.000056        -6.447977     -0.227208        605
      E3           -0.000572             -0.000347     T1       -0.000110   -0.000110        -7.299340     -0.224398        574
```

## Holdout confirmation
```
Strategy  mean_net_btc_T1T10  median_net_btc_T1T10 best_T  best_T_net_btc  T1_net_btc  mean_return_pct  worst_max_dd  Trades_T1
    BASE            0.000082              0.000028     T7        0.000521    0.000003         1.041647     -0.024442        270
      E1           -0.000062             -0.000105     T7        0.000460   -0.000213        -0.789921     -0.043466        378
      E2            0.000238              0.000231     T7        0.000508    0.000115         3.035746     -0.027592        202
      E3           -0.000181             -0.000217     T7        0.000357   -0.000210        -2.307103     -0.080112        466
      E4           -0.000099             -0.000112     T7        0.000363   -0.000179        -1.269276     -0.040195        330
      E5           -0.000152             -0.000192     T7        0.000120   -0.000255        -1.939776     -0.090855        491
```

## Next
Freeze entry `E2`; run Stage 2 to re-rank T1–T10 on that entry stream.
