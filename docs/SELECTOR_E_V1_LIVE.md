# Selector E-v1 Live Phase

Baseline tag: `baseline-selector-E-v1`

## Architecture

- **Live engine:** `SelectorLiveEngine` — Selector E chooses T1–T10; counterfactuals always recorded
- **Research:** rolling windows, concentration diagnostics, regret — does not alter E math
- **Safety:** independent `SelectorSafetyMonitor` (NORMAL / WARNING / HALT)
- **Storage:** `data/selector_live/` append-only, 365-day retention

## Enable live Selector E (paper only)

1. Set `selector_live.enabled: true` in `configs/selector_live_config.yaml`
2. Set `sim.enabled: false` in `configs/sim_config.yaml` (mutually exclusive with Adaptive V2)
3. Restart `btcc.service`

## Reports

```bash
python scripts/generate_selector_live_report.py
```

## Baseline policy

- Do **not** change E `rank_ewma` math without a new versioned experiment (E-v2, etc.)
- E decision memory remains configurable for future walk-forward tests (E-30, E-90, …)
- Historical 365d experiment continues under `results/selector_compare_365d_*`
