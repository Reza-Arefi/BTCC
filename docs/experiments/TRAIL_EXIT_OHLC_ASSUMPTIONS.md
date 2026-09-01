# Trailing-exit experiment — OHLC execution assumptions

This experiment uses the **existing 15-minute OHLC exit engine** without modification
to execution semantics.

## Per-bar order (unchanged)

1. Update MFE/MAE from bar high/low vs entry fill.
2. If trailing not active and `high >= activation`: activate trail, set initial trailing stop.
3. If trailing active: ratchet stop upward from bar high (never downward).
4. Evaluate whether active stop (initial SL or trailing) is touched via `low <= stop`.
5. Fixed take-profit is **disabled** for all T1–T10 (`take_profit_pct: null`).

## Same-candle conservative rule (unchanged)

When both stop and (hypothetical) TP would be touched in one bar, **stop first**.
For pure trailing strategies, conflict reduces to stop vs trail on the same bar.

Within a bar we only observe OHLC — not intrabar path. Order applied:

1. Activate/ratchet from **high**
2. Then evaluate stop against **low**

This may optimistic-activate trails before a dip that would have hit initial SL
on the same bar; this is documented legacy behavior and was **not changed** for
this experiment.

## Entry fill

Next-bar **open** after closed decision bar (unchanged).

## Entry eligibility (new for this experiment)

- `0.65 <= S < 0.85` on first cross into band
- No exhaustion rejection
- BTC.D contextual only (does not block)
- Max 10 common opportunities; one per pair
