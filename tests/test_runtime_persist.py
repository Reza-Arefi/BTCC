"""RuntimeState crash-recovery helpers."""

from btcc.runtime_persist import RuntimeState


def test_runtime_state_dedup_and_failure(tmp_path):
    p = tmp_path / "runtime_state.json"
    rt = RuntimeState(p)
    assert rt.last_decision_candle_ts() is None
    assert not rt.already_processed("2026-01-01 00:00:00+00:00")

    rt.mark_cycle_ok(
        decision_candle_ts="2026-01-01 00:00:00+00:00",
        n_ranked=20,
        weights_version="wv1",
        health_ok=True,
    )
    assert rt.already_processed("2026-01-01 00:00:00+00:00")
    assert rt.get("consecutive_failures") == 0

    n = rt.mark_cycle_failure("boom")
    assert n == 1
    n2 = rt.mark_cycle_failure("boom2")
    assert n2 == 2

    # Reload from disk
    rt2 = RuntimeState(p)
    assert rt2.already_processed("2026-01-01 00:00:00+00:00")
    assert rt2.get("last_weights_version") == "wv1"
