"""Entry-policy experiment unit tests."""

from __future__ import annotations

from btcc.config import load_config
from btcc.sim.config import load_sim_config
from btcc.sim.entry_policy import (
    CLS_LATE_ENTRY_ACCEPTED,
    CLS_LATE_ENTRY_REJECTED,
    CLS_NORMAL_ENTRY,
    POLICY_LATE_ALLOWED,
    POLICY_NORMAL,
    evaluate_entry_policy,
    is_late_extended,
)


def _sm_open():
    return {"signal_generated": True, "trade_suggested": True, "rejection_reason": None}


def test_normal_rejects_late():
    r = evaluate_entry_policy(
        policy=POLICY_NORMAL,
        sm_decision=_sm_open(),
        late_entry_score=0.9,
        late_entry_class="VERY_HIGH_LATE_ENTRY_RISK",
    )
    assert r["trade_suggested"] is False
    assert r["entry_classification"] == CLS_LATE_ENTRY_REJECTED


def test_late_allowed_accepts_recovered():
    r = evaluate_entry_policy(
        policy=POLICY_LATE_ALLOWED,
        sm_decision=_sm_open(),
        late_entry_score=0.9,
        late_entry_class="VERY_HIGH_LATE_ENTRY_RISK",
    )
    assert r["trade_suggested"] is True
    assert r["recovered_by_late_allowed"] is True
    assert r["entry_classification"] == CLS_LATE_ENTRY_ACCEPTED


def test_both_accept_non_late():
    for pol in (POLICY_NORMAL, POLICY_LATE_ALLOWED):
        r = evaluate_entry_policy(
            policy=pol,
            sm_decision=_sm_open(),
            late_entry_score=0.2,
            late_entry_class="NORMAL",
        )
        assert r["trade_suggested"] is True
        assert r["entry_classification"] == CLS_NORMAL_ENTRY
        assert r["recovered_by_late_allowed"] is False


def test_late_allowed_does_not_bypass_max_open():
    sm = {"signal_generated": True, "trade_suggested": False, "rejection_reason": "MAX_OPEN_TRADES"}
    r = evaluate_entry_policy(
        policy=POLICY_LATE_ALLOWED,
        sm_decision=sm,
        late_entry_score=0.9,
        late_entry_class="VERY_HIGH_LATE_ENTRY_RISK",
    )
    assert r["trade_suggested"] is False
    assert r["entry_classification"] == "MAX_OPEN_TRADES"


def test_late_allowed_does_not_bypass_btc_d():
    r = evaluate_entry_policy(
        policy=POLICY_LATE_ALLOWED,
        sm_decision=_sm_open(),
        late_entry_score=0.2,
        late_entry_class="NORMAL",
        health_allow_new_trades=False,
        health_btc_d_available=False,
    )
    assert r["trade_suggested"] is False
    assert r["entry_classification"] == "BTC_D_UNAVAILABLE"


def test_is_late_extended_threshold():
    assert is_late_extended(late_entry_score=0.75, late_entry_class="NORMAL", alert_threshold=0.75)
    assert not is_late_extended(late_entry_score=0.74, late_entry_class="NORMAL", alert_threshold=0.75)
    assert is_late_extended(late_entry_score=0.1, late_entry_class="HIGH_LATE_ENTRY_RISK")


def test_config_has_both_entry_policies():
    sim = load_sim_config()
    enabled = (sim.get("entry_policies") or {}).get("enabled") or []
    assert POLICY_NORMAL in enabled
    assert POLICY_LATE_ALLOWED in enabled


def test_three_exit_strategies_unchanged():
    sim = load_sim_config()
    assert set(sim["strategies"]) == {"strategy_1", "strategy_2", "strategy_3"}


def test_allow_trading_false():
    assert load_config()["safety"]["allow_trading"] is False


def test_threshold_still_060():
    assert float(load_sim_config()["long_threshold"]) == 0.60
