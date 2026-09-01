"""Tests for Selector E-v1 monitoring infrastructure."""

from __future__ import annotations

import pandas as pd

from btcc.monitoring.selector_safety import SafetyState, SelectorSafetyMonitor
from btcc.monitoring.selector_windows import selection_concentration_report
from btcc.sim.selector_live_config import load_selector_live_config, version_manifest


def test_live_config_loads():
    sim = load_selector_live_config()
    assert sim.get("selector_version") == "E-v1"
    assert len(sim.get("strategies") or {}) == 10
    assert sim.get("enabled") is False


def test_concentration_diagnosis_t1_dominant():
    sel = pd.DataFrame({
        "arm_label": ["E"] * 100,
        "selected_arm_label": ["T1"] * 99 + ["T7"],
    })
    c = selection_concentration_report(sel, selector="E")
    assert c["top_strategy"] == "T1"
    assert c["top_strategy_pct"] > 90
    assert "≈ T1" in c["diagnosis"]


def test_safety_halt_blocks_entries():
    mon = SelectorSafetyMonitor({"enabled": True, "thresholds": {"halt_consecutive_losses": 3}})
    legs = pd.DataFrame({
        "pnl_pct": [-0.01, -0.01, -0.01],
        "pnl_usd_equiv": [-1, -1, -1],
    })
    ev = mon.evaluate(legs)
    assert ev.state == SafetyState.HALT
    assert not mon.allow_new_entries()


def test_version_manifest():
    m = version_manifest()
    assert m["selector_version"] == "E-v1"
    assert m["selector_kind"] == "rank_ewma"
