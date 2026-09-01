#!/usr/bin/env python3
"""Generate Selector E-v1 live monitoring report (rolling windows + concentration)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from btcc.monitoring.selector_report import update_selector_live_analytics
    from btcc.sim.selector_live_config import load_selector_live_config

    sim = load_selector_live_config()
    root = update_selector_live_analytics(sim)
    dash = json.loads((root / "window_dashboard.json").read_text(encoding="utf-8"))
    print("SELECTOR E-v1 LIVE REPORT")
    print("=" * 72)
    print(f"Analytics: {root}")
    conc = json.loads((root / "live_metrics.json").read_text(encoding="utf-8")).get("E_concentration", {})
    print("\nE CONCENTRATION:", conc.get("diagnosis"))
    for k, v in (conc.get("frequency_pct") or {}).items():
        print(f"  {k}: {v}%")
    print("\nWINDOW DASHBOARD")
    print(f"{'Window':<16} {'Return%':>8} {'PF':>6} {'DD%':>7} {'WR%':>6} {'Trades':>7} {'Selected':>16} {'Regret%':>8}")
    for row in dash:
        print(
            f"{row.get('window',''):<16} "
            f"{row.get('E_return_pct') or '—':>8} "
            f"{row.get('PF') or '—':>6} "
            f"{row.get('max_dd_pct') or '—':>7} "
            f"{row.get('win_rate_pct') or '—':>6} "
            f"{row.get('n_trades') or 0:>7} "
            f"{str(row.get('selected_strategy','—')):>16} "
            f"{row.get('regret_pct') or '—':>8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
