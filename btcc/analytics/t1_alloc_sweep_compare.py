"""Compare T1 allocation-sweep arms in one results folder (overlay plots + tables).

Discovers ``results/t1_binance_fullbtc_thr0p65_max*_alloc*_*/`` runs and writes a
single compare directory with equity overlays and bar charts so all max_open /
allocation variants can be judged side-by-side.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btcc.analytics.capital import capital_daily_series, capital_drawdown_series

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

RUN_GLOB = "t1_binance_fullbtc_thr0p65_max*_alloc*"
ARM_RE = re.compile(
    r"t1_binance_fullbtc_thr0p65_max(?P<max>\d+)_alloc(?P<alloc>[^_]+)_(?P<stamp>\d{8}_\d{6})"
)

# Stable order matching the sweep launcher
MAX_OPEN_ORDER = (3, 4, 5, 6, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30)

# Distinct, colorblind-friendlier palette (no purple cluster)
ARM_COLORS = {
    3: "#00441b",
    4: "#006d2c",
    5: "#238b45",
    6: "#41ab5d",
    8: "#01665e",
    10: "#5ab4ac",
    13: "#8c510a",
    15: "#1b9e77",
    18: "#d95f02",
    20: "#7570b3",
    23: "#e7298a",
    25: "#66a61e",
    28: "#e6ab02",
    30: "#a6761d",
}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _parse_run_dir(path: Path) -> dict[str, Any] | None:
    m = ARM_RE.fullmatch(path.name)
    if not m:
        return None
    max_open = int(m.group("max"))
    alloc_tag = m.group("alloc")
    # alloc6p67 -> 6.67
    alloc_pct = float(alloc_tag.replace("p", "."))
    return {
        "run_dir": path,
        "max_open": max_open,
        "alloc_tag": alloc_tag,
        "alloc_pct": alloc_pct,
        "stamp": m.group("stamp"),
        "label": f"max{max_open} / {alloc_pct:g}%",
        "short_label": f"max{max_open}",
    }


def discover_t1_alloc_runs(results_dir: Path | None = None) -> list[dict[str, Any]]:
    """Latest non-aborted run per max_open (by stamp), sorted by MAX_OPEN_ORDER."""
    root = Path(results_dir or RESULTS)
    by_max: dict[int, dict[str, Any]] = {}
    for child in root.glob(RUN_GLOB):
        if not child.is_dir():
            continue
        if (child / "ABORTED.txt").exists():
            continue
        meta = _parse_run_dir(child)
        if meta is None:
            continue
        prev = by_max.get(meta["max_open"])
        if prev is None or meta["stamp"] >= prev["stamp"]:
            by_max[meta["max_open"]] = meta
    out = [by_max[m] for m in MAX_OPEN_ORDER if m in by_max]
    return out


def _t1_legs(legs: pd.DataFrame) -> pd.DataFrame:
    if legs.empty:
        return legs
    df = legs.copy()
    if "arm_key" in df.columns:
        df = df[df["arm_key"].astype(str) == "T1"]
    elif "strategy_key" in df.columns:
        df = df[df["strategy_key"].astype(str) == "trail_1"]
    if "closed" in df.columns:
        df = df[df["closed"] == True]  # noqa: E712
    if "entry_policy" not in df.columns:
        df["entry_policy"] = "COMMON"
    # capital_* keys off strategy_key — force T1 label for overlays
    df = df.copy()
    df["strategy_key"] = "T1"
    df["arm_key"] = "T1"
    return df


def _as_of_day(run_dir: Path, legs: pd.DataFrame) -> int:
    ck = sorted((run_dir / "daily_checkpoints").glob("day_*"))
    if ck:
        try:
            return max(int(p.name.split("_")[1]) for p in ck)
        except Exception:
            pass
    if not legs.empty and "day_number" in legs.columns:
        return int(pd.to_numeric(legs["day_number"], errors="coerce").max() or 0)
    return 0


def _metrics_from_legs(legs: pd.DataFrame, *, starting_capital: float, max_day: int) -> dict[str, Any]:
    if legs.empty:
        return {
            "n_trades": 0,
            "win_rate_pct": None,
            "profit_factor": None,
            "avg_trade_return_pct": None,
            "total_pnl_usd": 0.0,
            "final_equity_usd": starting_capital,
            "cumulative_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "as_of_day": max_day,
        }
    pnl = pd.to_numeric(legs.get("pnl_usd_equiv"), errors="coerce").fillna(0.0)
    wins = (pd.to_numeric(legs.get("pnl_pct"), errors="coerce") > 0).sum()
    gp = float(pnl[pnl > 0].sum())
    gl = float(abs(pnl[pnl < 0].sum()))
    daily = capital_daily_series(legs, starting_capital_usd=starting_capital, max_day=max_day or None)
    dd = capital_drawdown_series(daily, starting_capital_usd=starting_capital)
    max_dd = float(dd["drawdown_pct"].min()) if not dd.empty else 0.0
    total = float(pnl.sum())
    return {
        "n_trades": int(len(legs)),
        "win_rate_pct": 100.0 * float(wins) / len(legs) if len(legs) else None,
        "profit_factor": (gp / gl) if gl > 1e-9 else None,
        "avg_trade_return_pct": 100.0 * float(pd.to_numeric(legs.get("pnl_pct"), errors="coerce").mean()),
        "total_pnl_usd": total,
        "final_equity_usd": starting_capital + total,
        "cumulative_return_pct": 100.0 * total / starting_capital,
        "max_drawdown_pct": max_dd,
        "as_of_day": max_day,
    }


def collect_arm_series(runs: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (summary_table, daily_equity_long, drawdown_long)."""
    summary_rows: list[dict[str, Any]] = []
    daily_parts: list[pd.DataFrame] = []
    dd_parts: list[pd.DataFrame] = []

    for meta in runs:
        run_dir: Path = meta["run_dir"]
        legs_all = _safe_read_csv(run_dir / "strategy_legs.csv")
        legs = _t1_legs(legs_all)
        capital = 1000.0
        man = run_dir / "experiment_manifest.json"
        if man.exists():
            try:
                capital = float(json.loads(man.read_text(encoding="utf-8")).get("starting_capital_usd", 1000.0))
            except Exception:
                pass
        metrics_path = run_dir / "analytics" / "selector_metrics.json"
        if metrics_path.exists():
            try:
                mj = json.loads(metrics_path.read_text(encoding="utf-8"))
                capital = float(mj.get("starting_capital_usd", capital))
            except Exception:
                pass

        as_of = _as_of_day(run_dir, legs)
        notional = float(legs["notional_usd"].iloc[0]) if not legs.empty and "notional_usd" in legs.columns else capital / meta["max_open"]
        m = _metrics_from_legs(legs, starting_capital=capital, max_day=as_of)
        complete = (run_dir / "summary.json").exists()
        summary_rows.append(
            {
                **{k: meta[k] for k in ("max_open", "alloc_pct", "alloc_tag", "label", "short_label", "stamp")},
                "run_dir": str(run_dir),
                "notional_usd": notional,
                "starting_capital_usd": capital,
                "exposure_cap_pct": meta["alloc_pct"] * meta["max_open"],
                "complete": complete,
                **m,
            }
        )

        if legs.empty or as_of <= 0:
            continue
        daily = capital_daily_series(legs, starting_capital_usd=capital, max_day=as_of)
        if daily.empty:
            continue
        daily = daily.copy()
        daily["max_open"] = meta["max_open"]
        daily["label"] = meta["label"]
        daily["short_label"] = meta["short_label"]
        daily_parts.append(daily)
        dd = capital_drawdown_series(daily, starting_capital_usd=capital)
        dd = dd.copy()
        dd["max_open"] = meta["max_open"]
        dd["label"] = meta["label"]
        dd["short_label"] = meta["short_label"]
        dd_parts.append(dd)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values("max_open").reset_index(drop=True)
    daily_long = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    dd_long = pd.concat(dd_parts, ignore_index=True) if dd_parts else pd.DataFrame()
    return summary, daily_long, dd_long


def _style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _nudge_label_ys(ys: list[float], *, min_gap: float) -> list[float]:
    """Push nearby label y-positions apart (sorted low→high) so text does not stack."""
    if not ys:
        return []
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = [float(y) for y in ys]
    for k in range(1, len(order)):
        prev_i = order[k - 1]
        cur_i = order[k]
        if out[cur_i] - out[prev_i] < min_gap:
            out[cur_i] = out[prev_i] + min_gap
    return out


def _plot_overlay_return(daily: pd.DataFrame, out: Path, *, note: str) -> None:
    """Day-axis cumulative return curves — one colored line per max_open.

    Each curve gets a gray horizontal guide to both spines:
      left  = final cumulative return %
      right = max-open N
    """
    if daily.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 7.5))
    order = [m for m in MAX_OPEN_ORDER if m in set(int(x) for x in daily["max_open"].unique())]
    endpoints: list[tuple[int, float, float, str]] = []  # max_open, last_day, last_y, color

    for max_open in order:
        g = daily[daily["max_open"].astype(int) == int(max_open)].sort_values("day_number")
        if g.empty:
            continue
        color = ARM_COLORS.get(int(max_open), "#333333")
        xs = g["day_number"].astype(float)
        ys = g["cumulative_return_pct"].astype(float)
        alloc = 100.0 / float(max_open)
        if "label" in g.columns:
            try:
                alloc = float(str(g["label"].iloc[0]).split("/")[-1].strip().rstrip("%"))
            except Exception:
                pass
        ax.plot(
            xs,
            ys,
            label=f"max {max_open}  ({alloc:g}% / trade)",
            color=color,
            lw=2.4,
            solid_capstyle="round",
            zorder=3,
        )
        endpoints.append((int(max_open), float(xs.iloc[-1]), float(ys.iloc[-1]), color))

    ax.axhline(0.0, color="#666666", lw=0.9, ls="--", alpha=0.7, zorder=1)
    ax.set_title(f"T1 cumulative return % by day — all max-open arms\n{note}")
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative return (%)")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_color("#bbbbbb")

    if endpoints:
        x_min_data = float(daily["day_number"].min())
        x_max_data = max(e[1] for e in endpoints)
        pad = max(0.8, 0.05 * (x_max_data - x_min_data + 1))
        ax.set_xlim(x_min_data - 0.15 * pad, x_max_data + pad)

        # Keep full curve range visible (not just endpoint labels)
        series_lo = float(daily["cumulative_return_pct"].min())
        series_hi = float(daily["cumulative_return_pct"].max())
        true_ys = [e[2] for e in endpoints]
        curve_span = max(series_hi - series_lo, 1e-9)
        min_gap = max(0.06, 0.028 * curve_span)
        label_ys = _nudge_label_ys(true_ys, min_gap=min_gap)

        lo = min(series_lo, min(true_ys), min(label_ys))
        hi = max(series_hi, max(true_ys), max(label_ys))
        margin = 0.06 * max(hi - lo, 0.5)
        ax.set_ylim(lo - margin, hi + margin)

        x0, x1 = ax.get_xlim()
        # data-coordinate offset just outside the plot box for side labels
        x_left = x0 - 0.02 * (x1 - x0)
        x_right = x1 + 0.01 * (x1 - x0)

        for (max_open, _last_day, y_true, color), y_lab in zip(endpoints, label_ys):
            # Full-width gray guide at the true final cumulative value
            ax.plot([x0, x1], [y_true, y_true], color="#9a9a9a", lw=0.9, alpha=0.7, zorder=2)
            # Connector if label was nudged away from the guide
            if abs(y_lab - y_true) > 1e-6:
                ax.plot([x_left, x0], [y_lab, y_true], color="#c0c0c0", lw=0.6, zorder=2)
                ax.plot([x1, x_right], [y_true, y_lab], color="#c0c0c0", lw=0.6, zorder=2)

            # Left: cumulative return value
            ax.text(
                x_left,
                y_lab,
                f"{y_true:+.2f}%",
                ha="right",
                va="center",
                fontsize=8.5,
                color="#222222",
                clip_on=False,
                zorder=5,
            )
            # Right: max N
            ax.text(
                x_right,
                y_lab,
                f"{max_open}",
                ha="left",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color=color,
                clip_on=False,
                zorder=5,
            )

    ax.legend(
        title="Simultaneous trades",
        loc="upper left",
        fontsize=8.5,
        title_fontsize=9,
        framealpha=0.92,
    )
    fig.subplots_adjust(left=0.12, right=0.92, top=0.90, bottom=0.10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_overlay_drawdown(dd: pd.DataFrame, out: Path, *, note: str) -> None:
    if dd.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 7))
    order = [m for m in MAX_OPEN_ORDER if m in set(int(x) for x in dd["max_open"].unique())]
    for max_open in order:
        g = dd[dd["max_open"].astype(int) == int(max_open)].sort_values("day_number")
        if g.empty:
            continue
        color = ARM_COLORS.get(int(max_open), "#333333")
        ax.plot(
            g["day_number"].astype(float),
            g["drawdown_pct"].astype(float),
            label=f"max {max_open}",
            color=color,
            lw=2.4,
            solid_capstyle="round",
        )
    _style_axes(ax, f"T1 drawdown % by day — all max-open arms\n{note}", "Day", "Drawdown (%)")
    ax.legend(title="Simultaneous trades", loc="best", fontsize=9, title_fontsize=9, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_bar(
    summary: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    ylabel: str,
    out: Path,
    note: str,
) -> None:
    if summary.empty or value_col not in summary.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [str(x) for x in summary["short_label"]]
    vals = pd.to_numeric(summary[value_col], errors="coerce")
    colors = [ARM_COLORS.get(int(m), "#555555") for m in summary["max_open"]]
    bars = ax.bar(labels, vals.fillna(0.0), color=colors, edgecolor="white", width=0.72)
    for b, v in zip(bars, vals):
        if pd.isna(v):
            continue
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    _style_axes(ax, f"{title}\n{note}", "Arm (max simultaneous)", ylabel)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_compare_md(summary: pd.DataFrame, out: Path, *, note: str, compare_dir: Path) -> None:
    lines = [
        "# T1 allocation sweep — single-place compare",
        "",
        f"- **Entry:** S ≥ 0.65 (T1 only, selectors off)",
        f"- **Capital:** $1000 start; allocation = 100% / max_open",
        f"- **Status note:** {note}",
        f"- **Generated (UTC):** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Ranking (by cumulative return %)",
        "",
        "| rank | arm | alloc% | notional | trades | return% | maxDD% | win% | PF | day | done |",
        "|-----:|-----|-------:|---------:|-------:|--------:|-------:|-----:|---:|----:|:----:|",
    ]
    if not summary.empty:
        ranked = summary.sort_values("cumulative_return_pct", ascending=False).reset_index(drop=True)
        for i, r in ranked.iterrows():
            lines.append(
                "| {rank} | {lab} | {alloc:.2f} | ${notional:.2f} | {n} | {ret:.2f} | {dd:.2f} | {wr} | {pf} | {day} | {done} |".format(
                    rank=i + 1,
                    lab=r["label"],
                    alloc=float(r["alloc_pct"]),
                    notional=float(r["notional_usd"]),
                    n=int(r["n_trades"] or 0),
                    ret=float(r["cumulative_return_pct"] or 0),
                    dd=float(r["max_drawdown_pct"] or 0),
                    wr=f"{float(r['win_rate_pct']):.1f}" if pd.notna(r.get("win_rate_pct")) else "—",
                    pf=f"{float(r['profit_factor']):.2f}" if pd.notna(r.get("profit_factor")) else "—",
                    day=int(r.get("as_of_day") or 0),
                    done="yes" if r.get("complete") else "no",
                )
            )
    lines += [
        "",
        "## Plots in this folder",
        "",
        "- `plots/cumulative_return_bar.png` ← **day-axis curves (all max-open, colored)**",
        "- `plots/cumulative_return_overlay.png` (same curves)",
        "- `plots/drawdown_overlay.png`",
        "- `plots/cumulative_return_snapshot_bar.png` (end-point bars only)",
        "- `plots/final_equity_bar.png`",
        "- `plots/max_drawdown_bar.png`",
        "- `plots/win_rate_bar.png`",
        "- `plots/profit_factor_bar.png`",
        "- `plots/trade_count_bar.png`",
        "- `plots/avg_trade_return_bar.png`",
        "",
        f"Source run dirs are listed in `summary_table.csv`. Compare root: `{compare_dir}`",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def write_t1_alloc_sweep_compare(
    *,
    results_dir: Path | None = None,
    out_dir: Path | None = None,
    reuse_dir: bool = True,
) -> Path:
    """Build / refresh the unified compare directory. Returns path."""
    results_dir = Path(results_dir or RESULTS)
    runs = discover_t1_alloc_runs(results_dir)
    if not runs:
        raise FileNotFoundError(f"No T1 alloc sweep runs under {results_dir}/{RUN_GLOB}")

    if out_dir is None:
        # Prefer a stable folder name so the user always looks in one place
        stable = results_dir / "t1_alloc_sweep_compare_thr0p65"
        if reuse_dir:
            out_dir = stable
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_dir = results_dir / f"t1_alloc_sweep_compare_thr0p65_{stamp}"
    out_dir = Path(out_dir)
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    summary, daily, dd = collect_arm_series(runs)
    days = [int(x) for x in summary["as_of_day"].tolist()] if not summary.empty else [0]
    done_n = int(summary["complete"].sum()) if not summary.empty else 0
    note = f"as of day {min(days)}–{max(days)} · {done_n}/{len(summary)} runs complete · S≥0.65 · T1 only"

    summary.to_csv(out_dir / "summary_table.csv", index=False)
    if not daily.empty:
        daily.to_csv(out_dir / "daily_equity_overlay.csv", index=False)
    if not dd.empty:
        dd.to_csv(out_dir / "daily_drawdown_overlay.csv", index=False)

    _plot_overlay_return(daily, plots / "cumulative_return_overlay.png", note=note)
    # Primary view the user opens: day-axis curves (not a bar chart)
    _plot_overlay_return(daily, plots / "cumulative_return_bar.png", note=note)
    _plot_overlay_drawdown(dd, plots / "drawdown_overlay.png", note=note)
    _plot_bar(summary, value_col="final_equity_usd", title="Final equity ($)", ylabel="Equity (USD)", out=plots / "final_equity_bar.png", note=note)
    _plot_bar(summary, value_col="cumulative_return_pct", title="Cumulative return % (snapshot)", ylabel="Return (%)", out=plots / "cumulative_return_snapshot_bar.png", note=note)
    _plot_bar(summary, value_col="max_drawdown_pct", title="Max drawdown %", ylabel="Drawdown (%)", out=plots / "max_drawdown_bar.png", note=note)
    _plot_bar(summary, value_col="win_rate_pct", title="Win rate %", ylabel="Win rate (%)", out=plots / "win_rate_bar.png", note=note)
    _plot_bar(summary, value_col="profit_factor", title="Profit factor", ylabel="Profit factor", out=plots / "profit_factor_bar.png", note=note)
    _plot_bar(summary, value_col="n_trades", title="Closed trades", ylabel="Trades", out=plots / "trade_count_bar.png", note=note)
    _plot_bar(summary, value_col="avg_trade_return_pct", title="Avg trade return %", ylabel="Avg return (%)", out=plots / "avg_trade_return_bar.png", note=note)

    _write_compare_md(summary, out_dir / "COMPARE.md", note=note, compare_dir=out_dir)

    manifest = {
        "experiment": "t1_binance_thr0p65_alloc_sweep_compare",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "note": note,
        "n_arms": len(summary),
        "n_complete": done_n,
        "runs": summary.to_dict(orient="records") if not summary.empty else [],
        "plots": sorted(p.name for p in plots.glob("*.png")),
    }
    (out_dir / "compare_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # Symlink per-arm analytics plots folder for convenience (best-effort)
    links = out_dir / "per_arm_plots"
    links.mkdir(parents=True, exist_ok=True)
    for meta in runs:
        dest = links / meta["short_label"]
        src = meta["run_dir"] / "analytics" / "plots" / "selector"
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            dest.symlink_to(src.resolve() if src.exists() else meta["run_dir"])
        except OSError:
            pass

    return out_dir
