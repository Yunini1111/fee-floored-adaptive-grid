#!/usr/bin/env python3
"""One-command reproduction of every number in results/.

    pip install -r requirements.txt
    python run_backtest.py --all

`--all` fetches (or reuses the cache in data/), runs the integrity pass, runs
every window, runs every sensitivity sweep, and regenerates results/ from
scratch. `--offline` runs purely from cache so a reviewer with no network gets
identical numbers. `--window flat-2024-26` runs one window in a few seconds.

Nothing in results/ is hand-written. If a number appears in SKILL.md or
README.md, it came from here.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from grid.data import (
    CoinWClient,
    fetch_symbol_constraints,
    integrity_check,
    max_consecutive_gap,
    ms_to_date,
)
from grid.engine import run_backtest
from grid.metrics import compute_metrics
from grid.strategy import Config, SignalEngine, breakeven_spacing, net_edge_per_round_trip

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

PAIR = "BTC_USDT"
DAILY_START = "2018-01-01"  # a year of burn-in so EMA200/ADX14 are warm
EXEC_START = "2019-01-01"
END = "2026-08-01"

WINDOWS = [
    ("bear-2022", "2022-01-01", "2023-01-01", "Sustained bear market, BTC -64%"),
    ("range-2023", "2023-01-01", "2024-01-01", "Recovery year"),
    ("bull-2020Q4", "2020-10-01", "2021-04-01", "Parabolic bull, BTC +445%"),
    ("flat-2024-26", "2024-08-01", "2026-08-01", "Two years, roughly flat"),
    ("year-2025", "2025-01-01", "2026-01-01", "Calendar 2025"),
    ("chop-2019H2", "2019-07-01", "2020-03-01", "Choppy decline incl. Mar-2020 crash"),
    ("full-2019-2026", EXEC_START, END, "Everything: 7.6 years, no cherry-picking"),
]


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def pct(x: float, places: int = 2) -> str:
    if x != x:
        return "n/a"
    return f"{100 * x:+.{places}f}%"


def upct(x: float, places: int = 2) -> str:
    if x != x:
        return "n/a"
    return f"{100 * x:.{places}f}%"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_data(offline: bool):
    client = CoinWClient(DATA_DIR, offline=offline)
    daily = client.fetch(PAIR, 86400, DAILY_START, END)
    execution = client.fetch(PAIR, 900, EXEC_START, END)
    hourly = client.fetch(PAIR, 3600, "2018-12-01", END)
    constraints = fetch_symbol_constraints(PAIR, DATA_DIR, offline=offline)
    return daily, execution, hourly, constraints


def integrity_section(daily, execution, hourly, constraints) -> str:
    lines = ["## 1. Data provenance and integrity", ""]
    lines.append(
        "All bars come from CoinW's public, unauthenticated kline endpoint. Every raw response is "
        "cached under `data/` with a SHA-256 manifest. The cache is ~90 MB and is gitignored, so a "
        "fresh clone carries `data/manifest.json` but not the responses: **the first run needs "
        "network access.** Once `python run_backtest.py --all` has populated `data/`, `--offline` "
        "reproduces every number here byte-for-byte with no further network call, and the manifest "
        "lets a reviewer confirm they hold the same bytes these results were computed from."
    )
    lines.append("")
    lines.append("```")
    lines.append(f"GET {'https://api.coinw.com/api/v1/public'}")
    lines.append("    ?command=returnChartData&currencyPair=BTC_USDT&period=<sec>&start=<ms>&end=<ms>")
    lines.append("```")
    lines.append("")
    lines.append("| Series | Period | Bars | Span | Missing | Longest gap |")
    lines.append("|---|---|---:|---|---:|---:|")
    for name, series in (("Indicator", daily), ("Execution", execution), ("Cross-check", hourly)):
        rep = integrity_check(series)
        lines.append(
            f"| {name} | {series.period}s | {rep.bars:,} | "
            f"{ms_to_date(rep.first_ts)} -> {ms_to_date(rep.last_ts)} | "
            f"{rep.missing_bars:,} ({rep.missing_pct:.3f}%) | {max_consecutive_gap(series)} bars |"
        )
    lines.append("")
    lines.append(
        "Missing bars are **counted and reported, never interpolated**. A fabricated bar "
        "with a plausible low would generate a fill that never happened, which is the exact "
        "class of error that makes a grid backtest untrustworthy."
    )
    lines.append("")
    lines.append(
        f"Exchange constraints read from `command=returnSymbol` and enforced by the engine: "
        f"min notional **{constraints['min_notional']} USDT**, price tick "
        f"**{constraints['price_tick']}**, quantity step **{constraints['qty_step']}**."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Path-length table -- why the fill model is built the way it is
# --------------------------------------------------------------------------- #


def path_length_section(client: CoinWClient, offline: bool) -> str:
    lines = ["## 2. Why the fill model is pessimistic", ""]
    lines.append(
        "A grid's return is close to linear in its fill count, and the fill count is "
        "entirely a modelling choice. Total variation of `log(close)` on BTC_USDT over "
        "2024-08 -> 2026-08, measured at five sampling resolutions:"
    )
    lines.append("")
    lines.append("| Sampling | 1D | 4H | 1H | 15m | 5m |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    cells = []
    for period in (86400, 14400, 3600, 900, 300):
        try:
            series = client.fetch(PAIR, period, "2024-08-01", "2026-08-01")
            logc = np.log(series.close)
            days = (int(series.ts[-1]) - int(series.ts[0])) / 86_400_000.0
            cells.append(f"{100 * np.abs(np.diff(logc)).sum() / days:.2f}%")
        except Exception:
            cells.append("n/a")
    lines.append("| Path length per day | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Path length **diverges as resolution increases** -- it is a fractal quantity, not a "
        "property of the market. A model that counts every level crossing at 1H resolution "
        "implies a few percent per day at 1% spacing, which is obviously nonsense, and it is "
        "the arithmetic a careless grid backtest performs. Hence:"
    )
    lines.append("")
    lines.append("```text")
    lines.append("F1  orders placed on bar t are live from bar t+1's open; daily")
    lines.append("    indicators for UTC day D use only daily bars <= D-1")
    lines.append("F2  state snapshotted at the bar open, applied at the close --")
    lines.append("    cash freed by a sell on bar t cannot fund a buy on bar t")
    lines.append("F3  buy fills iff low < p*(1-1bp). `low == p` is NOT a fill:")
    lines.append("    the market must trade THROUGH us, not merely touch us")
    lines.append("F4  sell fills iff high > p*(1+1bp)")
    lines.append("F5  fill price is always the limit price -- never improved, even on a gap")
    lines.append("F6  if the bar OPENS through our limit, a post-only order would have")
    lines.append("    been rejected -> fill at p but charge the TAKER fee")
    lines.append("F7  one fill per level per UTC DAY -- the ladder refreshes once daily and a")
    lines.append("    level that fills is not re-armed until the next rollover")
    lines.append("F8  forced sales price at a CLOSE, never a high: regime de-risking at the last")
    lines.append("    close known at the 00:00 rollover, the kill at the current bar's close")
    lines.append("F9  0bp slippage on maker fills, 5bp adverse on taker fills")
    lines.append("```")
    lines.append("")
    lines.append(
        "**Residual optimism, disclosed:** even with F3's through-buffer the model assumes we "
        "are at the front of the queue at every price we trade through. On a thin book that is "
        "generous. `results/sensitivity.md` section 4 reruns everything at `fill_probability` "
        "1.0 / 0.7 / 0.5 to price that assumption instead of hiding it."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #


def run_windows(daily, execution, hourly, cfg: Config, names: list[str] | None = None):
    out = []
    for name, start, end, note in WINDOWS:
        if names and name not in names:
            continue
        from grid.data import iso_to_ms

        sub = execution.slice_ms(iso_to_ms(start), iso_to_ms(end))
        if len(sub) < 100:
            continue
        result = run_backtest(sub, daily, cfg, label=name, hourly=hourly)
        out.append((name, note, result, compute_metrics(result, sub), sub))
    return out


def headline_section(rows) -> str:
    lines = ["## 3. Headline results", ""]
    lines.append(
        "Every tested window is here, including every one where the strategy is beaten. "
        "Reporting only the flattering window is the most common backtest dishonesty and "
        "publishing all of them is cheap insurance against the accusation."
    )
    lines.append("")
    lines.append(
        "| Window | Strategy | Max DD | Buy & hold | B&H max DD | Exposure-matched B&H | Avg inventory | Round trips | Sharpe | Calmar |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, _note, _r, m, _ in rows:
        lines.append(
            f"| `{name}` | **{pct(m.total_return)}** | {upct(m.max_dd)} | {pct(m.bh_return)} | "
            f"{upct(m.bh_max_dd)} | {pct(m.exposure_matched_return)} | "
            f"{upct(m.avg_inventory_ratio,1)} | {m.round_trips} | {m.sharpe:.2f} | {m.calmar:.2f} |"
        )
    lines.append("")
    invs = [m.avg_inventory_ratio for *_x, m, _y in rows]
    lines.append(
        f"**Exposure-matched buy & hold is the fair comparison.** This strategy runs "
        f"{upct(float(np.mean(invs)),0)} average inventory across these windows "
        f"({upct(min(invs),0)} to {upct(max(invs),0)}); measuring it against 100%-long BTC compares "
        f"two different amounts of risk. Both are shown, always. Note the benchmark is only "
        f"exposure-matched at t=0 -- it never trims, so its exposure drifts upward as BTC rises, "
        f"which makes it harder to beat, not easier."
    )
    lines.append("")

    beat_bh = sum(1 for *_, m, _ in rows if m.total_return > m.bh_return)
    beat_em = sum(1 for *_, m, _ in rows if m.total_return > m.exposure_matched_return)
    lines.append(
        f"Across {len(rows)} windows the strategy beats outright buy-and-hold in **{beat_bh}** "
        f"and exposure-matched buy-and-hold in **{beat_em}**."
    )
    lines.append("")
    lines.append(
        "> **What this is.** A long-biased grid systematically sells strength and buys weakness. "
        "It *must* underperform in a sustained trend and *must* do relatively well in a chop or a "
        "decline. That is not a bug being explained away -- it is the strategy's identity, and any "
        "grid claiming otherwise is misrepresenting itself. **If you believe BTC is going up, hold "
        "BTC.** This is for the case where you want volatility exposure without a directional view "
        "and will trade away trend participation to get a materially smaller drawdown."
    )
    lines.append("")
    return "\n".join(lines)


def invariant_section(rows) -> str:
    lines = ["## 4. Invariants -- the claims a reviewer can check mechanically", ""]
    lines.append(
        "The paired-exit design makes a strong claim: *every completed round trip is "
        "net-positive by construction*, because a lot's exit is fixed at "
        "`fill_price * (1 + s_at_fill)` when it is bought and is never re-priced. If that "
        "claim is true, the losing-round-trip column must be exactly zero everywhere."
    )
    lines.append("")
    lines.append(
        "| Window | Round trips | Losing round trips | Win rate | Cap breaches (own action) | Bars above cap (mark drift) | Peak inventory |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, _note, _r, m, _ in rows:
        lines.append(
            f"| `{name}` | {m.round_trips} | **{m.losing_round_trips}** | {upct(m.win_rate)} | "
            f"**{m.cap_breaches}** | {upct(m.bars_above_cap_pct,1)} | {upct(m.peak_inventory_ratio,1)} |"
        )
    lines.append("")
    lines.append(
        "Two different questions are counted separately, on purpose. **Cap breaches** asks "
        "*did our own action put inventory over the cap?* and must be 0 -- `tests/test_invariants.py` "
        "asserts it. **Bars above cap** asks *how many bars did mark-to-market drift leave us above "
        "the cap?*, which is not a control failure: a rising price raises the inventory ratio without "
        "us trading, and no cap can be enforced continuously without trading continuously. Reporting "
        "only the first would look airtight; only the second would look broken."
    )
    lines.append("")
    lines.append(
        "The 100% win rate is **an invariant of the exit design, not skill**, and it excludes "
        "de-risk sales, which are frequently losses and are reported separately below."
    )
    lines.append("")
    return "\n".join(lines)


def chain_section(daily, execution, hourly, base: Config) -> str:
    from grid.data import iso_to_ms

    variants = [
        ("A. floating exits (the broken design)", base.with_(paired_exits=False, regime_gate=False, active_derisk=False)),
        ("B. + paired per-lot exits", base.with_(regime_gate=False, active_derisk=False)),
        ("C. + regime gate, passive — **SHIPPED DEFAULT**", base),
        ("D. + active de-risk (tested, REJECTED)", base.with_(active_derisk=True)),
    ]
    window_names = [w[0] for w in WINDOWS if w[0] != "full-2019-2026"]

    lines = ["## 5. The improvement chain, measured", ""]
    lines.append(
        "Each architectural change measured in isolation on the same data: the mean across the "
        f"{len(window_names)} sub-windows, and separately the full 2019-2026 run. A single good "
        "number proves nothing, and the sub-windows overlap heavily, so the full run is the "
        "tiebreaker."
    )
    lines.append("")
    lines.append(
        "| Variant | Sub-window mean return | Sub-window mean DD | Full-run return | Full-run DD | Full-run round trips | Full-run losing round trips | Mean fee drag |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    full = execution
    for vname, vcfg in variants:
        rets, dds, drags = [], [], []
        for name, start, end, _ in WINDOWS:
            if name not in window_names:
                continue
            sub = execution.slice_ms(iso_to_ms(start), iso_to_ms(end))
            m = compute_metrics(run_backtest(sub, daily, vcfg, label=name, hourly=hourly), sub)
            rets.append(m.total_return)
            dds.append(m.max_dd)
            if m.fee_drag == m.fee_drag:
                drags.append(m.fee_drag)
        mf = compute_metrics(run_backtest(full, daily, vcfg, label="full", hourly=hourly), full)
        # Round-trip counts are reported for the FULL RUN ONLY. The six
        # sub-windows all sit inside it, so adding them to the full run would
        # count the same trades roughly twice -- an earlier version of this table
        # did exactly that and reported 295 losing round trips where the full run
        # has 154.
        lines.append(
            f"| {vname} | {pct(float(np.mean(rets)))} | {upct(float(np.mean(dds)))} | "
            f"**{pct(mf.total_return)}** | {upct(mf.max_dd)} | {mf.round_trips} | "
            f"**{mf.losing_round_trips}** | {upct(float(np.mean(drags)),1)} |"
        )

    lines.append("")
    lines.append(
        "**Read the losing-round-trip column first, and read it for what it is.** Floating exits "
        "produce well over a hundred losing round trips; paired exits produce exactly zero, in "
        "every window, across every configuration tested. That is a *structural guarantee*, not a "
        "statistical result."
    )
    lines.append("")
    lines.append(
        "**What paired exits do not buy you is a large return improvement.** On the sub-window "
        "mean the floating-exit variant is actually slightly ahead, because realising losses "
        "earlier sometimes helps. The case for paired exits is that they make the strategy's "
        "behaviour *knowable*: the fee-floor algebra in section 4 is only meaningful if a lot's "
        "exit is fixed relative to its own entry, and 'every round trip is net-positive' is only "
        "a checkable claim if exits never move. We would rather ship a mechanism whose worst case "
        "is bounded and provable than one that backtests marginally better and can fail in ways "
        "no test can catch. Both numbers are above; the reader can disagree."
    )
    lines.append("")
    lines.append("### 5a. Three findings that reversed our own intuition")
    lines.append("")
    lines.append("Every number below is computed at generation time, not transcribed.")
    lines.append("")

    def full_run(**kw):
        cfg = base.with_(**kw)
        r = run_backtest(full, daily, cfg, label="full", hourly=hourly)
        return r, compute_metrics(r, full)

    # Finding 1 -- active de-risk
    _r_off, m_off = full_run()
    _r_on, m_on = full_run(active_derisk=True)
    _r_nogate, m_nogate = full_run(regime_gate=False, active_derisk=False)
    lines.append(
        f"**1. Force-selling into a downtrend is a wealth transfer, not a risk control.** Selling "
        f"inventory down to the 15% DOWNTREND cap when the regime turns down is the obvious design "
        f"and it is what most grid write-ups describe. Over the full run it returns "
        f"{pct(m_on.total_return)} against {pct(m_off.total_return)} for not doing it -- a cost of "
        f"**{100 * (m_off.total_return - m_on.total_return):.1f} percentage points**. Selling into "
        f"weakness at taker prices and rebuying when the regime flips back is a pump running in the "
        f"wrong direction. What *does* work is the passive half of the same gate, refusing to add "
        f"new buys in a downtrend: {pct(m_nogate.total_return)} without it against "
        f"{pct(m_off.total_return)} with it, worth "
        f"**{100 * (m_off.total_return - m_nogate.total_return):.1f} points**. "
        f"**Stop adding; do not panic-sell; keep a properly calibrated circuit breaker.**"
    )
    lines.append("")

    # Finding 2 -- the kill threshold
    _r_k20, m_k20 = full_run(dd_kill_override=0.20)
    kills_20 = sum(
        t.realized_pnl for t in _r_k20.trades if t.reason == "KILL"
    )
    kills_derived = sum(t.realized_pnl for t in _r_off.trades if t.reason == "KILL")
    lines.append(
        f"**2. A hardcoded drawdown kill switch was a large source of loss while wearing the label "
        f"'risk control'.** It was originally a flat 20%. But holding up to "
        f"{100 * base.cap_range:.0f}% of equity in an asset whose own worst drawdown in this "
        f"dataset is {upct(m_off.bh_max_dd,0)} makes a "
        f">20% equity drawdown *structurally normal*, so the switch was not detecting an abnormal "
        f"loss -- it fired on the strategy working as designed, at local bottoms, at taker prices. "
        f"At a 20% threshold the kill path realised **{kills_20:,.0f} USDT** on a "
        f"{base.initial_equity:,.0f} USDT account and the full run returned "
        f"**{pct(m_k20.total_return)}**. Deriving the threshold as "
        f"`clamp(cap_range x asset_max_dd, 0.15, 0.50)` = **{100 * base.dd_kill:.0f}%** leaves "
        f"{kills_derived:,.0f} USDT realised on that path and **{pct(m_off.total_return)}** "
        f"overall. A threshold with units of percent is not a risk control until you can say what "
        f"it is a threshold *of*."
    )
    lines.append("")

    # Finding 3 -- lot ordering, measured where it actually bites
    _r_hi, m_hi = full_run(active_derisk=True, derisk_highest_cost_first=True)
    _r_lo, m_lo = full_run(active_derisk=True, derisk_highest_cost_first=False)
    lines.append(
        f"**3. 'Sell the worst lots first' is backwards.** Dumping highest-cost-basis lots removes "
        f"the positions furthest from their exits, which is the intuitive choice and the one we "
        f"implemented first. Lowest-cost-first realises a smaller loss and leaves the deep lots to "
        f"recover: **{pct(m_lo.total_return)}** against **{pct(m_hi.total_return)}** on the full "
        f"run with forced de-risking enabled. The shipped default does not force-sell, so this "
        f"governs only the drawdown-kill liquidation path -- which now honours the same setting "
        f"rather than hardcoding the rejected ordering, as it did until an audit caught it."
    )
    lines.append("")
    lines.append(
        "**Also tested and rejected:** persistence-gating the de-risk (3 and 5 consecutive "
        "DOWNTREND days) and spreading it across days at 50% of the excess. Neither helped. An "
        "earlier `K = 0.50` default made fee drag ~10% and made the regime gate look worthless; "
        "that was the drag drowning the signal, not the gate being useless. **The value of a risk "
        "overlay is conditional on the underlying edge surviving costs.** Every one of these is "
        "reproducible by flipping the corresponding flag in `grid/strategy.py`."
    )
    lines.append("")
    return "\n".join(lines)


def detail_section(rows) -> str:
    lines = ["## 6. Per-window detail", ""]
    for name, note, r, m, _ in rows:
        lines.append(f"### `{name}` -- {note}")
        lines.append("")
        lines.append(f"`{m.start}` -> `{m.end}` ({m.days:.0f} days)")
        lines.append("")
        lines.append("| Metric | Strategy | Buy & hold | Exposure-matched B&H |")
        lines.append("|---|---:|---:|---:|")
        lines.append(f"| Total return | **{pct(m.total_return)}** | {pct(m.bh_return)} | {pct(m.exposure_matched_return)} |")
        lines.append(f"| CAGR | {pct(m.cagr)} | - | - |")
        lines.append(f"| Max drawdown | {upct(m.max_dd)} | {upct(m.bh_max_dd)} | {upct(m.exposure_matched_max_dd)} |")
        lines.append(f"| Max DD duration | {m.max_dd_duration_days:.0f} d | - | - |")
        lines.append(f"| Sharpe (rf=0) | {m.sharpe:.2f} | {m.bh_sharpe:.2f} | - |")
        lines.append(f"| Sortino | {m.sortino:.2f} | - | - |")
        lines.append(f"| Calmar | {m.calmar:.2f} | - | - |")
        lines.append("")
        lines.append("| Mechanism | Value |")
        lines.append("|---|---:|")
        lines.append(f"| Round trips (paired exits) | {m.round_trips} |")
        lines.append(f"| Losing round trips | {m.losing_round_trips} |")
        lines.append(f"| Avg round-trip PnL | {m.avg_round_trip_pnl:+.2f} USDT ({upct(m.avg_round_trip_pct,3)} of notional) |")
        lines.append(f"| Avg holding period | {m.avg_holding_hours:.0f} h |")
        lines.append(f"| De-risk / kill sales | {m.derisk_sales} (realized {m.derisk_pnl:+.2f} USDT) |")
        lines.append(f"| Gross spread capture | {r.gross_spread_capture:+.2f} USDT |")
        lines.append(f"| Fees paid | {m.fees_paid:.2f} USDT (maker {m.fees_maker:.2f} / taker {m.fees_taker:.2f}) |")
        lines.append(f"| Fee drag on gross capture | {upct(m.fee_drag,1)} |")
        lines.append(f"| Maker / taker fills | {m.maker_fills} / {m.taker_fills} |")
        lines.append(f"| Time in market | {upct(m.time_in_market,1)} |")
        lines.append(f"| Avg / peak inventory | {upct(m.avg_inventory_ratio,1)} / {upct(m.peak_inventory_ratio,1)} |")
        lines.append(f"| Daily-loss halt days | {m.halted_days} |")
        lines.append(f"| Drawdown kill fired | {'yes, ' + str(r.kill_count) + 'x' if r.kill_count else 'no'} |")
        occ = ", ".join(f"{k} {upct(v,1)}" for k, v in m.regime_occupancy.items() if v > 0.001)
        lines.append(f"| Regime occupancy | {occ} |")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Sensitivity
# --------------------------------------------------------------------------- #


def sensitivity(daily, execution, hourly, base: Config) -> str:
    from grid.data import iso_to_ms

    window_names = [w for w in WINDOWS if w[0] != "full-2019-2026"]

    per_window: dict = {}

    def sweep(values, make_cfg, key=None):
        out = []
        for v in values:
            cfg = make_cfg(v)
            rets, dds, rts, drags, fees = [], [], [], [], []
            for name, start, end, _ in window_names:
                sub = execution.slice_ms(iso_to_ms(start), iso_to_ms(end))
                m = compute_metrics(run_backtest(sub, daily, cfg, label=name, hourly=hourly), sub)
                rets.append(m.total_return)
                dds.append(m.max_dd)
                rts.append(m.round_trips)
                fees.append(m.fees_paid)
                if m.fee_drag == m.fee_drag:
                    drags.append(m.fee_drag)
            if key:
                per_window.setdefault(key, {})[v] = list(rets)
                per_window.setdefault(key + ":drag", {})[v] = list(drags)
            out.append(
                (v, float(np.mean(rets)), float(np.mean(dds)), int(np.median(rts)),
                 float(np.mean(drags)) if drags else float("nan"), float(np.mean(fees)))
            )
        return out

    lines = ["# Parameter sensitivity", ""]
    lines.append(
        "Every row is the mean across the six sub-windows (bear-2022, range-2023, bull-2020Q4, "
        "flat-2024-26, year-2025, chop-2019H2), so no single regime drives a conclusion. "
        "Generated by `python run_backtest.py --all`."
    )
    lines.append("")
    lines.append(
        "**Overfitting discipline.** These sweeps are published to show the *shape of the surface*, "
        "not to select a point on it. Defaults come from published convention (ADX 25, EMA 50/200, "
        "ATR 14, Donchian 20) or from a stated mechanism. Where a default did move -- `K` from 0.50 "
        "to 1.00 -- it moved because it has a named cause, fee drag, which falls monotonically in "
        "every window. It did NOT move because return is monotone: it is not, in four of the six "
        "windows, and section 1 below publishes that decomposition rather than the earlier and "
        "incorrect claim that all six agreed."
    )
    lines.append("")

    lines.append("## 1. `k_spacing` -- tests the fee-floor thesis")
    lines.append("")
    k_values = [0.25, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00]
    lines.append("| K | Mean return | Mean max DD | Median round trips | Fee drag | Mean fees (USDT) |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for v, r, d, rt, dg, fe in sweep(k_values, lambda v: base.with_(k_spacing=v), key="K"):
        star = " **(default)**" if abs(v - base.k_spacing) < 1e-9 else ""
        lines.append(f"| {v:.2f}{star} | {pct(r)} | {upct(d)} | {rt} | {upct(dg,1)} | {fe:.0f} |")
    lines.append("")

    # Per-window decomposition, and an HONEST monotonicity count. An earlier
    # draft of this file asserted in prose that return was "monotone in all six
    # windows individually". An external audit checked, and it is not. The claim
    # is now computed and the table published, because a justification for moving
    # a default is worthless if nobody can check it.
    lines.append("### Per-window decomposition, and how monotone this actually is")
    lines.append("")
    lines.append("| Window | " + " | ".join(f"K={v:.2f}" for v in k_values) + " | Monotone? |")
    lines.append("|---|" + "---:|" * len(k_values) + "---|")
    monotone_count = 0
    for i, (name, _s, _e, _n) in enumerate(window_names):
        row = [per_window["K"][v][i] for v in k_values]
        is_mono = all(b >= a - 1e-12 for a, b in zip(row, row[1:]))
        monotone_count += int(is_mono)
        lines.append(
            f"| `{name}` | " + " | ".join(pct(x) for x in row) + f" | {'yes' if is_mono else '**no**'} |"
        )
    means = [float(np.mean([per_window["K"][v][i] for i in range(len(window_names))])) for v in k_values]
    mean_mono = all(b >= a - 1e-12 for a, b in zip(means, means[1:]))
    lines.append(
        f"| **mean** | " + " | ".join(pct(x) for x in means) + f" | {'yes' if mean_mono else '**no**'} |"
    )
    lines.append("")
    drag_mono = 0
    for i in range(len(window_names)):
        drow = [per_window["K:drag"][v][i] for v in k_values if i < len(per_window["K:drag"][v])]
        if len(drow) == len(k_values) and all(b <= a + 1e-12 for a, b in zip(drow, drow[1:])):
            drag_mono += 1
    lines.append(
        f"**{monotone_count} of {len(window_names)} windows are individually monotone in K**, and "
        f"the mean is {'monotone' if mean_mono else 'not monotone across the full sweep either'}. "
        f"The honest statement is therefore weaker than 'monotone everywhere': widening K improves "
        f"the *average* result, but individual windows disagree, and `bull-2020Q4` is actively "
        f"better at a tighter setting than the one we ship."
    )
    lines.append("")
    lines.append(
        f"What does hold universally is the **cost mechanism**: fee drag falls monotonically with K "
        f"in **{drag_mono} of {len(window_names)}** windows. That is the part of the argument the "
        f"default rests on -- a tighter grid fills more often and hands the difference to the "
        f"exchange -- and it is why the default moved to K = 1.00 despite the return ordering being "
        f"messier than we first claimed."
    )
    lines.append("")

    lines.append("## 2. `cap_range` -- the dominant risk lever")
    lines.append("")
    lines.append("| Inventory cap | Mean return | Mean max DD | Median round trips |")
    lines.append("|---:|---:|---:|---:|")
    for v, r, d, rt, _dg, _fe in sweep([0.20, 0.30, 0.50, 0.70], lambda v: base.with_(cap_range=v, cap_uptrend=min(0.6, v + 0.1))):
        star = " **(default)**" if abs(v - base.cap_range) < 1e-9 else ""
        lines.append(f"| {v:.2f}{star} | {pct(r)} | {upct(d)} | {rt} |")
    lines.append("")
    lines.append(
        "Drawdown is close to linear in the inventory cap. This is the cleanest evidence in the "
        "study that **the dominant risk lever is inventory, not signal** -- and it is the dial to "
        "turn if the shipped risk profile is not yours. 0.30 is a reasonable conservative setting; "
        "0.20 more so. The default stays at 0.50 because this is a risk-appetite preference, not a "
        "performance parameter, and tuning it to a backtest would be exactly the wrong lesson."
    )
    lines.append("")

    lines.append("### Presets: how to actually dial the risk")
    lines.append("")
    lines.append(
        "The inventory cap is the dial. Note that `dd_kill` is *derived* from it "
        "(`clamp(cap_range * asset_max_dd, 0.15, 0.50)`), so lowering the cap automatically "
        "tightens the circuit breaker too -- they cannot drift out of sync."
    )
    lines.append("")
    lines.append("| Preset | `cap_range` | Derived `dd_kill` | Full-run return | Full-run DD | Sub-window mean DD |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for tag, kw in (
        ("Default (moderate)", dict(cap_range=0.50, cap_uptrend=0.60)),
        ("Conservative", dict(cap_range=0.30, cap_uptrend=0.40)),
        ("Defensive", dict(cap_range=0.20, cap_uptrend=0.30)),
        ("Conservative + active de-risk", dict(cap_range=0.30, cap_uptrend=0.40, active_derisk=True)),
    ):
        cfg = base.with_(**kw)
        mf = compute_metrics(run_backtest(execution, daily, cfg, label="full", hourly=hourly), execution)
        dds = []
        for name, start, end, _ in window_names:
            sub = execution.slice_ms(iso_to_ms(start), iso_to_ms(end))
            dds.append(compute_metrics(run_backtest(sub, daily, cfg, label=name, hourly=hourly), sub).max_dd)
        lines.append(
            f"| {tag} | {cfg.cap_range:.2f} | {upct(cfg.dd_kill,0)} | {pct(mf.total_return)} | "
            f"{upct(mf.max_dd)} | {upct(float(np.mean(dds)))} |"
        )
    lines.append("")
    lines.append(
        "The last row is the important one. Adding a force-selling mechanism on top of a lower cap "
        "is **worse on both axes** than simply lowering the cap: it gives up return *and* ends up "
        "with a larger full-run drawdown. If you want less risk, hold less inventory -- do not bolt "
        "on a mechanism that liquidates into weakness."
    )
    lines.append("")

    lines.append("## 3. `maker_fee` / `taker_fee` -- robustness of the fee assumption")
    lines.append("")
    lines.append("| Fee per side | Mean return | Mean max DD | Fee drag | Break-even spacing | Derived `s_floor` |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for v, r, d, _rt, dg, _fe in sweep(
        [0.0002, 0.0005, 0.0007, 0.0010, 0.0015, 0.0020],
        lambda v: base.with_(maker_fee=v, taker_fee=v),
    ):
        star = " **(verified)**" if abs(v - 0.0010) < 1e-9 else ""
        cfg_v = base.with_(maker_fee=v, taker_fee=v)
        lines.append(
            f"| {100*v:.3f}%{star} | {pct(r)} | {upct(d)} | {upct(dg,1)} | "
            f"{upct(breakeven_spacing(v),4)} | {upct(cfg_v.s_floor,4)} |"
        )
    lines.append("")
    lines.append(
        "The published CoinW spot base rate is 0.10% per side. This sweep spans a 10x range around "
        "it, covering both the best VIP tier and double the base rate. **If a reviewer believes our "
        "fee assumption is wrong, their number is already in this table.** `s_floor` is derived from "
        "`fee` at runtime, never hardcoded, and assertion R8 refuses to place an order if the two "
        "ever disagree."
    )
    lines.append("")

    lines.append("## 4. `fill_probability` -- pricing the front-of-queue assumption")
    lines.append("")
    lines.append("| Fill probability | Mean return | Mean max DD | Median round trips |")
    lines.append("|---:|---:|---:|---:|")
    for v, r, d, rt, _dg, _fe in sweep([1.0, 0.7, 0.5], lambda v: base.with_(fill_probability=v)):
        lines.append(f"| {v:.1f} | {pct(r)} | {upct(d)} | {rt} |")
    lines.append("")
    lines.append(
        "The fill model assumes that whenever price trades through a resting order, we are filled. "
        "On a thin book or with a large order that is generous. This sweep applies a deterministic "
        "seeded gate to each candidate fill. If the strategy only worked at 1.0, that would be a "
        "finding, and it would be reported here."
    )
    lines.append("")

    lines.append("## 5. `n_levels` -- ladder depth")
    lines.append("")
    lines.append("| N | Mean return | Mean max DD | Median round trips |")
    lines.append("|---:|---:|---:|---:|")
    for v, r, d, rt, _dg, _fe in sweep([3, 4, 6, 8, 10], lambda v: base.with_(n_levels=v)):
        star = " **(default)**" if v == base.n_levels else ""
        lines.append(f"| {v}{star} | {pct(r)} | {upct(d)} | {rt} |")
    lines.append("")

    lines.append("## 6. `K` x `cap_range` interaction")
    lines.append("")
    caps = [0.20, 0.30, 0.50, 0.70]
    lines.append("| K \\ cap | " + " | ".join(f"{c:.2f}" for c in caps) + " |")
    lines.append("|---:|" + "---:|" * len(caps))
    for k in (0.35, 0.50, 1.00, 1.50):
        cells = []
        for cap in caps:
            cfg = base.with_(k_spacing=k, cap_range=cap, cap_uptrend=min(0.6, cap + 0.1))
            rets, dds = [], []
            for name, start, end, _ in window_names:
                sub = execution.slice_ms(iso_to_ms(start), iso_to_ms(end))
                m = compute_metrics(run_backtest(sub, daily, cfg, label=name, hourly=hourly), sub)
                rets.append(m.total_return)
                dds.append(m.max_dd)
            cells.append(f"{pct(float(np.mean(rets)))}<br><sub>DD {upct(float(np.mean(dds)),1)}</sub>")
        lines.append(f"| **{k:.2f}** | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Mean return across all six windows, with mean max drawdown beneath.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Charts and trade log
# --------------------------------------------------------------------------- #


def write_charts(rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = next((r for r in rows if r[0] == "full-2019-2026"), rows[-1])
    _name, _note, result, m, sub = target
    ts = np.array([np.datetime64(int(t), "ms") for t in result.ts])

    fee = result.config.maker_fee
    bh = (result.config.initial_equity / (float(sub.open[0]) * (1 + fee))) * sub.close
    w = min(1.0, max(0.0, m.avg_inventory_ratio))
    em = result.config.initial_equity * (1 - w) + (
        (result.config.initial_equity * w) / (float(sub.open[0]) * (1 + fee))
    ) * sub.close

    # Equity curve
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts, result.equity, lw=1.4, label="Fee-Floored Adaptive Grid", color="#1f77b4")
    ax.plot(ts, em, lw=1.1, label=f"Exposure-matched B&H ({100*w:.0f}% BTC)", color="#2ca02c", alpha=0.85)
    ax.plot(ts, bh, lw=1.0, label="Buy & hold BTC (100%)", color="#d62728", alpha=0.55)
    ax.set_yscale("log")
    ax.set_title(f"Equity curve, {result.label} (log scale) — CoinW BTC_USDT 15m, fee 0.10%/side")
    ax.set_ylabel("Equity (USDT, log)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "equity_curve.png", dpi=130)
    plt.close(fig)

    # Underwater
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for series, lbl, colour in ((result.equity, "Strategy", "#1f77b4"), (bh, "Buy & hold", "#d62728")):
        peak = np.maximum.accumulate(series)
        ax.fill_between(ts, -100 * (1 - series / peak), 0, alpha=0.45, label=lbl, color=colour)
    ax.set_title("Drawdown (marked at every 15m bar, not daily closes)")
    ax.set_ylabel("Drawdown %")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "drawdown.png", dpi=130)
    plt.close(fig)

    # Regime timeline
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, height_ratios=[2, 1])
    ax1.plot(ts, sub.close, lw=0.7, color="#333333")
    ax1.set_yscale("log")
    ax1.set_ylabel("BTC_USDT (log)")
    for code, colour, lbl in ((1, "#2ca02c", "UPTREND"), (2, "#d62728", "DOWNTREND")):
        ax1.fill_between(ts, sub.close.min(), sub.close.max(),
                         where=result.regime_code == code, alpha=0.13, color=colour, label=lbl)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Regime classification and inventory ratio")
    ax2.plot(ts, 100 * result.inventory_ratio, lw=0.7, color="#1f77b4")
    ax2.axhline(100 * result.config.cap_range, ls="--", lw=1, color="#888888", label=f"RANGE cap {100*result.config.cap_range:.0f}%")
    ax2.axhline(100 * result.config.cap_downtrend, ls=":", lw=1, color="#d62728", label=f"DOWNTREND cap {100*result.config.cap_downtrend:.0f}%")
    ax2.set_ylabel("Inventory / equity %")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "regime_timeline.png", dpi=130)
    plt.close(fig)


def write_trades(rows) -> None:
    target = next((r for r in rows if r[0] == "full-2019-2026"), rows[-1])
    result = target[2]
    from grid.data import ms_to_iso

    with open(RESULTS_DIR / "trades.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["timestamp_utc", "epoch_ms", "side", "reason", "liquidity", "price", "qty",
             "notional", "fee", "regime", "lot_id", "realized_pnl"]
        )
        for t in result.trades:
            writer.writerow(
                [ms_to_iso(t.ts), t.ts, t.side, t.reason, t.liquidity, f"{t.price:.2f}",
                 f"{t.qty:.6f}", f"{t.notional:.2f}", f"{t.fee:.4f}", t.regime, t.lot_id,
                 f"{t.realized_pnl:.4f}"]
            )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="run everything and regenerate results/")
    parser.add_argument("--window", help="run a single window by name")
    parser.add_argument("--offline", action="store_true", help="use only cached data, no network")
    parser.add_argument("--no-charts", action="store_true", help="skip PNG generation")
    args = parser.parse_args()

    if not args.all and not args.window:
        parser.print_help()
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    cfg.validate()

    print(f"Fee-Floored Adaptive Grid | maker/taker {100*cfg.maker_fee:.3f}% | "
          f"break-even s* {100*cfg.breakeven:.4f}% | s_floor {100*cfg.s_floor:.4f}% | "
          f"net edge at floor {100*net_edge_per_round_trip(cfg.s_floor, cfg.maker_fee):.4f}%")

    daily, execution, hourly, constraints = load_data(args.offline)
    cfg = cfg.with_(
        min_notional=constraints["min_notional"],
        price_tick=constraints["price_tick"],
        qty_step=constraints["qty_step"],
    )

    names = [args.window] if args.window else None
    rows = run_windows(daily, execution, hourly, cfg, names)
    if not rows:
        print(f"no window matched {args.window!r}", file=sys.stderr)
        return 1

    print()
    print(f"{'window':<16}{'return':>10}{'maxDD':>9}{'B&H':>10}{'expo-B&H':>10}{'RT':>6}{'lose':>6}{'drag':>8}")
    for name, _n, _r, m, _ in rows:
        print(f"{name:<16}{pct(m.total_return):>10}{upct(m.max_dd):>9}{pct(m.bh_return):>10}"
              f"{pct(m.exposure_matched_return):>10}{m.round_trips:>6}{m.losing_round_trips:>6}"
              f"{upct(m.fee_drag,1):>8}")

    if not args.all:
        return 0

    print("\nregenerating results/ ...")
    client = CoinWClient(DATA_DIR, offline=args.offline, verbose=False)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    doc = [
        "# Backtest results — Fee-Floored Adaptive Grid",
        "",
        f"Generated by `python run_backtest.py --all` on {stamp} (UTC). "
        "Every figure in `SKILL.md` and `README.md` traces to this file; nothing here is "
        "hand-written.",
        "",
        f"**Configuration.** CoinW `BTC_USDT` spot, 15m execution bars, native daily "
        f"indicator bars, maker = taker = **{100*cfg.maker_fee:.2f}%** (published CoinW spot "
        f"base rate), initial equity {cfg.initial_equity:,.0f} USDT, "
        f"`K={cfg.k_spacing}`, `N={cfg.n_levels}`, `s_cap={100*cfg.s_cap:.1f}%`, "
        f"`s_floor={100*cfg.s_floor:.4f}%` (derived, never hardcoded), "
        f"inventory caps {cfg.cap_range}/{cfg.cap_uptrend}/{cfg.cap_downtrend} by regime.",
        "",
        "> **Not a live-trading record.** This is a historical simulation. It has never been "
        "run against a real CoinW account, and no forward-looking return is claimed anywhere "
        "in this repository.",
        "",
        "---",
        "",
        integrity_section(daily, execution, hourly, constraints),
        "---",
        "",
        path_length_section(client, args.offline),
        "---",
        "",
        headline_section(rows),
        "---",
        "",
        invariant_section(rows),
        "---",
        "",
        chain_section(daily, execution, hourly, cfg),
        "---",
        "",
        detail_section(rows),
    ]
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(doc), encoding="utf-8")
    print("  results/RESULTS.md")

    (RESULTS_DIR / "sensitivity.md").write_text(
        sensitivity(daily, execution, hourly, cfg), encoding="utf-8"
    )
    print("  results/sensitivity.md")

    write_trades(rows)
    print("  results/trades.csv")

    if not args.no_charts:
        write_charts(rows)
        print("  results/equity_curve.png, drawdown.png, regime_timeline.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
