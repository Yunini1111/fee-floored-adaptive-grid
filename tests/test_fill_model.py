"""Fill model F1-F9, tested on hand-built bars.

A grid's return is roughly linear in its fill count, so these rules ARE the
backtest's credibility. Each one is pinned here so it cannot be quietly loosened.
"""

import numpy as np
import pytest

from grid.data import Candles
from grid.engine import run_backtest
from grid.strategy import Config

DAY_MS = 86_400_000
BAR_MS = 900_000


def make_daily(closes, start_ms=0, atr_frac=0.04):
    """Daily bars with a controlled range so ATR% is predictable."""
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    return Candles(
        pair="TEST",
        period=86400,
        ts=np.arange(n, dtype=np.int64) * DAY_MS + start_ms,
        open=closes.copy(),
        high=closes * (1 + atr_frac / 2),
        low=closes * (1 - atr_frac / 2),
        close=closes.copy(),
        volume=np.ones(n),
    )


def make_exec(bars, start_ms):
    """bars: list of (open, high, low, close)."""
    arr = np.asarray(bars, dtype=float)
    n = len(bars)
    return Candles(
        pair="TEST",
        period=900,
        ts=np.arange(n, dtype=np.int64) * BAR_MS + start_ms,
        open=arr[:, 0],
        high=arr[:, 1],
        low=arr[:, 2],
        close=arr[:, 3],
        volume=np.ones(n),
    )


def flat_config(**kw):
    base = dict(
        min_notional=5.0,
        price_tick=0.01,
        qty_step=0.000001,
        initial_equity=10_000.0,
        n_levels=1,
        regime_gate=False,
        active_derisk=False,
        dd_kill_override=0.99,
        daily_loss_limit=0.99,
    )
    base.update(kw)
    return Config(**base)


# --------------------------------------------------------------------------- #
# F3 / F4 -- the market must trade THROUGH us
# --------------------------------------------------------------------------- #


def test_touching_the_limit_exactly_is_not_a_fill():
    """`low == p` must NOT fill. This single rule prevents a large class of
    manufactured returns: on a wick that prints exactly at our price we have no
    evidence our order was ahead of the queue."""
    daily = make_daily([100.0] * 260)
    start = 260 * DAY_MS

    # Find the level the strategy will place, then build a bar whose low is exactly it.
    from grid.strategy import SignalEngine

    sig = SignalEngine(daily, flat_config()).signal_for(start)
    level = sig.buy_levels[0]

    exact = make_exec([(100.0, 100.0, 100.0, 100.0)] * 3 + [(100.0, 100.0, level, 100.0)] * 3, start)
    through = make_exec(
        [(100.0, 100.0, 100.0, 100.0)] * 3 + [(100.0, 100.0, level * 0.99, 100.0)] * 3, start
    )

    r_exact = run_backtest(exact, daily, flat_config(), label="exact")
    r_through = run_backtest(through, daily, flat_config(), label="through")

    assert len([t for t in r_exact.trades if t.side == "BUY"]) == 0
    assert len([t for t in r_through.trades if t.side == "BUY"]) >= 1


def test_a_bar_strictly_through_the_level_does_fill():
    daily = make_daily([100.0] * 260)
    start = 260 * DAY_MS
    bars = [(100.0, 100.0, 100.0, 100.0)] * 2 + [(100.0, 100.0, 50.0, 60.0)] * 2
    result = run_backtest(make_exec(bars, start), daily, flat_config(), label="t")
    assert any(t.side == "BUY" for t in result.trades)


# --------------------------------------------------------------------------- #
# F1 -- order lag
# --------------------------------------------------------------------------- #


def test_no_fill_on_the_bar_that_generated_the_order():
    """Orders derived at the daily rollover are live from the NEXT bar."""
    daily = make_daily([100.0] * 260)
    start = 260 * DAY_MS
    # The very first bar of the day plunges. It must not fill anything.
    bars = [(100.0, 100.0, 1.0, 100.0), (100.0, 100.0, 100.0, 100.0)]
    result = run_backtest(make_exec(bars, start), daily, flat_config(), label="t")
    assert [t for t in result.trades if t.ts == start] == []


# --------------------------------------------------------------------------- #
# F5 / F6 -- no price improvement, gap-through pays taker
# --------------------------------------------------------------------------- #


def test_fill_price_is_the_limit_never_the_bar_low():
    """Even a bar that gaps far through us fills at our price, not a better one."""
    daily = make_daily([100.0] * 260)
    start = 260 * DAY_MS
    bars = [(100.0, 100.0, 100.0, 100.0)] * 2 + [(100.0, 100.0, 1.0, 2.0)] * 2
    result = run_backtest(make_exec(bars, start), daily, flat_config(), label="t")
    buys = [t for t in result.trades if t.side == "BUY"]
    assert buys
    assert buys[0].price > 50.0  # nowhere near the 1.0 low


def test_gap_through_open_is_charged_the_taker_fee():
    """A post-only order would have been REJECTED if the bar opened through it,
    so a real agent would have had to chase. Worst price and worst fee."""
    daily = make_daily([100.0] * 260)
    start = 260 * DAY_MS
    cfg = flat_config(maker_fee=0.0010, taker_fee=0.0050)
    # Bar 2 opens far below the level -> the resting buy was never placeable.
    bars = [(100.0, 100.0, 100.0, 100.0)] * 2 + [(50.0, 60.0, 40.0, 55.0)] * 2
    result = run_backtest(make_exec(bars, start), daily, cfg, label="t")
    buys = [t for t in result.trades if t.side == "BUY"]
    assert buys
    assert buys[0].liquidity == "TAKER"
    assert result.fees_taker > 0


def test_normal_fill_inside_the_bar_is_maker():
    daily = make_daily([100.0] * 260)
    start = 260 * DAY_MS
    bars = [(100.0, 100.0, 100.0, 100.0)] * 2 + [(100.0, 101.0, 80.0, 95.0)] * 2
    result = run_backtest(make_exec(bars, start), daily, flat_config(), label="t")
    buys = [t for t in result.trades if t.side == "BUY"]
    assert buys
    assert buys[0].liquidity == "MAKER"


# --------------------------------------------------------------------------- #
# F7 -- one fill per level per bar
# --------------------------------------------------------------------------- #


def test_a_level_cannot_fill_twice_in_one_day():
    daily = make_daily([100.0] * 260)
    start = 260 * DAY_MS
    bars = [(100.0, 100.0, 100.0, 100.0)] + [(100.0, 120.0, 50.0, 100.0)] * 20
    result = run_backtest(make_exec(bars, start), daily, flat_config(n_levels=1), label="t")
    day_buys = [t for t in result.trades if t.side == "BUY"]
    assert len(day_buys) <= 1


# --------------------------------------------------------------------------- #
# F2 -- atomic bar evaluation
# --------------------------------------------------------------------------- #


def test_lot_book_is_snapshotted_at_the_bar_open():
    """F2's other half: a same-bar sell must not free a lot slot or inventory
    headroom for a same-bar buy. Only the cash half of F2 was tested originally,
    and the lot-book half was silently unguarded.

    Constructed so the R3 lot ceiling is the binding constraint: with
    max_open_lots=1, a bar on which the single open lot's exit fills AND a buy
    level is crossed must NOT open a replacement lot on that same bar.
    """
    daily = make_daily([100.0] * 260)
    start = 260 * DAY_MS
    cfg = flat_config(n_levels=1, max_open_lots=1)

    from grid.strategy import SignalEngine

    level = SignalEngine(daily, cfg).signal_for(start).buy_levels[0]

    bars = [
        (100.0, 100.0, 100.0, 100.0),  # rollover, orders armed for next bar
        (100.0, 100.0, level * 0.98, 99.0),  # buy fills
        # Wide bar: high clears the lot's exit AND low re-crosses the buy level.
        (99.0, 200.0, level * 0.90, 99.0),
        (99.0, 200.0, level * 0.90, 99.0),
    ]
    result = run_backtest(make_exec(bars, start), daily, cfg, label="t")

    by_bar: dict[int, list[str]] = {}
    for t in result.trades:
        by_bar.setdefault(t.ts, []).append(t.side)
    for ts, sides in by_bar.items():
        assert not ("SELL" in sides and "BUY" in sides), (
            f"bar {ts} both closed and opened a lot at max_open_lots=1: "
            f"the sell freed a slot for the buy within the same bar"
        )


def test_cash_never_goes_negative():
    """Cash freed by a sell on bar t must not fund a buy on bar t."""
    rng = np.random.default_rng(2)
    daily = make_daily(100 * np.exp(np.cumsum(rng.normal(scale=0.02, size=400))))
    start = 400 * DAY_MS
    n = 3000
    walk = 100 * np.exp(np.cumsum(rng.normal(scale=0.004, size=n)))
    bars = [(p, p * 1.006, p * 0.994, p) for p in walk]
    cfg = flat_config(n_levels=6, initial_equity=1_000.0)
    result = run_backtest(make_exec(bars, start), daily, cfg, label="t")

    cash = cfg.initial_equity
    for t in sorted(result.trades, key=lambda x: x.ts):
        cash += (t.notional - t.fee) if t.side == "SELL" else -(t.notional + t.fee)
        assert cash > -1e-6, f"cash went negative at {t.ts}"


# --------------------------------------------------------------------------- #
# Exchange constraints
# --------------------------------------------------------------------------- #


def test_no_order_below_the_exchange_minimum_notional():
    rng = np.random.default_rng(4)
    daily = make_daily(100 * np.exp(np.cumsum(rng.normal(scale=0.02, size=300))))
    start = 300 * DAY_MS
    walk = 100 * np.exp(np.cumsum(rng.normal(scale=0.005, size=1500)))
    bars = [(p, p * 1.01, p * 0.99, p) for p in walk]
    cfg = flat_config(min_notional=5.0, initial_equity=200.0, n_levels=6)
    result = run_backtest(make_exec(bars, start), daily, cfg, label="t")
    for t in result.trades:
        if t.side == "BUY":
            assert t.notional >= cfg.min_notional - 1e-9


def test_quantities_respect_the_exchange_step():
    rng = np.random.default_rng(6)
    daily = make_daily(100 * np.exp(np.cumsum(rng.normal(scale=0.02, size=300))))
    start = 300 * DAY_MS
    walk = 100 * np.exp(np.cumsum(rng.normal(scale=0.005, size=1200)))
    bars = [(p, p * 1.01, p * 0.99, p) for p in walk]
    cfg = flat_config(qty_step=0.001, n_levels=4)
    result = run_backtest(make_exec(bars, start), daily, cfg, label="t")
    for t in result.trades:
        assert abs(round(t.qty / cfg.qty_step) - t.qty / cfg.qty_step) < 1e-6


def test_fill_probability_gate_reduces_fills_monotonically():
    rng = np.random.default_rng(8)
    daily = make_daily(100 * np.exp(np.cumsum(rng.normal(scale=0.02, size=400))))
    start = 400 * DAY_MS
    walk = 100 * np.exp(np.cumsum(rng.normal(scale=0.004, size=4000)))
    bars = [(p, p * 1.006, p * 0.994, p) for p in walk]
    ex = make_exec(bars, start)
    counts = [
        len(run_backtest(ex, daily, flat_config(n_levels=6, fill_probability=p), label="t").trades)
        for p in (1.0, 0.5)
    ]
    assert counts[0] > counts[1]
