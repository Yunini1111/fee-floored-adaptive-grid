"""Look-ahead is the failure mode that makes a backtest worthless while looking great.

The test is blunt on purpose: perturb every bar after time T by a large factor and
assert that not one decision at or before T changes. If any future information
leaks into a decision, a +/-30% shock to the future will move something.
"""

import numpy as np
import pytest

from grid.data import Candles
from grid.engine import run_backtest
from grid.strategy import Config, SignalEngine

DAY_MS = 86_400_000
BAR_MS = 900_000


def synthetic(seed=42, days=420, bars=6000):
    rng = np.random.default_rng(seed)
    dcloses = 100 * np.exp(np.cumsum(rng.normal(scale=0.02, size=days)))
    daily = Candles(
        pair="TEST",
        period=86400,
        ts=np.arange(days, dtype=np.int64) * DAY_MS,
        open=dcloses.copy(),
        high=dcloses * 1.02,
        low=dcloses * 0.98,
        close=dcloses.copy(),
        volume=np.ones(days),
    )
    start = days * DAY_MS
    walk = float(dcloses[-1]) * np.exp(np.cumsum(rng.normal(scale=0.004, size=bars)))
    execution = Candles(
        pair="TEST",
        period=900,
        ts=np.arange(bars, dtype=np.int64) * BAR_MS + start,
        open=walk.copy(),
        high=walk * 1.006,
        low=walk * 0.994,
        close=walk.copy(),
        volume=np.ones(bars),
    )
    return daily, execution


def perturb_after(candles: Candles, index: int, factor: float) -> Candles:
    out = Candles(
        pair=candles.pair,
        period=candles.period,
        ts=candles.ts.copy(),
        open=candles.open.copy(),
        high=candles.high.copy(),
        low=candles.low.copy(),
        close=candles.close.copy(),
        volume=candles.volume.copy(),
    )
    for arr in (out.open, out.high, out.low, out.close):
        arr[index:] *= factor
    return out


@pytest.mark.parametrize("factor", [1.3, 0.7])
def test_future_execution_bars_cannot_change_past_trades(factor):
    daily, execution = synthetic()
    cut = 3000
    cut_ts = int(execution.ts[cut])
    cfg = Config(min_notional=5.0, price_tick=0.01, qty_step=0.000001)

    base = run_backtest(execution, daily, cfg, label="base")
    shocked = run_backtest(perturb_after(execution, cut, factor), daily, cfg, label="shock")

    def before(result):
        return [
            (t.ts, t.side, t.reason, round(t.price, 6), round(t.qty, 8))
            for t in result.trades
            if t.ts < cut_ts
        ]

    assert before(base) == before(shocked)
    assert len(before(base)) > 5, "test is vacuous if nothing traded before the cut"


@pytest.mark.parametrize("factor", [1.4, 0.6])
def test_future_daily_bars_cannot_change_past_trades(factor):
    daily, execution = synthetic(seed=7)
    cut_day = 380
    cut_ts = cut_day * DAY_MS
    cfg = Config(min_notional=5.0, price_tick=0.01, qty_step=0.000001)

    base = run_backtest(execution, daily, cfg, label="base")
    shocked = run_backtest(execution, perturb_after(daily, cut_day, factor), cfg, label="shock")

    # Every execution bar precedes the perturbed daily bars only if it is before
    # cut_ts; the signal for a day reads bars <= D-1, so trades before cut_ts must
    # be untouched.
    def before(result):
        return [(t.ts, t.side, round(t.price, 6)) for t in result.trades if t.ts < cut_ts]

    assert before(base) == before(shocked)


def test_signal_for_a_day_never_reads_that_days_own_bar():
    """The structural guarantee: signal_for(day) resolves bars strictly before it."""
    daily, _ = synthetic(seed=11)
    cfg = Config()
    engine = SignalEngine(daily, cfg)

    day_index = 300
    day_start = int(daily.ts[day_index])
    signal = engine.signal_for(day_start)
    assert signal is not None
    assert signal.signal_bar_ts < day_start
    assert signal.signal_bar_ts == int(daily.ts[day_index - 1])


def test_mutating_the_current_day_bar_does_not_change_its_own_signal():
    daily, _ = synthetic(seed=13)
    cfg = Config()
    day_index = 300
    day_start = int(daily.ts[day_index])

    before = SignalEngine(daily, cfg).signal_for(day_start)

    mutated = perturb_after(daily, day_index, 2.0)
    after = SignalEngine(mutated, cfg).signal_for(day_start)

    assert before.anchor == pytest.approx(after.anchor)
    assert before.spacing == pytest.approx(after.spacing)
    assert before.regime == after.regime
    assert before.buy_levels == after.buy_levels


def test_indicator_arrays_are_causal_at_the_series_level():
    daily, _ = synthetic(seed=17)
    cfg = Config()
    cut = 350
    base = SignalEngine(daily, cfg)
    shocked = SignalEngine(perturb_after(daily, cut, 1.5), cfg)

    for name in ("atr_pct", "adx", "ema_fast", "ema_slow", "anchor_ema", "dc_high", "dc_low"):
        a = np.nan_to_num(getattr(base, name)[:cut])
        b = np.nan_to_num(getattr(shocked, name)[:cut])
        np.testing.assert_allclose(a, b, rtol=1e-10, err_msg=f"{name} leaked future data")
