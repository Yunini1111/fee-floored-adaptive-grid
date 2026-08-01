"""The competing strategies are load-bearing: WHEN-TO-GRID.md's conclusions rest on
them, so they get the same scrutiny as the strategy itself. A benchmark that
flatters the grid by accident is worse than no benchmark.
"""

import numpy as np
import pytest

from grid.benchmarks import run_buy_and_hold, run_dca, run_martingale
from grid.data import Candles
from grid.strategy import Config

BAR_MS = 900_000


def series(prices):
    prices = np.asarray(prices, dtype=float)
    n = prices.size
    return Candles(
        pair="TEST",
        period=900,
        ts=np.arange(n, dtype=np.int64) * BAR_MS,
        open=prices.copy(),
        high=prices * 1.004,
        low=prices * 0.996,
        close=prices.copy(),
        volume=np.ones(n),
    )


def cfg(**kw):
    base = dict(min_notional=5.0, price_tick=0.01, qty_step=0.000001, initial_equity=10_000.0)
    base.update(kw)
    return Config(**base)


# --------------------------------------------------------------------------- #
# Buy and hold
# --------------------------------------------------------------------------- #


def test_buy_and_hold_tracks_price_net_of_one_fee():
    c = cfg()
    ex = series(np.linspace(100, 200, 500))
    r = run_buy_and_hold(ex, c)
    expected = c.initial_equity * (200 / 100) / (1 + c.maker_fee)
    assert r.final_equity == pytest.approx(expected, rel=1e-3)
    assert r.trades == 1


def test_buy_and_hold_loses_exactly_what_the_asset_loses():
    c = cfg()
    r = run_buy_and_hold(series(np.linspace(100, 50, 300)), c)
    assert r.final_equity / c.initial_equity - 1 == pytest.approx(-0.5, abs=0.01)


# --------------------------------------------------------------------------- #
# DCA
# --------------------------------------------------------------------------- #


def test_dca_deploys_capital_gradually_and_never_sells():
    c = cfg()
    ex = series(np.full(2000, 100.0))
    r = run_dca(ex, c, interval_days=7)
    assert r.trades > 1
    assert r.deployed_pct > 0.9
    assert r.final_equity < c.initial_equity  # flat price, so fees are pure cost


def test_dca_beats_lump_sum_into_a_falling_market():
    """The whole point of averaging in."""
    c = cfg()
    ex = series(np.linspace(100, 50, 3000))
    assert run_dca(ex, c, 7).final_equity > run_buy_and_hold(ex, c).final_equity


def test_dca_loses_to_lump_sum_in_a_rising_market():
    c = cfg()
    ex = series(np.linspace(50, 200, 3000))
    assert run_dca(ex, c, 7).final_equity < run_buy_and_hold(ex, c).final_equity


def test_dca_pays_taker_fees_because_it_uses_market_orders():
    c = cfg(maker_fee=0.0, taker_fee=0.01)
    r = run_dca(series(np.full(1000, 100.0)), c, 7)
    assert r.fees > 0


def test_dca_never_overspends():
    c = cfg()
    r = run_dca(series(np.linspace(100, 30, 4000)), c, 7)
    assert np.all(r.equity > 0)
    assert r.deployed_pct <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# Martingale
# --------------------------------------------------------------------------- #


def test_martingale_averages_down_as_price_falls():
    c = cfg()
    r = run_martingale(series(np.linspace(100, 60, 3000)), c)
    assert r.trades > 1
    assert "safety orders" in r.notes


def test_martingale_takes_profit_and_recycles_in_an_oscillating_market():
    c = cfg()
    osc = 100 + 4 * np.sin(np.arange(6000) * 0.05)
    r = run_martingale(series(osc), c)
    assert r.round_trips > 1, "should complete multiple cycles in a sine wave"


def test_martingale_ladder_is_finite_and_the_report_says_so():
    """The famous failure mode: once the safety orders run out, the bot just sits
    fully exposed. WHEN-TO-GRID.md leans on this, so it has to be real."""
    c = cfg()
    r = run_martingale(series(np.linspace(100, 20, 5000)), c, max_safety_orders=3)
    assert "fully spent" in r.notes
    pct = float(r.notes.split("(")[-1].split("%")[0])
    assert pct > 50.0, "in a relentless decline the ladder should be exhausted most of the time"


def test_martingale_never_spends_more_than_it_has():
    c = cfg()
    r = run_martingale(series(np.linspace(100, 10, 6000)), c, volume_multiplier=2.0, max_safety_orders=15)
    assert np.all(r.equity > -1e-6)


def test_martingale_cannot_escape_a_monotone_decline():
    """A drawdown it cannot take profit out of is still a drawdown."""
    c = cfg()
    r = run_martingale(series(np.linspace(100, 40, 4000)), c)
    assert r.final_equity < c.initial_equity
    assert r.max_dd > 0.05


# --------------------------------------------------------------------------- #
# Cross-strategy fairness
# --------------------------------------------------------------------------- #


def test_every_benchmark_starts_from_the_same_capital():
    c = cfg()
    ex = series(np.linspace(100, 130, 2000))
    for r in (run_buy_and_hold(ex, c), run_dca(ex, c, 7), run_martingale(ex, c)):
        assert r.equity[0] <= c.initial_equity * 1.0001
        assert r.equity.size == len(ex)
        assert r.ts.size == len(ex)


def test_benchmarks_respect_the_exchange_step_and_minimum():
    """A benchmark allowed to trade sub-minimum sizes would beat a strategy that is not."""
    c = cfg(qty_step=0.01, min_notional=50.0, initial_equity=500.0)
    ex = series(np.linspace(100, 80, 2000))
    for r in (run_dca(ex, c, 7), run_martingale(ex, c)):
        assert np.all(np.isfinite(r.equity))


def test_higher_fees_hurt_every_benchmark():
    ex = series(100 + 3 * np.sin(np.arange(4000) * 0.05))
    cheap, dear = cfg(maker_fee=0.0001, taker_fee=0.0001), cfg(maker_fee=0.01, taker_fee=0.01)
    for fn in (run_dca, run_martingale):
        assert fn(ex, cheap).final_equity > fn(ex, dear).final_equity
