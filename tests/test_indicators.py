"""Indicators checked against hand-computed fixtures and known identities.

The point of hand-writing these was auditability, which is worthless without
tests that pin the definitions down. In particular: Wilder's RMA is NOT an EMA of
the same period, and getting that wrong is the most common way a published "ATR"
fails to reconcile with a chart.
"""

import numpy as np
import pytest

from grid import indicators as ind


# --------------------------------------------------------------------------- #
# EMA / RMA
# --------------------------------------------------------------------------- #


def test_ema_warmup_is_nan_then_seeds_with_sma():
    values = np.arange(1.0, 11.0)
    out = ind.ema(values, 3)
    assert np.all(np.isnan(out[:2]))
    assert out[2] == pytest.approx(2.0)  # SMA of 1,2,3


def test_ema_recursion_matches_hand_computation():
    values = np.array([1.0, 2.0, 3.0, 10.0])
    out = ind.ema(values, 3)
    alpha = 2.0 / 4.0
    assert out[3] == pytest.approx(alpha * 10.0 + (1 - alpha) * 2.0)


def test_ema_of_a_constant_is_that_constant():
    out = ind.ema(np.full(50, 7.5), 10)
    assert out[-1] == pytest.approx(7.5)


def test_wilder_rma_differs_from_ema_of_same_period():
    """RMA(n) behaves like EMA(2n-1). If these ever coincide, ATR is wrong."""
    rng = np.random.default_rng(0)
    values = np.cumsum(rng.normal(size=200)) + 100
    assert not np.allclose(ind.wilder_rma(values, 14)[-1], ind.ema(values, 14)[-1])
    assert ind.wilder_rma(values, 14)[-1] == pytest.approx(ind.ema(values, 27)[-1], rel=0.02)


def test_wilder_rma_recursion_matches_hand_computation():
    values = np.array([2.0, 4.0, 6.0, 20.0])
    out = ind.wilder_rma(values, 3)
    assert out[2] == pytest.approx(4.0)  # mean of 2,4,6
    assert out[3] == pytest.approx((4.0 * 2 + 20.0) / 3)


# --------------------------------------------------------------------------- #
# True range / ATR
# --------------------------------------------------------------------------- #


def test_true_range_takes_the_max_of_the_three_definitions():
    high = np.array([10.0, 12.0])
    low = np.array([8.0, 11.0])
    close = np.array([9.0, 11.5])
    tr = ind.true_range(high, low, close)
    assert tr[0] == pytest.approx(2.0)  # no prior close -> H-L
    assert tr[1] == pytest.approx(3.0)  # |12 - 9| beats 12-11 and |11-9|


def test_true_range_captures_a_gap_the_high_low_range_misses():
    high = np.array([100.0, 90.0])
    low = np.array([98.0, 89.0])
    close = np.array([99.0, 89.5])
    tr = ind.true_range(high, low, close)
    assert tr[1] == pytest.approx(10.0)  # |89 - 99|, not the 1.0 intrabar range


def test_atr_of_constant_range_bars_equals_that_range():
    n = 60
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    close = np.full(n, 100.0)
    assert ind.atr(high, low, close, 14)[-1] == pytest.approx(2.0)


def test_atr_is_never_negative():
    rng = np.random.default_rng(3)
    close = np.cumsum(rng.normal(size=300)) + 500
    high = close + np.abs(rng.normal(size=300))
    low = close - np.abs(rng.normal(size=300))
    out = ind.atr(high, low, close, 14)
    assert np.all(out[np.isfinite(out)] >= 0)


# --------------------------------------------------------------------------- #
# ADX
# --------------------------------------------------------------------------- #


def test_adx_is_bounded_0_to_100():
    rng = np.random.default_rng(7)
    close = np.cumsum(rng.normal(size=500)) + 1000
    high = close + np.abs(rng.normal(size=500))
    low = close - np.abs(rng.normal(size=500))
    adx, plus_di, minus_di = ind.adx(high, low, close, 14)
    for arr in (adx, plus_di, minus_di):
        finite = arr[np.isfinite(arr)]
        assert finite.size > 0
        assert finite.min() >= -1e-9
        assert finite.max() <= 100 + 1e-9


def test_adx_is_high_on_a_clean_monotone_trend():
    n = 200
    close = np.linspace(100, 300, n)
    high, low = close + 1.0, close - 1.0
    adx, plus_di, minus_di = ind.adx(high, low, close, 14)
    assert adx[-1] > 60
    assert plus_di[-1] > minus_di[-1]


def test_adx_direction_flips_on_a_downtrend():
    n = 200
    close = np.linspace(300, 100, n)
    high, low = close + 1.0, close - 1.0
    _adx, plus_di, minus_di = ind.adx(high, low, close, 14)
    assert minus_di[-1] > plus_di[-1]


def test_adx_is_low_on_a_tight_oscillation():
    n = 400
    close = 100 + np.sin(np.arange(n) * 0.6) * 0.5
    high, low = close + 0.2, close - 0.2
    adx, _, _ = ind.adx(high, low, close, 14)
    assert adx[-1] < 30


# --------------------------------------------------------------------------- #
# Donchian / rolling
# --------------------------------------------------------------------------- #


def test_rolling_max_min_are_inclusive_of_the_current_bar():
    values = np.array([1.0, 5.0, 3.0, 2.0, 9.0])
    assert ind.rolling_max(values, 3)[2] == pytest.approx(5.0)
    assert ind.rolling_max(values, 3)[4] == pytest.approx(9.0)
    assert ind.rolling_min(values, 3)[4] == pytest.approx(2.0)
    assert np.all(np.isnan(ind.rolling_max(values, 3)[:2]))


def test_donchian_brackets_the_price():
    rng = np.random.default_rng(11)
    close = np.cumsum(rng.normal(size=200)) + 300
    high, low = close + 1, close - 1
    upper, lower = ind.donchian(high, low, 20)
    ok = np.isfinite(upper)
    assert np.all(upper[ok] >= close[ok])
    assert np.all(lower[ok] <= close[ok])


# --------------------------------------------------------------------------- #
# Parkinson -- this is where the sqrt rescaling actually lives
# --------------------------------------------------------------------------- #


def test_parkinson_rescaling_uses_sqrt_of_the_period_ratio():
    """5-minute bars to daily is sqrt(288) = 16.97, not sqrt(24) = 4.90.

    Using sqrt(24) on 5m data -- a real and widely-copied mistake -- understates
    daily volatility by a factor of 3.46.
    """
    n = 500
    rng = np.random.default_rng(5)
    close = 100 * np.exp(np.cumsum(rng.normal(scale=0.001, size=n)))
    high, low = close * 1.002, close * 0.998

    at_5m = ind.parkinson_sigma(high, low, 288, bar_seconds=300, horizon_seconds=86400)[-1]
    per_bar = ind.parkinson_sigma(high, low, 288, bar_seconds=300, horizon_seconds=300)[-1]
    assert at_5m / per_bar == pytest.approx(np.sqrt(288.0), rel=1e-9)
    assert np.sqrt(288.0) / np.sqrt(24.0) == pytest.approx(3.4641, abs=1e-4)


def test_parkinson_recovers_a_known_sigma():
    """For a bar whose H/L ratio is fixed, the estimator is analytic."""
    n = 100
    ratio = 1.01
    high = np.full(n, 100.0 * ratio)
    low = np.full(n, 100.0)
    expected = np.sqrt(np.log(ratio) ** 2 / (4 * np.log(2)))
    got = ind.parkinson_sigma(high, low, 50, bar_seconds=3600, horizon_seconds=3600)[-1]
    assert got == pytest.approx(expected, rel=1e-9)


def test_parkinson_is_zero_for_flat_bars():
    out = ind.parkinson_sigma(np.full(50, 100.0), np.full(50, 100.0), 20, 3600)
    assert out[-1] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Causality -- the property everything else depends on
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fn", ["ema", "wilder_rma"])
def test_single_series_indicators_ignore_the_future(fn):
    rng = np.random.default_rng(13)
    values = np.cumsum(rng.normal(size=200)) + 100
    func = getattr(ind, fn)
    base = func(values, 14)
    perturbed = values.copy()
    perturbed[120:] *= 1.5
    after = func(perturbed, 14)
    np.testing.assert_allclose(base[:120], after[:120], rtol=1e-12)


def test_atr_and_adx_ignore_the_future():
    rng = np.random.default_rng(17)
    close = np.cumsum(rng.normal(size=300)) + 500
    high, low = close + 2, close - 2
    cut = 200

    atr_base = ind.atr(high, low, close, 14)
    adx_base, _, _ = ind.adx(high, low, close, 14)

    close2, high2, low2 = close.copy(), high.copy(), low.copy()
    close2[cut:] *= 1.4
    high2[cut:] *= 1.4
    low2[cut:] *= 1.4

    atr_after = ind.atr(high2, low2, close2, 14)
    adx_after, _, _ = ind.adx(high2, low2, close2, 14)

    np.testing.assert_allclose(atr_base[:cut], atr_after[:cut], rtol=1e-12)
    np.testing.assert_allclose(
        np.nan_to_num(adx_base[:cut]), np.nan_to_num(adx_after[:cut]), rtol=1e-12
    )
