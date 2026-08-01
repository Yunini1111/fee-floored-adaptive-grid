"""The claims this strategy makes about itself, asserted against the real dataset.

These are the tests that turn "the design guarantees X" into something a reviewer
can run. They use the cached CoinW data if it is present and skip cleanly if not,
so `pytest` works on a fresh clone before `run_backtest.py` has ever been run.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from grid.data import CoinWClient, DataError, integrity_check, iso_to_ms
from grid.engine import run_backtest
from grid.metrics import compute_metrics
from grid.strategy import Config, Regime, SignalEngine, net_edge_per_round_trip

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Imported rather than duplicated: hardcoding the window here let the tests drift
# away from what the study actually ran, and the warm-up test below then passed
# against the wrong dates.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_backtest import DAILY_START, END, EXEC_START, PAIR  # noqa: E402


@pytest.fixture(scope="module")
def market():
    client = CoinWClient(DATA_DIR, offline=True, verbose=False)
    try:
        daily = client.fetch(PAIR, 86400, DAILY_START, END)
        execution = client.fetch(PAIR, 900, EXEC_START, END)
    except DataError as exc:
        pytest.skip(f"cached CoinW data unavailable ({exc}); run `python run_backtest.py --all`")
    return daily, execution


@pytest.fixture(scope="module")
def full_run(market):
    daily, execution = market
    cfg = Config()
    result = run_backtest(execution, daily, cfg, label="full")
    return result, compute_metrics(result, execution), execution


# --------------------------------------------------------------------------- #
# The headline structural claim
# --------------------------------------------------------------------------- #


def test_every_completed_round_trip_is_net_positive(full_run):
    """The paired-exit guarantee. A single violation is a code defect, not a
    market event, and it invalidates the fee-floor argument entirely."""
    result, metrics, _ = full_run
    assert result.round_trips > 100, "not enough round trips for this to mean anything"
    assert metrics.losing_round_trips == 0
    assert min(result.round_trip_pnls) > 0
    assert metrics.win_rate == pytest.approx(1.0)


def test_round_trip_pnl_matches_the_fee_formula(full_run):
    """Realized PnL per round trip should equal q*P*[s - f(2+s)] to within
    rounding. If it does not, either the fee accounting or the exit pricing is wrong."""
    result, _, _ = full_run
    cfg = result.config
    exits = [t for t in result.trades if t.reason == "EXIT"]
    assert exits
    entries = {t.lot_id: t for t in result.trades if t.side == "BUY"}
    checked = 0
    for exit_trade in exits[:400]:
        entry = entries.get(exit_trade.lot_id)
        if entry is None:
            continue
        s = exit_trade.price / entry.price - 1.0
        predicted = entry.qty * entry.price * net_edge_per_round_trip(s, cfg.maker_fee)
        if exit_trade.liquidity == "MAKER" and entry.liquidity == "MAKER":
            assert exit_trade.realized_pnl == pytest.approx(predicted, rel=1e-6, abs=1e-6)
            checked += 1
    assert checked > 50


def test_no_exit_is_ever_priced_below_its_own_entry(full_run):
    result, _, _ = full_run
    entries = {t.lot_id: t for t in result.trades if t.side == "BUY"}
    for t in result.trades:
        if t.reason == "EXIT":
            assert t.price > entries[t.lot_id].price


# --------------------------------------------------------------------------- #
# Risk overlay
# --------------------------------------------------------------------------- #


def test_inventory_cap_is_never_breached_by_our_own_action(full_run):
    """R1, evaluated on decision-time prices.

    The gate and this counter both use the bar's OPEN for existing inventory and
    the fill price for the new lot. Neither reads the bar's close, which has not
    happened at the moment of the decision -- an earlier version did, which made
    a zero here circular rather than meaningful.
    """
    result, metrics, _ = full_run
    assert result.cap_breaches == 0
    assert metrics.cap_breaches == 0


def test_mark_drift_above_the_cap_is_reported_and_non_zero(full_run):
    """The honest other half. If this were also zero, the cap would be being
    enforced with hindsight rather than by refusing trades."""
    result, metrics, _ = full_run
    assert result.bars_above_cap > 0
    assert 0.0 < metrics.bars_above_cap_pct < 1.0


def test_kill_switch_honours_the_lot_ordering_setting(market):
    """The kill path used to hardcode highest-cost-first while the write-up said
    the flag governed it. Both orderings must now be reachable."""
    daily, execution = market
    sub = execution.slice_ms(iso_to_ms("2021-11-01"), iso_to_ms("2022-12-01"))
    orders = {}
    for highest_first in (True, False):
        cfg = Config(derisk_highest_cost_first=highest_first, dd_kill_override=0.12)
        result = run_backtest(sub, daily, cfg, label="kill")
        kills = [t for t in result.trades if t.reason == "KILL"]
        orders[highest_first] = [t.lot_id for t in kills]
    assert orders[True], "no kill fired; test is vacuous"
    assert orders[True] != orders[False]


def test_inventory_never_exceeds_the_absolute_ceiling_by_more_than_drift(full_run):
    result, metrics, _ = full_run
    assert metrics.peak_inventory_ratio <= result.config.cap_absolute + 0.15


def test_open_lots_never_exceed_the_hard_limit(market):
    daily, execution = market
    cfg = Config()
    sub = execution.slice_ms(iso_to_ms("2022-01-01"), iso_to_ms("2023-01-01"))
    result = run_backtest(sub, daily, cfg, label="bear")
    open_ids = set()
    peak = 0
    for t in sorted(result.trades, key=lambda x: x.ts):
        if t.side == "BUY":
            open_ids.add(t.lot_id)
        else:
            open_ids.discard(t.lot_id)
        peak = max(peak, len(open_ids))
    assert peak <= cfg.max_open_lots


def test_no_buy_exceeds_the_per_level_equity_cap(full_run):
    """R2. Checked against the equity curve at the time of each fill."""
    result, _, execution = full_run
    cfg = result.config
    ts_index = {int(t): i for i, t in enumerate(result.ts)}
    for t in result.trades:
        if t.side != "BUY":
            continue
        i = ts_index.get(t.ts)
        if i is None or i == 0:
            continue
        equity = result.equity[i - 1]
        assert t.notional <= cfg.max_level_pct * equity * 1.05 + cfg.min_notional


def test_spacing_always_clears_the_fee_hurdle(market):
    """R8, checked over every day in the dataset rather than at one point."""
    daily, _ = market
    cfg = Config()
    engine = SignalEngine(daily, cfg)
    checked = 0
    for i in range(250, len(daily), 7):
        signal = engine.signal_for(int(daily.ts[i]))
        if signal is None:
            continue
        assert signal.spacing >= cfg.s_floor - 1e-12
        assert net_edge_per_round_trip(signal.spacing, cfg.maker_fee) > 0
        checked += 1
    assert checked > 300


def test_no_buys_are_placed_in_a_confirmed_downtrend(full_run):
    """The passive half of the regime gate -- the half that measurably works."""
    result, _, _ = full_run
    downtrend_buys = [t for t in result.trades if t.side == "BUY" and t.regime == "DOWNTREND"]
    assert downtrend_buys == []


# --------------------------------------------------------------------------- #
# Accounting
# --------------------------------------------------------------------------- #


def test_equity_reconciles_with_cash_plus_inventory(full_run):
    """Independent replay of the trade log must land on the reported final equity."""
    result, metrics, execution = full_run
    cfg = result.config
    cash = cfg.initial_equity
    qty = 0.0
    for t in sorted(result.trades, key=lambda x: x.ts):
        if t.side == "BUY":
            cash -= t.notional + t.fee
            qty += t.qty
        else:
            cash += t.notional - t.fee
            qty -= t.qty
    assert qty >= -1e-9
    replayed = cash + qty * float(execution.close[-1])
    assert replayed == pytest.approx(metrics.final_equity, rel=1e-6)


def test_fees_are_positive_and_split_correctly(full_run):
    result, metrics, _ = full_run
    assert result.fees_maker > 0
    assert result.fees_paid == pytest.approx(result.fees_maker + result.fees_taker)
    total = sum(t.fee for t in result.trades)
    assert total == pytest.approx(result.fees_paid, rel=1e-9)
    assert metrics.maker_fills + metrics.taker_fills == len(result.trades)


def test_fee_drag_is_in_a_sane_range(full_run):
    _result, metrics, _ = full_run
    assert 0.0 < metrics.fee_drag < 0.25


def test_cash_is_never_negative_on_the_real_dataset(full_run):
    result, _, _ = full_run
    cfg = result.config
    cash = cfg.initial_equity
    for t in sorted(result.trades, key=lambda x: x.ts):
        cash += (t.notional - t.fee) if t.side == "SELL" else -(t.notional + t.fee)
        assert cash > -1e-6


# --------------------------------------------------------------------------- #
# Data hygiene
# --------------------------------------------------------------------------- #


def test_cached_series_pass_the_integrity_check(market):
    daily, execution = market
    for series in (daily, execution):
        report = integrity_check(series)
        assert report.bars > 1000
        assert report.missing_pct < 1.0
        assert report.misaligned_bars == 0


def test_no_forward_filled_bars(market):
    """A forward-filled series shows runs of identical OHLC. Real data does not."""
    _daily, execution = market
    identical = (
        (execution.open == execution.high)
        & (execution.high == execution.low)
        & (execution.low == execution.close)
    )
    assert identical.mean() < 0.01


def test_regime_occupancy_matches_the_documented_split(market):
    daily, _ = market
    cfg = Config()
    engine = SignalEngine(daily, cfg)
    counts = {r: 0 for r in Regime}
    for i in range(250, len(daily)):
        signal = engine.signal_for(int(daily.ts[i]))
        if signal:
            counts[signal.regime] += 1
    total = sum(counts.values())
    assert total > 2000
    assert 0.45 < counts[Regime.RANGE] / total < 0.65
    assert 0.20 < counts[Regime.UPTREND] / total < 0.35
    assert 0.10 < counts[Regime.DOWNTREND] / total < 0.25


# --------------------------------------------------------------------------- #
# Data-layer contract
# --------------------------------------------------------------------------- #


def test_fetch_returns_exactly_the_requested_window(market):
    """Regression: chunks are cached at calendar-year granularity, so a cached
    chunk can hold a SUPERSET of what was asked for. Returning that silently made
    the backtest run over a window nobody requested and misreported its own
    provenance. The cache may be a superset; `fetch` must be exact.
    """
    client = CoinWClient(DATA_DIR, offline=True, verbose=False)
    for start, end in (("2018-03-05", "2026-08-01"), ("2022-01-01", "2023-01-01")):
        series = client.fetch("BTC_USDT", 900, start, end)
        assert int(series.ts[0]) >= iso_to_ms(start), (
            f"fetch({start}..{end}) leaked bars from before {start}"
        )
        assert int(series.ts[-1]) < iso_to_ms(end), (
            f"fetch({start}..{end}) leaked bars at or after {end}"
        )


def test_full_run_starts_where_the_indicators_are_warm(market):
    """The execution series must not begin before EMA200/ADX14 are usable, and it
    must not begin materially later either -- starting late silently discards the
    2018 bear market, which is the most demanding window in the record."""
    daily, execution = market
    engine = SignalEngine(daily, Config())
    warm = int(np.argmax(np.isfinite(engine.adx) & np.isfinite(engine.ema_slow)))
    first_exec_day = int(execution.ts[0]) // 86_400_000 * 86_400_000
    warm_day = int(daily.ts[warm])
    assert first_exec_day > warm_day, "execution starts before indicators are warm"
    assert first_exec_day - warm_day <= 40 * 86_400_000, (
        "execution starts more than 40 days after warm-up; earlier data is being wasted"
    )
