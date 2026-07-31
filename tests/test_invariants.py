"""The claims this strategy makes about itself, asserted against the real dataset.

These are the tests that turn "the design guarantees X" into something a reviewer
can run. They use the cached CoinW data if it is present and skip cleanly if not,
so `pytest` works on a fresh clone before `run_backtest.py` has ever been run.
"""

from pathlib import Path

import numpy as np
import pytest

from grid.data import CoinWClient, DataError, integrity_check, iso_to_ms
from grid.engine import run_backtest
from grid.metrics import compute_metrics
from grid.strategy import Config, Regime, SignalEngine, net_edge_per_round_trip

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def market():
    client = CoinWClient(DATA_DIR, offline=True, verbose=False)
    try:
        daily = client.fetch("BTC_USDT", 86400, "2018-01-01", "2026-08-01")
        execution = client.fetch("BTC_USDT", 900, "2019-01-01", "2026-08-01")
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
    """R1. Mark-to-market drift above the cap is expected and counted separately."""
    result, metrics, _ = full_run
    assert result.cap_breaches == 0
    assert metrics.cap_breaches == 0


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
