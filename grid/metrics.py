"""Performance metrics, including the benchmark that actually makes the comparison fair.

Two conventions worth stating up front, because both are places where crypto
backtests routinely flatter themselves:

1.  Max drawdown is computed on the mark-to-market equity curve at EVERY
    execution bar, not on daily closes. Daily-close drawdown is systematically
    smaller and is the friendlier number.

2.  The headline benchmark is buy-and-hold, but the FAIR benchmark is
    exposure-matched buy-and-hold. This strategy runs roughly 30-40% average
    inventory; comparing its absolute return against 100%-long BTC compares two
    different amounts of risk. Both are reported side by side, always.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import Candles
from .engine import BacktestResult

__all__ = ["Metrics", "compute_metrics", "max_drawdown", "verify_cap_compliance"]


def verify_cap_compliance(result) -> int:
    """Replay the trade log and count R1 violations. Returns the breach count.

    This exists because the obvious way to "check" the inventory cap -- re-testing
    the gate's own expression a few lines after the gate -- is a tautology. Same
    variables, same comparison, so it can never fire, and a zero from it means
    nothing at all.

    So this rebuilds the book from the trade log alone, using only the
    decision-time inputs the engine recorded (the bar's open and the cap in force
    at that moment), and re-derives the post-trade inventory ratio independently.
    It shares no arithmetic with the gate, so it CAN fail, which is the entire
    point of having it.
    """
    trades = sorted(result.trades, key=lambda t: (t.ts, 0 if t.side == "SELL" else 1))
    context = {(ts, lot_id): (bar_open, cap) for ts, lot_id, bar_open, cap in result.cap_context}

    cash = result.config.initial_equity
    holdings: dict[int, float] = {}
    breaches = 0

    for trade in trades:
        if trade.side == "SELL":
            cash += trade.notional - trade.fee
            holdings.pop(trade.lot_id, None)
            continue

        key = (trade.ts, trade.lot_id)
        cash -= trade.notional + trade.fee
        holdings[trade.lot_id] = trade.qty

        if key not in context:
            continue
        bar_open, cap = context[key]
        # Value everything the way the decision had to: prior lots at the bar's
        # open, the new lot at what we actually paid.
        inventory = sum(
            qty * (trade.price if lot_id == trade.lot_id else bar_open)
            for lot_id, qty in holdings.items()
        )
        equity = cash + inventory
        if equity > 0 and inventory > cap * equity * (1 + 1e-6):
            breaches += 1

    return breaches

REGIME_NAMES = ("RANGE", "UPTREND", "DOWNTREND", "INSUFFICIENT_HISTORY")


def max_drawdown(equity: np.ndarray) -> tuple[float, int, int]:
    """Returns (max_drawdown_fraction, peak_index, trough_index)."""
    if equity.size == 0:
        return 0.0, 0, 0
    running_peak = np.maximum.accumulate(equity)
    drawdown = np.where(running_peak > 0, 1.0 - equity / running_peak, 0.0)
    trough = int(np.argmax(drawdown))
    peak = int(np.argmax(equity[: trough + 1])) if trough > 0 else 0
    return float(drawdown[trough]), peak, trough


def _drawdown_duration_days(ts: np.ndarray, equity: np.ndarray) -> float:
    """Longest peak-to-recovery stretch, in calendar days.

    An unrecovered drawdown is measured to the end of the test, which is the
    conservative reading.
    """
    if equity.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(equity)
    underwater = equity < running_peak
    longest = 0
    start = None
    for i, wet in enumerate(underwater):
        if wet and start is None:
            start = i
        elif not wet and start is not None:
            longest = max(longest, int(ts[i] - ts[start]))
            start = None
    if start is not None:
        longest = max(longest, int(ts[-1] - ts[start]))
    return longest / 86_400_000.0


def _daily_series(ts: np.ndarray, equity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Last equity mark of each UTC day. Used for Sharpe/Sortino only."""
    days = ts // 86_400_000
    _, last_idx = np.unique(days[::-1], return_index=True)
    idx = np.sort(equity.size - 1 - last_idx)
    return ts[idx], equity[idx]


def _sharpe(daily_equity: np.ndarray) -> float:
    """Annualised Sharpe from daily marks, risk-free rate = 0 (stated, not assumed away)."""
    if daily_equity.size < 3:
        return 0.0
    rets = np.diff(daily_equity) / daily_equity[:-1]
    sd = float(np.std(rets, ddof=1))
    return float(np.mean(rets) / sd * np.sqrt(365.0)) if sd > 0 else 0.0


def _sortino(daily_equity: np.ndarray) -> float:
    if daily_equity.size < 3:
        return 0.0
    rets = np.diff(daily_equity) / daily_equity[:-1]
    downside = rets[rets < 0]
    if downside.size < 2:
        return 0.0
    dd = float(np.std(downside, ddof=1))
    return float(np.mean(rets) / dd * np.sqrt(365.0)) if dd > 0 else 0.0


@dataclass
class Metrics:
    label: str
    start: str
    end: str
    days: float

    total_return: float
    cagr: float
    max_dd: float
    max_dd_duration_days: float
    sharpe: float
    sortino: float
    calmar: float

    round_trips: int
    win_rate: float
    losing_round_trips: int
    avg_round_trip_pnl: float
    avg_round_trip_pct: float
    avg_holding_hours: float
    derisk_sales: int
    derisk_pnl: float

    fees_paid: float
    fees_maker: float
    fees_taker: float
    maker_fills: int
    taker_fills: int
    gross_capture: float
    fee_drag: float

    time_in_market: float
    avg_inventory_ratio: float
    peak_inventory_ratio: float
    cap_breaches: int
    bars_above_cap_pct: float
    halted_days: int
    killed: bool

    regime_occupancy: dict = field(default_factory=dict)

    bh_return: float = 0.0
    bh_max_dd: float = 0.0
    bh_sharpe: float = 0.0
    exposure_matched_return: float = 0.0
    exposure_matched_max_dd: float = 0.0

    final_equity: float = 0.0
    initial_equity: float = 0.0
    n_buys: int = 0

    def as_row(self) -> dict:
        return {
            "window": self.label,
            "return": self.total_return,
            "max_dd": self.max_dd,
            "sharpe": self.sharpe,
            "calmar": self.calmar,
            "round_trips": self.round_trips,
            "fee_drag": self.fee_drag,
            "bh_return": self.bh_return,
            "bh_max_dd": self.bh_max_dd,
            "em_bh_return": self.exposure_matched_return,
            "avg_inv": self.avg_inventory_ratio,
        }


def compute_metrics(result: BacktestResult, execution: Candles) -> Metrics:
    from .data import ms_to_iso  # local import keeps the module import graph flat

    cfg = result.config
    equity = result.equity
    ts = result.ts
    n = equity.size

    initial = cfg.initial_equity
    final = float(equity[-1]) if n else initial
    total_return = final / initial - 1.0
    days = (int(ts[-1]) - int(ts[0])) / 86_400_000.0 if n > 1 else 0.0
    cagr = (final / initial) ** (365.25 / days) - 1.0 if days > 1 and final > 0 else 0.0

    dd, _, _ = max_drawdown(equity)
    dd_days = _drawdown_duration_days(ts, equity)
    _, daily_equity = _daily_series(ts, equity)
    sharpe = _sharpe(daily_equity)
    sortino = _sortino(daily_equity)
    calmar = cagr / dd if dd > 0 else 0.0

    pnls = np.asarray(result.round_trip_pnls, dtype=np.float64)
    losing = int(np.count_nonzero(pnls <= 0))
    win_rate = float(np.count_nonzero(pnls > 0) / pnls.size) if pnls.size else 0.0
    avg_rt = float(pnls.mean()) if pnls.size else 0.0

    buy_trades = [t for t in result.trades if t.side == "BUY"]
    exit_trades = [t for t in result.trades if t.reason == "EXIT"]
    avg_rt_pct = (
        float(np.mean([t.realized_pnl / t.notional for t in exit_trades if t.notional > 0]))
        if exit_trades
        else 0.0
    )
    maker_fills = sum(1 for t in result.trades if t.liquidity == "MAKER")
    taker_fills = sum(1 for t in result.trades if t.liquidity == "TAKER")

    # Fee drag is defined against the GRID MECHANISM's gross spread capture --
    # the sum of qty*(exit-entry) over closed round trips -- not against total
    # realized PnL. Total realized PnL includes de-risk sales, which are losses by
    # design, and mixing them in produces a meaningless (often negative) ratio.
    gross = result.gross_spread_capture
    fee_drag = result.round_trip_fees / gross if gross > 0 else float("nan")

    inv = result.inventory_ratio
    time_in_market = float(np.count_nonzero(inv > 1e-9) / n) if n else 0.0
    avg_inv = float(inv.mean()) if n else 0.0
    peak_inv = float(inv.max()) if n else 0.0

    occupancy = {}
    if n:
        for code, name in enumerate(REGIME_NAMES):
            occupancy[name] = float(np.count_nonzero(result.regime_code == code) / n)

    result.cap_breaches = verify_cap_compliance(result)

    # --- benchmarks ---------------------------------------------------------
    fee = cfg.maker_fee
    entry = float(execution.open[0])
    closes = execution.close

    bh_qty = initial / (entry * (1.0 + fee))
    bh_equity = bh_qty * closes
    bh_return = float(bh_equity[-1] / initial - 1.0)
    bh_dd, _, _ = max_drawdown(bh_equity)
    _, bh_daily = _daily_series(ts, bh_equity)
    bh_sharpe = _sharpe(bh_daily)

    # Exposure-matched: hold `avg_inv` of equity in BTC from bar 0, rest idle in cash.
    w = min(1.0, max(0.0, avg_inv))
    em_qty = (initial * w) / (entry * (1.0 + fee))
    em_equity = initial * (1.0 - w) + em_qty * closes
    em_return = float(em_equity[-1] / initial - 1.0)
    em_dd, _, _ = max_drawdown(em_equity)

    return Metrics(
        label=result.label,
        start=ms_to_iso(int(ts[0])) if n else "",
        end=ms_to_iso(int(ts[-1])) if n else "",
        days=days,
        total_return=total_return,
        cagr=cagr,
        max_dd=dd,
        max_dd_duration_days=dd_days,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        round_trips=result.round_trips,
        win_rate=win_rate,
        losing_round_trips=losing,
        avg_round_trip_pnl=avg_rt,
        avg_round_trip_pct=avg_rt_pct,
        avg_holding_hours=float(np.mean(result.holding_hours)) if result.holding_hours else 0.0,
        derisk_sales=result.derisk_sales,
        derisk_pnl=result.derisk_pnl,
        fees_paid=result.fees_paid,
        fees_maker=result.fees_maker,
        fees_taker=result.fees_taker,
        maker_fills=maker_fills,
        taker_fills=taker_fills,
        gross_capture=gross,
        fee_drag=fee_drag,
        time_in_market=time_in_market,
        avg_inventory_ratio=avg_inv,
        peak_inventory_ratio=peak_inv,
        cap_breaches=result.cap_breaches,
        bars_above_cap_pct=float(result.bars_above_cap / n) if n else 0.0,
        halted_days=result.halted_days,
        killed=result.killed,
        regime_occupancy=occupancy,
        bh_return=bh_return,
        bh_max_dd=bh_dd,
        bh_sharpe=bh_sharpe,
        exposure_matched_return=em_return,
        exposure_matched_max_dd=em_dd,
        final_equity=final,
        initial_equity=initial,
        n_buys=len(buy_trades),
    )
