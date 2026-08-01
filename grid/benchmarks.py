"""Competing retail strategies, simulated on the same data with the same cost model.

These are NOT part of the Skill. They exist so that "should I run a grid?" can be
answered against the things a retail trader would actually run instead: periodic
DCA, and a DCA-martingale bot of the kind Pionex / 3Commas / OKX all ship.

Fairness rules, applied identically to every strategy here and to the grid:
  - same starting capital, same window, same maker/taker fees
  - DCA and martingale entries are MARKET orders, so they pay the TAKER fee plus
    adverse slippage. That is how these bots actually execute. The grid pays
    maker on most fills, which is a genuine structural advantage of resting
    orders and is not a modelling favour.
  - equity is marked at every execution bar, so drawdown is comparable
  - no strategy is allowed to spend cash it does not have

The martingale parameters are the conventional retail defaults rather than
anything tuned. They are stated in the output so a reader can disagree with them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import Candles
from .strategy import Config, floor_to_step

__all__ = ["BenchmarkResult", "run_buy_and_hold", "run_dca", "run_martingale"]


@dataclass
class BenchmarkResult:
    label: str
    equity: np.ndarray
    ts: np.ndarray
    trades: int = 0
    fees: float = 0.0
    final_equity: float = 0.0
    max_dd: float = 0.0
    deployed_pct: float = 0.0  # fraction of capital actually put to work
    round_trips: int = 0
    notes: str = ""
    final_btc: float = 0.0
    detail: dict = field(default_factory=dict)


def _finish(label, ts, equity, trades, fees, cfg, btc_qty, price_last, deployed, rt=0, notes=""):
    eq = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = float(np.max(np.where(peak > 0, 1.0 - eq / peak, 0.0))) if eq.size else 0.0
    return BenchmarkResult(
        label=label,
        equity=eq,
        ts=ts,
        trades=trades,
        fees=fees,
        final_equity=float(eq[-1]) if eq.size else cfg.initial_equity,
        max_dd=dd,
        deployed_pct=deployed,
        round_trips=rt,
        notes=notes,
        final_btc=float(eq[-1]) / price_last if price_last > 0 else 0.0,
    )


def run_buy_and_hold(execution: Candles, cfg: Config) -> BenchmarkResult:
    """Buy everything at the first bar's open, hold. The null hypothesis."""
    entry = float(execution.open[0])
    qty = floor_to_step(cfg.initial_equity / (entry * (1 + cfg.maker_fee)), cfg.qty_step)
    fee = qty * entry * cfg.maker_fee
    cash = cfg.initial_equity - qty * entry - fee
    equity = cash + qty * execution.close
    return _finish(
        "Buy & hold", execution.ts, equity, 1, fee, cfg, qty, float(execution.close[-1]), 1.0,
        notes="Single maker buy at the first bar, never sold.",
    )


def run_dca(execution: Candles, cfg: Config, interval_days: int = 7) -> BenchmarkResult:
    """Periodic DCA (定投): buy a fixed slice every `interval_days`, never sell.

    Capital is divided evenly across the number of purchases in the window, so
    the strategy is fully deployed by the end and idle cash earns nothing --
    which is what actually happens to a retail DCA plan.
    """
    ts = execution.ts
    step_ms = interval_days * 86_400_000
    buy_at = list(range(0, len(ts), max(1, int(step_ms // (execution.period * 1000)))))
    if not buy_at:
        buy_at = [0]
    slice_usdt = cfg.initial_equity / len(buy_at)

    cash = cfg.initial_equity
    qty = 0.0
    fees = 0.0
    trades = 0
    equity = np.empty(len(ts), dtype=np.float64)
    buy_set = set(buy_at)

    for i in range(len(ts)):
        if i in buy_set:
            # Market order: taker fee and adverse slippage, as a real DCA does.
            price = float(execution.close[i]) * (1 + cfg.taker_slippage)
            want = min(slice_usdt, cash / (1 + cfg.taker_fee))
            q = floor_to_step(want / price, cfg.qty_step)
            if q > 0 and q * price >= cfg.min_notional:
                fee = q * price * cfg.taker_fee
                cash -= q * price + fee
                qty += q
                fees += fee
                trades += 1
        equity[i] = cash + qty * float(execution.close[i])

    return _finish(
        f"DCA every {interval_days}d", ts, equity, trades, fees, cfg, qty,
        float(execution.close[-1]), 1.0 - cash / cfg.initial_equity,
        notes=f"{trades} market buys of {slice_usdt:,.0f} USDT, never sold.",
    )


def run_martingale(
    execution: Candles,
    cfg: Config,
    base_pct: float = 0.02,
    safety_deviation: float = 0.025,
    volume_multiplier: float = 1.5,
    step_multiplier: float = 1.0,
    take_profit: float = 0.015,
    max_safety_orders: int = 8,
) -> BenchmarkResult:
    """DCA-martingale bot (馬丁): average down on a ladder, exit the whole cycle at TP.

    The defaults are the conventional retail ones, not tuned: a base order, up to
    eight safety orders each 1.5x the previous, triggered every 2.5% below the
    last entry, and a 1.5% take-profit on the AVERAGE cost of the whole cycle.

    The failure mode this exposes is the one martingale is famous for: the ladder
    is finite. Once all safety orders are spent the position simply sits, fully
    exposed, until price returns to average cost -- which in a sustained decline
    can be never.
    """
    ts = execution.ts
    op, hi, lo, cl = execution.open, execution.high, execution.low, execution.close

    cash = cfg.initial_equity
    qty = 0.0
    cost = 0.0
    fees = 0.0
    trades = 0
    cycles = 0
    safety_used = 0
    last_entry = 0.0
    max_safety_reached = 0
    bars_maxed = 0
    equity = np.empty(len(ts), dtype=np.float64)

    def market_buy(usdt: float, price: float) -> None:
        nonlocal cash, qty, cost, fees, trades
        px = price * (1 + cfg.taker_slippage)
        want = min(usdt, cash / (1 + cfg.taker_fee))
        q = floor_to_step(want / px, cfg.qty_step)
        if q <= 0 or q * px < cfg.min_notional:
            return
        fee = q * px * cfg.taker_fee
        cash -= q * px + fee
        qty += q
        cost += q * px
        fees += fee
        trades += 1

    for i in range(len(ts)):
        price = float(cl[i])

        if qty <= 0:
            # Open a new cycle with the base order.
            market_buy(cfg.initial_equity * base_pct, price)
            if qty > 0:
                last_entry = cost / qty
                safety_used = 0
                cycles += 1
        else:
            avg = cost / qty
            # Take profit closes the ENTIRE cycle at once.
            if float(hi[i]) >= avg * (1 + take_profit):
                exit_px = avg * (1 + take_profit) * (1 - cfg.taker_slippage)
                proceeds = qty * exit_px
                fee = proceeds * cfg.taker_fee
                cash += proceeds - fee
                fees += fee
                trades += 1
                qty = 0.0
                cost = 0.0
                safety_used = 0
            else:
                trigger = last_entry * (1 - safety_deviation * (step_multiplier**safety_used))
                if float(lo[i]) < trigger and safety_used < max_safety_orders:
                    size = cfg.initial_equity * base_pct * (volume_multiplier ** (safety_used + 1))
                    before = qty
                    market_buy(size, trigger)
                    if qty > before:
                        safety_used += 1
                        last_entry = trigger
                        max_safety_reached = max(max_safety_reached, safety_used)
                if safety_used >= max_safety_orders:
                    bars_maxed += 1

        equity[i] = cash + qty * price

    deployed = 1.0 - cash / cfg.initial_equity if qty > 0 else 0.0
    return _finish(
        "Martingale (DCA bot)", ts, equity, trades, fees, cfg, qty, float(cl[-1]),
        max(0.0, deployed), rt=cycles,
        notes=(
            f"base {base_pct:.0%}, {max_safety_orders} safety orders x{volume_multiplier} "
            f"every {safety_deviation:.1%}, TP {take_profit:.1%}. "
            f"{cycles} cycles; ladder fully spent on {bars_maxed:,} bars "
            f"({100 * bars_maxed / max(1, len(ts)):.1f}% of the time)."
        ),
    )
