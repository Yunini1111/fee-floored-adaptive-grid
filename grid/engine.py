"""Backtest engine: bar loop, per-lot ledger, and a deliberately pessimistic fill model.

A grid's return is close to linear in its fill count, and the fill count is
entirely a modelling choice. That makes this file, not the strategy file, the
place where a grid backtest's credibility is won or lost.

The problem, stated plainly
---------------------------
An OHLC bar records four prices and no path. A 15-minute bar with low = 62,400
tells you the price reached 62,400. It does NOT tell you how much volume traded
there, whether our order was ahead of it in the queue, or whether the touch was a
single wick print. Measured on BTC_USDT 2024-08..2026-07, the total variation of
log(close) is 1.69%/day sampled daily, 15.58%/day sampled at 15m and 26.73%/day
sampled at 5m -- path length diverges with resolution because it is a fractal
quantity. A model that counts every level crossing at 1h resolution implies
roughly +3.1%/day net at 1% spacing, which is obvious nonsense, and it is exactly
the arithmetic a careless grid backtest performs.

So the rules below are all biased against the strategy:

    F1  Order lag           orders placed on bar t are live from bar t+1's open.
                            Daily indicators for UTC day D use only bars <= D-1.
    F2  Atomic bar          cash AND the lot book are snapshotted at the bar open,
                            every fill is evaluated against that snapshot, and all
                            changes apply at the close. A sell on bar t frees
                            neither cash, nor a lot slot, nor inventory headroom
                            for a buy on bar t.
    F3  Buy fill            low < p*(1-eps), eps = 1bp. `low == p` is NOT a fill:
                            the market must trade THROUGH us, not merely touch us.
    F4  Sell fill           high > p*(1+eps).
    F5  No improvement      fill price is always the limit price, never the bar's
                            open or close, even when the bar gapped far through us.
    F6  Gap-through penalty if the bar OPENS through our limit, a post-only order
                            would have been rejected and a real agent would have
                            had to chase -> fill at p but charge the TAKER fee.
    F7  One fill per level per UTC DAY. The ladder is refreshed once daily (R7),
                            and a level that fills is not re-armed until the next
                            rollover. Stricter than the usual per-bar rule.
    F8  Forced sales        regime de-risking prices at the last CLOSE known at the
                            00:00 rollover, i.e. the previous bar's close, with the
                            taker fee. The drawdown kill prices at the current bar's
                            close. Neither ever prices at the high.
    F9  Slippage            0bp maker, 5bp taker, applied adversely. The 5bp is an
                            ASSUMPTION, not a measurement -- there is no order-book
                            data to calibrate it against.

    Risk gates are evaluated on prices knowable at the moment of the decision --
    the bar's open for existing inventory, the fill price for the lot being
    opened. Never the bar's close, which has not happened yet.

Residual optimism is disclosed rather than hidden: even with F3's through-buffer
the model assumes we are at the front of the queue at every price we trade
through. `Config.fill_probability` exists to price that assumption -- rerun at
0.7 and 0.5 and publish all three.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .data import Candles, utc_day_start_ms
from .strategy import (
    Config,
    DailySignal,
    Regime,
    SignalEngine,
    floor_to_step,
    level_notional,
    round_up_to_tick,
)

__all__ = ["BacktestResult", "Lot", "Trade", "run_backtest"]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass
class Lot:
    """One open position, with its own exit price attached at fill time."""

    lot_id: int
    qty: float
    entry_price: float
    entry_ts: int
    entry_fee: float
    exit_price: float
    entry_regime: str
    entry_spacing: float


@dataclass
class Trade:
    ts: int
    side: str  # BUY | SELL
    price: float
    qty: float
    notional: float
    fee: float
    liquidity: str  # MAKER | TAKER
    reason: str  # GRID | EXIT | DERISK | KILL
    regime: str
    lot_id: int
    realized_pnl: float = 0.0  # populated on the closing leg only


@dataclass
class BacktestResult:
    config: Config
    label: str
    ts: np.ndarray  # execution bar timestamps
    equity: np.ndarray  # marked at every execution bar close
    price: np.ndarray  # close of every execution bar
    inventory_ratio: np.ndarray
    regime_code: np.ndarray  # 0 RANGE, 1 UPTREND, 2 DOWNTREND, 3 INSUFFICIENT
    trades: list[Trade] = field(default_factory=list)
    fees_maker: float = 0.0
    fees_taker: float = 0.0
    realized_pnl: float = 0.0
    round_trips: int = 0
    round_trip_pnls: list[float] = field(default_factory=list)
    derisk_sales: int = 0
    derisk_pnl: float = 0.0
    gross_spread_capture: float = 0.0  # sum of qty*(exit-entry) over closed round trips
    round_trip_fees: float = 0.0  # both legs of those same round trips
    # cap_breaches is NOT filled in by the gate. Re-testing the gate's own
    # expression a few lines after the gate is a tautology: same variables, same
    # comparison, so it can never fire and reporting it as a check is theatre.
    # It is computed after the run by `metrics.verify_cap_compliance`, which
    # replays the trade log from scratch and can therefore actually fail.
    cap_breaches: int = 0
    bars_above_cap: int = 0  # bars above cap from mark drift alone -- informational
    # (bar_ts, lot_id, bar_open, cap_in_force) for every buy, so the replay can
    # reconstruct the decision-time state without trusting the engine's own maths.
    cap_context: list = field(default_factory=list)
    holding_hours: list[float] = field(default_factory=list)
    killed: bool = False
    kill_ts: int | None = None
    kill_count: int = 0
    halted_days: int = 0
    signals: list[DailySignal] = field(default_factory=list)

    @property
    def fees_paid(self) -> float:
        return self.fees_maker + self.fees_taker

    @property
    def gross_capture(self) -> float:
        """What the strategy would have earned at zero fees."""
        return self.realized_pnl + self.fees_paid


_REGIME_CODE = {
    Regime.RANGE: 0,
    Regime.UPTREND: 1,
    Regime.DOWNTREND: 2,
    Regime.INSUFFICIENT_HISTORY: 3,
}


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


def run_backtest(
    execution: Candles,
    daily: Candles,
    cfg: Config,
    label: str = "",
    hourly: Candles | None = None,
) -> BacktestResult:
    """Run the strategy over `execution`, taking daily signals from `daily`.

    `daily` should start well before `execution` so EMA200 and ADX14 are fully
    warm before the first execution bar -- otherwise the headline number is
    contaminated by burn-in.
    """
    cfg.validate()
    signals = SignalEngine(daily, cfg, hourly=hourly)
    rng = np.random.default_rng(cfg.seed)

    n = len(execution)
    ts, op, hi, lo, cl = execution.ts, execution.open, execution.high, execution.low, execution.close

    cash = cfg.initial_equity
    lots: list[Lot] = []
    next_lot_id = 1

    resting_buys: list[tuple[float, float]] = []  # (price, notional) -- live next bar
    pending_buys: list[tuple[float, float]] = []  # placed this bar, not yet live (F1)
    filled_levels_today: set[float] = set()
    missed_today: set[int] = set()  # lot ids whose exit excursion we missed today

    signal: DailySignal | None = None
    current_day = -1
    frozen_anchor: float | None = None
    freeze_days_left = 0

    equity_hwm = cfg.initial_equity
    day_start_equity = cfg.initial_equity
    halted_today = False
    killed = False
    kill_ts: int | None = None
    kill_day: int | None = None
    halted_days = 0
    downtrend_days = 0

    out = BacktestResult(
        config=cfg,
        label=label,
        ts=ts,
        equity=np.empty(n, dtype=np.float64),
        price=cl,
        inventory_ratio=np.empty(n, dtype=np.float64),
        regime_code=np.full(n, 3, dtype=np.int8),
    )

    def mark(price: float) -> tuple[float, float]:
        inv_qty = sum(lot.qty for lot in lots)
        inv_val = inv_qty * price
        return inv_val, cash + inv_val

    def sell_lot(lot: Lot, price: float, when: int, reason: str, liquidity: str) -> float:
        """Close `lot` at `price`. Returns realized PnL net of both legs' fees."""
        nonlocal cash
        if liquidity == "TAKER":
            price *= 1.0 - cfg.taker_slippage
            fee_rate = cfg.taker_fee
        else:
            fee_rate = cfg.maker_fee
        notional = lot.qty * price
        fee = notional * fee_rate
        cash += notional - fee
        pnl = (notional - fee) - (lot.qty * lot.entry_price + lot.entry_fee)

        if liquidity == "TAKER":
            out.fees_taker += fee
        else:
            out.fees_maker += fee
        out.realized_pnl += pnl
        out.trades.append(
            Trade(
                ts=when,
                side="SELL",
                price=price,
                qty=lot.qty,
                notional=notional,
                fee=fee,
                liquidity=liquidity,
                reason=reason,
                regime=signal.regime.value if signal else "NA",
                lot_id=lot.lot_id,
                realized_pnl=pnl,
            )
        )
        out.holding_hours.append((when - lot.entry_ts) / 3_600_000.0)
        return pnl

    for i in range(n):
        bar_ts = int(ts[i])
        day = utc_day_start_ms(bar_ts)

        # ------------------------------------------------------------------ #
        # Daily rollover at 00:00 UTC
        # ------------------------------------------------------------------ #
        if day != current_day:
            current_day = day
            mark_price = float(cl[i - 1]) if i > 0 else float(op[i])
            inv_val, equity = mark(mark_price)
            day_start_equity = equity
            halted_today = False
            filled_levels_today.clear()
            missed_today.clear()

            # R5 is specified as requiring an explicit human re-arm, which a
            # backtest cannot observe. We model an operator who reviews and
            # re-arms `dd_kill_rearm_days` after the kill, and reset the
            # high-water mark so the same drawdown cannot re-trigger instantly.
            # This is an ASSUMPTION about operator behaviour, not a measurement;
            # results with re-arm disabled are published alongside.
            if killed and kill_day is not None and cfg.dd_kill_rearm_days > 0:
                if (day - kill_day) >= cfg.dd_kill_rearm_days * 86_400_000:
                    killed = False
                    kill_day = None
                    equity_hwm = equity

            new_signal = signals.signal_for(day, frozen_anchor=frozen_anchor)
            if new_signal is not None:
                signal = new_signal

                # Anti-martingale: a downside range break freezes the anchor so
                # the ladder cannot chase price down. An UPSIDE break is benign
                # -- inventory has already exited above -- and is only logged.
                if freeze_days_left > 0:
                    freeze_days_left -= 1
                    if freeze_days_left == 0:
                        frozen_anchor = None
                elif signal.broke_lower and signal.regime is Regime.DOWNTREND:
                    frozen_anchor = signal.anchor
                    freeze_days_left = cfg.anchor_freeze_days

                # The floating-exit ablation: reprice every open lot onto the
                # current grid. This is the design that loses money, kept so the
                # improvement chain can be measured rather than asserted.
                if not cfg.paired_exits:
                    new_exit = round_up_to_tick(
                        signal.anchor * (1.0 + signal.spacing), cfg.price_tick
                    )
                    for lot in lots:
                        lot.exit_price = new_exit

                # R1 + active de-risk. Measured finding: a gate that only blocks
                # new buys changes bear-2022 results by 0.12 points, i.e. nothing.
                # The gate only earns its place if it actively reduces inventory.
                if signal.regime is Regime.DOWNTREND:
                    downtrend_days += 1
                else:
                    downtrend_days = 0

                cap = signal.inventory_cap
                persistent = (
                    signal.regime is not Regime.DOWNTREND
                    or downtrend_days >= cfg.derisk_persistence_days
                )
                if cfg.active_derisk and persistent and equity > 0 and inv_val > cap * equity:
                    excess = (inv_val - cap * equity) * cfg.derisk_max_fraction
                    sign = -1.0 if cfg.derisk_highest_cost_first else 1.0
                    for lot in sorted(lots, key=lambda x: sign * x.entry_price):
                        if excess <= 0:
                            break
                        pnl = sell_lot(lot, mark_price, bar_ts, "DERISK", "TAKER")
                        lots.remove(lot)
                        out.derisk_sales += 1
                        out.derisk_pnl += pnl
                        excess -= lot.qty * mark_price

                # Refresh the buy ladder once per day (R7: nothing rests > 24h).
                resting_buys = []
                pending_buys = []
                if signal.regime is not Regime.DOWNTREND and not killed:
                    inv_val, equity = mark(mark_price)
                    per_level = level_notional(
                        cfg,
                        equity=equity,
                        inventory_value=inv_val,
                        inventory_cap=cap,
                        atr_now=signal.atr_pct,
                        atr_ref=signal.atr_ref,
                        n_levels=len(signal.buy_levels),
                    )
                    if per_level > 0 and len(lots) < cfg.max_open_lots:
                        pending_buys = [(p, per_level) for p in signal.buy_levels]

        if signal is None:
            out.equity[i] = cash
            out.inventory_ratio[i] = 0.0
            continue

        out.regime_code[i] = _REGIME_CODE[signal.regime]

        # ------------------------------------------------------------------ #
        # F2: snapshot at the bar open
        # ------------------------------------------------------------------ #
        bar_open, bar_high, bar_low, bar_close = (
            float(op[i]),
            float(hi[i]),
            float(lo[i]),
            float(cl[i]),
        )
        snapshot_cash = cash
        committed = 0.0
        # F2: the lot book as it stood at the OPEN. Reading `lots` directly after
        # this bar's exits have run would let a sell on bar t free an R3 lot slot
        # and inventory headroom for a buy on bar t -- the same violation the cash
        # snapshot exists to prevent, just via a different field.
        lots_at_open = list(lots)
        opened_this_bar: list[Lot] = []

        # ---- exits first (they are always live and never re-priced) -------- #
        for lot in list(lots):
            target = lot.exit_price
            if bar_high > target * (1.0 + cfg.fill_epsilon):
                if lot.lot_id in missed_today:
                    continue
                if cfg.fill_probability < 1.0 and rng.random() > cfg.fill_probability:
                    # We were not at the front of the queue for this excursion.
                    # Retiring the chance for the rest of the day rather than
                    # retrying next bar is what makes this knob mean anything: a
                    # per-bar retry converges back to "everything fills eventually"
                    # and measures only delay, not queue position.
                    missed_today.add(lot.lot_id)
                    continue
                # F6: a bar that opens above our sell would have rejected a
                # post-only order; charge taker.
                liquidity = "TAKER" if bar_open > target else "MAKER"
                entry_fee, entry_price, qty = lot.entry_fee, lot.entry_price, lot.qty
                pnl = sell_lot(lot, target, bar_ts, "EXIT", liquidity)
                lots.remove(lot)
                out.round_trips += 1
                out.round_trip_pnls.append(pnl)
                out.gross_spread_capture += qty * (out.trades[-1].price - entry_price)
                out.round_trip_fees += entry_fee + out.trades[-1].fee

        # ---- then entries, against the OPEN snapshot ----------------------- #
        if not killed and not halted_today:
            for price, notional in list(resting_buys):
                if price in filled_levels_today:  # F7
                    continue
                if bar_low >= price * (1.0 - cfg.fill_epsilon):  # F3
                    continue
                if cfg.fill_probability < 1.0 and rng.random() > cfg.fill_probability:
                    filled_levels_today.add(price)  # excursion missed, see note above
                    continue
                if len(lots_at_open) + len(opened_this_bar) >= cfg.max_open_lots:  # R3, on the open book
                    continue

                liquidity = "TAKER" if bar_open < price else "MAKER"  # F6
                fill_price = price * (1.0 + cfg.taker_slippage) if liquidity == "TAKER" else price
                fee_rate = cfg.taker_fee if liquidity == "TAKER" else cfg.maker_fee

                qty = floor_to_step(notional / fill_price, cfg.qty_step)
                if qty <= 0 or qty * fill_price < cfg.min_notional:
                    continue
                cost = qty * fill_price * (1.0 + fee_rate)
                if snapshot_cash - committed < cost:  # F2 solvency on the snapshot
                    continue

                # R1, evaluated ONLY on prices knowable at the moment of the fill:
                # the bar's open for inventory already held, and the fill price
                # itself for the lot being opened.
                #
                # An earlier version marked both at the bar's CLOSE. That drove the
                # reported breach count to zero, but only because the gate was
                # reading a price that does not exist yet -- the engine was using
                # future information to guarantee the very number it reports. A
                # control that can only be satisfied with hindsight is not a control.
                # Marking at the open is what a live agent can actually do, and the
                # residual drift between the gate and the close-marked ratio is
                # reported separately as `bars_above_cap`.
                #
                # Cash committed earlier this bar is already spent. Proceeds from
                # exits earlier this bar are deliberately NOT counted (F2).
                inventory_at_open = sum(lot.qty for lot in lots_at_open) * bar_open
                opened_value = sum(l.qty * l.entry_price for l in opened_this_bar)
                post_inventory = inventory_at_open + opened_value + qty * fill_price
                post_cash = snapshot_cash - committed - cost
                post_equity = post_cash + post_inventory
                if post_equity <= 0 or post_inventory > signal.inventory_cap * post_equity:
                    continue

                committed += cost
                cash -= cost
                fee = qty * fill_price * fee_rate
                if liquidity == "TAKER":
                    out.fees_taker += fee
                else:
                    out.fees_maker += fee

                exit_price = round_up_to_tick(
                    fill_price * (1.0 + signal.spacing), cfg.price_tick
                )
                new_lot = Lot(
                    lot_id=next_lot_id,
                    qty=qty,
                    entry_price=fill_price,
                    entry_ts=bar_ts,
                    entry_fee=fee,
                    exit_price=exit_price,
                    entry_regime=signal.regime.value,
                    entry_spacing=signal.spacing,
                )
                lots.append(new_lot)
                opened_this_bar.append(new_lot)
                out.cap_context.append((bar_ts, next_lot_id, bar_open, signal.inventory_cap))
                out.trades.append(
                    Trade(
                        ts=bar_ts,
                        side="BUY",
                        price=fill_price,
                        qty=qty,
                        notional=qty * fill_price,
                        fee=fee,
                        liquidity=liquidity,
                        reason="GRID",
                        regime=signal.regime.value,
                        lot_id=next_lot_id,
                    )
                )
                next_lot_id += 1
                filled_levels_today.add(price)
                resting_buys.remove((price, notional))

        # F1: orders placed during this bar's rollover become live next bar.
        if pending_buys:
            resting_buys = pending_buys
            pending_buys = []

        # ------------------------------------------------------------------ #
        # Mark, then evaluate the risk overlay for the NEXT bar
        # ------------------------------------------------------------------ #
        inv_val, equity = mark(bar_close)
        out.equity[i] = equity
        out.inventory_ratio[i] = inv_val / equity if equity > 0 else 0.0
        equity_hwm = max(equity_hwm, equity)

        # Two different questions, deliberately counted separately, and neither
        # is allowed to launder the other.
        #
        # cap_breaches (incremented at fill time, above) asks "did our own action
        # break R1 given what we knew when we acted?" It must be 0, and a test
        # asserts it. Critically it is evaluated on decision-time prices, so it
        # cannot be satisfied by hindsight.
        #
        # bars_above_cap asks "how many bars did mark-to-market drift leave the
        # close-marked ratio above the cap?" This is NOT a control failure: a
        # rising price lifts the inventory ratio without us trading, and no cap
        # can be enforced continuously without trading continuously. It is
        # expected to be non-zero, and reporting it is the honest half of the
        # picture.
        if equity > 0 and inv_val > (signal.inventory_cap + 1e-9) * equity:
            out.bars_above_cap += 1

        # R4 daily loss limit: stop buying for the rest of the UTC day, keep exits.
        if not halted_today and equity < (1.0 - cfg.daily_loss_limit) * day_start_equity:
            halted_today = True
            halted_days += 1
            resting_buys = []

        # R5 drawdown kill switch. Must be ACTIVE: a passive version that only
        # blocks new buys was measured to produce byte-identical results.
        if not killed and equity <= (1.0 - cfg.dd_kill) * equity_hwm:
            killed = True
            kill_ts = bar_ts if kill_ts is None else kill_ts
            kill_day = current_day
            out.kill_count += 1
            resting_buys = []
            target_inv = cfg.cap_downtrend * equity
            kill_sign = -1.0 if cfg.derisk_highest_cost_first else 1.0
            for lot in sorted(lots, key=lambda x: kill_sign * x.entry_price):
                if inv_val <= target_inv:
                    break
                pnl = sell_lot(lot, bar_close, bar_ts, "KILL", "TAKER")
                lots.remove(lot)
                out.derisk_sales += 1
                out.derisk_pnl += pnl
                inv_val -= lot.qty * bar_close
            inv_val, equity = mark(bar_close)
            out.equity[i] = equity
            out.inventory_ratio[i] = inv_val / equity if equity > 0 else 0.0

        if signal is not None and (i == n - 1 or utc_day_start_ms(int(ts[i + 1])) != current_day):
            out.signals.append(signal)

    out.killed = killed
    out.kill_ts = kill_ts
    out.halted_days = halted_days
    return out
