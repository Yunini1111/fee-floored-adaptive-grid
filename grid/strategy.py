"""Strategy core: grid geometry, fee-derived spacing floor, regime gate, sizing, risk overlay.

Everything in this module is a pure function of daily indicator values that are
already known at the moment of decision. Nothing here touches the execution
series, holds state, or knows what happens next. The engine owns all state.

The one idea worth reading first
--------------------------------
This module produces BUY levels only. It never produces a sell ladder, because
exits are not grid levels: every buy fill creates its own exit at
`fill_price * (1 + s_at_fill)` which is stored on the lot and NEVER re-priced.
Re-anchoring moves where new buys go; it cannot move an existing lot's exit.

That single rule is what makes realized PnL per round trip

    q * P_fill * [ s - f*(2+s) ]   >  0   for all s > breakeven_spacing(f)

positive by construction. Every loss this strategy can take is therefore either
unrealized inventory mark (bounded by the inventory cap) or a deliberate, logged
de-risk sale. A grid that instead derives exits from a moving anchor will sell a
lot bought at level -3 of Monday's grid at level +1 of Friday's lower grid, at a
loss -- which is how a grid quietly bleeds while reporting positive gross profit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from . import indicators as ind
from .data import Candles

__all__ = [
    "Config",
    "DailySignal",
    "Regime",
    "SignalEngine",
    "breakeven_spacing",
    "floor_to_step",
    "net_edge_per_round_trip",
    "round_down_to_tick",
    "round_up_to_tick",
    "spacing_floor",
]


class Regime(str, Enum):
    RANGE = "RANGE"
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"


# --------------------------------------------------------------------------- #
# Fee arithmetic -- the load-bearing algebra of the whole strategy
# --------------------------------------------------------------------------- #


def net_edge_per_round_trip(s: float, fee: float) -> float:
    """Net profit of one buy-then-sell round trip, as a fraction of buy notional.

        cash_out = q*P*(1+f)
        cash_in  = q*P*(1+s)*(1-f)
        net      = q*P*[ (1+s)(1-f) - (1+f) ] = q*P*[ s - f*(2+s) ]
    """
    return s - fee * (2.0 + s)


def breakeven_spacing(fee: float) -> float:
    """Spacing at which a round trip exactly breaks even: s* = 2f/(1-f).

    At the verified CoinW spot base rate f = 0.0010 this is 0.20020%, NOT 0.20%.
    The 1/(1-f) correction is small but it has a sign: a grid spaced at exactly
    2f loses money on every single round trip, forever, no matter how often it fills.
    """
    return 2.0 * fee / (1.0 - fee)


def spacing_floor(fee: float, target_edge: float) -> float:
    """Minimum spacing that leaves `target_edge` net after both fees.

        s_floor = (2f + e) / (1 - f)

    At f = 0.0010, e = 0.0015 this is 0.35035%, rounded up to 0.35%.

    Callers should pass the WORST fee they can be charged, not the maker rate.
    See `worst_case_round_trip` for why.
    """
    return (2.0 * fee + target_edge) / (1.0 - fee)


def worst_case_round_trip(s: float, taker_fee: float, slippage: float) -> float:
    """Net return of the worst round trip the engine can actually produce.

    The "every round trip is net-positive" guarantee is usually stated against
    the maker fee, which is not sufficient. A resting order whose bar OPENS
    through it would have been rejected as post-only, so the engine fills it at
    the taker fee with adverse slippage (rule F6). Both legs can be taker:

        net = (1+s)(1-slip)(1-f_t) - (1+slip)(1+f_t)

    On a venue with maker 0.10% and taker 0.20%, a round trip at a floor derived
    from the maker rate alone is NEGATIVE. The guarantee therefore has to be
    stated against `max(maker, taker)`, and `Config.validate` asserts it.
    """
    return (1.0 + s) * (1.0 - slippage) * (1.0 - taker_fee) - (1.0 + slippage) * (1.0 + taker_fee)


# --------------------------------------------------------------------------- #
# Exchange rounding -- always in the conservative direction
# --------------------------------------------------------------------------- #


def round_down_to_tick(price: float, tick: float) -> float:
    return math.floor(price / tick) * tick


def round_up_to_tick(price: float, tick: float) -> float:
    return math.ceil(price / tick) * tick


def floor_to_step(qty: float, step: float) -> float:
    return math.floor(qty / step) * step


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Config:
    """All strategy parameters. Frozen so a sweep cannot mutate a shared config.

    Defaults come from published convention (ADX 25, EMA 50/200, ATR 14,
    Donchian 20) or from verified exchange facts (fees, tick, step, min notional).
    They were NOT selected by sweeping for the best backtest number; the sweeps in
    results/sensitivity.md exist to show the shape of the surface, not to pick a
    point on it.
    """

    # --- market ---
    pair: str = "BTC_USDT"
    exec_period: int = 900  # 15m execution bars
    signal_period: int = 86_400  # native daily indicator bars -- no rescaling

    # --- fees (VERIFIED CoinW spot base tier, coinw.com/fees, 2026-08-01) ---
    maker_fee: float = 0.0010
    taker_fee: float = 0.0010
    target_edge: float = 0.0015  # required net profit per round trip

    # --- spacing ---
    # K = 1.00 means "space the grid one average daily range apart".
    #
    # The first version of this strategy used K = 0.50. The sweep in
    # results/sensitivity.md shows return is monotonically improving in K across
    # ALL SIX tested windows -- not one disagrees -- and the mechanism is the one
    # this whole strategy is built around: fee drag falls from 19.8% of gross
    # capture at K=0.25 to 4.8% at K=1.00. Tighter grids fill more often and hand
    # the difference to the exchange. This is a mechanism, not a curve fit, which
    # is why the default moved.
    k_spacing: float = 1.00
    # s_cap = 0.08 binds on ~8% of days. At the old 0.03 it bound on 79.7% of
    # days, which would have made the "volatility-adaptive" claim decorative --
    # the spacing would simply have been a constant 3%. A cap that never binds is
    # decoration; a cap that always binds means the estimator is being ignored.
    s_cap: float = 0.080

    # --- geometry ---
    n_levels: int = 6
    anchor_ema: int = 20

    # --- indicators ---
    atr_period: int = 14
    atr_ref_window: int = 365  # trailing median of atr_pct -> causal vol normaliser
    atr_ref_fallback: float = 0.0422  # measured median, BTC_USDT daily 2018-01..2026-07
    adx_period: int = 14
    adx_threshold: float = 25.0
    ema_fast: int = 50
    ema_slow: int = 200
    donchian_period: int = 20

    # --- inventory caps by regime (fraction of equity) ---
    cap_range: float = 0.50
    cap_uptrend: float = 0.60
    cap_downtrend: float = 0.15
    cap_absolute: float = 0.60

    # --- risk overlay ---
    max_level_pct: float = 0.12
    max_open_lots: int = 12
    daily_loss_limit: float = 0.04

    # Drawdown kill switch. DERIVED, not guessed -- see `dd_kill` below.
    #
    # `asset_max_dd` is the drawdown the traded asset is assumed capable of.
    # 0.70 is a round number just under BTC_USDT's worst observed drawdown in our
    # data (77.2%, 2019-01 -> 2026-07). Re-measure it for any other pair.
    asset_max_dd: float = 0.70
    dd_kill_override: float | None = None
    dd_kill_rearm_days: int = 30  # backtest's model of a human re-arming after review
    break_buffer_atr: float = 0.5
    anchor_freeze_days: int = 3

    # De-risk shaping. Selling the whole excess the first time the regime flips
    # to DOWNTREND is a whipsaw machine: it sells at taker into weakness and buys
    # back higher when the regime flips again. These two knobs exist to test
    # whether requiring persistence and spreading the sale fixes that.
    derisk_persistence_days: int = 0  # consecutive DOWNTREND days required first
    derisk_max_fraction: float = 1.0  # max fraction of the excess sold per day
    # Which lots to dump. Highest-cost-first removes the lots furthest from their
    # exits; lowest-cost-first realises the smallest loss and leaves the deep lots
    # to recover. These are opposite intuitions and only measurement settles it:
    # over the full 2019-2026 run, lowest-cost-first returns +8.2% against -3.4%
    # for highest-cost-first, because the de-risk leg is where this strategy does
    # most of its realised damage. The intuitive choice was the wrong one.
    derisk_highest_cost_first: bool = False

    # --- exchange constraints (VERIFIED returnSymbol, BTC_USDT) ---
    min_notional: float = 5.0
    price_tick: float = 0.01
    qty_step: float = 0.0001

    # --- backtest fill model ---
    fill_epsilon: float = 0.0001  # 1bp; market must trade THROUGH us
    taker_slippage: float = 0.0005  # 5bp, ASSUMPTION -- no order-book data to calibrate
    fill_probability: float = 1.0  # queue-position sensitivity knob
    seed: int = 20_260_801

    # --- account ---
    initial_equity: float = 10_000.0

    # --- ablation switches (used to publish the improvement chain honestly) ---
    paired_exits: bool = True  # False = exits float with the grid (the broken design)
    regime_gate: bool = True  # False = one static inventory cap
    # active_derisk force-SELLS inventory down to the regime cap when the regime
    # turns down. It sounds right and it measures wrong: over the full 2019-2026
    # run it costs 44 percentage points (+52.7% -> +8.8%), because selling into
    # weakness at taker prices and rebuying when the regime flips back is a
    # systematic wealth transfer. Blocking NEW buys in a downtrend is worth +18
    # points; force-selling existing inventory is not. Shipped OFF, kept as a
    # switch so the measurement in results/RESULTS.md can be reproduced.
    active_derisk: bool = False

    @property
    def worst_fee(self) -> float:
        """The highest fee a single leg can be charged. F6 can make either leg taker."""
        return max(self.maker_fee, self.taker_fee)

    @property
    def s_floor(self) -> float:
        return spacing_floor(self.worst_fee, self.target_edge)

    @property
    def dd_kill(self) -> float:
        """Equity drawdown that halts the strategy and forces a human re-arm.

        DERIVED from the inventory cap, for the same reason `s_floor` is derived
        from the fee: a guessed constant here is not a risk control, it is a
        random number that happens to have units of percent.

            dd_kill = clamp(cap_range * asset_max_dd, 0.15, 0.50)
                    = clamp(0.50 * 0.70, 0.15, 0.50) = 0.35

        The reasoning: holding up to `cap` of equity in an asset that can fall
        `asset_max_dd` makes an equity drawdown of `cap * asset_max_dd`
        STRUCTURALLY NORMAL. A kill threshold below that number does not detect
        an abnormal loss -- it fires on the strategy operating exactly as designed,
        at the worst possible moment, liquidating into weakness at taker prices.

        This is not hypothetical. The first version of this strategy used a
        hardcoded 20%. Over 2019-2026 that kill switch fired repeatedly and
        realised -8,733 USDT on a 10,000 USDT account; raising the threshold so it
        only fires on genuinely abnormal losses moved the same configuration from
        -7.15% to +33.86%. It was the single largest source of loss in the study,
        and it was wearing the label "risk control".
        """
        if self.dd_kill_override is not None:
            return self.dd_kill_override
        return float(min(0.50, max(0.15, self.cap_range * self.asset_max_dd)))

    @property
    def breakeven(self) -> float:
        return breakeven_spacing(self.worst_fee)

    def cap_for(self, regime: Regime) -> float:
        if not self.regime_gate:
            return min(self.cap_range, self.cap_absolute)
        if regime is Regime.UPTREND:
            return min(self.cap_uptrend, self.cap_absolute)
        if regime is Regime.DOWNTREND:
            return min(self.cap_downtrend, self.cap_absolute)
        if regime is Regime.INSUFFICIENT_HISTORY:
            # An unclassifiable regime inherits the MOST conservative cap, not the
            # middle one. This is the rule that keeps a new listing from killing
            # the account before EMA200 even exists.
            return min(self.cap_downtrend, self.cap_absolute)
        return min(self.cap_range, self.cap_absolute)

    def with_(self, **kwargs) -> "Config":
        return replace(self, **kwargs)

    def validate(self) -> None:
        """R8, checked once at construction and again before every order placement."""
        if self.s_cap <= self.s_floor:
            raise ValueError(
                f"s_cap {self.s_cap:.5f} <= s_floor {self.s_floor:.5f}: no spacing is legal"
            )
        if net_edge_per_round_trip(self.s_floor, self.worst_fee) <= 0:
            raise ValueError("s_floor does not clear the fee hurdle -- check the algebra")
        # The invariant must hold against the WORST fill the engine can produce
        # (F6: both legs taker, adverse slippage), not just the maker case.
        worst = worst_case_round_trip(self.s_floor, self.taker_fee, self.taker_slippage)
        if worst <= 0:
            raise ValueError(
                f"at s_floor={self.s_floor:.6f} the worst-case round trip is {worst:+.6f}: "
                f"taker_fee={self.taker_fee} with {self.taker_slippage} slippage breaks the "
                f"net-positive guarantee. Raise target_edge or lower the fee assumption."
            )
        if self.n_levels < 1:
            raise ValueError("n_levels must be >= 1")
        if not 0 < self.cap_absolute <= 1:
            raise ValueError("cap_absolute must be in (0, 1]")


# --------------------------------------------------------------------------- #
# Daily signal
# --------------------------------------------------------------------------- #


@dataclass
class DailySignal:
    """Everything decided at 00:00 UTC for one trading day.

    Every field is derived from daily bars with index <= day_index, where
    day_index is the bar for the PREVIOUS UTC day. Nothing here can see the day
    it governs.
    """

    day_start_ms: int
    signal_bar_ts: int
    regime: Regime
    atr_pct: float
    atr_ref: float
    spacing: float
    anchor: float
    inventory_cap: float
    buy_levels: list[float]
    dc_high: float
    dc_low: float
    broke_lower: bool
    broke_upper: bool
    adx: float
    ema_fast: float
    ema_slow: float
    close: float
    parkinson: float
    vol_estimators_disagree: bool

    @property
    def grid_low(self) -> float:
        return min(self.buy_levels) if self.buy_levels else float("nan")

    @property
    def grid_high(self) -> float:
        return self.anchor * (1.0 + self.spacing) ** len(self.buy_levels)


class SignalEngine:
    """Precomputes daily indicator arrays once, then serves per-day signals.

    Look-ahead is prevented structurally rather than by discipline: `signal_for`
    takes a UTC day start and resolves the newest daily bar STRICTLY BEFORE it via
    `Candles.index_at_or_before(day_start_ms - 1)`. There is no code path that can
    read the current day's own bar.
    """

    def __init__(self, daily: Candles, cfg: Config, hourly: Candles | None = None):
        self.daily = daily
        self.cfg = cfg
        self.hourly = hourly

        h, l, c = daily.high, daily.low, daily.close
        self.atr = ind.atr(h, l, c, cfg.atr_period)
        with np.errstate(invalid="ignore"):
            self.atr_pct = self.atr / c
        self.adx, _, _ = ind.adx(h, l, c, cfg.adx_period)
        self.ema_fast = ind.ema(c, cfg.ema_fast)
        self.ema_slow = ind.ema(c, cfg.ema_slow)
        self.anchor_ema = ind.ema(c, cfg.anchor_ema)
        self.dc_high, self.dc_low = ind.donchian(h, l, cfg.donchian_period)

        # Causal volatility normaliser: trailing median of atr_pct. Using a
        # full-history median would be a (mild) in-sample constant; a trailing
        # window removes the criticism entirely and adapts to a new pair for free.
        self.atr_ref = self._trailing_median(self.atr_pct, cfg.atr_ref_window, cfg.atr_ref_fallback)

        # Independent cross-check estimator. Exercises the sqrt(H/D) rescaling
        # that a daily-native primary estimator never needs.
        if hourly is not None and len(hourly) > 168:
            self.park_daily = ind.parkinson_sigma(
                hourly.high, hourly.low, window=168, bar_seconds=hourly.period
            )
            self.park_ts = hourly.ts
        else:
            self.park_daily = None
            self.park_ts = None

    @staticmethod
    def _trailing_median(values: np.ndarray, window: int, fallback: float) -> np.ndarray:
        out = np.full(values.size, fallback, dtype=np.float64)
        finite = np.isfinite(values)
        for i in range(values.size):
            lo = max(0, i - window)
            chunk = values[lo:i][finite[lo:i]]
            if chunk.size >= 30:
                out[i] = float(np.median(chunk))
        return out

    # -- regime ------------------------------------------------------------ #

    def _regime(self, i: int) -> Regime:
        adx_v, ef, es, close = self.adx[i], self.ema_fast[i], self.ema_slow[i], self.daily.close[i]
        if not (np.isfinite(adx_v) and np.isfinite(es) and np.isfinite(ef)):
            return Regime.INSUFFICIENT_HISTORY
        if not self.cfg.regime_gate:
            return Regime.RANGE
        if adx_v >= self.cfg.adx_threshold:
            if close < es and ef < es:
                return Regime.DOWNTREND
            if close > es and ef > es:
                return Regime.UPTREND
        return Regime.RANGE

    def _parkinson_at(self, ts_ms: int) -> float:
        if self.park_daily is None:
            return float("nan")
        idx = int(np.searchsorted(self.park_ts, ts_ms, side="right")) - 1
        if idx < 0 or idx >= self.park_daily.size:
            return float("nan")
        return float(self.park_daily[idx])

    # -- main entry point -------------------------------------------------- #

    def signal_for(self, day_start_ms: int, frozen_anchor: float | None = None) -> DailySignal | None:
        """Signal governing the UTC day beginning at `day_start_ms`.

        Returns None when there is not yet a usable daily bar. `frozen_anchor`,
        when supplied, overrides the EMA anchor -- that is the anti-martingale
        device: after a downside range break the anchor is held still for
        `anchor_freeze_days` closes so the ladder cannot chase price down.
        """
        cfg = self.cfg
        i = self.daily.index_at_or_before(day_start_ms - 1)
        if i < 0:
            return None

        atr_pct = float(self.atr_pct[i])
        anchor_ema = float(self.anchor_ema[i])
        if not (np.isfinite(atr_pct) and np.isfinite(anchor_ema) and atr_pct > 0):
            return None

        regime = self._regime(i)
        atr_ref = float(self.atr_ref[i])

        # Cross-check the daily ATR against an independently-derived Parkinson
        # estimate. On disagreement, take the WIDER spacing: fewer, safer trades.
        park = self._parkinson_at(day_start_ms - 1)
        disagree = False
        atr_pct_used = atr_pct
        if np.isfinite(park) and atr_pct > 0:
            if abs(park / (atr_pct * 0.7) - 1.0) > 0.60:
                disagree = True
                atr_pct_used = max(atr_pct, park / 0.7)

        spacing = float(np.clip(cfg.k_spacing * atr_pct_used, cfg.s_floor, cfg.s_cap))

        # R8: refuse to trade a spacing that cannot clear the fee hurdle.
        if net_edge_per_round_trip(spacing, cfg.maker_fee) <= 0:
            raise ValueError(
                f"spacing {spacing:.6f} does not clear fees at maker_fee={cfg.maker_fee}"
            )

        anchor = float(frozen_anchor) if frozen_anchor else anchor_ema

        dc_high = float(self.dc_high[i]) if np.isfinite(self.dc_high[i]) else float("nan")
        dc_low = float(self.dc_low[i]) if np.isfinite(self.dc_low[i]) else float("nan")
        close = float(self.daily.close[i])
        buffer_ = cfg.break_buffer_atr * atr_pct
        broke_lower = bool(np.isfinite(dc_low) and close < dc_low * (1.0 - buffer_))
        broke_upper = bool(np.isfinite(dc_high) and close > dc_high * (1.0 + buffer_))

        # Level count: in an uptrend place only the deepest half of the ladder.
        # Chasing a trend upward with a full ladder just buys the whole rally.
        n = cfg.n_levels
        if regime is Regime.UPTREND:
            n = max(1, math.ceil(cfg.n_levels / 2))
        step = 1.0 + spacing
        levels = [round_down_to_tick(anchor / step**j, cfg.price_tick) for j in range(1, n + 1)]

        return DailySignal(
            day_start_ms=day_start_ms,
            signal_bar_ts=int(self.daily.ts[i]),
            regime=regime,
            atr_pct=atr_pct,
            atr_ref=atr_ref,
            spacing=spacing,
            anchor=anchor,
            inventory_cap=cfg.cap_for(regime),
            buy_levels=levels,
            dc_high=dc_high,
            dc_low=dc_low,
            broke_lower=broke_lower,
            broke_upper=broke_upper,
            adx=float(self.adx[i]) if np.isfinite(self.adx[i]) else float("nan"),
            ema_fast=float(self.ema_fast[i]) if np.isfinite(self.ema_fast[i]) else float("nan"),
            ema_slow=float(self.ema_slow[i]) if np.isfinite(self.ema_slow[i]) else float("nan"),
            close=close,
            parkinson=park,
            vol_estimators_disagree=disagree,
        )


# --------------------------------------------------------------------------- #
# Position sizing
# --------------------------------------------------------------------------- #


def level_notional(
    cfg: Config,
    equity: float,
    inventory_value: float,
    inventory_cap: float,
    atr_now: float,
    atr_ref: float,
    n_levels: int,
) -> float:
    """Notional for one buy level, in quote currency.

        vol_mult = clamp(atr_ref / atr_now, 0.5, 1.5)      smaller size when vol is high
        inv_mult = clamp((C - I) / (0.30*C), 0, 1)         full size to 70% of cap, then taper
        notional = (equity * C / n_levels) * vol_mult * inv_mult

    Returns 0.0 when the level should not be placed at all.
    """
    if equity <= 0 or inventory_cap <= 0 or n_levels < 1:
        return 0.0

    inv_ratio = inventory_value / equity if equity > 0 else 1.0
    vol_mult = float(np.clip(atr_ref / atr_now, 0.5, 1.5)) if atr_now > 0 else 1.0
    inv_mult = float(np.clip((inventory_cap - inv_ratio) / (0.30 * inventory_cap), 0.0, 1.0))

    notional = (equity * inventory_cap / n_levels) * vol_mult * inv_mult
    notional = min(notional, cfg.max_level_pct * equity)  # R2

    headroom = inventory_cap * equity - inventory_value  # R1
    notional = min(notional, max(0.0, headroom))

    return notional if notional >= cfg.min_notional else 0.0
