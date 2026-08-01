---
name: fee-floored-adaptive-grid
description: >
  Submission for the CWC AI Trading Skill Challenge.
  This Skill defines a spot grid strategy whose level spacing is derived from measured volatility
  and floored by the exchange fee, so that every completed round trip is net-positive by
  construction. It ships with a reproducible backtest on eight years of CoinW BTC_USDT data,
  including the windows where the strategy loses.
---

# Fee-Floored Adaptive Grid

> Submission for the CWC AI Trading Skill Challenge.
> This Skill is designed to be structured, reviewed, and executed by an AI Agent, and every
> quantitative claim in it is reproducible with one command.

## 1. Skill Name

**Fee-Floored Adaptive Grid**

## 2. Strategy Type

**Grid**

A long-biased spot grid. It does not forecast direction. It places a ladder of buy limit orders
below a volatility-scaled anchor and pairs each filled lot with its own sell order, harvesting
oscillation as realised cash flow.

Two properties separate it from a conventional grid. First, **the spacing is floored by the fee**:
the minimum spacing is derived from the exchange's fee schedule rather than chosen — from
`max(maker, taker)`, because a gap-through fill is charged as taker — so no level can be placed
that loses money on a completed round trip. Second, **exits are attached to lots, not to
the grid**: when a lot fills, its sell price is fixed at that moment and is never re-priced, even
when the grid re-anchors. A grid that instead derives exits from a moving anchor will sell a lot
bought at a deep level into a lower grid, at a loss, while still reporting positive gross profit.

## 3. Applicable Market

- **Markets:** Crypto **spot** only. No leverage, no perpetuals, therefore no liquidation risk.
- **Exchange:** CoinW. All market data comes from public, unauthenticated endpoints.
- **Trading pair:** `BTC_USDT` (calibrated); any pair from `command=returnSymbol` after recalibration.
- **Execution timeframe:** 15m
- **Signal timeframe:** 1D (native daily candles; no resampling and no volatility rescaling)
- **Best suited for:** Range-bound, choppy, and declining markets. Measured on CoinW data, it beat
  buy-and-hold in the 2022 bear (-33.4% against -64.3%), in the choppy 2019H2-2020Q1 window
  (+11.6% against -21.6%), and year-to-date in 2026 (-11.2% against -28.3% while BTC sits
  roughly 50% below its 2025 high).
- **Less suited for:** Sustained trending markets. It **loses badly** to buy-and-hold in a bull run
  (2020Q4-2021Q1: +10.8% against +444.5%). Selling into strength is what a grid does; this is the
  price of the strategy, not a defect in it.

## 4. Core Logic

The strategy answers one question:

**At what prices should we bid, how far apart, and how much inventory may we hold while doing it?**

```text
Signal source (all from daily bars strictly <= D-1; nothing reads the day it governs):
  atr_pct  = ATR(14) / close            Wilder, on native daily candles
  A        = EMA(20) of daily closes    the grid anchor
  ADX14, EMA50, EMA200                  regime classification
  DC_high, DC_low = Donchian(20)        range-break invalidation

Spacing, floored by the fee:
  f        = max(maker, taker) = 0.0010  (CoinW spot base rate; a gap-through leg pays taker)
  s_floor  = (2f + e) / (1 - f) = 0.35%  e = 0.0015 required net edge
  s        = clamp(K * atr_pct, s_floor, 8.00%)     K = 1.00
  -> net profit per round trip = s - f*(2+s), positive for all s > 2f/(1-f) = 0.2002%

Geometry (geometric, so net edge is identical at every level):
  buy_j    = A / (1 + s)^j     for j = 1..N        N = 6
  exit(lot) = fill_price * (1 + s_at_fill)         FIXED at fill, never re-priced

Rule 1: RANGE       ADX14 < 25, or price and EMA50 not aligned with EMA200
  -> No trend worth respecting.
  -> Place all N buy levels. Inventory cap 50% of equity.

Rule 2: UPTREND     ADX14 >= 25 and close > EMA200 and EMA50 > EMA200
  -> Do not fight the trend by laddering into every dip.
  -> Place only the deepest ceil(N/2) levels. Inventory cap 60% of equity.

Rule 3: DOWNTREND   ADX14 >= 25 and close < EMA200 and EMA50 < EMA200
  -> STOP ADDING. Place no new buy orders at all.
  -> Existing per-lot exits stay live. Do NOT force-sell inventory.
  -> Inventory cap 15% of equity, enforced by refusing to buy, not by dumping.

Risk Overlay (all caps hard; any breach halts new buys immediately):
  R1  inventory <= cap * equity, evaluated on the post-trade book; absolute ceiling 60%
  R2  per-level notional <= 12% of equity
  R3  <= 6 resting buy orders, <= 12 open lots
  R4  equity < 96% of the 00:00 UTC mark -> cancel buys, keep exits, resume next UTC day
  R5  equity <= (1 - dd_kill) * high-water mark -> liquidate to 15%, HALT, human re-arm
        dd_kill = clamp(cap_range * asset_max_dd, 0.15, 0.50) = 35%   [DERIVED, not chosen]
  R6  data-quality halt: stale, missing, malformed, or empty data -> place nothing
  R7  cancel and re-place any resting order older than 24h
  R8  assert s >= (2f + e)/(1 - f) before EVERY order; on violation place nothing
```

## 5. Why Floor the Spacing at the Fee?

Because a grid that ignores fees is not a strategy, it is a rebate programme for the exchange.

A buy at `P` and a sell at `P(1+s)`, both paying maker fee `f`, nets
`q·P·[s − f(2+s)]`. Setting that to zero gives break-even spacing `s* = 2f/(1−f)`. At CoinW's
published 0.10% spot rate that is **0.2002%**, not 0.20% — and a grid spaced at exactly `2f` loses
on every round trip, forever, regardless of how often it fills. Fee drag, the share of gross spread
capture consumed by fees, is `f(2+s)/s`: **80% at 0.25% spacing, 4.8% at the 4.2% median spacing
this strategy actually uses.**

This is not theoretical, but it is also not as clean as we first claimed. Sweeping `K` across six
independent windows, **fee drag falls monotonically in all six**, from 20.0% at `K=0.25` to 2.9% at
`K=2.00`, and the average return improves. The *return* ordering is messier: only **2 of the 6**
windows are individually monotone, and `bull-2020Q4` is actually better at a tighter setting than
the one shipped. An earlier draft of this Skill asserted monotonicity in all six; an audit checked,
and it was wrong. The default therefore rests on the **cost mechanism**, which does hold everywhere
— tighter grids fill more often and hand the difference to the exchange — and not on a universal
ordering of returns. The full per-window decomposition is computed into `results/sensitivity.md`.

The volatility input is deliberately boring. CoinW serves **native daily candles**, so daily ATR is
read directly and there is no rescaling step to get wrong. For reference, converting a volatility
measured on bars of length `D` to horizon `H` requires `σ_H = σ_D·√(H/D)`; 5-minute bars to daily is
`√288 = 16.97`, and using `√24 = 4.90` — the factor for *hourly* bars — understates daily volatility
by **3.46×**, collapsing an "adaptive" grid onto its own clamp. The safest way to avoid that error
is to not perform the calculation.

## 6. Agent Execution Flow

```text
Step 1: Fetch market data              (3 unauthenticated GETs per cycle, 10 req/s limit)
- GET returnChartData  period=86400  last 400 days   -> indicators
- GET returnChartData  period=900    last 7 days     -> execution bars
- GET returnSymbol                                   -> tick, step, min notional, pair state
- An explicit User-Agent header is REQUIRED; the Python default returns HTTP 403.
- Run every R6 check. Any failure -> emit HALTED_DATA and stop. Never trade on partial data.

Step 2: Calculate indicators           (daily bars <= D-1 only; zero look-ahead)
- atr_pct, EMA20 anchor, EMA50, EMA200, ADX14, Donchian(20)
- Parkinson high-low estimator on 1h bars as an independent cross-check;
  on >60% disagreement, use the WIDER spacing and flag VOL_ESTIMATOR_DISAGREE
- s = clamp(K * atr_pct, s_floor, s_cap);  assert R8 before continuing

Step 3: Apply signal rules
- Classify regime -> RANGE / UPTREND / DOWNTREND / INSUFFICIENT_HISTORY
- Check Donchian lower break; if broken in a downtrend, FREEZE the anchor for 3 daily
  closes so the ladder cannot chase price down
- Evaluate R1-R8; cancel resting buys older than 24h
- Place buy levels sized by equity, volatility and inventory headroom
- Ensure every open lot has a live exit; NEVER re-price an existing lot's exit

Step 4: Produce output
- Current state, regime, spacing and its binding constraint
- Grid levels and open lots with their fixed exits
- Executable actions, invalidation conditions, and anything needing escalation
```

**Autonomous vs escalated.** The Agent may place, cancel and refresh orders, size within the caps,
classify regime, and trigger the R4 and R6 halts. It must **stop and escalate to a human** on the R5
drawdown kill, on any structural invalidation, on a pair whose `state != 1`, and on a detected fee-tier
change. **The Agent may lower a risk cap but may never raise one** — caps are a one-way ratchet toward
safety unless a human intervenes.

## 7. Core Parameters

| Parameter | Default Value | Meaning | Adjustment Notes |
|---|---:|---|---|
| `k_spacing` | 1.00 | Spacing = K × daily ATR% | Fee drag falls monotonically in all six tested windows (20.0% → 2.9%); return improves on average but only 2 of 6 windows are individually monotone |
| `s_floor` | *derived* 0.35% | `(2f + e)/(1 − f)` | Never hardcode. Recompute whenever the fee or target edge changes |
| `s_cap` | 8.00% | Maximum spacing | Binds ~8% of days. At 3% it bound 80% of days and made adaptation decorative |
| `n_levels` | 6 | Buy levels per side | Range half-width ≈ `N × K × atr_pct` ≈ ±25% at median volatility |
| `maker_fee` / `taker_fee` | 0.0010 | CoinW spot base rate (verified) | **Set to your actual tier.** Lowering is safe; raising re-derives `s_floor` |
| `target_edge` | 0.0015 | Required net profit per round trip | Higher means fewer, fatter trades. `validate()` **rejects** any value where the worst-case round trip (both legs taker, adverse slippage) is not positive — at the shipped fees that is anything at or below 0.0010 |
| `cap_range` / `cap_uptrend` / `cap_downtrend` | 0.50 / 0.60 / 0.15 | Inventory cap by regime | **The dominant risk lever.** Full-run max drawdown runs 22.4% / 26.1% / 41.9% at caps of 0.20 / 0.30 / 0.50 |
| `dd_kill` | *derived* 35% | Drawdown kill switch | `clamp(cap_range × asset_max_dd, 0.15, 0.50)`. Must exceed `cap × asset drawdown`, or it fires on normal operation |
| `adx_threshold` | 25 | Trend-strength gate | Wilder's conventional value; not tuned. Median ADX14 on BTC_USDT daily 2018-2026 is 26.5 |
| `ema_fast` / `ema_slow` | 50 / 200 | Trend-direction gate | Conventional pair; not tuned |
| `max_level_pct` | 0.12 | Per-level notional ceiling | Hard cap |
| `donchian_period` | 20 | Range-break lookback | Used for invalidation only, never for anchoring |

## 8. Standard Output Format

```text
Fee-Floored Adaptive Grid | [Date] (UTC)

PAIR          BTC_USDT (spot)
STATE         ACTIVE / HALTED_DATA / HALTED_LOSS / KILLED / INSUFFICIENT_HISTORY
REGIME        RANGE / UPTREND / DOWNTREND   (ADX14 NN.N | EMA50 vs EMA200)
VOLATILITY    ATR14 N.NN%/day               (Parkinson cross-check N.NN% — AGREE / DISAGREE)
SPACING       s = N.NN%                     (K x ATR%, floor 0.35%, cap 8.00% — which bound, if any)
              net edge per round trip N.NN% | fee drag N.N%
ANCHOR        A = NNNNN.NN                  (EMA20 of daily closes)
GRID          N buy levels: NNNNN / NNNNN / NNNNN / ...
              covered range NNNNN - NNNNN (+/-NN.N%)
INVENTORY     N.NNNN BTC = NNNN USDT        (NN.N% of equity | cap NN.N% | headroom NNNN)
OPEN LOTS     N   fixed exits: NNNNN / NNNNN / ...
EQUITY        NNNNN USDT                    (cash NNNNN + inventory NNNN)
              high-water NNNNN | drawdown N.N% | kill at 35.0%
RISK CHECKS   R1 ok  R2 ok  R3 ok(N/12)  R4 ok  R5 ok  R6 ok  R7 ok  R8 ok

Suggested Action:
[Example: Place BUY 0.0161 BTC @ 62,161.00 post-only; keep 4 exits unchanged.]

Invalidation Condition:
[Example: Daily close below Donchian-20 low minus 0.5 ATR while in DOWNTREND
 -> freeze anchor for 3 closes and stop adding.]

Confidence:
__%

Risk Notice:
This output is for strategy demonstration only and does not constitute investment advice.
```

## 9. Risk Notice

**Applicable conditions.** Spot only, no leverage, one pair at a time, on an account whose entire
balance the operator can afford to lose. It requires a venue with maker fees at or below roughly
0.10% per side and a pair with at least 200 days of daily history. It is designed for range-bound,
choppy and declining markets.

**Invalidation conditions — the strategy is not working; stop and review.**

- **Range break.** A daily close below the Donchian-20 low minus 0.5 ATR while in DOWNTREND. The
  anchor freezes for 3 daily closes so the ladder cannot chase price down.
- **Inertia.** Fewer than 20 completed round trips in 90 days: the spacing is too wide, or price has
  left the grid.
- **Cost dominance.** Fees exceeding 40% of gross spread capture over 90 days: the spacing is too
  tight for the live fee tier.
- **Purpose failure.** A 180-day return more than 25 points below buy-and-hold *while* max drawdown
  exceeds 25%: the strategy is delivering neither the return nor the lower drawdown it exists for.
- **Broken invariant.** Any completed round trip closing at a loss outside a logged de-risk event.
  This is impossible under the paired-exit design, so it is a code defect, not a market event, and
  it is a hard stop.
- **Stale fee assumption.** A live fee tier different from the configured `maker_fee`. `s_floor` is
  derived from the fee; if they disagree, every level is mispriced.

**Maximum risk assumptions.**

- **This strategy is structurally long BTC. It is not market-neutral, not hedged, and not
  risk-free.** In the worst tested window it lost **33.4%** of the account, with a maximum drawdown
  of **35.5%**, and over the full 2018-2026 run its maximum drawdown was **41.9%**.
- The realistic worst case is worse than the backtest. The kill switch assumes a fill; a venue
  outage, a withdrawal freeze, or a gap through every level can prevent one. **Assume total loss of
  the allocated capital is possible.**
- Inventory is capped at 50-60% of equity by rule, so roughly half the account can be exposed to a
  single asset's drawdown at any moment.
- The backtest models a human re-arming the kill switch 30 days after it fires. A real operator may
  re-arm sooner, later, or never; this materially affects outcomes.
- Fill modelling assumes we are at the front of the queue whenever price trades through a resting
  order. On a thin book this is optimistic; results at 70% and 50% fill probability are published.
- Slippage on taker fills is assumed at 5bp. That is an assumption, not a measurement — no
  order-book data exists to calibrate it.

**What this Skill does not claim.**

- **It has never been run against a real CoinW account.** It is documentation plus a historical
  simulation, and nothing here is a live-trading record.
- **It does not beat holding BTC.** Over the full 2018-2026 run it returned **+30.8%** against
  **+446.1%** for buy-and-hold. It won 3 of 7 windows against buy-and-hold and 3 of 7 against an
  exposure-matched benchmark. **If you believe BTC is going up, hold BTC.** This strategy is for
  wanting volatility exposure without a directional view, and accepting capped upside in exchange
  for roughly half the drawdown (41.9% against 77.2% over the full run).
- No forward-looking return is projected. Past behaviour on historical data is not a forecast.
- The 100% win rate on completed round trips is **an invariant of the exit design, not skill**. It
  excludes de-risk and kill-switch sales, which are frequently losses and are reported separately.
- Moving averages and ADX are lagging and generate false signals in sideways markets. The regime
  gate has **no statistically significant directional predictive power**: over 475 DOWNTREND days
  the mean forward 5-day return is −0.38% against a 7.31% standard deviation, giving t = −1.13,
  which is not significant. What DOWNTREND days reliably have is higher realised volatility (median
  daily ATR 5.01% against 3.86% in RANGE) — which is where a long-biased grid accumulates inventory
  fastest. **It is a risk control, not an alpha source.**
- If market data is delayed, missing, or unreliable, the Agent must stop generating trading actions
  until data quality is restored.
- This Skill is provided for educational and activity demonstration purposes only. It does not
  constitute investment advice, financial advice, or trading advice.

## 10. Submission Checklist

- Skill name — §1
- Strategy type — §2
- Applicable market — §3
- Core logic — §4
- Core parameters — §7
- Risk notice — §9
- Public GitHub link — §11
- Agent execution flow — §6
- Standard output format — §8
- Invalidation conditions — §9
- Backtest and simulation results — `results/RESULTS.md`, reproducible with one command
- Parameter sensitivity and risk boundaries — `results/sensitivity.md`
- Trade-by-trade case review — `results/CASE-REVIEW.md`, generated from the trade log
- When a grid is and is not the right instrument — `results/WHEN-TO-GRID.md`
- Full trade log — `results/trades.csv`

## 11. Public GitHub Link

```text
https://github.com/Yunini1111/fee-floored-adaptive-grid
```

## 12. Disclaimer

This Skill is a submission to the CWC AI Trading Skill Challenge and is provided for educational and
demonstration purposes only.

It does not represent a commitment that the strategy will be listed, supported, executed, or
productized by CoinW. It does not constitute investment advice or a guarantee of returns.

Users are responsible for their own research, risk assessment, and trading decisions.
