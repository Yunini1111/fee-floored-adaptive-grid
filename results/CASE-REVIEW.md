# Case review — what the strategy actually did

Generated from `results/trades.csv` by `python run_backtest.py --all`. Every episode here is extracted from the run rather than written by hand, so it cannot drift away from the numbers it describes.

> **This is a review of a historical simulation, not of live trading.** The Skill has never been run against a real CoinW account. What follows is the trade-by-trade behaviour of the backtest on real CoinW BTC_USDT data, which is the closest honest equivalent.

---

## Episode 1 — the drawdown kill switch, June 2022

The single worst event in the whole run. At `2022-06-18T06:45:00Z` the grid sleeve hit its 35% drawdown limit and liquidated **10 lots simultaneously** at 19,367.06 USDT, realising **-3,203.76 USDT** on a 10,000 USDT account.

| Lot | Bought | Entry price | Qty | Liquidated at | Realised |
|---:|---|---:|---:|---:|---:|
| 161 | 2021-11-18 | 59,797.85 | 0.0334 | 19,367.06 | -1,353.03 |
| 165 | 2021-11-21 | 58,652.69 | 0.0230 | 19,367.06 | -905.36 |
| 166 | 2021-11-22 | 58,477.08 | 0.0083 | 19,367.06 | -325.26 |
| 163 | 2021-11-19 | 59,040.64 | 0.0052 | 19,367.06 | -206.71 |
| 167 | 2021-11-23 | 57,857.63 | 0.0044 | 19,367.06 | -169.70 |
| 170 | 2021-11-27 | 56,358.05 | 0.0029 | 19,367.06 | -107.49 |
| 164 | 2021-11-20 | 58,750.75 | 0.0013 | 19,367.06 | -51.30 |
| 172 | 2021-12-04 | 54,763.73 | 0.0011 | 19,367.06 | -39.02 |
| 168 | 2021-11-24 | 57,651.76 | 0.0006 | 19,367.06 | -23.02 |
| 169 | 2021-11-25 | 57,402.33 | 0.0006 | 19,367.06 | -22.87 |

**What it shows.** Every one of these lots was bought between `2021-11-18` and `2021-12-04` — the November 2021 top — and held all the way down. This is the strategy's core weakness in one table: a long-biased grid buys the whole way into a bear market, and the deepest lots never reach their exits.

**What a reviewer should take from it.** The circuit breaker did its job, and its job was expensive. Note the design decision behind the threshold: it is *derived* as `clamp(cap_range x asset_max_dd, 0.15, 0.50)`, not chosen. An earlier version used a flat 20%, which fires on drawdowns that a 50% inventory cap makes structurally normal — so it liquidated repeatedly, at local bottoms, at taker prices. `results/RESULTS.md` section 5a measures exactly what that cost.

## Episode 2 — did the regime gate actually stop anything?

**Buy orders filled while the regime was DOWNTREND, across the entire run: 0.** This is the passive half of the regime gate, and it is trivially checkable — filter `trades.csv` for `side=BUY` and `regime=DOWNTREND` and count the rows.

Grid entries per calendar year:

| Year | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Grid buys | 70 | 42 | 60 | 25 | 59 | 87 | 73 | 14 |

The collapse in 2022 is the gate refusing to add into a confirmed downtrend. Note what it does **not** do: it never force-sells. `results/RESULTS.md` section 5a shows that adding forced selling on top of this gate costs tens of percentage points — stopping is worth a lot, panicking is worth less than nothing.

## Episode 3 — an ordinary winning cycle, start to finish

The median round trip of 408, so it is representative rather than cherry-picked.

```text
BUY   2023-02-09T19:45:00Z   0.0630 BTC @ 21,935.84   MAKER   fee 1.3820
      regime at entry: RANGE
      exit target fixed AT THIS MOMENT at 22,570.74  (+2.89%)
SELL  2023-02-15T12:00:00Z   0.0630 BTC @ 22,570.74   MAKER   fee 1.4220
      held 136 h    realised +37.19 USDT
      fees consumed 7.0% of the gross spread
```

**The point of this episode is the third line.** The exit was fixed the instant the buy filled, and nothing that happened afterwards — re-anchoring, a regime change, a volatility spike — was allowed to move it. That is why every completed round trip in this run is net-positive: all 408 of them, zero exceptions.

## Episode 4 — where the model charges itself

**194 of 848 fills (23%) were charged the taker fee**, not because the strategy places market orders — it never does — but because the bar *opened* through the resting limit. A real post-only order would have been rejected there and the agent would have had to chase, so the model fills at the limit price and charges taker plus adverse slippage. Worst price and worst fee.

| Time | Side | Price | Qty | Fee | Reason |
|---|---|---:|---:|---:|---|
| 2019-01-11T00:15:00Z | BUY | 3,594.98 | 0.2157 | 0.7754 | GRID |
| 2019-01-14T00:15:00Z | BUY | 3,537.26 | 0.2260 | 0.7994 | GRID |
| 2019-01-29T00:15:00Z | BUY | 3,450.75 | 0.0203 | 0.0701 | GRID |
| 2019-01-30T00:15:00Z | BUY | 3,436.51 | 0.0091 | 0.0313 | GRID |
| 2019-07-15T00:15:00Z | BUY | 10,347.75 | 0.1011 | 1.0462 | GRID |

Total fees over the run: **740.70 USDT** (maker 635.57 / taker 105.13), which is 4.8% of gross spread capture. A grid that ignores this line is not a strategy, it is a rebate programme for the exchange.

## What an AI Agent would have done differently at each point

The Skill is written to be executed by an agent, so the interesting question is where the agent acts alone and where it must stop and ask.

| Episode | Agent acts autonomously | Agent must escalate |
|---|---|---|
| Episode 1 (kill switch) | Liquidates the grid sleeve to the 15% floor and sets `state = KILLED` | **Yes.** Does not auto-resume. A human must review and re-arm — this is the one place where 'just keep running' is the wrong default |
| Episode 2 (regime gate) | Classifies regime, stops placing buys, keeps every existing exit live | No |
| Episode 3 (normal cycle) | Sizes, places, and pairs the exit within hard caps | No |
| Episode 4 (taker fills) | Pays the fee and logs it | Only if the live fee tier differs from the configured rate, because `s_floor` is derived from it and every level would be mispriced |

**The rule that matters most:** the agent may *lower* any risk cap on its own authority but may **never raise one**. Caps are a one-way ratchet toward safety. The failure mode everyone worries about — an agent quietly widening its own risk limits — is closed off by construction rather than by prompt wording.
