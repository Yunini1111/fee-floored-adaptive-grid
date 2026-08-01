# Example: Fee-Floored Adaptive Grid

This file summarises the Skill for readers skimming the repository.
For the canonical version, see [`../SKILL.md`](../SKILL.md).

## Summary

The Fee-Floored Adaptive Grid places a geometric ladder of buy limit orders below a volatility-scaled
anchor on CoinW spot, and pairs every filled lot with its own sell order at a price fixed the moment
it fills.

It is designed to be transparent and checkable: the two claims it makes about itself are both
mechanically verifiable rather than statistical.

## Key Idea

- **Spacing is floored by the fee.** Break-even spacing is `2f/(1−f)`, which at CoinW's 0.10% spot
  rate is 0.2002%. The minimum spacing is derived from the live fee rate, never hardcoded.
- **Exits belong to lots, not to the grid.** A lot's sell price is fixed at fill time and never
  re-priced, so every completed round trip is net-positive by construction.
- **The regime gate stops adding, it does not force-sell.** Refusing to buy in a downtrend is worth
  double-digit percentage points over the full backtest; force-selling into that same downtrend
  costs more still. Current figures: [`../results/RESULTS.md`](../results/RESULTS.md) section 5a.

## Suggested Use

Range-bound, choppy and declining spot markets, on capital the operator can afford to lose in full.

It is **not** suitable as a way to outperform holding the asset in a trending market. Over the full
2018-2026 backtest it is beaten badly by buy-and-hold on return, while running roughly half the
maximum drawdown. The trade being offered is capped upside for materially less drawdown. Headline
figures are in [`../README.md`](../README.md) and
[`../results/RESULTS.md`](../results/RESULTS.md) — deliberately not duplicated here, because a
hand-copied number in a secondary file is exactly what goes stale.

## Important Notice

This is a strategy example and a historical simulation only. It has never been run against a real
CoinW account. It is not investment advice and should not be used for real trading without
independent testing, risk assessment, and proper controls.

The complete risk notice — applicable conditions, invalidation conditions, and maximum risk
assumptions — is in [`../SKILL.md` §9](../SKILL.md#9-risk-notice).
