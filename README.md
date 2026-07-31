# Fee-Floored Adaptive Grid

A spot grid strategy for CoinW, submitted to the **CWC AI Trading Skill Challenge**.
The Skill itself is [`SKILL.md`](SKILL.md). Everything else in this repository exists to let you
check it.

> **Read this first.** This is a **sideways- and down-market instrument**. It converts realised
> volatility into realised cash flow and caps upside in exchange for materially lower drawdown.
> Over the full 2019-2026 backtest it returned **+56.1%** while buy-and-hold BTC returned
> **+1,598.9%** — with a maximum drawdown of **41.9%** against buy-and-hold's **77.2%**.
> **If you believe BTC is going up, hold BTC.** This strategy is for wanting exposure to volatility
> without a directional view, and being willing to give up trend participation to get roughly half
> the drawdown. It is structurally long BTC, it is not hedged, and it can lose a third of the
> account in a bear market — it did, in 2022.

---

## What makes it different

**1. The spacing is floored by the fee, not chosen.**
A buy at `P` and a sell at `P(1+s)`, both paying maker fee `f`, nets `q·P·[s − f(2+s)]`.
Break-even spacing is therefore `s* = 2f/(1−f)` — at CoinW's published 0.10% spot rate, **0.2002%**,
not 0.20%. A grid spaced at exactly `2f` loses money on *every* round trip, forever, no matter how
often it fills. This strategy derives its minimum spacing from the live fee rate at runtime and
refuses to place an order that violates it.

**2. Exits belong to lots, not to the grid.**
When a buy fills, its sell price is fixed at `fill_price × (1 + s_at_fill)` and is **never
re-priced**. Re-anchoring the grid moves only where *new* buys go. This makes "every completed round
trip is net-positive" a structural guarantee rather than a hope — and `tests/test_invariants.py`
asserts it across all 408 round trips of the full run. The ablation that lets exits float with the
grid produces **154 losing round trips** on that same run, and turns +39.9% into **−37.6%**.

**3. The volatility input has no rescaling step.**
CoinW serves native daily candles, so daily ATR is read directly. Converting volatility from bars of
length `D` to horizon `H` needs `σ_H = σ_D·√(H/D)` — 5-minute bars to daily is `√288 = 16.97`, and
using `√24 = 4.90` (the factor for *hourly* bars) understates daily volatility by **3.46×**,
silently collapsing an "adaptive" grid onto its clamp bounds. The safest way to avoid that error is
to not perform the calculation.

## Results

CoinW `BTC_USDT` spot, 15m execution bars, maker = taker = 0.10%, 10,000 USDT initial equity.
Every window tested is shown, including the ones where the strategy is beaten badly.

| Window | Strategy | Max DD | Buy & hold | Exposure-matched B&H | Round trips | Losing |
|---|---:|---:|---:|---:|---:|---:|
| `bear-2022` | −33.36% | 35.46% | −64.25% | −24.10% | 16 | **0** |
| `range-2023` | +12.32% | 6.92% | +155.39% | +27.37% | 70 | **0** |
| `bull-2020Q4` | +10.82% | 2.17% | +444.53% | +4.59% | 14 | **0** |
| `flat-2024-26` | −5.81% | 29.70% | −2.71% | −0.88% | 117 | **0** |
| `year-2025` | −0.55% | 17.91% | −6.45% | −1.98% | 68 | **0** |
| `chop-2019H2` | +11.55% | 13.22% | −21.55% | −5.76% | 54 | **0** |
| **`full-2019-2026`** | **+56.13%** | **41.86%** | **+1598.88%** | **+457.63%** | **408** | **0** |

It beats outright buy-and-hold in 3 of 7 windows and the exposure-matched benchmark in 3 of 7.
**Exposure-matched buy-and-hold is the fair comparison** — the strategy runs ~29% average inventory
over the full run (1% to 38% by window), so measuring it against 100%-long BTC compares two
different amounts of risk. Note the benchmark is only exposure-matched at t=0: it never trims, so
its exposure drifts upward as BTC rises, which makes it *harder* to beat, not easier.

Full detail, provenance and per-window decomposition: [`results/RESULTS.md`](results/RESULTS.md).
Parameter sweeps and risk boundaries: [`results/sensitivity.md`](results/sensitivity.md).
**Trade-by-trade case review:** [`results/CASE-REVIEW.md`](results/CASE-REVIEW.md) — the June 2022
kill-switch event lot by lot, proof that the regime gate filled **zero** buys in a downtrend across
the whole run, an ordinary winning cycle end to end, and where the model charges itself.
Every fill: [`results/trades.csv`](results/trades.csv).

## Three findings that reversed our own intuition

These are in the repository because they were surprising, not because they were flattering.

**Force-selling into a downtrend is a wealth transfer, not a risk control.** Selling inventory down
to a 15% cap when the regime turns down is the obvious design and it costs **41.9 percentage points**
over the full run (+14.3% against +56.1%). Selling into weakness at taker prices and rebuying when
the regime flips back is a pump running in the wrong direction. What *does* work is the passive half
of the same gate — refusing to add new buys in a downtrend — worth **16.2 points**. **Stop adding;
do not panic-sell.**

**A hardcoded drawdown kill switch was a large source of loss, while wearing the label "risk
control".** It was originally a flat 20%. But holding up to 50% of equity in an asset whose own worst drawdown
over this dataset is 77.2% makes a >20% equity drawdown *structurally normal*, so the switch fired on the strategy
working as designed, at local bottoms, at taker prices — realising **−9,955 USDT** on a 10,000 USDT
account, with the full run returning +15.3%. Deriving it as
`clamp(cap_range × asset_max_dd, 0.15, 0.50)` = 35% leaves −3,204 USDT on that path and **+56.1%**
overall. **A threshold with units of percent is not a risk control until you can say what it is a
threshold *of*.**

**"Sell the worst lots first" is backwards.** Dumping highest-cost-basis lots is the intuitive
choice; lowest-cost-first realises a smaller loss and leaves the deep lots to recover, worth
**+14.3% against −2.9%** on the full run with forced de-risking enabled.

Also tested and rejected: persistence-gating the de-risk, spreading it across days, and a tighter
`K = 0.50` default whose ~10% fee drag made the regime gate look worthless. **The value of a risk
overlay is conditional on the underlying edge surviving costs.**

**And one claim we got wrong ourselves.** An earlier draft justified moving the `K` default by
asserting that return improves monotonically in all six test windows. An external audit checked the
per-window decomposition and it does not — only **2 of 6** are individually monotone, and
`bull-2020Q4` is actively better at a tighter setting. What does hold in all six is fee drag falling
monotonically (20.0% → 2.9%), so the default now rests on the cost mechanism and the full
decomposition is computed into `results/sensitivity.md` rather than asserted in prose. The same
audit found the inventory-cap gate was reading the bar's *close* — a price that has not happened at
the moment of the decision — which made a reported "0 cap breaches" circular. The gate now uses only
the bar's open and the fill price.

## Reproduce it

```bash
pip install -r requirements.txt && python run_backtest.py --all
```

That fetches, runs the integrity pass, runs every window and every sweep, and regenerates `results/`
from scratch. `--window flat-2024-26` runs one window in seconds.

**About `data/` and `--offline`.** The raw kline cache is roughly 90 MB, so it is gitignored — a
fresh clone contains `data/manifest.json` (the SHA-256 and fetch time of every response the published
results were computed from) but not the responses themselves. So **the first run must have network
access**. Once `--all` has populated `data/`, `--offline` reproduces every number byte-for-byte with
no further network call, and the manifest lets you confirm you have the same bytes we did.

```bash
python -m pytest tests/ -v
```

107 tests covering the fee algebra, indicator definitions against hand-computed fixtures, every fill
rule, the no-look-ahead property, and the structural invariants. **18 of them exercise the real
CoinW dataset and will skip until `run_backtest.py --all` has been run once** — including the ones
that assert the net-positive round-trip guarantee, so run it before judging that claim.

Three dependencies (`requests`, `numpy`, `matplotlib`). No pandas, no TA-Lib, no backtesting
framework — every indicator is hand-written in `grid/indicators.py` so a reviewer can audit what
"ATR(14)" means here without cloning a dependency tree.

## How the backtest tries not to lie to you

A grid's return is close to linear in its fill count, and the fill count is entirely a modelling
choice. Total variation of `log(close)` on BTC_USDT over 2024-2026 is **1.69%/day** sampled daily
but **26.73%/day** sampled at 5m — path length diverges with resolution because it is a fractal
quantity. A model that counts every level crossing manufactures returns that do not exist. So:

- Orders placed on bar *t* are live from bar *t+1*; daily indicators for day *D* read only bars ≤ *D−1*.
- A resting buy fills only if `low < price × (1 − 1bp)`. **`low == price` is not a fill** — the
  market must trade *through* us, not merely touch us.
- Fill price is always the limit price, never improved, even when the bar gaps far through it.
- If a bar *opens* through our limit, a post-only order would have been rejected, so we charge the
  **taker** fee — worst price and worst fee.
- Cash freed by a sell on bar *t* cannot fund a buy on bar *t*.
- Missing bars are counted and reported, **never interpolated**. A fabricated bar with a plausible
  low generates a fill that never happened.
- Exchange constraints are read from `returnSymbol` and enforced: 5 USDT minimum, 0.01 tick,
  0.0001 step.

**Residual optimism, disclosed:** the model still assumes front-of-queue position whenever price
trades through a resting order. `results/sensitivity.md` reruns everything at 70% and 50% fill
probability so you can price that assumption rather than trust it.

## Repository layout

```text
SKILL.md                  the submission itself — 12 sections, competition template order
README.md                 this file
run_backtest.py           the one command
grid/
  data.py                 CoinW client: caching, chunking, integrity checks
  indicators.py           ATR / ADX / EMA / Donchian / Parkinson, hand-written
  strategy.py             geometry, fee floor, regime gate, sizing, risk overlay
  engine.py               bar loop, per-lot ledger, fill model
  metrics.py              metrics and the exposure-matched benchmark
tests/                    107 tests
data/                     cached raw CoinW responses + SHA-256 manifest
results/                  generated: RESULTS.md, sensitivity.md, CASE-REVIEW.md,
                          trades.csv, 3 charts
```

## Applying it to CWC or any new listing

As of 2026-08-01, `command=returnCurrencies` lists **CWC** as a registered currency (`symbolId 2018`,
ERC20) with deposits and withdrawals disabled, and `command=returnSymbol` returns **417 spot pairs,
none of which is a CWC pair**. Both raw responses are committed to `data/` so you can check this
yourself without a network call. This Skill therefore ships calibrated for `BTC_USDT`. No listing,
date, price, or supply is claimed here.

What ships instead is the procedure, which applies to any newly listed token the moment its pair
appears:

1. **Refuse to trade until the data exists.** ATR(14) and EMA(200) need 14 and 200 daily bars. Below
   20 daily bars the Skill returns `INSUFFICIENT_HISTORY` and places nothing. This is a hard block
   and it is the single most common way a grid bot dies on a new listing.
2. **Bootstrap conservatively (days 20-200).** EMA200 does not exist, so no regime can be classified
   — do not pretend one. Run RANGE only, with the **DOWNTREND** cap of 15% for the whole period. An
   unclassifiable regime inherits the most conservative cap, not the middle one.
3. **Re-measure volatility on that pair.** New listings routinely run 3-10× BTC's daily ATR.
4. **Raise `s_cap`** to about `3 ×` the pair's median ATR%, or a 8% cap on a token with 20%/day
   volatility turns the grid into a fee-paying machine.
5. **Re-read `returnSymbol`** for that pair's own minimum notional, tick and step. Do not reuse BTC's.
6. **Halve the allocation and cap exposure in absolute USDT**, not only as a fraction of equity — new
   listings have thin books and a percentage cap on a small account can still be a large share of
   visible depth.

**Why this is relevant to Agentic trading**, independent of any specific token: the entire execution
loop is specified against **public, unauthenticated endpoints** — three GETs per cycle, no API key,
no order-book access, no proprietary data. Any agent can run this Skill against CoinW today on any of
the 417 live pairs, and the same Skill works unchanged on a new pair the moment `returnSymbol` reports
it. The Agent decides and acts within hard numeric caps, and **may lower a cap but may never raise
one** — the failure mode everyone worries about is closed off by construction rather than by prompt.

## Risk notice

Crypto assets are highly volatile and may result in partial or total loss of funds. This strategy is
**structurally long BTC**: it is not hedged, not market-neutral, and not risk-free. Its worst tested
window lost **33.4%** of the account and its worst tested drawdown was **41.9%**.

**This has never been run against a real CoinW account.** It is documentation plus a historical
simulation. Historical simulation is not a forecast, and no forward-looking return is projected
anywhere in this repository. The full risk notice, including applicable conditions, invalidation
conditions and maximum risk assumptions, is [`SKILL.md` §9](SKILL.md#9-risk-notice).

Conduct your own research and evaluate your own risk tolerance before using any strategy.

## Disclaimer

This repository is provided for educational and activity demonstration purposes only. It does not
constitute investment advice, financial advice, trading advice, or a recommendation to buy or sell
any crypto asset.

It does not represent a commitment that the strategy will be listed, supported, executed, or
productized by CoinW. CoinW reserves the right to interpret the final rules of the CWC AI Trading
Skill Challenge.

Licensed under the [MIT License](LICENSE).

---
---

# 費用地板自適應網格（繁體中文）

一個給 CoinW 的現貨網格策略，投稿至 **CWC AI Trading Skill Challenge**。
Skill 本體是 [`SKILL.md`](SKILL.md)，這個倉庫的其他內容都是為了讓你可以親自驗證它。

> **請先讀這段。** 這是一個**適用於盤整與下跌行情的工具**。它把已實現波動率轉換成已實現現金流，
> 代價是放棄上漲參與度，換取明顯較低的回撤。在 2019-2026 完整回測中它的報酬是 **+56.1%**，
> 而單純持有 BTC 是 **+1,598.9%**；但最大回撤是 **41.9%**，持有 BTC 則是 **77.2%**。
> **如果你認為 BTC 會漲，那就直接持有 BTC。** 這個策略適合的情境是：你想要暴露在波動率上但不做方向判斷，
> 並且願意用放棄趨勢行情來換取大約一半的回撤。它在結構上是淨多頭、沒有避險，
> 在熊市可能虧掉三分之一的帳戶——2022 年就是這樣。

## 三個關鍵差異

**1. 網格間距由手續費決定，不是隨便選的。**
在 `P` 買進、在 `P(1+s)` 賣出，兩邊都付 maker 費 `f`，淨利是 `q·P·[s − f(2+s)]`。
因此損益兩平間距是 `s* = 2f/(1−f)`——以 CoinW 公告的現貨 0.10% 費率計算是 **0.2002%**，不是 0.20%。
間距剛好等於 `2f` 的網格，**每一次來回都在虧錢**，無論成交多少次都一樣。
本策略在執行時從實際費率推導出最小間距，並拒絕掛出任何違反它的委託。

**2. 出場價綁在「每一筆持倉」上，不綁在網格上。**
買單成交時，它的賣出價就固定在 `成交價 × (1 + 當下間距)`，**永遠不會被重新定價**。
重新錨定網格只會改變「新的買單掛在哪」。這讓「每一次完成的來回都是淨賺」成為結構性保證，
而不是期望——`tests/test_invariants.py` 針對完整回測的全部 408 次來回做了斷言。
對照組（讓出場價隨網格浮動）在同一段期間產生了 **154 次虧損的來回**，並把 +39.9% 變成 **−37.6%**。

**3. 波動率計算沒有任何換算步驟。**
CoinW 直接提供原生日線，所以日 ATR 是直接讀出來的。把長度 `D` 的 K 線波動率換算到 `H` 期間需要
`σ_H = σ_D·√(H/D)`——5 分鐘換算到日線是 `√288 = 16.97`，若誤用 `√24 = 4.90`（那是「小時線」的係數），
日波動率會被低估 **3.46 倍**，讓一個號稱「自適應」的網格默默塌縮到它的上下限。
避免這個錯誤最保險的方法，就是根本不做這個計算。

## 回測結果

CoinW `BTC_USDT` 現貨，15 分鐘執行 K 線，maker = taker = 0.10%，初始資金 10,000 USDT。
**所有測試過的區間都列出來了，包含策略被打得很慘的那幾個。**

| 區間 | 策略 | 最大回撤 | 買入持有 | 曝險對齊的買入持有 | 完成來回 | 虧損來回 |
|---|---:|---:|---:|---:|---:|---:|
| `bear-2022` 熊市 | −33.36% | 35.46% | −64.25% | −24.10% | 16 | **0** |
| `range-2023` 復甦 | +12.32% | 6.92% | +155.39% | +27.37% | 70 | **0** |
| `bull-2020Q4` 大多頭 | +10.82% | 2.17% | +444.53% | +4.59% | 14 | **0** |
| `flat-2024-26` 橫盤兩年 | −5.81% | 29.70% | −2.71% | −0.88% | 117 | **0** |
| `year-2025` | −0.55% | 17.91% | −6.45% | −1.98% | 68 | **0** |
| `chop-2019H2` 震盪下跌 | +11.55% | 13.22% | −21.55% | −5.76% | 54 | **0** |
| **`full-2019-2026` 完整期間** | **+56.13%** | **41.86%** | **+1598.88%** | **+457.63%** | **408** | **0** |

7 個區間中，贏過單純買入持有的有 3 個，贏過曝險對齊基準的有 3 個。
**曝險對齊的買入持有才是公平的比較對象**——本策略完整期間平均庫存約 29%（各區間 1%～38%），
拿它去比 100% 滿倉 BTC 是在比較兩種不同的風險量。附帶說明：該基準只在 t=0 時曝險對齊，
之後不再調整，所以 BTC 上漲時它的曝險比例會往上飄，這讓它**更難**被贏過，而不是更容易。

## 三個推翻我們自己直覺的發現

**在下跌趨勢中強制賣出是財富轉移，不是風控。** 在趨勢轉空時把庫存砍到 15% 上限是最直覺的設計，
但在完整回測中它的代價是 **41.9 個百分點**（+14.3% 對 +56.1%）。用 taker 價格賣在弱勢、然後在趨勢翻回時買回，
是一個方向相反的抽水機。真正有效的是同一個閘門的被動面——**在下跌趨勢中不再加碼**，價值 **16.2 個百分點**。
**停止加碼，但不要恐慌性賣出。**

**回撤斷路器曾經是最大的虧損來源，卻掛著「風控」的名字。** 它原本寫死在 20%。
但持有最多 50% 權益在一個「自身最大回撤達 77.2%」的資產上，**超過 20% 的權益回撤本來就是結構性正常的**——
所以這個開關偵測到的不是異常，而是策略正常運作，並且總是在局部底部、用 taker 價格觸發，
在 10,000 USDT 的帳戶上實現了 **−9,955 USDT** 的虧損，完整回測報酬只有 +15.3%。改成推導式的
`clamp(cap_range × asset_max_dd, 0.15, 0.50)` = 35% 之後，同一路徑只實現 −3,204 USDT，整體變成 **+56.1%**。
**一個只有百分比單位的門檻，在你說得出它是「什麼東西的門檻」之前，都不算風控。**

**「先賣最爛的持倉」是反的。** 先倒掉成本最高的持倉是直覺選擇；先賣成本最低的反而實現較小的虧損，
並且把深水區的持倉留著等反彈——在開啟強制減倉的情況下，完整回測 **+14.3% 對 −2.9%**。

**還有一個我們自己講錯的說法。** 早期版本用「報酬在全部六個測試區間都隨 K 單調改善」來合理化把
`K` 預設值調大。外部稽核去逐區間驗算，結果不成立——**只有 6 個裡的 2 個**是單調的，而且
`bull-2020Q4` 在較窄的間距下表現反而更好。六個區間都成立的是**手續費損耗單調下降**（20.0% → 2.9%），
所以現在預設值改為建立在成本機制上，而且完整的逐區間分解是在產生報告時**計算出來**的，
不再是寫死在文字裡的斷言。同一次稽核還發現庫存上限的檢查讀了 K 線的**收盤價**——那是在下單當下
根本還不存在的價格——導致回報的「0 次違規」是循環論證。現在該檢查只使用開盤價與成交價。

## 重現結果

```bash
pip install -r requirements.txt && python run_backtest.py --all
```

原始 K 線快取約 90 MB，因此被 gitignore：剛 clone 下來的倉庫只有 `data/manifest.json`（記錄每一筆回應的
SHA-256 與抓取時間），**第一次執行需要網路**。跑過一次 `--all` 之後，`--offline` 就能完全用快取重現
每一個數字，不再需要任何網路呼叫。

```bash
python -m pytest tests/ -v
```

107 個測試，涵蓋手續費代數、指標定義（對照手算基準）、每一條成交規則、無未來函數性質，以及結構性不變量。
其中 18 個會使用真實 CoinW 資料，在你跑過一次 `run_backtest.py --all` 之前會 skip。

## 風險聲明

加密資產波動劇烈，可能導致部分或全部資金損失。本策略在結構上是**淨多頭 BTC**：沒有避險、非市場中性、
也不是無風險。最差的測試區間虧損 **33.4%**，最大回撤 **41.9%**。

**本策略從未在真實 CoinW 帳戶上運行過。** 它是文件加上歷史模擬。歷史模擬不是預測，
本倉庫任何地方都沒有對未來報酬做出預測。完整的風險聲明（包含適用條件、失效條件與最大風險假設）
請見 [`SKILL.md` §9](SKILL.md#9-risk-notice)。

使用任何策略前，請自行研究並評估自身的風險承受度。

## 免責聲明

本倉庫僅供教育與活動展示用途，不構成投資建議、財務建議、交易建議，也不構成買賣任何加密資產的推薦。

本倉庫不代表 CoinW 承諾將此策略上架、支援、執行或產品化。CoinW 保留對 CWC AI Trading Skill Challenge
最終規則的解釋權。

授權條款：[MIT License](LICENSE)。
