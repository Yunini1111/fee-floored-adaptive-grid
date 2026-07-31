"""Fee-Floored Adaptive Grid — a spot grid strategy for CoinW, with a real backtest.

Modules
-------
data        CoinW public kline client: caching, chunking, integrity checks.
indicators  ATR / ADX / EMA / Donchian / Parkinson, hand-written and auditable.
strategy    Grid geometry, fee-derived spacing floor, regime gate, sizing, risk overlay.
engine      Bar loop, per-lot paired-exit ledger, conservative fill model.
metrics     Performance metrics and the exposure-matched benchmark.
"""

__version__ = "1.0.0"
