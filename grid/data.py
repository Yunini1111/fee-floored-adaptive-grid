"""CoinW public market-data client.

Endpoint (public, unauthenticated, 10 req/s):
    GET https://api.coinw.com/api/v1/public
        ?command=returnChartData&currencyPair=BTC_USDT&period=<sec>&start=<ms>&end=<ms>

Three gotchas this module exists to handle. All three were observed against the
live endpoint on 2026-08-01, not inferred from documentation.

1.  The default Python User-Agent gets HTTP 403. Any explicit UA works. A naive
    implementation fails immediately and confusingly, so the UA header is set in
    the single request function and covered by a smoke test.

2.  A request implying more than roughly 250k-265k rows returns
        {"code": "200", "success": true, "data": []}
    -- an EMPTY ARRAY with a success code, not an error. Treating that as "no
    trades in this window" would silently delete years of history from a
    backtest. Every non-empty window therefore asserts len(data) > 0, and we
    chunk well under the observed cap.

3.  Intraday series contain missing bars (the daily series has exactly one gap in
    3,270 bars; 15m has more). We NEVER forward-fill. A missing bar is a bar on
    which no order could have filled; fabricating one manufactures fills, which
    is precisely the class of error that makes a grid backtest untrustworthy.

The published period whitelist is incomplete. Verified working values are in
PERIODS below -- note that 3600 (1h) and 86400 (1d) both work despite being
absent from the public docs page. Native daily candles are why this strategy has
no volatility-rescaling step at all.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

BASE_URL = "https://api.coinw.com/api/v1/public"

USER_AGENT = "fee-floored-adaptive-grid/1.0 (research backtest)"

# Verified working period values, in seconds. 120 and 259200 return
# {"code": -3, "msg": "API access error"}; everything below returned correctly
# spaced bars, and the 86400 bars were cross-checked against the 4h bars inside
# the same UTC day (open, max-high and min-low all agreed).
PERIODS = (60, 180, 300, 900, 1800, 3600, 7200, 14400, 21600, 43200, 86400, 604800)

# Observed row cap is between 250,419 (returned fine) and 265,440 (returned an
# empty array). We chunk at 200,000 to stay comfortably clear of it.
MAX_ROWS_PER_REQUEST = 200_000

REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 3
RATE_LIMIT_SLEEP_S = 0.15  # 10 req/s is the documented limit; we use well under it


class DataError(RuntimeError):
    """Raised when data cannot be fetched or fails an integrity check.

    Never swallowed. A backtest on incomplete data is worse than no backtest.
    """


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #


@dataclass
class IntegrityReport:
    """Result of the pre-backtest integrity pass. Quoted verbatim in RESULTS.md."""

    pair: str
    period: int
    bars: int
    first_ts: int
    last_ts: int
    expected_bars: int
    missing_bars: int
    missing_timestamps: list[int] = field(default_factory=list)
    duplicate_timestamps: int = 0
    misaligned_bars: int = 0

    @property
    def missing_pct(self) -> float:
        return 100.0 * self.missing_bars / self.expected_bars if self.expected_bars else 0.0

    def summary(self) -> str:
        first = ms_to_iso(self.first_ts)
        last = ms_to_iso(self.last_ts)
        return (
            f"{self.pair} period={self.period}s | {self.bars:,} bars | {first} -> {last} | "
            f"missing {self.missing_bars:,}/{self.expected_bars:,} ({self.missing_pct:.3f}%)"
        )


@dataclass
class Candles:
    """OHLCV series as parallel numpy arrays. Deliberately not a DataFrame.

    No pandas anywhere in this project: a reviewer should be able to audit every
    numeric step without cloning a dependency tree.
    """

    pair: str
    period: int
    ts: np.ndarray  # int64, milliseconds, strictly increasing
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    provenance: str = ""

    def __len__(self) -> int:
        return int(self.ts.size)

    def __getitem__(self, sl: slice) -> "Candles":
        return Candles(
            pair=self.pair,
            period=self.period,
            ts=self.ts[sl],
            open=self.open[sl],
            high=self.high[sl],
            low=self.low[sl],
            close=self.close[sl],
            volume=self.volume[sl],
            provenance=self.provenance,
        )

    def slice_ms(self, start_ms: int, end_ms: int) -> "Candles":
        """Half-open [start_ms, end_ms) slice."""
        lo = int(np.searchsorted(self.ts, start_ms, side="left"))
        hi = int(np.searchsorted(self.ts, end_ms, side="left"))
        return self[lo:hi]

    def index_at_or_before(self, ts_ms: int) -> int:
        """Index of the newest bar with ts <= ts_ms, or -1 if none.

        The workhorse of look-ahead avoidance: daily indicators for UTC day D are
        always read through this with ts = start-of-day-D minus one millisecond.
        """
        return int(np.searchsorted(self.ts, ts_ms, side="right")) - 1


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #


def iso_to_ms(iso: str) -> int:
    """'2024-01-01' or '2024-01-01T00:00:00' -> epoch milliseconds (UTC)."""
    text = iso.strip()
    if len(text) == 10:
        text += "T00:00:00"
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp() * 1000)


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ms_to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def utc_day_start_ms(ts_ms: int) -> int:
    """Start-of-UTC-day containing ts_ms. The strategy re-anchors at 00:00 UTC."""
    return (ts_ms // 86_400_000) * 86_400_000


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class CoinWClient:
    """Fetches and caches CoinW klines.

    Every raw response is written to `cache_dir` verbatim alongside its SHA-256,
    so the whole study is reproducible offline and a reviewer can diff the exact
    bytes the numbers were computed from.
    """

    def __init__(self, cache_dir: Path | str, offline: bool = False, verbose: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.verbose = verbose
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self._last_request_at = 0.0

    # -- logging ----------------------------------------------------------- #

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[data] {message}", flush=True)

    # -- raw HTTP ---------------------------------------------------------- #

    def _get(self, params: dict) -> dict:
        """One GET with backoff. Returns the parsed JSON body."""
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < RATE_LIMIT_SLEEP_S:
            time.sleep(RATE_LIMIT_SLEEP_S - elapsed)

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT_S)
                self._last_request_at = time.monotonic()
                if response.status_code == 403:
                    raise DataError(
                        "HTTP 403 from CoinW. This is what happens when the User-Agent "
                        "header is missing or is the Python default; check USER_AGENT."
                    )
                if response.status_code != 200:
                    raise DataError(f"HTTP {response.status_code}: {response.text[:200]}")
                body = response.json()
                if body.get("success") is not True or str(body.get("code")) != "200":
                    raise DataError(f"API error: code={body.get('code')} msg={body.get('msg')}")
                return body
            except (requests.RequestException, ValueError, DataError) as exc:
                last_error = exc
                if isinstance(exc, DataError) and "403" in str(exc):
                    raise  # a UA problem will not fix itself by retrying
                if attempt < MAX_RETRIES - 1:
                    backoff = 2**attempt
                    self._log(f"attempt {attempt + 1}/{MAX_RETRIES} failed ({exc}); retry in {backoff}s")
                    time.sleep(backoff)
        raise DataError(f"failed after {MAX_RETRIES} attempts: {last_error}")

    # -- cache ------------------------------------------------------------- #

    def _cache_path(self, pair: str, period: int, label: str) -> Path:
        return self.cache_dir / f"{pair}_{period}_{label}.json"

    def _write_cache(self, path: Path, payload: dict) -> None:
        text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        path.write_text(text, encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self._update_manifest(path.name, digest, payload["_meta"])

    def _update_manifest(self, filename: str, digest: str, meta: dict) -> None:
        manifest_path = self.cache_dir / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
        manifest[filename] = {"sha256": digest, **meta}
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    # -- chunking ---------------------------------------------------------- #

    @staticmethod
    def _chunks(start_ms: int, end_ms: int, period: int) -> list[tuple[int, int, str]]:
        """Split [start_ms, end_ms) into requests of at most MAX_ROWS_PER_REQUEST bars.

        Calendar-year boundaries are used when a full year fits, purely so cache
        filenames read as `BTC_USDT_900_2019.json` rather than epoch soup. For
        very fine periods where a year does not fit, falls back to calendar months.
        """
        bars_per_year = int(365.25 * 86400 / period)
        if end_ms - start_ms <= MAX_ROWS_PER_REQUEST * period * 1000:
            return [(start_ms, end_ms, f"{ms_to_date(start_ms)}_{ms_to_date(end_ms)}")]

        use_years = bars_per_year <= MAX_ROWS_PER_REQUEST
        out: list[tuple[int, int, str]] = []
        cursor = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)

        while cursor < end_dt:
            if use_years:
                nxt = cursor.replace(
                    year=cursor.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0
                )
                label = str(cursor.year)
            else:
                if cursor.month == 12:
                    nxt = cursor.replace(year=cursor.year + 1, month=1, day=1)
                else:
                    nxt = cursor.replace(month=cursor.month + 1, day=1)
                nxt = nxt.replace(hour=0, minute=0, second=0, microsecond=0)
                label = cursor.strftime("%Y-%m")
            chunk_end = min(nxt, end_dt)
            out.append((int(cursor.timestamp() * 1000), int(chunk_end.timestamp() * 1000), label))
            cursor = chunk_end
        return out

    # -- public API -------------------------------------------------------- #

    def fetch(self, pair: str, period: int, start: str, end: str) -> Candles:
        """Fetch [start, end) for `pair` at `period` seconds, using cache when present."""
        if period not in PERIODS:
            raise DataError(f"period {period} is not in the verified whitelist {PERIODS}")

        start_ms, end_ms = iso_to_ms(start), iso_to_ms(end)
        if end_ms <= start_ms:
            raise DataError(f"empty window: {start} -> {end}")

        rows: list[dict] = []
        for chunk_start, chunk_end, label in self._chunks(start_ms, end_ms, period):
            rows.extend(self._fetch_chunk(pair, period, chunk_start, chunk_end, label))

        if not rows:
            raise DataError(f"no rows returned for {pair} {period}s {start}..{end}")

        # Chunks are cached at calendar-year granularity, so a cached chunk can
        # legitimately hold a SUPERSET of what was asked for -- ask for
        # 2018-03-05 after something else already cached all of 2018 and the raw
        # rows come back starting 2018-01-01. Returning that silently would mean
        # `fetch` does not honour its own contract, and a backtest would run over
        # a window nobody requested. The cache may be a superset; `fetch` is exact.
        candles = _rows_to_candles(rows, pair, period).slice_ms(start_ms, end_ms)
        if len(candles) == 0:
            raise DataError(f"no bars inside {start}..{end} for {pair} {period}s")
        self._log(f"{pair} {period}s -> {len(candles):,} bars {start}..{end}")
        return candles

    def _fetch_chunk(
        self, pair: str, period: int, start_ms: int, end_ms: int, label: str
    ) -> list[dict]:
        path = self._cache_path(pair, period, label)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            meta = payload.get("_meta", {})
            # The reverse of the superset case: a chunk cached from a request that
            # began mid-year does NOT cover a later request for the whole year.
            # Serving it would silently drop bars, so re-fetch instead.
            covered_start = meta.get("start_ms")
            covered_end = meta.get("end_ms")
            if covered_start is None or covered_end is None:
                return payload["response"]["data"]
            if covered_start <= start_ms and covered_end >= end_ms:
                return payload["response"]["data"]
            self._log(
                f"cache {path.name} covers {ms_to_date(covered_start)}..{ms_to_date(covered_end)} "
                f"but {ms_to_date(start_ms)}..{ms_to_date(end_ms)} was requested; refetching"
            )
            if self.offline:
                raise DataError(
                    f"--offline: cached {path.name} does not cover the requested window"
                )

        if self.offline:
            raise DataError(
                f"--offline was requested but {path.name} is not cached. "
                f"Run once without --offline to populate data/."
            )

        params = {
            "command": "returnChartData",
            "currencyPair": pair,
            "period": period,
            "start": start_ms,
            "end": end_ms,
        }
        self._log(f"GET {pair} {period}s {ms_to_date(start_ms)}..{ms_to_date(end_ms)}")
        body = self._get(params)
        data = body.get("data") or []

        # Gotcha 2. An over-cap request answers 200/success with an empty array.
        # Any non-empty window that comes back empty is a hard error, never
        # "there were no trades".
        if not data:
            expected = (end_ms - start_ms) // (period * 1000)
            raise DataError(
                f"CoinW returned an EMPTY data array for a window expecting ~{expected:,} bars "
                f"({pair} {period}s {ms_to_date(start_ms)}..{ms_to_date(end_ms)}). This is the "
                f"documented over-cap silent-failure mode. Reduce MAX_ROWS_PER_REQUEST."
            )

        payload = {
            "_meta": {
                "pair": pair,
                "period": period,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "rows": len(data),
                "url": f"{BASE_URL}?command=returnChartData&currencyPair={pair}"
                f"&period={period}&start={start_ms}&end={end_ms}",
                "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "response": body,
        }
        self._write_cache(path, payload)
        return data


# --------------------------------------------------------------------------- #
# Parsing and integrity
# --------------------------------------------------------------------------- #


def _rows_to_candles(rows: list[dict], pair: str, period: int) -> Candles:
    ts = np.fromiter((int(r["date"]) for r in rows), dtype=np.int64, count=len(rows))
    op = np.fromiter((float(r["open"]) for r in rows), dtype=np.float64, count=len(rows))
    hi = np.fromiter((float(r["high"]) for r in rows), dtype=np.float64, count=len(rows))
    lo = np.fromiter((float(r["low"]) for r in rows), dtype=np.float64, count=len(rows))
    cl = np.fromiter((float(r["close"]) for r in rows), dtype=np.float64, count=len(rows))
    vol = np.fromiter((float(r.get("volume", 0.0)) for r in rows), dtype=np.float64, count=len(rows))

    order = np.argsort(ts, kind="stable")
    ts, op, hi, lo, cl, vol = ts[order], op[order], hi[order], lo[order], cl[order], vol[order]

    keep = np.ones(ts.size, dtype=bool)
    keep[1:] = ts[1:] != ts[:-1]  # drop duplicate timestamps from chunk boundaries

    return Candles(
        pair=pair,
        period=period,
        ts=ts[keep],
        open=op[keep],
        high=hi[keep],
        low=lo[keep],
        close=cl[keep],
        volume=vol[keep],
        provenance=f"CoinW returnChartData {pair} period={period}",
    )


def integrity_check(candles: Candles, strict: bool = True) -> IntegrityReport:
    """Validate a series before it is allowed anywhere near the backtest.

    Deliberately does NOT repair anything. Missing bars are counted and reported,
    never interpolated -- a fabricated bar with a plausible low would generate a
    fill that never existed.
    """
    n = len(candles)
    if n == 0:
        raise DataError("empty series")

    ts, op, hi, lo, cl = candles.ts, candles.open, candles.high, candles.low, candles.close
    step = candles.period * 1000

    if not np.all(np.diff(ts) > 0):
        raise DataError("timestamps are not strictly increasing after sort and dedupe")

    bad_range = (hi < np.maximum(op, cl)) | (lo > np.minimum(op, cl)) | (hi < lo)
    if np.any(bad_range):
        idx = int(np.argmax(bad_range))
        raise DataError(
            f"{int(bad_range.sum())} degenerate bar(s); first at {ms_to_iso(int(ts[idx]))} "
            f"O={op[idx]} H={hi[idx]} L={lo[idx]} C={cl[idx]}"
        )

    if np.any(lo <= 0) or np.any(cl <= 0):
        raise DataError("non-positive price found")

    misaligned = int(np.count_nonzero(ts % step != 0))
    if misaligned and strict:
        raise DataError(f"{misaligned} bar(s) not aligned to the {candles.period}s grid")

    expected = int((ts[-1] - ts[0]) // step) + 1
    missing_count = expected - n
    missing_ts: list[int] = []
    if missing_count > 0:
        full = np.arange(ts[0], ts[-1] + step, step, dtype=np.int64)
        missing = np.setdiff1d(full, ts, assume_unique=False)
        missing_ts = [int(x) for x in missing[:200]]  # cap the report, not the count

    return IntegrityReport(
        pair=candles.pair,
        period=candles.period,
        bars=n,
        first_ts=int(ts[0]),
        last_ts=int(ts[-1]),
        expected_bars=expected,
        missing_bars=max(0, missing_count),
        missing_timestamps=missing_ts,
        misaligned_bars=misaligned,
    )


def max_consecutive_gap(candles: Candles) -> int:
    """Longest run of consecutive missing bars. Feeds risk check R6."""
    if len(candles) < 2:
        return 0
    step = candles.period * 1000
    strides = np.diff(candles.ts) // step
    return int(strides.max() - 1) if strides.size else 0


def fetch_symbol_constraints(pair: str, cache_dir: Path | str, offline: bool = False) -> dict:
    """Exchange constraints for `pair` from command=returnSymbol.

    Verified for BTC_USDT on 2026-08-01: minBuyAmount 5.0 USDT, pricePrecision 2
    (0.01 tick), countPrecision 4 (0.0001 step). These are hard exchange limits;
    an order violating any of them is rejected, so the backtest honours them too.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "returnSymbol.json"

    if path.exists():
        body = json.loads(path.read_text(encoding="utf-8"))["response"]
    elif offline:
        raise DataError("--offline requested but returnSymbol.json is not cached")
    else:
        client = CoinWClient(cache_dir, offline=False, verbose=False)
        body = client._get({"command": "returnSymbol"})
        path.write_text(
            json.dumps(
                {
                    "_meta": {
                        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "url": f"{BASE_URL}?command=returnSymbol",
                    },
                    "response": body,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    rows = body.get("data") or []
    if isinstance(rows, dict):
        rows = list(rows.values())
    for row in rows:
        if row.get("currencyPair") == pair:
            price_precision = int(row.get("pricePrecision", 2))
            count_precision = int(row.get("countPrecision", 4))
            return {
                "pair": pair,
                "min_notional": float(row.get("minBuyAmount", 5.0)),
                "price_tick": 10.0**-price_precision,
                "qty_step": 10.0**-count_precision,
                "state": int(row.get("state", 0)),
                "source": "CoinW returnSymbol",
            }
    raise DataError(f"{pair} not found among {len(rows)} pairs in returnSymbol")


def default_window_ms(days_back: int, end: str | None = None) -> tuple[int, int]:
    """Convenience [start, end) for live use."""
    end_ms = iso_to_ms(end) if end else int(time.time() * 1000)
    return end_ms - days_back * 86_400_000, end_ms


__all__ = [
    "BASE_URL",
    "PERIODS",
    "USER_AGENT",
    "MAX_ROWS_PER_REQUEST",
    "Candles",
    "CoinWClient",
    "DataError",
    "IntegrityReport",
    "fetch_symbol_constraints",
    "integrity_check",
    "iso_to_ms",
    "max_consecutive_gap",
    "ms_to_date",
    "ms_to_iso",
    "utc_day_start_ms",
]
