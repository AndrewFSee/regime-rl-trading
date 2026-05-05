"""Macro / cross-asset data loader.

Fetches auxiliary daily series intended to provide *independent* predictive
information beyond pure SPY OHLCV technicals. All series come from yfinance
(no API keys required):

- Equity vol     : ^VIX, ^VIX3M
- Cross-asset   : TLT (long Treasuries), GLD, UUP (USD), HYG/LQD (credit)
- Treasuries    : ^TNX (10Y), ^IRX (13W), ^FVX (5Y), ^TYX (30Y) yields in %

Derived series (added by ``add_derived``):
- VIX_TS    : ^VIX / ^VIX3M  (term structure: <1 contango, >1 stress)
- HYG_LQD   : HYG / LQD       (credit risk proxy; replaces HY OAS)
- T10Y2Y    : ^TNX - ^FVX     (5Y used as a 2Y proxy since yfinance lacks ^IRY)
- T10Y3M    : ^TNX - ^IRX     (yield-curve slope, recession indicator)

Disk-caches each fetch under ``data/cache/macro/`` so repeated runs are free.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CACHE_DIR = Path("data") / "cache" / "macro"

# Reasonable default macro panel.  All free, all daily, all yfinance.
DEFAULT_YF_TICKERS: list[str] = [
    "^VIX",     # equity vol
    "^VIX3M",   # 3-month VIX (term-structure denominator)
    "TLT",      # long-duration Treasuries
    "GLD",      # gold
    "UUP",      # USD index proxy
    "HYG",      # high-yield credit
    "LQD",      # investment-grade credit
    "^TNX",     # 10Y Treasury yield (in %)
    "^IRX",     # 13W Treasury yield (in %)  – short end
    "^FVX",     # 5Y Treasury yield (in %)   – belly
]

DEFAULT_FRED_SERIES: list[str] = []  # Deprecated; FRED CSV endpoint unreliable. Kept for back-compat.


# ----------------------------------------------------------------------
# Cache helpers
# ----------------------------------------------------------------------

def _cache_path(cache_dir: Path, key: str, start: str, end: str) -> Path:
    safe = key.replace("^", "").replace("/", "_")
    return cache_dir / f"{safe}__{start}__{end}.parquet"


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
    except Exception:
        # Fall back to CSV if pyarrow / fastparquet missing.
        df.to_csv(path.with_suffix(".csv"))


# ----------------------------------------------------------------------
# Fetchers
# ----------------------------------------------------------------------

def _fetch_yf_close(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch daily Close for a single ticker via yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for macro loader.") from exc

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        raise ValueError(f"No yfinance data for '{ticker}' in {start}..{end}.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if "Close" not in raw.columns:
        raise ValueError(f"yfinance response for '{ticker}' has no Close column.")
    return raw["Close"].astype(float).rename(ticker)


def _fetch_fred_csv(series_id: str, start: str, end: str) -> pd.Series:
    """Deprecated: FRED CSV endpoint silently caps some series to ~1y of data.

    Kept as a stub that always raises so any leftover callers fail loudly
    rather than silently bfilling a constant value across years of history.
    Use yfinance treasuries (^TNX, ^IRX, ^FVX) and HYG/LQD ratio instead.
    """
    raise RuntimeError(
        f"FRED CSV path is deprecated (series '{series_id}'); "
        "the public CSV endpoint silently caps some series to ~1 year, "
        "leading to bfilled constants. Use yfinance treasuries instead."
    )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

class MacroLoader:
    """Fetch & cache a panel of macro / cross-asset daily series."""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        yf_tickers: list[str] | None = None,
        fred_series: list[str] | None = None,
        cache_dir: str | os.PathLike | None = None,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.yf_tickers = list(yf_tickers) if yf_tickers is not None else list(DEFAULT_YF_TICKERS)
        self.fred_series = list(fred_series) if fred_series is not None else list(DEFAULT_FRED_SERIES)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR

    # ------------------------------------------------------------------

    def _fetch_one(self, key: str, kind: str) -> pd.Series:
        path = _cache_path(self.cache_dir, key, self.start_date, self.end_date)
        cached = _read_cache(path)
        if cached is not None:
            return cached.iloc[:, 0].rename(key)

        if kind == "yf":
            series = _fetch_yf_close(key, self.start_date, self.end_date)
        elif kind == "fred":
            series = _fetch_fred_csv(key, self.start_date, self.end_date)
        else:
            raise ValueError(f"Unknown source kind: {kind!r}")

        _write_cache(path, series.to_frame())
        return series

    def fetch(self, *, use_cache: bool = True) -> pd.DataFrame:
        """Fetch all configured series and return as a single aligned DataFrame.

        Series are forward-filled to a business-day index spanning
        ``[start_date, end_date]`` so they can be merged with daily OHLCV.
        Missing leading values (before a series starts) remain NaN — caller
        should drop them or align to a primary asset's index.
        """
        if not use_cache:
            # Force a re-fetch by ignoring the cache lookup.
            self.cache_dir = self.cache_dir  # no-op; caller controls cache_dir
            # Implementation detail: we still write fresh data to cache.

        series_list: list[pd.Series] = []
        errors: list[str] = []

        for t in self.yf_tickers:
            try:
                if not use_cache:
                    path = _cache_path(self.cache_dir, t, self.start_date, self.end_date)
                    if path.exists():
                        path.unlink()
                series_list.append(self._fetch_one(t, "yf"))
            except Exception as exc:  # pragma: no cover - network errors
                errors.append(f"{t}: {exc}")

        for s in self.fred_series:
            try:
                if not use_cache:
                    path = _cache_path(self.cache_dir, s, self.start_date, self.end_date)
                    if path.exists():
                        path.unlink()
                series_list.append(self._fetch_one(s, "fred"))
            except Exception as exc:  # pragma: no cover - network errors
                errors.append(f"{s}: {exc}")

        if not series_list:
            raise RuntimeError(
                "MacroLoader fetched no series. Errors: " + "; ".join(errors)
            )

        df = pd.concat(series_list, axis=1)
        # Align to business-day index spanning the full requested range so
        # downstream merges have a stable calendar.
        bdays = pd.bdate_range(self.start_date, self.end_date)
        df = df.reindex(bdays).ffill()
        df.index.name = "Date"

        if errors:
            # Surface partial-fetch warnings but don't fail the run.
            print(f"[MacroLoader] partial fetch, missing: {errors}")

        return df

    # ------------------------------------------------------------------
    # Convenience: derived ratios / spreads useful as features
    # ------------------------------------------------------------------

    @staticmethod
    def add_derived(df: pd.DataFrame) -> pd.DataFrame:
        """Append common derived series (ratios / spreads) where inputs exist."""
        out = df.copy()
        if "^VIX" in out.columns and "^VIX3M" in out.columns:
            # <1.0 = contango (calm), >1.0 = backwardation (stress).
            out["VIX_TS"] = out["^VIX"] / out["^VIX3M"]
        if "HYG" in out.columns and "LQD" in out.columns:
            # Risk-on/risk-off credit ratio (HY OAS proxy).
            out["HYG_LQD"] = out["HYG"] / out["LQD"]
        if "^TNX" in out.columns and "^FVX" in out.columns:
            # 10Y - 5Y spread (proxy for 10Y-2Y; yfinance lacks 2Y).
            out["T10Y5Y"] = out["^TNX"] - out["^FVX"]
        if "^TNX" in out.columns and "^IRX" in out.columns:
            # 10Y - 3M slope: classic recession / yield-curve indicator.
            out["T10Y3M"] = out["^TNX"] - out["^IRX"]
        return out

    # ------------------------------------------------------------------
    # Stationarity-aware feature transforms
    # ------------------------------------------------------------------

    # Per-column transform plan. Choose the gentlest transform that produces
    # an approximately stationary series, and add an N-day log-return
    # companion for non-stationary price levels so memory is preserved.
    _ETF_PRICES = ("TLT", "GLD", "UUP", "HYG", "LQD")
    _VOL_LEVELS = ("^VIX", "^VIX3M")
    _YIELD_LEVELS = ("^TNX", "^IRX", "^FVX")
    _RAW_RATIOS = ("VIX_TS", "HYG_LQD", "T10Y5Y", "T10Y3M")

    @staticmethod
    def compute_features(
        df: pd.DataFrame,
        return_window: int = 20,
    ) -> pd.DataFrame:
        """Transform a raw macro panel into approximately-stationary features.

        Per-column rules:
        - ETF prices (TLT, GLD, UUP, HYG, LQD)
            -> ``<sym>_logret{N}`` : N-day log return (stationary, momentum-like)
            -> ``<sym>_log_norm``  : log price minus its first valid value
                                     (preserves memory; gently non-stationary
                                     but trends slowly relative to log returns)
        - VIX, VIX3M : ``log(<sym>)`` (mean-reverting; log compresses spikes)
        - Treasury yields (^TNX, ^IRX, ^FVX) : raw level (already bounded)
        - Spreads / ratios (VIX_TS, HYG_LQD, T10Y5Y, T10Y3M) : raw level

        Output column names use only ASCII / underscores so they survive any
        pandas/numpy round-trips. Series are forward-filled, then any leading
        NaNs are replaced with 0 so the env can consume them directly.
        """
        out_cols: dict[str, pd.Series] = {}

        def _slug(name: str) -> str:
            return name.replace("^", "").replace("/", "_")

        for col in df.columns:
            s = df[col].astype(float)

            if col in MacroLoader._ETF_PRICES:
                logp = np.log(s.replace(0.0, np.nan))
                first_valid = logp.dropna().iloc[0] if logp.dropna().size else 0.0
                out_cols[f"{_slug(col)}_log_norm"] = (logp - first_valid).rename(
                    f"{_slug(col)}_log_norm"
                )
                logret = logp.diff(return_window)
                out_cols[f"{_slug(col)}_logret{return_window}"] = logret.rename(
                    f"{_slug(col)}_logret{return_window}"
                )
            elif col in MacroLoader._VOL_LEVELS:
                lv = np.log(s.clip(lower=1e-6))
                out_cols[f"{_slug(col)}_log"] = lv.rename(f"{_slug(col)}_log")
            elif col in MacroLoader._YIELD_LEVELS or col in MacroLoader._RAW_RATIOS:
                out_cols[_slug(col)] = s.rename(_slug(col))
            else:
                # Unknown column: pass through as raw level. Caller can override.
                out_cols[_slug(col)] = s.rename(_slug(col))

        feats = pd.DataFrame(out_cols, index=df.index)
        return feats.ffill().fillna(0.0)
