"""Market data loader with yfinance backend."""
from __future__ import annotations

from typing import Optional

import pandas as pd


class DataLoader:
    """
    Fetches OHLCV data for one or more tickers and provides train/test splits.

    Parameters
    ----------
    tickers:    List of ticker symbols (e.g. ``["SPY", "QQQ"]``).
    start_date: ISO date string, e.g. ``"2015-01-01"``.
    end_date:   ISO date string, e.g. ``"2023-12-31"``.
    train_ratio: Fraction of rows used for training (default ``0.8``).
    interval:   yfinance interval string (default ``"1d"``).
    """

    def __init__(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        train_ratio: float = 0.8,
        interval: str = "1d",
    ) -> None:
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.train_ratio = train_ratio
        self.interval = interval
        self._data: dict[str, pd.DataFrame] = {}
        self._macro: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_data(self) -> dict[str, pd.DataFrame]:
        """
        Download OHLCV data via yfinance for all configured tickers.

        Returns
        -------
        dict mapping ticker → DataFrame[Open, High, Low, Close, Volume].
        """
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for DataLoader.fetch_data(). "
                "Install with: pip install yfinance"
            ) from exc

        self._data = {}
        for ticker in self.tickers:
            raw = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                progress=False,
                auto_adjust=True,
            )
            if raw.empty:
                raise ValueError(f"No data returned for ticker '{ticker}'.")

            # Flatten multi-level columns that yfinance sometimes returns
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
            self._data[ticker] = raw

        return self._data

    def load_from_dataframe(
        self, data: pd.DataFrame, ticker: str = "SYNTHETIC"
    ) -> dict[str, pd.DataFrame]:
        """
        Load data directly from an existing DataFrame (useful for tests /
        offline usage).

        Parameters
        ----------
        data:   DataFrame with Open, High, Low, Close, Volume columns.
        ticker: Key to store the data under.
        """
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")
        self._data[ticker] = data.copy()
        return self._data

    def get_train_test_split(
        self, ticker: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (train_df, test_df) for the given ticker.

        Raises
        ------
        KeyError if the ticker has not been loaded yet.
        """
        if ticker not in self._data:
            available = list(self._data.keys())
            raise KeyError(
                f"Ticker '{ticker}' not loaded. Available: {available}. "
                "Call fetch_data() or load_from_dataframe() first."
            )
        df = self._data[ticker]
        split_idx = int(len(df) * self.train_ratio)
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    def get_walk_forward_splits(
        self,
        ticker: str,
        train_size: int,
        test_size: int,
        step: int | None = None,
        embargo: int = 0,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Return rolling-window walk-forward (train, test) splits.

        Parameters
        ----------
        ticker:      Ticker key to split.
        train_size:  Number of bars in each training window.
        test_size:   Number of bars in each test window (immediately following
                     the train window plus ``embargo`` bars).
        step:        Stride between successive train-window starts. Defaults to
                     ``test_size`` (non-overlapping test windows).
        embargo:     Bars to drop between train and test to reduce leakage from
                     rolling features that span both segments.

        Returns
        -------
        List of (train_df, test_df) tuples in chronological order.
        """
        if ticker not in self._data:
            raise KeyError(f"Ticker '{ticker}' not loaded.")
        if train_size <= 0 or test_size <= 0:
            raise ValueError("train_size and test_size must be positive")
        df = self._data[ticker]
        n = len(df)
        stride = step if step is not None else test_size
        if stride <= 0:
            raise ValueError("step must be positive")

        splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        start = 0
        while True:
            train_end = start + train_size
            test_start = train_end + embargo
            test_end = test_start + test_size
            if test_end > n:
                break
            train_df = df.iloc[start:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            splits.append((train_df, test_df))
            start += stride
        return splits

    def get_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Return the full DataFrame for a ticker, or None if not loaded."""
        return self._data.get(ticker)

    # ------------------------------------------------------------------
    # Macro / cross-asset panel (opt-in)
    # ------------------------------------------------------------------

    def fetch_macro(
        self,
        yf_tickers: list[str] | None = None,
        fred_series: list[str] | None = None,
        cache_dir: str | None = None,
        use_cache: bool = True,
        add_derived: bool = True,
    ) -> pd.DataFrame:
        """Fetch the macro / cross-asset panel and store it on the loader.

        The panel spans the same ``start_date``..``end_date`` window as the
        primary OHLCV data and is reindexed to business days. Use
        :meth:`get_macro` to retrieve it after fetching, or
        :meth:`get_macro_aligned` to get it joined to a specific ticker.
        """
        from .macro_loader import MacroLoader

        ml = MacroLoader(
            start_date=self.start_date,
            end_date=self.end_date,
            yf_tickers=yf_tickers,
            fred_series=fred_series,
            cache_dir=cache_dir,
        )
        df = ml.fetch(use_cache=use_cache)
        if add_derived:
            df = MacroLoader.add_derived(df)
        self._macro = df
        return df

    def get_macro(self) -> Optional[pd.DataFrame]:
        """Return the cached macro panel, or None if :meth:`fetch_macro` not called."""
        return self._macro

    def get_macro_aligned(self, ticker: str) -> pd.DataFrame:
        """Return the macro panel reindexed to ``ticker``'s OHLCV index (ffill).

        Raises ``KeyError`` if the ticker has not been loaded, ``RuntimeError``
        if the macro panel has not been fetched yet.
        """
        if ticker not in self._data:
            raise KeyError(f"Ticker '{ticker}' not loaded.")
        if self._macro is None:
            raise RuntimeError("Macro panel not fetched. Call fetch_macro() first.")
        return self._macro.reindex(self._data[ticker].index).ffill().bfill()
