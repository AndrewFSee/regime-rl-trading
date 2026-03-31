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

    def get_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Return the full DataFrame for a ticker, or None if not loaded."""
        return self._data.get(ticker)
