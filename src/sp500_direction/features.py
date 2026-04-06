from typing import List, Tuple

import numpy as np
import pandas as pd

from src.data_loader import download_single_ticker

TECHNICAL_FEATURES = ["RSI_14", "MACD_Signal", "SMA_Cross", "Volatility_10"]


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def build_sp500_features(config) -> pd.DataFrame:
    df = download_single_ticker(
        config.sp500_ticker,
        config.sp500_start_date,
        config.sp500_end_date,
        "data/raw/SP500.csv",
    )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["Return"] = df["Close"].pct_change()

    # Lagged returns
    for lag in config.sp500_lag_days:
        df[f"Return_Lag_{lag}"] = df["Return"].shift(lag)

    # RSI (14-day) — momentum oscillator, scaled 0-100
    df["RSI_14"] = _compute_rsi(df["Close"], 14)

    # MACD signal crossover — difference between MACD line and signal line
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD_Signal"] = macd_line - signal_line

    # SMA crossover — SMA(5) minus SMA(20), positive means short-term above long-term
    sma5 = df["Close"].rolling(5).mean()
    sma20 = df["Close"].rolling(20).mean()
    df["SMA_Cross"] = (sma5 - sma20) / sma20

    # Volatility — 10-day rolling standard deviation of returns
    df["Volatility_10"] = df["Return"].rolling(10).std()

    # Target: 1 if tomorrow's close > today's close
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.dropna().reset_index(drop=True)

    keep_cols = ["Date"] + get_feature_columns(config) + ["Target"]
    return df[keep_cols]


def get_feature_columns(config) -> List[str]:
    lag_cols = [f"Return_Lag_{n}" for n in config.sp500_lag_days]
    return lag_cols + TECHNICAL_FEATURES


def time_split(df: pd.DataFrame, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
