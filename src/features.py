import numpy as np
import pandas as pd


BASE_FEATURE_COLUMNS = [
    "Close",
    "Volume",
    "Return",
    "MA_5",
    "MA_10",
    "MA_20",
    "EMA_12",
    "EMA_26",
    "MACD",
    "MACD_Signal",
    "Volatility_10",
    "Volume_Change",
    "RSI_14",
    "BB_Middle",
    "BB_Upper",
    "BB_Lower",
]


MARKET_FEATURE_COLUMNS = [
    "SP500_Close",
    "SP500_Return",
    "NASDAQ_Close",
    "NASDAQ_Return",
]


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Return"] = df["Close"].pct_change()
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()

    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    rolling_std = df["Close"].rolling(20).std()
    df["BB_Middle"] = df["MA_20"]
    df["BB_Upper"] = df["MA_20"] + (2 * rolling_std)
    df["BB_Lower"] = df["MA_20"] - (2 * rolling_std)

    df["Volatility_10"] = df["Return"].rolling(10).std()
    df["Volume_Change"] = df["Volume"].pct_change()
    df["RSI_14"] = compute_rsi(df["Close"], 14)

    return df


def merge_market_features(stock_df: pd.DataFrame, market_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = stock_df.copy()

    if "^GSPC" in market_data:
        sp = market_data["^GSPC"][["Date", "Close"]].copy()
        sp.columns = ["Date", "SP500_Close"]
        df = df.merge(sp, on="Date", how="left")
        df["SP500_Return"] = df["SP500_Close"].pct_change()

    if "^IXIC" in market_data:
        nd = market_data["^IXIC"][["Date", "Close"]].copy()
        nd.columns = ["Date", "NASDAQ_Close"]
        df = df.merge(nd, on="Date", how="left")
        df["NASDAQ_Return"] = df["NASDAQ_Close"].pct_change()

    return df


def finalize_feature_frame(df: pd.DataFrame, use_market_features: bool = True) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    feature_columns = BASE_FEATURE_COLUMNS.copy()

    if use_market_features:
        for col in MARKET_FEATURE_COLUMNS:
            if col in df.columns:
                feature_columns.append(col)

    df = df.dropna().reset_index(drop=True)
    return df, feature_columns