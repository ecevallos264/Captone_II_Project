from pathlib import Path
import pandas as pd
import yfinance as yf


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def download_single_ticker(ticker: str, start_date: str, end_date: str, save_path: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    df = _flatten_columns(df)

    if df.empty:
        raise ValueError(f"No data returned for ticker={ticker}")

    df = df.reset_index()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    return df


def download_many_tickers(tickers: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    output = {}
    for ticker in tickers:
        output[ticker] = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        output[ticker] = _flatten_columns(output[ticker]).reset_index()
    return output


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"])