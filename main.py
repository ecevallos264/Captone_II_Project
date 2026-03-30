from config import CONFIG
from src.pipeline.train import run_training_for_ticker
from src.pipeline.predict import run_prediction_for_ticker
from src.pipeline.backtest import run_backtest_for_ticker


def main() -> None:
    for ticker in CONFIG.tickers:
        print(f"\n{'=' * 70}")
        print(f"Running pipeline for {ticker}")
        print(f"{'=' * 70}\n")
        run_training_for_ticker(ticker)
        run_prediction_for_ticker(ticker)
        run_backtest_for_ticker(ticker)


if __name__ == "__main__":
    main()