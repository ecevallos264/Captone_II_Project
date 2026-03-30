from pathlib import Path
import joblib
import pandas as pd

from config import CONFIG
from src.utils import ensure_dir, save_json, set_seed
from src.data_loader import download_single_ticker, download_many_tickers
from src.features import add_technical_features, merge_market_features, finalize_feature_frame
from src.preprocess import prepare_lstm_data, flatten_for_baseline, inverse_close_from_scaled
from src.baselines import train_linear_regression
from src.models.lstm_model import create_lstm_model, train_lstm
from src.evaluation.metrics import rmse, mae, mape, r2
from src.evaluation.plots import plot_actual_vs_predicted


def run_training_for_ticker(ticker: str) -> None:
    set_seed(CONFIG.random_seed)

    ensure_dir("data/raw")
    ensure_dir("outputs/models")
    ensure_dir("outputs/metrics")
    ensure_dir("outputs/plots")

    raw_path = f"data/raw/{ticker}.csv"
    stock_df = download_single_ticker(ticker, CONFIG.start_date, CONFIG.end_date, raw_path)

    market_data = {}
    if CONFIG.use_market_features:
        market_data = download_many_tickers(CONFIG.market_tickers, CONFIG.start_date, CONFIG.end_date)

    stock_df = add_technical_features(stock_df)
    if CONFIG.use_market_features:
        stock_df = merge_market_features(stock_df, market_data)

    feature_df, feature_columns = finalize_feature_frame(stock_df, CONFIG.use_market_features)

    x_train, y_train, x_test, y_test, scaler, _ = prepare_lstm_data(
        feature_df,
        feature_columns,
        CONFIG.sequence_length,
        CONFIG.test_size,
    )

    model = create_lstm_model((x_train.shape[1], x_train.shape[2]))
    train_lstm(model, x_train, y_train, CONFIG.epochs, CONFIG.batch_size)

    lstm_pred_scaled = model.predict(x_test, verbose=0).reshape(-1)
    y_test_actual = inverse_close_from_scaled(y_test, scaler, len(feature_columns))
    lstm_pred_actual = inverse_close_from_scaled(lstm_pred_scaled, scaler, len(feature_columns))

    x_train_flat = flatten_for_baseline(x_train)
    x_test_flat = flatten_for_baseline(x_test)
    baseline_model = train_linear_regression(x_train_flat, y_train)
    baseline_pred_scaled = baseline_model.predict(x_test_flat)
    baseline_pred_actual = inverse_close_from_scaled(baseline_pred_scaled, scaler, len(feature_columns))

    metrics = {
        "ticker": ticker,
        "lstm": {
            "RMSE": rmse(y_test_actual, lstm_pred_actual),
            "MAE": mae(y_test_actual, lstm_pred_actual),
            "MAPE": mape(y_test_actual, lstm_pred_actual),
            "R2": r2(y_test_actual, lstm_pred_actual),
        },
        "linear_regression_baseline": {
            "RMSE": rmse(y_test_actual, baseline_pred_actual),
            "MAE": mae(y_test_actual, baseline_pred_actual),
            "MAPE": mape(y_test_actual, baseline_pred_actual),
            "R2": r2(y_test_actual, baseline_pred_actual),
        },
    }

    model.save(f"outputs/models/{ticker}_lstm.keras")
    joblib.dump(scaler, f"outputs/models/{ticker}_scaler.pkl")
    joblib.dump(feature_columns, f"outputs/models/{ticker}_feature_columns.pkl")
    joblib.dump(baseline_model, f"outputs/models/{ticker}_baseline.pkl")

    pd.DataFrame([
        {
            "ticker": ticker,
            "model": "LSTM",
            **metrics["lstm"],
        },
        {
            "ticker": ticker,
            "model": "LinearRegression",
            **metrics["linear_regression_baseline"],
        },
    ]).to_csv(f"outputs/metrics/{ticker}_metrics.csv", index=False)

    save_json(metrics, f"outputs/metrics/{ticker}_metrics.json")

    plot_actual_vs_predicted(
        y_test_actual,
        lstm_pred_actual,
        baseline_pred_actual,
        f"outputs/plots/{ticker}_actual_vs_predicted.png",
    )

    print(f"Training complete for {ticker}")
    print(metrics)