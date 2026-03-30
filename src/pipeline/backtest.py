import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from config import CONFIG
from src.data_loader import load_csv, download_many_tickers
from src.features import add_technical_features, merge_market_features, finalize_feature_frame
from src.evaluation.metrics import rmse, mae, mape, r2
from src.utils import save_json, ensure_dir


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0

    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    return float(np.mean(true_direction == pred_direction))


def make_sequences(values: np.ndarray, seq_len: int):
    x, y = [], []
    for i in range(seq_len, len(values)):
        x.append(values[i - seq_len:i])
        y.append(values[i, 0])
    return np.array(x), np.array(y)


def inverse_close(values_scaled: np.ndarray, scaler, n_features: int) -> np.ndarray:
    values_scaled = np.asarray(values_scaled).reshape(-1)
    dummy = np.zeros((len(values_scaled), n_features))
    dummy[:, 0] = values_scaled
    return scaler.inverse_transform(dummy)[:, 0]


def run_backtest_for_ticker(ticker: str) -> None:
    ensure_dir("outputs/backtests")

    model = load_model(f"outputs/models/{ticker}_lstm.keras")
    scaler = joblib.load(f"outputs/models/{ticker}_scaler.pkl")
    feature_columns = joblib.load(f"outputs/models/{ticker}_feature_columns.pkl")

    df = load_csv(f"data/raw/{ticker}.csv")
    df = add_technical_features(df)

    if CONFIG.use_market_features:
        market_data = download_many_tickers(CONFIG.market_tickers, CONFIG.start_date, CONFIG.end_date)
        df = merge_market_features(df, market_data)

    df, _ = finalize_feature_frame(df, CONFIG.use_market_features)

    feature_df = df[feature_columns].copy()
    scaled_values = scaler.transform(feature_df)

    split_idx = int(len(scaled_values) * (1 - CONFIG.test_size))
    test_values = scaled_values[split_idx - CONFIG.sequence_length:]

    x_test, y_test = make_sequences(test_values, CONFIG.sequence_length)
    y_pred_scaled = model.predict(x_test, verbose=0).reshape(-1)

    y_test_actual = inverse_close(y_test, scaler, len(feature_columns))
    y_pred_actual = inverse_close(y_pred_scaled, scaler, len(feature_columns))

    results_df = pd.DataFrame({
        "Actual_Close": y_test_actual,
        "Predicted_Close": y_pred_actual,
        "Absolute_Error": np.abs(y_test_actual - y_pred_actual),
        "Percent_Error": np.abs((y_test_actual - y_pred_actual) / (y_test_actual + 1e-10)) * 100,
    })
    results_df.to_csv(f"outputs/backtests/{ticker}_backtest_predictions.csv", index=False)

    summary = {
        "ticker": ticker,
        "RMSE": rmse(y_test_actual, y_pred_actual),
        "MAE": mae(y_test_actual, y_pred_actual),
        "MAPE": mape(y_test_actual, y_pred_actual),
        "R2": r2(y_test_actual, y_pred_actual),
        "Directional_Accuracy": directional_accuracy(y_test_actual, y_pred_actual),
        "num_test_predictions": int(len(y_test_actual)),
    }
    save_json(summary, f"outputs/backtests/{ticker}_backtest_summary.json")

    print(f"Backtest complete for {ticker}")
    print(summary)