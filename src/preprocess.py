import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


TARGET_COLUMN = "Close"


def build_sequences(data: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(seq_len, len(data)):
        x.append(data[i - seq_len:i])
        y.append(data[i, 0])
    return np.array(x), np.array(y)


def prepare_lstm_data(
    df: pd.DataFrame,
    feature_columns: list[str],
    seq_len: int,
    test_size: float,
):
    feature_df = df[feature_columns].copy()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feature_df)

    split_idx = int(len(scaled) * (1 - test_size))
    train_data = scaled[:split_idx]
    test_data = scaled[split_idx - seq_len:]

    x_train, y_train = build_sequences(train_data, seq_len)
    x_test, y_test = build_sequences(test_data, seq_len)

    return x_train, y_train, x_test, y_test, scaler, feature_df


def flatten_for_baseline(x_sequences: np.ndarray) -> np.ndarray:
    return x_sequences.reshape(x_sequences.shape[0], -1)


def inverse_close_from_scaled(pred_scaled: np.ndarray, scaler, n_features: int) -> np.ndarray:
    pred_scaled = np.asarray(pred_scaled).reshape(-1)
    dummy = np.zeros((len(pred_scaled), n_features))
    dummy[:, 0] = pred_scaled
    return scaler.inverse_transform(dummy)[:, 0]