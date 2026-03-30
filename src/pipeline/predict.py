import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from config import CONFIG
from src.data_loader import load_csv, download_many_tickers
from src.features import add_technical_features, merge_market_features, finalize_feature_frame
from src.models.volatility import estimate_volatility
from src.simulation.monte_carlo import run_monte_carlo
from src.evaluation.plots import plot_monte_carlo, plot_confidence_band
from src.utils import save_json


def run_prediction_for_ticker(ticker: str) -> None:
    model = load_model(f"outputs/models/{ticker}_lstm.keras")
    scaler = joblib.load(f"outputs/models/{ticker}_scaler.pkl")
    feature_columns = joblib.load(f"outputs/models/{ticker}_feature_columns.pkl")

    df = load_csv(f"data/raw/{ticker}.csv")
    df = add_technical_features(df)

    if CONFIG.use_market_features:
        market_data = download_many_tickers(CONFIG.market_tickers, CONFIG.start_date, CONFIG.end_date)
        df = merge_market_features(df, market_data)

    df, _ = finalize_feature_frame(df, CONFIG.use_market_features)

    recent = df[feature_columns].tail(CONFIG.sequence_length).values
    recent_scaled = scaler.transform(recent)
    x_input = np.expand_dims(recent_scaled, axis=0)

    next_close_scaled = model.predict(x_input, verbose=0)[0][0]

    dummy = np.zeros((1, len(feature_columns)))
    dummy[0, 0] = next_close_scaled
    next_close_pred = scaler.inverse_transform(dummy)[0, 0]

    last_close = df["Close"].iloc[-1]
    predicted_daily_return = (next_close_pred - last_close) / last_close
    daily_volatility = estimate_volatility(df["Return"])

    paths = run_monte_carlo(
        last_price=float(last_close),
        predicted_daily_return=float(predicted_daily_return),
        daily_volatility=float(daily_volatility),
        days=CONFIG.forecast_days,
        n_paths=CONFIG.monte_carlo_paths,
        seed=CONFIG.random_seed,
    )

    plot_monte_carlo(paths, f"outputs/plots/{ticker}_monte_carlo.png")
    plot_confidence_band(paths, f"outputs/plots/{ticker}_confidence_band.png")

    forecast_summary = {
        "ticker": ticker,
        "last_close": float(last_close),
        "predicted_next_close": float(next_close_pred),
        "predicted_daily_return": float(predicted_daily_return),
        "estimated_daily_volatility": float(daily_volatility),
        "forecast_days": CONFIG.forecast_days,
        "final_day_5th_percentile": float(np.percentile(paths[-1], 5)),
        "final_day_50th_percentile": float(np.percentile(paths[-1], 50)),
        "final_day_95th_percentile": float(np.percentile(paths[-1], 95)),
    }

    save_json(forecast_summary, f"outputs/forecasts/{ticker}_forecast_summary.json")
    pd.DataFrame(paths).to_csv(f"outputs/forecasts/{ticker}_monte_carlo_paths.csv", index=False)

    print(f"Prediction complete for {ticker}")
    print(forecast_summary)