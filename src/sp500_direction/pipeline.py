import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG
from src.sp500_direction.features import build_sp500_features, get_feature_columns, time_split
from src.sp500_direction.model import train_logistic_model, save_model
from src.sp500_direction.evaluate import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_coefficients,
    plot_cumulative_accuracy,
    save_metrics_csv,
)


def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def _save_json(data: dict, path: str):
    _ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def run_sp500_direction_pipeline() -> None:
    random.seed(CONFIG.random_seed)
    np.random.seed(CONFIG.random_seed)

    _ensure_dir("outputs/models")
    _ensure_dir("outputs/metrics")
    _ensure_dir("outputs/plots")
    _ensure_dir("outputs/backtests")

    # Build features
    print("Downloading S&P 500 data and building features...")
    df = build_sp500_features(CONFIG)
    feature_cols = get_feature_columns(CONFIG)
    print(f"  Dataset: {len(df)} samples, {len(feature_cols)} features")

    # Time-based train/test split
    train_df, test_df = time_split(df, CONFIG.sp500_test_ratio)
    X_train = train_df[feature_cols].values
    y_train = train_df["Target"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["Target"].values
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    # Scale features (fit on train only to avoid data leakage)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train
    print("Training Logistic Regression model...")
    model = train_logistic_model(
        X_train, y_train,
        C=CONFIG.sp500_logistic_C,
        max_iter=CONFIG.sp500_logistic_max_iter,
        random_state=CONFIG.random_seed,
    )

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate
    metrics = compute_classification_metrics(y_test, y_pred, y_prob)

    # Naive baseline (always predict "up")
    up_ratio = float(y_test.mean())
    metrics["naive_baseline_accuracy"] = max(up_ratio, 1 - up_ratio)
    metrics["test_up_ratio"] = up_ratio

    # Save model
    save_model(model, "outputs/models/SP500_direction_logistic.pkl")

    # Save metrics
    _save_json(metrics, "outputs/metrics/SP500_direction_metrics.json")
    save_metrics_csv(metrics, "outputs/metrics/SP500_direction_metrics.csv")

    # Generate plots
    print("Generating plots...")
    plot_confusion_matrix(y_test, y_pred, "outputs/plots/SP500_direction_confusion_matrix.png")
    plot_roc_curve(y_test, y_prob, "outputs/plots/SP500_direction_roc_curve.png")
    plot_feature_coefficients(feature_cols, model.coef_, "outputs/plots/SP500_direction_coefficients.png")
    plot_cumulative_accuracy(
        test_df["Date"].values, y_test, y_pred,
        "outputs/plots/SP500_direction_cumulative_accuracy.png",
    )

    # Save backtest CSV
    backtest_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "Actual_Direction": y_test,
        "Predicted_Direction": y_pred,
        "Predicted_Probability": y_prob,
        "Correct": (y_test == y_pred),
    })
    backtest_df.to_csv("outputs/backtests/SP500_direction_backtest.csv", index=False)

    # Print summary
    print(f"\n{'─' * 50}")
    print("S&P 500 Direction Prediction — Results")
    print(f"{'─' * 50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Naive baseline (always predict majority): {metrics['naive_baseline_accuracy']:.4f}")
    print(f"\n{metrics['classification_report']}")
