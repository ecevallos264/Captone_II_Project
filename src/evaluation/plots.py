from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)


def plot_actual_vs_predicted(actual, predicted, baseline_pred, save_path: str):
    _ensure_parent(save_path)

    plt.figure(figsize=(10, 6))
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="LSTM Predicted")
    plt.plot(baseline_pred, label="Linear Regression Baseline")
    plt.title("Actual vs Predicted Closing Price")
    plt.xlabel("Test Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_monte_carlo(paths: np.ndarray, save_path: str):
    _ensure_parent(save_path)

    plt.figure(figsize=(10, 6))
    plt.plot(paths[:, :50], alpha=0.35)
    plt.title("Monte Carlo Forecast Paths")
    plt.xlabel("Future Day")
    plt.ylabel("Simulated Price")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confidence_band(paths: np.ndarray, save_path: str):
    _ensure_parent(save_path)

    days = np.arange(paths.shape[0])
    median = np.percentile(paths, 50, axis=1)
    lower = np.percentile(paths, 5, axis=1)
    upper = np.percentile(paths, 95, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(days, median, label="Median Forecast")
    plt.fill_between(days, lower, upper, alpha=0.25, label="90% Interval")
    plt.title("Forecast Confidence Interval")
    plt.xlabel("Future Day")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()