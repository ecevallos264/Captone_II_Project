from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay,
)


def _ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def compute_classification_metrics(y_true, y_pred, y_prob) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=["Down", "Up"]),
    }


def plot_confusion_matrix(y_true, y_pred, save_path: str):
    _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Down", "Up"], cmap="Blues", ax=ax
    )
    ax.set_title("S&P 500 Direction — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path: str):
    _ensure_parent(save_path)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("S&P 500 Direction — ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_coefficients(feature_names: list, coefficients: np.ndarray, save_path: str):
    _ensure_parent(save_path)
    coefs = coefficients.flatten()
    sorted_idx = np.argsort(np.abs(coefs))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in sorted_idx], coefs[sorted_idx])
    ax.set_xlabel("Coefficient Value")
    ax.set_title("S&P 500 Direction — Logistic Regression Coefficients")
    ax.axvline(x=0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_cumulative_accuracy(dates, y_true, y_pred, save_path: str):
    _ensure_parent(save_path)
    correct = (np.array(y_true) == np.array(y_pred)).astype(float)
    cumulative_acc = np.cumsum(correct) / np.arange(1, len(correct) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, cumulative_acc, label="Cumulative Accuracy")
    ax.axhline(y=0.5, color="r", linestyle="--", label="50% Baseline")
    ax.set_xlabel("Date")
    ax.set_ylabel("Accuracy")
    ax.set_title("S&P 500 Direction — Cumulative Accuracy Over Test Period")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_metrics_csv(metrics: dict, save_path: str):
    _ensure_parent(save_path)
    row = {k: v for k, v in metrics.items() if k not in ("confusion_matrix", "classification_report")}
    pd.DataFrame([row]).to_csv(save_path, index=False)
