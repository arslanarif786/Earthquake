"""Part 2, Task C: metrics, threshold search, cross-validation."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score


def regression_report(y_true, y_pred) -> dict:
    """Return regression metrics as plain Python floats."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_report_dict(y_true, y_pred, y_proba) -> dict:
    """Return common binary-classification metrics as plain Python floats."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        # ROC AUC is undefined when y_true contains only one class.
        auc = 0.5
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
    }


def best_threshold(y_true, y_proba, metric: str = "f1") -> float:
    """Search thresholds from 0.05 through 0.95 and maximize a metric."""
    scorers = {
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
    }
    if metric not in scorers:
        raise ValueError("metric must be 'f1', 'precision', or 'recall'")
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    best_t, best_score = 0.5, -1.0
    for threshold in np.round(np.arange(0.05, 0.951, 0.05), 2):
        predictions = (y_proba >= threshold).astype(int)
        score = float(scorers[metric](y_true, predictions))
        if score > best_score:
            best_t, best_score = float(threshold), score
    return best_t


def cross_validate_model(pipe, X, y, scoring: str, cv: int = 5) -> dict:
    """Cross-validate a full pipeline and summarize its scores."""
    scores = cross_val_score(pipe, X, y, scoring=scoring, cv=cv)
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": [float(score) for score in scores],
    }
