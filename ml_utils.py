"""Shared validation and evaluation; the app never fits an estimator."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

TARGET = "spam"
# Map display names to the saved model files used by training and the app.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def validate_data(frame, features):
    # Reject empty data and ambiguous column headers.
    if frame.empty:
        raise ValueError("The CSV must contain at least one data row.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    # Require exactly the trained feature columns and the target label.
    expected = list(features) + [TARGET]
    missing = sorted(set(expected) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected))
    if missing or extra:
        raise ValueError(f"CSV columns do not match. Missing: {missing}; unexpected: {extra}.")
    # Arrange columns in the expected order and convert values to numbers.
    try:
        numeric = frame[expected].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("All feature values must be numeric and spam must be 0 or 1.") from exc
    # Reject missing/infinite values, invalid labels and negative features.
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Missing values and infinite values are not allowed.")
    if not numeric[TARGET].isin([0, 1]).all():
        raise ValueError("The spam target must contain only 0 (non-spam) and 1 (spam).")
    if (numeric[features] < 0).any().any():
        raise ValueError("Spambase feature values must be non-negative.")
    # Return model inputs separately from the actual spam labels.
    return numeric[features], numeric[TARGET].astype(int)


def evaluate(model, X, y):
    # Predict labels using the fitted model without retraining it.
    predictions = model.predict(X)
    # Use decision scores (LR/SVM) or spam probabilities for ROC AUC.
    scores = (model.decision_function(X) if hasattr(model, "decision_function")
              else model.predict_proba(X)[:, list(model.classes_).index(1)])
    # Calculate six metrics; AUC needs both classes, and spam (1) is positive.
    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(y, scores) if len(np.unique(y)) == 2 else float("nan"),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y, predictions),
    }
    # Share metrics, predicted labels and scores with reports and charts.
    return metrics, predictions, scores
