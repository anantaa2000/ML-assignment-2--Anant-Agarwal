"""Download UCI Spambase, split once, train five pipelines and save evaluation."""
import argparse
import hashlib
import io
import json
import platform
from pathlib import Path
import urllib.request
import zipfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

from ml_utils import MODEL_FILES, TARGET, evaluate, validate_data

ROOT = Path(__file__).resolve().parent
URL = "https://archive.ics.uci.edu/static/public/94/spambase.zip"
SEED = 42


def main():
    # Allow an optional fresh download instead of reusing the local dataset.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Refresh the cached UCI archive")
    args = parser.parse_args()

    # Download the official UCI archive if needed.
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    archive = data_dir / "spambase.zip"
    if args.download or not archive.exists():
        print(f"Downloading {URL}", flush=True)
        request = urllib.request.Request(URL, headers={"User-Agent": "ML-assignment/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            archive.write_bytes(response.read())

    # Read feature names and email records directly from the ZIP archive.
    with zipfile.ZipFile(archive) as z:
        names = z.read("spambase.names").decode()
        features = [line.split(":")[0].strip() for line in names.splitlines()
                    if ":" in line and line.strip().endswith("continuous.") and not line.startswith("|")]
        frame = pd.read_csv(io.BytesIO(z.read("spambase.data")), header=None,
                            names=features + [TARGET])
        (data_dir / "spambase.names").write_text(names)

    # Check dataset dimensions and separate valid inputs (X) from labels (y).
    if len(features) != 57 or len(frame) != 4601:
        raise ValueError("Unexpected UCI dataset dimensions")
    X, y = validate_data(frame, features)
    # Identical feature vectors stay together, including any conflicting labels.
    # This avoids an optimistic score from repeated emails across the split.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    # Use one fold as the test set (about 20%); the remaining rows train all models.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    assert not set(groups[train_idx]) & set(groups[test_idx])
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

    # Export held-out records for the app and create artifact directories.
    frame.iloc[test_idx].to_csv(ROOT / "test_data.csv", index=False)
    model_dir = ROOT / "model"
    result_dir = ROOT / "results"
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    # Bundle scaling with models that need it; scalers learn only during fit().
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=SEED)),
        "Decision Tree": make_pipeline(DecisionTreeClassifier(max_depth=12, min_samples_leaf=3, random_state=SEED)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "Naive Bayes": make_pipeline(GaussianNB()),
        "Support Vector Machine": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale", random_state=SEED)),
    }
    rows, details = [], {}
    for name, model in models.items():
        # Fit a new model on training data, then evaluate on unseen test rows.
        print(f"Training {name}...", flush=True)
        model.fit(X_train, y_train)
        metrics, predictions, _ = evaluate(model, X_test, y_test)
        rows.append({"Model": name, **metrics})
        details[name] = {"confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
                         "classification_report": classification_report(y_test, predictions, labels=[0, 1], output_dict=True, zero_division=0),
                         "parameters": str(model)}
        # Save the fitted pipeline, including any preprocessing, for the app.
        joblib.dump(model, model_dir / MODEL_FILES[name], compress=3)

    # Store all six metrics and detailed classification results for each model.
    comparison = pd.DataFrame(rows)
    comparison.to_csv(result_dir / "metrics.csv", index=False)
    (result_dir / "details.json").write_text(json.dumps(details, indent=2))

    # Record dataset identity, split details and package versions for reproducibility.
    metadata = {
        "dataset": "UCI Spambase", "source": URL,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "features": features, "target": TARGET, "seed": SEED,
        "rows": len(frame), "feature_count": len(features),
        "train_rows": len(train_idx), "test_rows": len(test_idx),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
        "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().items()},
        "unique_feature_vectors": int(len(np.unique(groups))),
        "train_test_feature_overlap": 0,
        "split": "First fold of StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42); identical feature vectors grouped",
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "winner_by_test_f1": str(comparison.loc[comparison.F1.idxmax(), "Model"]),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    # Preserve the exact row assignments so the split can be checked later.
    np.savez_compressed(result_dir / "split_indices.npz", train=train_idx, test=test_idx)

    # Print a compact comparison after all models have been saved.
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Saved models, test_data.csv, metadata and results.")


if __name__ == "__main__":
    main()
