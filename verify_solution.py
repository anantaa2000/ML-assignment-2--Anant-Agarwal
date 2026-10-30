"""Integration checks for persisted models, split integrity and Streamlit UI."""
import io
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from streamlit.testing.v1 import AppTest

from ml_utils import MODEL_FILES, evaluate, validate_data

ROOT = Path(__file__).resolve().parent


def main():
    metadata = json.loads((ROOT / "model/metadata.json").read_text())
    features = metadata["features"]
    test = pd.read_csv(ROOT / "test_data.csv")
    X, y = validate_data(test, features)
    recorded = pd.read_csv(ROOT / "results/metrics.csv").set_index("Model")
    split = np.load(ROOT / "results/split_indices.npz")
    with zipfile.ZipFile(ROOT / "data/spambase.zip") as archive:
        source = pd.read_csv(io.BytesIO(archive.read("spambase.data")), header=None, names=features + ["spam"])
    pd.testing.assert_frame_equal(test, source.iloc[split["test"]].reset_index(drop=True))
    train_hash = set(pd.util.hash_pandas_object(source.iloc[split["train"]][features], index=False))
    test_hash = set(pd.util.hash_pandas_object(X, index=False))
    assert not train_hash & test_hash, "Repeated feature vectors leaked across partitions"
    assert set(split["train"]) | set(split["test"]) == set(range(len(source)))
    for name, file in MODEL_FILES.items():
        model = joblib.load(ROOT / "model" / file)
        metrics, predictions, scores = evaluate(model, X, y)
        np.testing.assert_allclose(list(metrics.values()), recorded.loc[name].to_numpy(dtype=float), atol=1e-12)
        tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
        np.testing.assert_allclose(metrics["Accuracy"], (tn + tp) / len(y))
        np.testing.assert_allclose(metrics["Precision"], tp / (tp + fp))
        np.testing.assert_allclose(metrics["Recall"], tp / (tp + fn))
        np.testing.assert_allclose(metrics["AUC"], roc_auc_score(y, scores))
        single = y == 0
        assert np.isnan(evaluate(model, X[single], y[single])[0]["AUC"])
        reversed_X, _ = validate_data(test[list(reversed(test.columns))], features)
        np.testing.assert_array_equal(model.predict(reversed_X), predictions)
    invalid_frames = [test.drop(columns=features[0]), test.assign(spam=2),
                      test.assign(**{features[0]: np.nan}), test.assign(**{features[0]: np.inf}),
                      test.assign(**{features[0]: "bad"}), test.iloc[:0], test.assign(unexpected=1)]
    for invalid in invalid_frames:
        try:
            validate_data(invalid, features)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid CSV was accepted")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception, app.exception
    assert not app.error, app.error
    assert len(app.metric) == 6
    assert [tab.label for tab in app.tabs] == ["Single model details", "All model comparison"]
    for name in MODEL_FILES:
        app.selectbox[0].select(name).run()
        assert not app.exception, app.exception
        assert app.metric[0].value == f"{recorded.loc[name, 'Accuracy']:.4f}"
    for metric in ("AUC", "MCC", "F1"):
        app.selectbox[1].select(metric).run()
        assert not app.exception, app.exception
    # Exercise upload parsing and full UI evaluation (AppTest has no file-uploader setter).
    from unittest.mock import patch
    for raw, expected_error in [(test.head(50).to_csv(index=False).encode(), False),
                                (b"spam,spam\n0,1\n", True),
                                (test.drop(columns=features[0]).to_csv(index=False).encode(), True)]:
        class Upload:
            def getvalue(self):
                return raw
        with patch("streamlit.file_uploader", return_value=Upload()):
            uploaded_app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        assert not uploaded_app.exception, uploaded_app.exception
        assert bool(uploaded_app.error) == expected_error
        if not expected_error:
            with patch("streamlit.file_uploader", return_value=Upload()):
                uploaded_app.selectbox[1].select("AUC").run()
            assert not uploaded_app.exception, uploaded_app.exception
            assert not uploaded_app.error, uploaded_app.error
    print("PASS: five persisted models reproduce metrics; no split leakage; CSV validation, single-class AUC, all model selections and uploaded CSV flows verified.")


if __name__ == "__main__":
    main()
