"""Streamlit evaluation UI. Only locally saved, trusted models are loaded."""
import csv
import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from ml_utils import MODEL_FILES, evaluate, validate_data
from ui import render_header, single_model_view, comparison_view

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Assignment 2 | Anant Agarwal | BITS Pilani", page_icon=str(ROOT / "assets/bits-pilani-logo.png"), layout="wide")


@st.cache_resource
def load_models():
    return {name: joblib.load(ROOT / "model" / file) for name, file in MODEL_FILES.items()}


def read_csv_bytes(raw):
    # pandas renames duplicate headers; reject them before pandas parses the data.
    header = next(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    if len(header) != len(set(header)):
        raise ValueError("Duplicate column names are not allowed.")
    return pd.read_csv(io.BytesIO(raw))


def main():
    render_header()
    st.html('<div class="section-label">Classification experiment / UCI Spambase</div>')
    st.title("Email spam classification")
    st.write("Explore five classifiers, inspect their predictions, and compare performance on labeled test data.")
    try:
        metadata = json.loads((ROOT / "model" / "metadata.json").read_text())
        models = load_models()
    except FileNotFoundError:
        st.error("Saved artifacts are missing. Run python train.py before starting the app.")
        st.stop()
    with st.sidebar:
        st.header("Test dataset")
        st.caption("Use the bundled held-out sample or upload your own labeled test records.")
        uploaded = st.file_uploader("Upload test CSV", type=["csv"])
        st.caption("Include all 57 numeric features and spam (0 = non-spam, 1 = spam). Models are never retrained here.")
        st.download_button("Download sample test CSV", (ROOT / "test_data.csv").read_bytes(),
                           file_name="test_data.csv", mime="text/csv")
    try:
        frame = read_csv_bytes(uploaded.getvalue()) if uploaded else pd.read_csv(ROOT / "test_data.csv")
        if len(frame) > 20000:
            raise ValueError("Upload at most 20,000 test rows to keep evaluation responsive.")
        X, y = validate_data(frame, metadata["features"])
    except (ValueError, UnicodeError, StopIteration, pd.errors.ParserError) as exc:
        st.error(f"Cannot evaluate this CSV: {exc}")
        st.stop()
    st.html(f'<div class="dataset-strip"><span><strong>{"Uploaded test data" if uploaded else "Bundled held-out test data"}</strong></span><span><strong>{len(X):,}</strong> emails</span><span><strong>{len(X.columns)}</strong> features</span><span><strong>{int((y == 1).sum()):,}</strong> spam</span><span><strong>{int((y == 0).sum()):,}</strong> non-spam</span></div>')
    if y.nunique() < 2:
        st.warning("This CSV contains one target class. AUC is undefined; the other metrics are still shown.")
    results = {}
    for name, model in models.items():
        results[name] = evaluate(model, X, y)
    single_tab, comparison_tab = st.tabs(["Single model details", "All model comparison"])
    with single_tab:
        single_model_view(results, models, frame, y)
    with comparison_tab:
        comparison_view(results, y)
    with st.expander("Dataset and experiment details"):
        st.write(f"Training rows: {metadata['train_rows']:,}; bundled test rows: {metadata['test_rows']:,}. Seed: {metadata['seed']}.")
        st.write("Identical feature vectors stay in the same split. Scaling is fitted only on training data and saved inside each applicable pipeline. Hyperparameters were fixed before test evaluation.")
        st.markdown("[UCI dataset and feature definitions](https://archive.ics.uci.edu/dataset/94/spambase)")
        st.dataframe(frame.head(), width="stretch")
    st.html('<footer class="app-footer">Anant Agarwal · 2026AJ05360 &nbsp; | &nbsp; Machine Learning · Assignment 2</footer>')


if __name__ == "__main__":
    main()
