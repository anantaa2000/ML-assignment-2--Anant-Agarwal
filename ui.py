"""Presentation helpers for the BITS assignment dashboard."""
import base64
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, precision_recall_curve, roc_curve)

ROOT = Path(__file__).resolve().parent
alt.data_transformers.disable_max_rows()
RED = "#B52232"
NAVY = "#17365D"
MODEL_COLORS = [RED, NAVY, "#B47B13", "#468582", "#77648D"]
SHORT_NAMES = {"Logistic Regression": "Logistic Regression", "Decision Tree": "Decision Tree",
               "KNN": "KNN", "Naive Bayes": "Naive Bayes", "Support Vector Machine": "SVM"}
METRIC_HELP = {
    "Accuracy": "Fraction of all emails classified correctly.",
    "AUC": "ROC area using continuous scores. Undefined when only one target class is present.",
    "Precision": "Of emails predicted as spam, the fraction that actually are spam.",
    "Recall": "Of actual spam emails, the fraction detected.",
    "F1": "Harmonic mean of spam precision and recall.",
    "MCC": "Agreement across all four confusion-matrix cells; ranges from −1 to 1.",
}


def render_header():
    st.html("""<style>
    .stApp { background: #fff; }
    [data-testid="stHeader"] { background: rgba(255,255,255,.97); }
    .block-container { padding-top: 4.5rem; padding-bottom: 2rem; max-width: 1440px; }
    [data-testid="stSidebar"] { background: #f7f7f5; border-right: 1px solid #e1e3e5; }
    h1, h2, h3 { color: #17365d; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.12rem !important; }
    .assignment-header { display: flex; align-items: center; gap: 34px; padding: 10px 0 26px; }
    .assignment-header img { width: 235px; height: auto; flex-shrink: 0; }
    .assignment-heading { border-left: 1px solid #d8dde2; padding-left: 32px; }
    .assignment-heading h1 { font-family: Georgia, serif; font-size: 2.5rem; line-height: 1.15; padding: 0; margin: 0 0 10px; color: #17365d; }
    .student-name { font-size: 1.13rem; font-weight: 600; margin: 0 0 2px; color: #26384d; }
    .student-id { font-size: .9rem; letter-spacing: .07em; color: #6a717a; margin: 0; }
    .course-label { margin-left: auto; text-align: right; color: #596575; font-size: .86rem; line-height: 1.8; }
    .course-label strong { color: #17365d; font-size: .96rem; }
    .bits-rule { display: flex; height: 4px; margin-bottom: 28px; }
    .bits-rule span:nth-child(1) { background: #17365d; width: 55%; }
    .bits-rule span:nth-child(2) { background: #b52232; width: 30%; }
    .bits-rule span:nth-child(3) { background: #e6b441; width: 15%; }
    .section-label { text-transform: uppercase; letter-spacing: .12em; color: #b52232; font-weight: 600; font-size: .74rem; margin-bottom: 8px; }
    .dataset-strip { display: flex; flex-wrap: wrap; gap: 12px 30px; padding: 14px 18px; margin: 18px 0 24px; background: #f7f7f5; border-left: 3px solid #b52232; font-size: .88rem; color: #596575; }
    .dataset-strip strong { color: #17365d; }
    [data-testid="stMetric"] { border: 1px solid #e2e5e8; border-top: 3px solid #b52232; padding: 14px 12px; background: #fff; }
    [data-testid="stMetricValue"] { font-size: 1.65rem; color: #17365d; }
    .comparison-note { border-left: 3px solid #b52232; padding: 12px 18px; background: #fbf6f6; margin: 8px 0 24px; line-height: 1.7; }
    .app-footer { border-top: 1px solid #e2e5e8; margin-top: 32px; padding-top: 16px; color: #747c85; font-size: .78rem; }
    button[data-baseweb="tab"] { font-size: 1rem; padding: 14px 22px; }
    @media (max-width: 760px) {
      .assignment-header { flex-wrap: wrap; gap: 20px; }
      .assignment-header img { width: 190px; }
      .assignment-heading { padding-left: 20px; }
      .assignment-heading h1 { font-size: 1.9rem; }
      .course-label { display: none; }
      .block-container { padding-top: 4.5rem; }
    }
    @media (max-width: 450px) {
      .assignment-heading { border-left: 0; padding-left: 0; width: 100%; }
    }
    </style>""")
    logo = base64.b64encode((ROOT / "assets/bits-pilani-logo.png").read_bytes()).decode()
    st.html(f"""<header class="assignment-header">
      <img src="data:image/png;base64,{logo}" alt="BITS Pilani logo">
      <div class="assignment-heading"><h1>Assignment 2</h1>
        <p class="student-name">Anant Agarwal</p><p class="student-id">2026AJ05360</p></div>
      <div class="course-label"><strong>Machine Learning</strong><br>M.Tech · Artificial Intelligence &amp; Machine Learning<br>Work Integrated Learning Programmes</div>
      </header><div class="bits-rule" aria-hidden="true"><span></span><span></span><span></span></div>""")


def finish_chart(chart):
    return (chart.configure_view(stroke=None)
            .configure_axis(labelColor="#596575", titleColor=NAVY, gridColor="#eceef0", labelFontSize=11, titleFontSize=12)
            .configure_legend(title=None, labelColor=NAVY, orient="bottom", labelLimit=220)
            .configure_title(color=NAVY, anchor="start", fontSize=14))


def show_chart(chart):
    st.altair_chart(finish_chart(chart), use_container_width=True, theme=None)


def curve_chart(results, y, kind):
    rows = []
    for name, (_, _, scores) in results.items():
        if kind == "roc":
            x, value, _ = roc_curve(y, scores)
        else:
            value, x, _ = precision_recall_curve(y, scores)
        # Order preserves vertical steps at repeated x coordinates.
        rows.extend({"Model": name, "x": float(a), "y": float(b), "order": i}
                    for i, (a, b) in enumerate(zip(x, value)))
    x_title = "False positive rate" if kind == "roc" else "Recall"
    y_title = "True positive rate (recall)" if kind == "roc" else "Precision"
    domain = list(results)
    palette = [MODEL_COLORS[list(SHORT_NAMES).index(name)] for name in domain]
    curve = alt.Chart(pd.DataFrame(rows)).mark_line(strokeWidth=2.3).encode(
        x=alt.X("x:Q", title=x_title, scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("y:Q", title=y_title, scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Model:N", scale=alt.Scale(domain=domain, range=palette)),
        order="order:Q", tooltip=["Model:N", alt.Tooltip("x:Q", title=x_title, format=".3f"), alt.Tooltip("y:Q", title=y_title, format=".3f")])
    if kind == "roc":
        baseline = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="#a6acb3", strokeDash=[5, 5]).encode(x="x:Q", y="y:Q")
    else:
        baseline = alt.Chart(pd.DataFrame({"baseline": [float(y.mean())]})).mark_rule(color="#a6acb3", strokeDash=[5, 5]).encode(y="baseline:Q")
    return (baseline + curve).properties(height=300)


def single_model_view(results, models, frame, y):
    st.write("")
    selected = st.selectbox("Choose a model", list(models))
    metrics, predictions, scores = results[selected]
    for col, (label, value) in zip(st.columns(6), metrics.items()):
        col.metric(label, "N/A" if pd.isna(value) else f"{value:.4f}", help=METRIC_HELP[label])
    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ConfusionMatrixDisplay.from_predictions(y, predictions, labels=[0, 1], display_labels=["Non-spam", "Spam"], cmap="Reds", colorbar=False, ax=ax)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Rows are actual labels; columns are predictions. Off-diagonal cells are errors.")
    with right:
        st.subheader("Classification report")
        report = classification_report(y, predictions, labels=[0, 1], target_names=["Non-spam", "Spam"], output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.style.format(precision=4), width="stretch")
        tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
        st.write(f"**{fp:,} legitimate emails flagged as spam** · **{fn:,} spam emails missed**")
        st.caption("Spam (1) is the positive class. Undefined precision, recall and F1 use 0. AUC uses continuous scores; it is N/A for a single-class test set.")
    if y.nunique() == 2:
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("ROC curve")
            show_chart(curve_chart({selected: results[selected]}, y, "roc"))
            st.caption("A curve nearer the top-left corner indicates better separation. The dashed diagonal is chance ranking.")
        with right:
            st.subheader("Precision–recall curve")
            show_chart(curve_chart({selected: results[selected]}, y, "pr"))
            st.caption("Shows the trade-off as the decision threshold varies. The dashed line is the spam prevalence in this CSV.")
    output = frame.copy()
    output["prediction"] = predictions
    output["spam_score"] = scores
    st.download_button("Download model predictions", output.to_csv(index=False), "predictions.csv", "text/csv")
    st.caption("Spam scores are decision margins for Logistic Regression/SVM and spam probabilities for the other models. Margins are not probabilities.")


def comparison_view(results, y):
    st.write("")
    st.subheader("Five models. One test dataset.")
    st.write("Compare every classifier on the same rows. Use the metric selector to explore different priorities; F1 is the criterion used here to identify the highest-scoring model.")
    comparison = pd.DataFrame([{"Model": name, **result[0]} for name, result in results.items()])
    best_f1 = comparison.F1.max()
    winners = comparison.loc[np.isclose(comparison.F1, best_f1), "Model"].tolist()
    st.html(f'<div class="comparison-note"><strong>Highest F1 on this CSV: {" / ".join(winners)}</strong><br>F1 {best_f1:.4f} · {len(y):,} test records. This result describes this dataset; it does not establish a universal best model.</div>')
    st.subheader("Metric comparison")
    st.dataframe(comparison.style.format({key: "{:.4f}" for key in METRIC_HELP}, na_rep="N/A")
                 .highlight_max(subset=list(METRIC_HELP), color="#f8e7e9"), hide_index=True, width="stretch")
    st.caption("Higher is better for all six metrics. MCC ranges from −1 to 1; the other metrics range from 0 to 1. Highlighted cells show column maxima.")
    st.download_button("Download all model metrics", comparison.to_csv(index=False), "model_comparison.csv", "text/csv")
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Rank by metric")
        metric = st.selectbox("Comparison metric", list(METRIC_HELP), index=4)
        st.caption(METRIC_HELP[metric])
        ranked = comparison[["Model", metric]].dropna().sort_values(metric, ascending=False).copy()
        ranked.insert(0, "Rank", ranked[metric].rank(method="min", ascending=False).astype(int))
        st.dataframe(ranked.style.format({metric: "{:.4f}"}), hide_index=True, width="stretch")
    with right:
        if ranked.empty:
            st.info("AUC needs both spam and non-spam labels. Upload a CSV with both classes to compare AUC.")
        else:
            bars = alt.Chart(ranked).mark_bar(size=24).encode(
                x=alt.X(f"{metric}:Q", scale=alt.Scale(domain=[-1 if metric == "MCC" else 0, 1])),
                y=alt.Y("Model:N", sort=ranked.Model.tolist(), title=None),
                color=alt.condition(alt.datum[metric] == float(ranked[metric].max()), alt.value(RED), alt.value(NAVY)),
                tooltip=["Model:N", alt.Tooltip(f"{metric}:Q", format=".4f")])
            labels = bars.mark_text(align="left", dx=5, color=NAVY).encode(text=alt.Text(f"{metric}:Q", format=".4f"))
            show_chart((bars + labels).properties(height=280, padding={"right": 42}))
    st.subheader("Score overview")
    long = comparison.melt(id_vars="Model", var_name="Metric", value_name="Score").dropna()
    # MCC is colored on its own valid scale rather than implying it is bounded by zero.
    heat = alt.Chart(long).mark_rect(stroke="white", strokeWidth=3).encode(
        x=alt.X("Metric:N", sort=list(METRIC_HELP), title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Model:N", sort=list(results), title=None),
        color=alt.Color("Score:Q", scale=alt.Scale(domain=[min(0, float(long.Score.min())), 1], range=["#fff4f3", RED]), legend=None),
        tooltip=["Model:N", "Metric:N", alt.Tooltip("Score:Q", format=".4f")])
    text = heat.mark_text(fontSize=13).encode(text=alt.Text("Score:Q", format=".4f"), color=alt.condition(alt.datum.Score > .65, alt.value("white"), alt.value(NAVY)))
    show_chart((heat + text).properties(height=250))
    st.caption("Darker red indicates a higher score. Missing AUC values are omitted. Read MCC on its −1 to 1 scale.")
    if y.nunique() == 2:
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("ROC curves · all models")
            show_chart(curve_chart(results, y, "roc"))
            st.caption("Compare ranking performance across decision thresholds; the dashed diagonal represents chance.")
        with right:
            st.subheader("Precision–recall curves · all models")
            show_chart(curve_chart(results, y, "pr"))
            st.caption("Useful when weighing missed spam against legitimate emails incorrectly flagged. Dashed line: spam prevalence.")
    st.subheader("Where the models make mistakes")
    errors = []
    counts = []
    for name, (_, predictions, _) in results.items():
        tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
        counts.append({"Model": name, "Correct non-spam (TN)": tn, "False spam alarms (FP)": fp,
                       "Missed spam (FN)": fn, "Correct spam (TP)": tp})
        errors += [{"Model": name, "Error": "Legitimate email flagged (FP)", "Count": int(fp)},
                   {"Model": name, "Error": "Spam missed (FN)", "Count": int(fn)}]
    show_chart(alt.Chart(pd.DataFrame(errors)).mark_bar(size=25).encode(
        x=alt.X("Count:Q", title="Number of test emails"),
        y=alt.Y("Model:N", sort=list(results), title=None),
        color=alt.Color("Error:N", scale=alt.Scale(domain=["Legitimate email flagged (FP)", "Spam missed (FN)"], range=[RED, NAVY])),
        tooltip=["Model:N", "Error:N", "Count:Q"]).properties(height=250))
    st.caption("Shorter bars mean fewer errors on this CSV. Red counts legitimate emails flagged as spam; navy counts spam that was missed.")
    with st.expander("Confusion counts for every model"):
        st.dataframe(pd.DataFrame(counts), hide_index=True, width="stretch")
