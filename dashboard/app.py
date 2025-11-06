
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from src.config import DATA_DIR, MODELS_DIR, GROUP_COLS, NUM_COLS, CAT_COLS, TARGET_COL
from src.data import load_data, split_xy
from src.model import build_pipeline, fit_and_eval, save_pipeline, load_pipeline
from src.fairness import compute_fairness, binarize
from src.explanations import build_contrastive_explainer

st.set_page_config(page_title="Fair Triage Support", layout="wide")
st.title("Fair Triage Support — Research Prototype")

# Sidebar
with st.sidebar:
    st.header("Data & Model")
    data_path = st.text_input("CSV path", value=str(DATA_DIR / "triage_synth.csv"))
    model_kind = st.selectbox("Model", ["logreg", "gb"], index=0)
    group_col = st.selectbox("Group for fairness", GROUP_COLS, index=2)
    if st.button("Train / Retrain"):
        df = load_data(data_path)
        X, y = split_xy(df, TARGET_COL)
        pipe = build_pipeline(NUM_COLS, CAT_COLS, model_kind=model_kind)
        m = int(0.8 * len(df))
        X_train, y_train = X.iloc[:m], y[:m]
        X_val, y_val = X.iloc[m:], y[m:]
        metrics = fit_and_eval(pipe, X_train, y_train, X_val, y_val)
        save_pipeline(pipe, MODELS_DIR / f"pipeline_{model_kind}.joblib")
        st.success(f"Trained {model_kind}. AUC={metrics['auc']:.3f} ACC={metrics['acc']:.3f} Brier={metrics['brier']:.3f}")

# Load data & model
default_path = DATA_DIR / "triage_synth.csv"
if not default_path.exists():
    st.warning("Generate or provide a CSV dataset first (scripts/generate_synthetic_data.py).")
    st.stop()

df = load_data(str(default_path))
X, y = split_xy(df, TARGET_COL)
pipeline_path = MODELS_DIR / "pipeline_logreg.joblib"
if not pipeline_path.exists():
    st.info("No trained model found—use the sidebar to train one (logreg).")
    st.stop()

pipe = load_pipeline(pipeline_path)

proba = pipe.predict_proba(X)[:, 1]
pred = binarize(proba, threshold=0.5)

st.subheader("1) Patient-level scoring & contrastive explanation")
prep = pipe.named_steps["prep"]
X_embed = prep.transform(X)
feature_names = list(prep.get_feature_names_out())
explain_fn = build_contrastive_explainer(X_embed, pred)

idx = st.number_input("Row index", min_value=0, max_value=len(df)-1, value=0)
if st.button("Explain this case"):
    res = explain_fn(int(idx), feature_names, X_embed, proba)
    st.write(res["text"])
    st.json(res["deltas"])

st.subheader("2) Subgroup fairness widget")
g = df[group_col].astype(str)
fair = compute_fairness(y_true=y, y_pred=pred, sensitive_series=g)
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Overall metrics**")
    st.json(fair["overall"])
with c2:
    st.markdown("**By-group metrics**")
    st.json(fair["by_group"])
st.markdown(
    f"**Demographic parity difference:** {fair['demographic_parity_difference']:.3f}  \n"
    f"**Equalized odds difference:** {fair['equalized_odds_difference']:.3f}"
)

st.subheader("3) Bias-check before finalising")
ack = st.checkbox("I reviewed similar cases and subgroup parity.")
if ack:
    st.success("Bias-check acknowledged (research logging only).")

st.caption("Research prototype. Not for clinical use.")
