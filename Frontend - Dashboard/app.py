import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="LSTM Customer Behaviour", layout="centered")

# ---------------------------
# Paths & model loading
# ---------------------------
ART_DIR = "artifacts_lstm"
MODEL_PATHS = [
    os.path.join(ART_DIR, "lstm_model.keras"),
    os.path.join(ART_DIR, "lstm_model.h5"),
    "lstm_model.keras",
    "lstm_model.h5",
]

def load_first_existing_model(paths):
    for p in paths:
        if os.path.exists(p):
            st.info(f"Loading model from: `{p}`")
            return tf.keras.models.load_model(p)
    st.error("❌ No model file found (looked for .keras or .h5). Place it in `artifacts_lstm/` or project root.")
    st.stop()

model = load_first_existing_model(MODEL_PATHS)

# ---------------------------
# Try to load preprocessing artifacts (optional)
# ---------------------------
ohe, scaler, meta = None, None, None
raw_mode_available = False
meta_path = os.path.join(ART_DIR, "meta.json")
ohe_path  = os.path.join(ART_DIR, "ohe.joblib")
sc_path   = os.path.join(ART_DIR, "scaler.joblib")

if os.path.exists(meta_path) and os.path.exists(ohe_path) and os.path.exists(sc_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    ohe     = joblib.load(ohe_path)
    scaler  = joblib.load(sc_path)
    T       = int(meta["T"])
    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]
    raw_mode_available = True
    st.success("Raw features mode available (found meta.json, ohe.joblib, scaler.joblib).")
else:
    # fallback: only model available
    st.warning(
        "Preprocessing artifacts not found. "
        "Switching to **Preprocessed sequence mode** (upload numeric sequence already transformed)."
    )
    # You must specify T (sequence length) and expected feature_dim yourself if meta.json not present.
    # If unknown, set sensible defaults; but ideally keep meta.json.
    # Here we default to T=5 and ask user for feature_dim when uploading.
    T = 5
    num_cols, cat_cols = [], []

# ---------------------------
# Utility functions
# ---------------------------
def _right_pad_last_T(X_mat, T):
    """Right-pad to length T (zeros at the end). X_mat shape: (k, feat_dim)."""
    feat_dim = X_mat.shape[1]
    if X_mat.shape[0] > T:
        X_mat = X_mat[-T:, :]
    if X_mat.shape[0] < T:
        pad = np.zeros((T - X_mat.shape[0], feat_dim), dtype=X_mat.dtype)
        X_mat = np.vstack([X_mat, pad])
    return X_mat

def preprocess_sequence_raw(df_seq, ohe, scaler, num_cols, cat_cols, T):
    """
    df_seq: DataFrame with columns num_cols + cat_cols in chronological order (oldest -> newest)
    returns: array shape (1, T, feature_dim)
    """
    # enforce types
    for c in num_cols:
        df_seq[c] = pd.to_numeric(df_seq[c], errors="coerce").fillna(0)
    for c in cat_cols:
        df_seq[c] = df_seq[c].astype(str).fillna("Unknown")

    X_num = scaler.transform(df_seq[num_cols])
    X_cat = ohe.transform(df_seq[cat_cols])  # handle_unknown='ignore' was used during fitting
    X = np.hstack([X_num, X_cat]).astype("float32")

    X = _right_pad_last_T(X, T)   # (T, feat_dim)
    return X[np.newaxis, ...]     # (1, T, feat_dim)

def predict_from_sequence_matrix(X_seq):
    """
    X_seq: np.array shape (1, T, feature_dim)
    """
    prob = float(model.predict(X_seq, verbose=0).ravel()[0])
    pred = int(prob >= 0.5)
    return prob, pred

# ---------------------------
# UI
# ---------------------------
st.title("🧠 LSTM Customer Behaviour Predictor")

tabs = ["Preprocessed sequence (numeric)", "Raw features (with artifacts)"]
if not raw_mode_available:
    tabs = ["Preprocessed sequence (numeric)"]
tab = st.tabs(tabs)

# ------------------------------------------------------------
# TAB 1: Preprocessed sequence mode (no artifacts required)
# ------------------------------------------------------------
with tab[0]:
    st.subheader("Preprocessed sequence (numeric)")
    st.write(f"Upload a CSV containing **up to {T} rows** (timesteps) of an already **preprocessed numeric feature vector**.")
    st.write("The app will right-pad zeros up to T if fewer rows are provided.")

    uploaded = st.file_uploader("Upload preprocessed sequence CSV", type=["csv"], key="pre_seq")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(df.head())

        # sanity: must be numeric
        if not np.all([np.issubdtype(dt, np.number) for dt in df.dtypes.values]):
            st.error("CSV must contain only numeric columns in this mode. Use Raw features mode if you have artifacts.")
        else:
            if len(df) > T:
                st.info(f"More than T={T} rows uploaded. Using the last {T} rows.")
                df = df.tail(T)

            X_mat = df.to_numpy(dtype="float32")
            X_mat = _right_pad_last_T(X_mat, T)
            X_seq = X_mat[np.newaxis, ...]  # (1, T, feat_dim)

            prob, pred = predict_from_sequence_matrix(X_seq)

            st.subheader("Prediction")
            st.metric("Probability (Class 1)", f"{prob:.3f}")
            st.write("**Predicted class:**", f"{pred} → {'Positive behaviour' if pred==1 else 'Negative behaviour'}")
            thr = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01, key="thr_pre")
            st.write(f"With threshold {thr:.2f} → Predicted class = **{int(prob >= thr)}**")
            st.progress(min(max(prob, 0.0), 1.0))

    # template download
    st.markdown("**Download a blank numeric template**")
    template_cols = st.number_input("Feature dimension (columns) for numeric template", min_value=1, value=117, step=1)
    if st.button("Generate numeric template"):
        numeric_template = pd.DataFrame(np.zeros((T, int(template_cols))), columns=[f"f{i}" for i in range(int(template_cols))])
        st.download_button(
            "Download numeric_template.csv",
            numeric_template.to_csv(index=False).encode(),
            file_name="numeric_template.csv",
            mime="text/csv",
        )

# ----------------------------------------------------------------
# TAB 2: Raw features (requires ohe/scaler/meta artifacts)
# ----------------------------------------------------------------
if raw_mode_available and len(tab) > 1:
    with tab[1]:
        st.subheader("Raw features (with preprocessing artifacts)")
        st.write(f"Upload a CSV with **up to {T} rows** (timesteps) that includes these columns:")

        st.code("NUMERIC:\n" + ", ".join(num_cols) + "\n\nCATEGORICAL:\n" + ", ".join(cat_cols))

        upl_raw = st.file_uploader("Upload raw features CSV", type=["csv"], key="raw_seq")

        if upl_raw is not None:
            dfr = pd.read_csv(upl_raw)
            st.write("Preview:")
            st.dataframe(dfr.head())

            # check required columns
            required = set(num_cols + cat_cols)
            missing = [c for c in required if c not in dfr.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                if len(dfr) > T:
                    st.info(f"More than T={T} rows uploaded. Using the last {T} rows.")
                    dfr = dfr.tail(T)

                X_seq = preprocess_sequence_raw(dfr.copy(), ohe, scaler, num_cols, cat_cols, T)
                prob, pred = predict_from_sequence_matrix(X_seq)

                st.subheader("Prediction")
                st.metric("Probability (Class 1)", f"{prob:.3f}")
                st.write("**Predicted class:**", f"{pred} → {'Positive behaviour' if pred==1 else 'Negative behaviour'}")
                thr = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01, key="thr_raw")
                st.write(f"With threshold {thr:.2f} → Predicted class = **{int(prob >= thr)}**")
                st.progress(min(max(prob, 0.0), 1.0))

        # template for raw features
        if st.button("Download raw features template"):
            template = pd.DataFrame(columns=num_cols + cat_cols)
            st.download_button(
                "Download raw_template.csv",
                template.to_csv(index=False).encode(),
                file_name="raw_template.csv",
                mime="text/csv",
            )

st.caption("Tip: Keep values realistic (matching training distributions). Categorical values must match the training vocabulary.")
