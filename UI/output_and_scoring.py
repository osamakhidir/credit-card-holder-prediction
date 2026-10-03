"""
output_and_scoring.py  ->  save in the UI/ folder next to app.py

Provides two Streamlit render functions:
    render_output_browser()  - drop-down browser for the output sub-folders
    render_scoring()         - upload a file and score it with a saved model

Run the app from the project root:  streamlit run UI/app.py
"""

import base64
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# PATHS  (edit here if your layout differs)
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # ...\UI
PROJECT_DIR = os.path.dirname(BASE_DIR)                    # project root
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")

# Label shown in the drop-down  ->  folder name inside output/
OUTPUT_FOLDERS = {
    "Feature Importance": "feature_importance",
    "Lift Table": "lift_table",
    "Plots and Data Exploration": "plots and data exploration",
    "Correlation Output": "correlation output",
}

TARGET = "credit_card_holder_flag"
LEAKAGE_COLS = ["has_credit_card", TARGET]   # always excluded from features


# --------------------------------------------------------------------------
# 1) OUTPUT FOLDER BROWSER
# --------------------------------------------------------------------------
def _list_files(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and not f.startswith("~$")
    )


def _download_button(path, key):
    with open(path, "rb") as fh:
        st.download_button(
            "Download this file",
            data=fh.read(),
            file_name=os.path.basename(path),
            key=key,
        )


def _show_file(path):
    """Render one file according to its extension."""
    ext = os.path.splitext(path)[1].lower()
    name = os.path.basename(path)

    if ext == ".csv":
        st.dataframe(pd.read_csv(path))

    elif ext in (".xlsx", ".xlsm", ".xls"):
        sheets = pd.ExcelFile(path).sheet_names
        sheet = (
            st.selectbox("Sheet", sheets, key=f"sheet_{path}")
            if len(sheets) > 1 else sheets[0]
        )
        st.dataframe(pd.read_excel(path, sheet_name=sheet))

    elif ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        st.image(path, caption=name)

    elif ext == ".pdf":
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("utf-8")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            f'width="100%" height="800" type="application/pdf"></iframe>',
            unsafe_allow_html=True,
        )

    elif ext in (".txt", ".md", ".log"):
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            st.text(fh.read())

    elif ext == ".html":
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            st.components.v1.html(fh.read(), height=800, scrolling=True)

    else:
        st.info(f"Preview not supported for '{ext}' files. Use the download button.")


def render_output_browser():
    st.subheader("Model Outputs")

    folder_label = st.selectbox(
        "Select output folder",
        list(OUTPUT_FOLDERS.keys()),
        key="out_folder_select",
    )
    folder_path = os.path.join(OUTPUT_DIR, OUTPUT_FOLDERS[folder_label])

    if not os.path.isdir(folder_path):
        st.warning(f"Folder not found: {folder_path}")
        return

    files = _list_files(folder_path)
    if not files:
        st.info("This folder is empty.")
        return

    file_name = st.selectbox(
        f"Select a file ({len(files)} available)",
        files,
        key=f"out_file_select_{folder_label}",
    )
    file_path = os.path.join(folder_path, file_name)

    st.caption(file_path)
    _show_file(file_path)
    _download_button(file_path, key=f"dl_{folder_label}_{file_name}")


# --------------------------------------------------------------------------
# 2) UPLOAD + SCORE
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_pickle(path):
    return joblib.load(path)


def _unpack_model(obj):
    """Return (model, scaler, features) from a raw model or a dict bundle."""
    if isinstance(obj, dict):
        model = obj.get("model") or obj.get("clf") or obj.get("estimator")
        return model, obj.get("scaler"), obj.get("features") or obj.get("feature_names")
    return obj, None, None


def _expected_features(model, bundled_features=None):
    """Return (feature_list, needs_constant)."""
    if bundled_features:
        return list(bundled_features), False
    if hasattr(model, "feature_names_in_"):                      # sklearn
        return list(model.feature_names_in_), False
    inner = getattr(model, "model", None)                        # statsmodels
    if inner is not None and hasattr(inner, "exog_names"):
        names = list(inner.exog_names)
        return [n for n in names if n != "const"], "const" in names
    return None, False


def _predict_proba(model, X, needs_const):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if needs_const:
        X = X.copy()
        X.insert(0, "const", 1.0)
    return np.asarray(model.predict(X), dtype=float)


def _read_upload(uploaded):
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)


def render_scoring():
    st.subheader("Score a New File")

    model_files = sorted(
        f for f in os.listdir(MODEL_DIR) if f.lower().endswith(".pkl")
    ) if os.path.isdir(MODEL_DIR) else []

    if not model_files:
        st.warning(f"No .pkl model files found in: {MODEL_DIR}")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        model_name = st.selectbox("Model", model_files, key="score_model_select")
    with col2:
        threshold = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01,
                              key="score_threshold")

    uploaded = st.file_uploader(
        "Upload a CSV or Excel file to score",
        type=["csv", "xlsx", "xls"],
        key="score_uploader",
    )
    if uploaded is None:
        st.caption("The file must have the same columns as the training data "
                   "(same encoding/scaling as `train_set.csv`).")
        return

    try:
        df = _read_upload(uploaded)
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        return

    st.write(f"Loaded **{len(df):,}** rows and **{df.shape[1]}** columns.")
    st.dataframe(df.head(20))

    # ---- load model -------------------------------------------------------
    raw = _load_pickle(os.path.join(MODEL_DIR, model_name))
    model, scaler, bundled_features = _unpack_model(raw)
    features, needs_const = _expected_features(model, bundled_features)

    if features is None:
        st.error("Could not determine the model's feature names from the .pkl file.")
        return

    # Optional scaler (only if one was saved with the model)
    apply_scaler = False
    if scaler is not None:
        apply_scaler = st.checkbox(
            "Apply saved scaler (tick only if the uploaded file is NOT already scaled)",
            value=False, key="score_apply_scaler",
        )

    if not st.button("Score file", type="primary", key="score_button"):
        return

    # ---- build feature matrix --------------------------------------------
    X = df.drop(columns=LEAKAGE_COLS, errors="ignore").copy()

    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols)

    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    missing = [c for c in features if c not in X.columns]
    if missing:
        st.warning(
            f"{len(missing)} expected column(s) missing from the upload; filled with 0: "
            f"{', '.join(missing[:15])}{' ...' if len(missing) > 15 else ''}"
        )
    X = X.reindex(columns=features, fill_value=0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    if apply_scaler:
        cols = list(getattr(scaler, "feature_names_in_", features))
        cols = [c for c in cols if c in X.columns]
        X[cols] = scaler.transform(X[cols])

    # ---- score ------------------------------------------------------------
    try:
        proba = _predict_proba(model, X, needs_const)
    except Exception as exc:
        st.error(f"Scoring failed: {exc}")
        return

    scored = df.copy()
    scored.insert(0, "predicted_flag", (proba >= threshold).astype(int))
    scored.insert(0, "score_probability", proba)

    st.success("Scoring complete.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Rows scored", f"{len(scored):,}")
    m2.metric("Predicted holders", f"{int(scored['predicted_flag'].sum()):,}")
    m3.metric("Predicted holder rate", f"{scored['predicted_flag'].mean():.1%}")

    # If the upload already has the true target, show quick performance
    if TARGET in df.columns and df[TARGET].nunique() == 2:
        from sklearn.metrics import roc_auc_score, confusion_matrix
        y = df[TARGET].astype(int)
        st.write(f"**ROC AUC vs. `{TARGET}`:** {roc_auc_score(y, proba):.4f}")
        cm = confusion_matrix(y, scored["predicted_flag"])
        st.dataframe(
            pd.DataFrame(cm, index=["Actual 0", "Actual 1"],
                         columns=["Pred 0", "Pred 1"])
        )

    st.dataframe(scored)

    st.download_button(
        "Download scored file (CSV)",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name=f"scored_{os.path.splitext(uploaded.name)[0]}.csv",
        mime="text/csv",
        key="score_download",
    )


# --------------------------------------------------------------------------
# Stand-alone test:  streamlit run UI/output_and_scoring.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Outputs and Scoring", layout="wide")
    t1, t2 = st.tabs(["Model Outputs", "Score New File"])
    with t1:
        render_output_browser()
    with t2:
        render_scoring()
