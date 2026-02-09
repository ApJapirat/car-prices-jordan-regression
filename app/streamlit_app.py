# app/streamlit_app.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="Cat Weight Predictor", page_icon="🐾")

# Paths + expected schema--
DATA_PATH = os.getenv("DATA_PATH", "data/cats_dataset.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/cat_weight_model.joblib")
META_PATH = os.getenv("META_PATH", "models/metadata.json")

EXPECTED_FEATURES = ["Age", "Breed", "Color", "Gender"]
EXPECTED_TARGET = "Weight"


def _load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"Age (Years)": "Age", "Weight (kg)": "Weight"})
    for c in ["Breed", "Color", "Gender"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def train_and_save():
    """Train Linear Regression in this runtime and save artifacts."""
    df = _load_df()

    missing = [c for c in (EXPECTED_FEATURES + [EXPECTED_TARGET]) if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df = df.dropna(subset=EXPECTED_FEATURES + [EXPECTED_TARGET]).copy()
    X = df[EXPECTED_FEATURES]
    y = df[EXPECTED_TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=X["Breed"]  #สัดส่วน Breed ใน train/test ใกล้กัน
    )

    num_cols = ["Age"]
    cat_cols = ["Breed", "Color", "Gender"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("model", LinearRegression())
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    meta = {
        "features": EXPECTED_FEATURES,
        "target": EXPECTED_TARGET,
        "breed_options": sorted(df["Breed"].unique().tolist()),
        "color_options": sorted(df["Color"].unique().tolist()),
        "gender_options": sorted(df["Gender"].unique().tolist()),
        "metrics": {"r2": r2, "mae": mae, "rmse": rmse},
        "trained_on": "streamlit_cloud_runtime",
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return pipe, meta


@st.cache_resource
def load_or_train():
    """Load artifacts; if missing/incompatible/schema mismatch -> retrain."""
    try:
        pipe = joblib.load(MODEL_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # ✅ schema mismatch (เช่นเคยแก้ features) -> retrain
        if meta.get("features") != EXPECTED_FEATURES or meta.get("target") != EXPECTED_TARGET:
            raise ValueError("Feature/target mismatch -> retrain")

        return pipe, meta

    except Exception:
        return train_and_save()


# UI
st.title("🐾 Cat Weight Predictor (Regression)")
st.caption("Linear Regression + One-Hot Encoding (Supervised Learning)")

pipe, meta = load_or_train()

with st.sidebar:
    st.subheader("Model Info")
    st.write(f"Target: **{meta['target']}**")
    st.write("Features: " + ", ".join(meta["features"]))

    m = meta.get("metrics", {})
    st.markdown("**Test Metrics (hold-out test set):**")
    st.write(f"- R²  : {m.get('r2', 0):.3f}")
    st.write(f"- MAE : {m.get('mae', 0):.3f}")
    st.write(f"- RMSE: {m.get('rmse', 0):.3f}")

    if st.button("🔁 Retrain model (cloud)", help="Use when you update data or want fresh metrics."):
        load_or_train.clear()
        pipe, meta = train_and_save()
        st.success("Retrained successfully! Refresh/rerun if needed.")
        st.stop()

# keep prediction result after rerun
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "last_input" not in st.session_state:
    st.session_state.last_input = None

st.subheader("Input Features")

with st.form("predict_form"):
    age = st.number_input("Age (Years)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
    breed = st.selectbox("Breed", options=meta["breed_options"])
    color = st.selectbox("Color", options=meta["color_options"])
    gender = st.selectbox("Gender", options=meta["gender_options"])
    submit = st.form_submit_button("Predict Weight (kg)")

if submit:
    input_df = pd.DataFrame([{
        "Age": float(age),
        "Breed": str(breed),
        "Color": str(color),
        "Gender": str(gender),
    }])

    pred = float(pipe.predict(input_df)[0])

    st.session_state.last_pred = pred
    st.session_state.last_input = input_df

if st.session_state.last_pred is not None:
    st.success(f"✅ Predicted Weight: **{st.session_state.last_pred:.2f} kg**")
    with st.expander("Show input data"):
        st.dataframe(st.session_state.last_input, use_container_width=True)
