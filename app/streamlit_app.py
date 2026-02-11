# app/streamlit_app.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import sys
sys.path.append(os.path.abspath("."))

from src.features import build_features


st.set_page_config(page_title="Car Price Predictor (Jordan)", page_icon="🚗")

DATA_PATH = "data/car_prices_jordan.csv"
MODEL_PATH = "models/car_price_model.joblib"
META_PATH = "models/metadata.json"


def train_and_save_in_cloud():
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    df_raw = pd.read_csv(DATA_PATH)
    df = build_features(df_raw)

    features = ["Brand", "Model", "Property", "Year", "PowerCC", "Turbo"]
    target = "Price"

    df = df.dropna(subset=features + [target]).copy()
    X = df[features]
    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_cols = ["Year", "PowerCC", "Turbo"]
    cat_cols = ["Brand", "Model", "Property"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", Ridge(alpha=1.0)),
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    # dependent dropdown mappings
    brands = sorted(df["Brand"].dropna().unique().tolist())

    brand_to_models = {}
    brand_model_to_props = {}
    for b in brands:
        models = sorted(df.loc[df["Brand"] == b, "Model"].dropna().unique().tolist())
        brand_to_models[b] = models
        for m in models:
            props = sorted(
                df.loc[(df["Brand"] == b) & (df["Model"] == m), "Property"]
                .dropna().unique().tolist()
            )
            brand_model_to_props[f"{b}|||{m}"] = props

    meta = {
        "features": features,
        "target": target,
        "metrics": {"r2": float(r2), "mae": float(mae), "rmse": float(rmse)},
        "options": {
            "brands": brands,
            "brand_to_models": brand_to_models,
            "brand_model_to_props": brand_model_to_props,
        }
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return pipe, meta


@st.cache_resource
def load_or_train():
    try:
        pipe = joblib.load(MODEL_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return pipe, meta
    except Exception:
        return train_and_save_in_cloud()


# ===== UI =====
st.title("🚗 Car Price Predictor (Jordan)")
st.caption("Multiple Regression (Linear) with 6 features: Brand, Model, Property, Year, PowerCC, Turbo")

pipe, meta = load_or_train()

with st.sidebar:
    st.subheader("Model Info")
    st.write("Target:", meta["target"])
    st.write("Features:", ", ".join(meta["features"]))
    st.write("Test Metrics (hold-out test set):")
    st.metric("R²", f"{meta['metrics']['r2']:.3f}")
    st.metric("MAE", f"{meta['metrics']['mae']:.0f}")
    st.metric("RMSE", f"{meta['metrics']['rmse']:.0f}")

    if st.button("🔁 Retrain model (cloud)"):
        load_or_train.clear()
        pipe, meta = train_and_save_in_cloud()
        st.success("Retrained! (ลองกด Rerun / Refresh หน้าเว็บอีกที)")


# โหลด data สำหรับทำ “ตัวเลือกตามจริง”
df_raw = pd.read_csv(DATA_PATH)
df_feat = build_features(df_raw).dropna(subset=["Brand", "Model", "Property", "Year", "PowerCC", "Turbo", "Price"]).copy()

st.subheader("Input Features")

brands = meta["options"]["brands"]
brand_to_models = meta["options"]["brand_to_models"]
brand_model_to_props = meta["options"]["brand_model_to_props"]

# --- Step 1: Brand -> Model -> Property ---
brand = st.selectbox("Brand", options=brands)

models_for_brand = brand_to_models.get(brand, [])
model = st.selectbox("Model", options=models_for_brand if models_for_brand else ["(no models)"])

props_for_model = brand_model_to_props.get(f"{brand}|||{model}", [])
prop = st.selectbox("Property (transmission)", options=props_for_model if props_for_model else ["(no property)"])

# --- Step 2: derive valid Year/PowerCC/Turbo from real rows ---
subset = df_feat[
    (df_feat["Brand"] == brand) &
    (df_feat["Model"] == model) &
    (df_feat["Property"] == prop)
].copy()

has_rows = len(subset) > 0

# Year options
if has_rows:
    year_options = sorted(subset["Year"].astype(int).unique().tolist())
    default_year = int(year_options[-1])  # ใช้ปีล่าสุดเป็น default
    year = st.selectbox("Year", options=year_options, index=year_options.index(default_year))
else:
    year = st.number_input("Year", min_value=1990, max_value=2026, value=2020, step=1)

# PowerCC options
if has_rows:
    # บางแถว PowerCC อาจเป็น float แต่จริงๆคือ CC integer
    power_options = sorted(subset["PowerCC"].round(0).astype(int).unique().tolist())
    default_power = int(np.median(power_options)) if power_options else 1500
    power_cc = st.selectbox("Power (CC)", options=power_options, index=power_options.index(default_power) if default_power in power_options else 0)
else:
    power_cc = st.number_input("Power (CC)", min_value=0.0, max_value=8000.0, value=1500.0, step=50.0)

# Turbo options
if has_rows:
    turbo_options = sorted(subset["Turbo"].astype(int).unique().tolist())  # [0] or [1] or [0,1]
    default_turbo = 1 if 1 in turbo_options else 0
    turbo = st.selectbox("Turbo", options=turbo_options, format_func=lambda x: "Yes" if x == 1 else "No",
                         index=turbo_options.index(default_turbo))
else:
    turbo = st.selectbox("Turbo", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# --- Predict ---
if st.button("Predict Price", type="primary"):
    input_df = pd.DataFrame([{
        "Brand": brand,
        "Model": model,
        "Property": prop,
        "Year": int(year),
        "PowerCC": float(power_cc),
        "Turbo": int(turbo),
    }])

    pred = float(pipe.predict(input_df)[0])
    st.success(f"✅ Predicted Price: **{pred:,.0f} JOD**")

    with st.expander("Show input data"):
        st.dataframe(input_df)

    with st.expander("Show selected raw rows (for sanity check)"):
        show = df_feat[
            (df_feat["Brand"] == brand) &
            (df_feat["Model"] == model) &
            (df_feat["Property"] == prop)
        ][["Model", "Brand", "Property", "Power", "Year", "PowerCC", "Turbo", "Price"]].head(10)
        st.dataframe(show if len(show) else pd.DataFrame({"note": ["No exact row match (still ok for prediction)"]}))

# UI hint
if has_rows:
    st.caption(f"🔗 Linked inputs: found {len(subset)} rows for this Brand+Model+Property, so Year/CC/Turbo are constrained to valid values.")
else:
    st.caption("⚠️ No exact row match for Brand+Model+Property, so Year/CC/Turbo are free inputs (fallback).")
