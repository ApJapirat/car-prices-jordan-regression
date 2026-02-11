# app/ui.py
import numpy as np
import pandas as pd
import streamlit as st


def render_sidebar(meta: dict):
    """Sidebar: show model info + retrain button (return True if retrain clicked)."""
    with st.sidebar:
        st.subheader("Model Info")
        st.write("Target:", meta["target"])
        st.write("Features:", ", ".join(meta["features"]))

        st.write("Test Metrics (hold-out test set):")
        st.metric("R²", f"{meta['metrics']['r2']:.3f}")
        st.metric("MAE", f"{meta['metrics']['mae']:.0f}")
        st.metric("RMSE", f"{meta['metrics']['rmse']:.0f}")

        retrain_clicked = st.button("🔁 Retrain model (cloud)")
        return retrain_clicked


def render_inputs(meta: dict, df_feat: pd.DataFrame):
    """
    Main UI: Brand -> Model -> Property แล้ว Year/PowerCC/Turbo "เชื่อม" ตามแถวจริง
    Return: input_df (พร้อม predict) และ subset rows สำหรับ sanity check
    """
    st.subheader("Input Features")

    brands = meta["options"]["brands"]
    brand_to_models = meta["options"]["brand_to_models"]
    brand_model_to_props = meta["options"]["brand_model_to_props"]

    # 1) Brand -> Model -> Property
    brand = st.selectbox("Brand", options=brands)

    models_for_brand = brand_to_models.get(brand, [])
    model = st.selectbox("Model", options=models_for_brand if models_for_brand else ["(no models)"])

    props_for_model = brand_model_to_props.get(f"{brand}|||{model}", [])
    prop = st.selectbox("Property (transmission)", options=props_for_model if props_for_model else ["(no property)"])

    # 2) Filter real rows to link Year/PowerCC/Turbo
    subset = df_feat[
        (df_feat["Brand"] == brand) &
        (df_feat["Model"] == model) &
        (df_feat["Property"] == prop)
    ].copy()

    has_rows = len(subset) > 0

    # Year
    if has_rows:
        year_options = sorted(subset["Year"].astype(int).unique().tolist())
        default_year = int(year_options[-1])  # ล่าสุด
        year = st.selectbox("Year", options=year_options, index=year_options.index(default_year))
    else:
        year = st.number_input("Year", min_value=1990, max_value=2026, value=2020, step=1)

    # PowerCC
    if has_rows:
        power_options = sorted(subset["PowerCC"].round(0).astype(int).unique().tolist())
        default_power = int(np.median(power_options)) if power_options else 1500
        power_cc = st.selectbox(
            "Power (CC)",
            options=power_options if power_options else [1500],
            index=power_options.index(default_power) if default_power in power_options else 0
        )
    else:
        power_cc = st.number_input("Power (CC)", min_value=0.0, max_value=8000.0, value=1500.0, step=50.0)

    # Turbo
    if has_rows:
        turbo_options = sorted(subset["Turbo"].astype(int).unique().tolist())  # [0] or [1] or [0,1]
        default_turbo = 1 if 1 in turbo_options else 0
        turbo = st.selectbox(
            "Turbo",
            options=turbo_options,
            index=turbo_options.index(default_turbo),
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
    else:
        turbo = st.selectbox("Turbo", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    input_df = pd.DataFrame([{
        "Brand": brand,
        "Model": model,
        "Property": prop,
        "Year": int(year),
        "PowerCC": float(power_cc),
        "Turbo": int(turbo),
    }])

    # hint text
    if has_rows:
        st.caption(f"🔗 Linked inputs: found {len(subset)} rows for this Brand+Model+Property, so Year/CC/Turbo are constrained to valid values.")
    else:
        st.caption("⚠️ No exact row match for Brand+Model+Property, so Year/CC/Turbo are free inputs (fallback).")

    return input_df, subset


def render_output(pred: float, input_df: pd.DataFrame, subset: pd.DataFrame):
    st.success(f"✅ Predicted Price: **{pred:,.0f} JOD**")

    with st.expander("Show input data"):
        st.dataframe(input_df)

    with st.expander("Show selected raw rows (for sanity check)"):
        if len(subset):
            show = subset[["Model", "Brand", "Property", "Power", "Year", "PowerCC", "Turbo", "Price"]].head(10)
            st.dataframe(show)
        else:
            st.dataframe(pd.DataFrame({"note": ["No exact row match (still ok for prediction)"]}))
