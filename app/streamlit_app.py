# app/streamlit_app.py
import streamlit as st
from src.core import load_or_train, train_model, load_dataset, save_artifacts, predict_weight

st.set_page_config(page_title="Cat Weight Predictor", page_icon="🐾")

st.title("🐾 Cat Weight Predictor (Regression)")
st.caption("Predict cat weight (kg) using Age, Breed, Color, Gender (Linear Regression)")

@st.cache_resource
def get_pipe_and_meta():
    return load_or_train()

pipe, meta = get_pipe_and_meta()

with st.sidebar:
    st.subheader("Model Info")
    st.write("Target:", meta["target"])
    st.write("Features:", ", ".join(meta["features"]))
    st.write("Test Metrics (hold-out test set):")
    st.write(f"R² : {meta['metrics']['r2']:.3f}")
    st.write(f"MAE: {meta['metrics']['mae']:.3f}")
    st.write(f"RMSE: {meta['metrics']['rmse']:.3f}")

    if st.button("🔁 Retrain model (cloud)"):
        df = load_dataset()
        pipe2, meta2 = train_model(df)
        save_artifacts(pipe2, meta2)
        get_pipe_and_meta.clear()
        st.success("Retrained! Please rerun the app.")
        st.stop()

# keep result after rerun
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
    submitted = st.form_submit_button("Predict Weight (kg)")

if submitted:
    pred, input_df = predict_weight(pipe, age, breed, color, gender)
    st.session_state.last_pred = pred
    st.session_state.last_input = input_df

if st.session_state.last_pred is not None:
    st.success(f"✅ Predicted Weight: **{st.session_state.last_pred:.2f} kg**")

    with st.expander("Show input data"):
        st.dataframe(st.session_state.last_input, use_container_width=True)
