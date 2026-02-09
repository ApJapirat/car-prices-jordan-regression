# src/train.py
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Config
DATA_PATH = os.getenv("DATA_PATH", "data/cats_dataset.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/cat_weight_model.joblib")
META_PATH = os.getenv("META_PATH", "models/metadata.json")

FEATURES = ["Age", "Breed", "Color", "Gender"]
TARGET = "Weight"


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # rename columns เผื่อชื่อในไฟล์เป็นแบบนี้
    df = df.rename(columns={
        "Age (Years)": "Age",
        "Weight (kg)": "Weight",
    })

    # clean categorical
    for c in ["Breed", "Color", "Gender"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


def train(df: pd.DataFrame):
    # กันกรณีคอลัมน์หาย
    missing = [c for c in (FEATURES + [TARGET]) if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df = df.dropna(subset=FEATURES + [TARGET]).copy()

    X = df[FEATURES]
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
        ("model", LinearRegression()),
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    metrics = {"r2": r2, "mae": mae, "rmse": rmse}

    meta = {
        "features": FEATURES,
        "target": TARGET,
        "breed_options": sorted(df["Breed"].unique().tolist()),
        "color_options": sorted(df["Color"].unique().tolist()),
        "gender_options": sorted(df["Gender"].unique().tolist()),
        "metrics": metrics,
    }

    return pipe, meta


def save(pipe, meta, model_path: str, meta_path: str):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)

    joblib.dump(pipe, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    pipe, meta = train(df)

    print("=== Evaluation on Test Set ===")
    print(f"R^2  : {meta['metrics']['r2']:.4f}")
    print(f"MAE  : {meta['metrics']['mae']:.4f}")
    print(f"RMSE : {meta['metrics']['rmse']:.4f}")

    save(pipe, meta, MODEL_PATH, META_PATH)

    print("\nSaved:")
    print(f"- {MODEL_PATH}")
    print(f"- {META_PATH}")
