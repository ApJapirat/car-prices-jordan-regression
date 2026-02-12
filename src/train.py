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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.features import build_features


DATA_PATH = os.getenv("DATA_PATH", "data/car_prices_jordan.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/car_price_model.joblib")
META_PATH = os.getenv("META_PATH", "models/metadata.json")


def train_and_save(
    data_path: str = DATA_PATH,
    model_path: str = MODEL_PATH,
    meta_path: str = META_PATH,
    random_state: int = 42,
):
    # Load + feature engineering
    df_raw = pd.read_csv(data_path)
    df = build_features(df_raw)

    use_year = True  

    if use_year:
        features = ["Brand", "Model", "Property", "Year", "PowerCC", "Turbo"]  # 6
    else:
        features = ["Brand", "Model", "Property", "PowerCC", "Turbo"]  # 5

    target = "Price"

    # Clean NA
    needed_cols = features + [target]
    df = df.dropna(subset=needed_cols).copy()

    X = df[features]
    y = df[target].astype(float)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Preprocess
    if use_year:
        num_cols = ["Year", "PowerCC", "Turbo"]
        cat_cols = ["Brand", "Model", "Property"]
    else:
        num_cols = ["PowerCC", "Turbo"]
        cat_cols = ["Brand", "Model", "Property"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    # Ridge = linear regression family
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", Ridge(alpha=1.0)),
        ]
    )

    # Train
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # Save model
    os.makedirs(os.path.dirname(model_path) or "models", exist_ok=True)
    joblib.dump(pipe, model_path)

    # Build dependent dropdown options
    brands = sorted(df["Brand"].dropna().unique().tolist())

    brand_to_models = {}
    brand_model_to_props = {}

    for b in brands:
        models = sorted(df.loc[df["Brand"] == b, "Model"].dropna().unique().tolist())
        brand_to_models[b] = models
        for m in models:
            props = sorted(
                df.loc[(df["Brand"] == b) & (df["Model"] == m), "Property"]
                .dropna()
                .unique()
                .tolist()
            )
            brand_model_to_props[f"{b}|||{m}"] = props

    meta = {
        "features": features,
        "target": target,
        "use_year": bool(use_year),
        "metrics": {"r2": r2, "mae": mae, "rmse": rmse},
        "options": {
            "brands": brands,
            "brand_to_models": brand_to_models,
            "brand_model_to_props": brand_model_to_props,
        },
    }

    os.makedirs(os.path.dirname(meta_path) or "models", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return pipe, meta


def main():
    pipe, meta = train_and_save()
    print("=== Evaluation on Test Set ===")
    print(f"R^2  : {meta['metrics']['r2']:.4f}")
    print(f"MAE  : {meta['metrics']['mae']:.4f}")
    print(f"RMSE : {meta['metrics']['rmse']:.4f}")
    print("\nSaved:")
    print(f"- {MODEL_PATH}")
    print(f"- {META_PATH}")


if __name__ == "__main__":
    main()
