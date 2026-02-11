# src/train.py
import os
import re
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge  # ยังเป็น linear regression family
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Load dataset
DATA_PATH = os.getenv("DATA_PATH", "data/car_prices_jordan.csv")
df = pd.read_csv(DATA_PATH)

# Clean / Feature engineering 
def extract_year(model: str):
    m = re.search(r"(19|20)\d{2}", str(model))
    return int(m.group()) if m else np.nan

def extract_cc(power: str):
    m = re.search(r"(\d+)\s*CC", str(power).upper())
    return float(m.group(1)) if m else np.nan

df["Price"] = df["Price"].astype(str).str.replace(",", "", regex=False).astype(float)
df["Brand"] = df["Model"].astype(str).str.strip().str.split().str[0]
df["Year"] = df["Model"].apply(extract_year)
df["Property"] = df["Property"].astype(str).str.strip().str.lower()
df["PowerCC"] = df["Power"].apply(extract_cc)
df["Turbo"] = df["Power"].astype(str).str.contains("TURBO", case=False, na=False).astype(int)

features = ["Brand", "Property", "Year", "PowerCC", "Turbo"]
target = "Price"

df = df.dropna(subset=features + [target]).copy()

X = df[features]
y = df[target].astype(float)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess + Model
num_cols = ["Year", "PowerCC", "Turbo"]
cat_cols = ["Brand", "Property"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

# Ridge = linear regression
model = Ridge(alpha=1.0)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model),
])

# Train
pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

print("=== Evaluation on Test Set ===")
print(f"R^2  : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

# Save model + metadata
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/car_price_model.joblib"
META_PATH = "models/metadata.json"

joblib.dump(pipe, MODEL_PATH)

metadata = {
    "features": features,
    "target": target,
    "brand_options": sorted(df["Brand"].unique().tolist()),
    "property_options": sorted(df["Property"].unique().tolist()),
    "metrics": {"r2": float(r2), "mae": float(mae), "rmse": float(rmse)},
}

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("\nSaved:")
print(f"- {MODEL_PATH}")
print(f"- {META_PATH}")
