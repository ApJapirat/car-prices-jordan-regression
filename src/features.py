# src/features.py
import re
import numpy as np
import pandas as pd


def extract_year_from_model(model: str):
    m = re.search(r"(19|20)\d{2}", str(model))
    return int(m.group()) if m else np.nan


def extract_cc_from_power(power: str):
    m = re.search(r"(\d+)\s*CC", str(power).upper())
    return float(m.group(1)) if m else np.nan


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering ให้เหมือนกันทั้ง train และ app
    คอลัมน์ในไฟล์: Model, Property, Power, Price
    """
    out = df.copy()

    # price -> float
    out["Price"] = (
        out["Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")

    # brand จากคำแรกของ Model
    out["Model"] = out["Model"].astype(str).str.strip()
    out["Brand"] = out["Model"].str.split().str[0].fillna("Unknown")

    # year จาก model string
    out["Year"] = out["Model"].apply(extract_year_from_model)

    # property ทำให้สะอาด
    out["Property"] = out["Property"].astype(str).str.strip().str.lower()

    # power cc / turbo
    out["Power"] = out["Power"].astype(str).str.strip()
    out["PowerCC"] = out["Power"].apply(extract_cc_from_power)
    out["Turbo"] = out["Power"].str.contains("TURBO", case=False, na=False).astype(int)

    return out
