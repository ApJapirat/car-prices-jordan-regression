# src/features.py
import re
import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for car_prices_jordan.csv

    Input columns expected:
    - Model (text: brand + model + year mixed)
    - Property (transmission/trim)
    - Power (text: "2000 CC", "1500 CC Turbo", etc.)
    - Price (number or "12,000")

    Output columns added:
    - Brand, Model, Property, Year, PowerCC, Turbo, Price (clean)
    """
    out = df.copy()

    # Clean Price
    if "Price" in out.columns:
        out["Price"] = (
            out["Price"]
            .astype(str)
            .str.replace(",", "", regex=False)
        )
        out["Price"] = pd.to_numeric(out["Price"], errors="coerce")

    # Model text 
    if "Model" not in out.columns:
        out["Model"] = np.nan

    out["Model"] = out["Model"].astype(str).str.strip()

    # Brand = first token of Model
    out["Brand"] = out["Model"].str.split().str[0].fillna("Unknown").astype(str).str.strip()

    # Extract Year from Model string
    def extract_year(model_text: str):
        m = re.search(r"(19|20)\d{2}", str(model_text))
        return int(m.group()) if m else np.nan

    out["Year"] = out["Model"].apply(extract_year)

    # Property 
    if "Property" not in out.columns:
        out["Property"] = np.nan
    out["Property"] = out["Property"].astype(str).str.strip().str.lower()

    #Power / CC / Turbo 
    if "Power" not in out.columns:
        out["Power"] = np.nan
    out["Power"] = out["Power"].astype(str).str.strip()

    def extract_cc(power_text: str):
        # supports: "2000 CC", "1500 CC Turbo", "2000CC", etc.
        m = re.search(r"(\d+)\s*CC", str(power_text).upper())
        return float(m.group(1)) if m else np.nan

    out["PowerCC"] = out["Power"].apply(extract_cc)
    out["Turbo"] = out["Power"].astype(str).str.contains("TURBO", case=False, na=False).astype(int)

    return out
