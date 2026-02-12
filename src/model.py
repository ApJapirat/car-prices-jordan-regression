# src/model.py
import json
import joblib

from src.train import main as train_main

MODEL_PATH = "models/car_price_model.joblib"
META_PATH = "models/metadata.json"


def load_model():
    pipe = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return pipe, meta


def load_or_train():
    try:
        return load_model()
    except Exception:
        train_main()
        return load_model()


def retrain():
    train_main()
    return load_model()
