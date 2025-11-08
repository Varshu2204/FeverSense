import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "model.pkl"
FEATURES_PATH = ROOT / "model" / "features.json"

LABEL_MAP = {0: "Mild", 1: "Moderate", 2: "Severe"}

def load_model():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        print("Model or features file not found. Run trainer.py first.")
        return None, []
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)
    return model, features

def preprocess_input(data: dict, features: list):
    row = {}
    for feat in features:
        if feat in data:
            row[feat] = data[feat]
        else:
            if feat in ["cough", "rash", "headache", "travel_history"]:
                row[feat] = int(data.get(feat, 0))
            elif feat == "days_of_fever":
                row[feat] = int(data.get(feat, 0))
            elif feat == "age":
                row[feat] = int(data.get("age", 30))
            else:
                row[feat] = float(data.get(feat, 36.6))
    return pd.DataFrame([row], columns=features)

def predict_with_model(model, features, input_json):
    if model is None:
        return {"error": "Model not loaded. Run trainer.py to create a model."}

    X = preprocess_input(input_json, features)
    try:
        probs = model.predict_proba(X)[0]
        pred = int(np.argmax(probs))
        proba = float(np.max(probs))
    except Exception:
        pred = int(model.predict(X)[0])
        proba = None

    return {
        "predicted_class": pred,
        "predicted_label": LABEL_MAP.get(pred, "Unknown"),
        "probability": proba,
        "input_features": X.to_dict(orient="records")[0]
    }
