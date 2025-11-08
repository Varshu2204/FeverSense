# app.py — FINAL VERSION (Stable + Working Delete Function)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import sqlite3
import traceback
from datetime import datetime
import numpy as np

app = Flask(__name__)
CORS(app)

MODEL_DIR = r"C:\Users\varsh\OneDrive\Desktop\feversense\backend\model"
DB_PATH = os.path.join(os.path.dirname(__file__), "feversense.db")

# ---------------- Load Models ----------------
try:
    triage_model = joblib.load(os.path.join(MODEL_DIR, "fever_triage_model.pkl"))
    med_model = joblib.load(os.path.join(MODEL_DIR, "fever_medication_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "fever_scaler.pkl"))
    feature_encoders = joblib.load(os.path.join(MODEL_DIR, "feature_encoders.pkl"))
    severity_encoder = joblib.load(os.path.join(MODEL_DIR, "severity_encoder.pkl"))
    med_encoder = joblib.load(os.path.join(MODEL_DIR, "med_encoder.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    print("✅ Models & artifacts loaded successfully.")
except Exception as e:
    print("❌ Error loading artifacts:", e)
    traceback.print_exc()
    raise

# ---------------- Initialize Database ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            gender TEXT,
            temperature REAL,
            heart_rate REAL,
            blood_pressure TEXT,
            days_fever INTEGER,
            triage TEXT,
            medication TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------- Utility Functions ----------------
RENAME_MAP = {
    "Air_Quality_Index": "AQI",
    "Days_of_Fever": "Days_Fever",
    "Temperature_C": "Temperature",
    "temperature": "Temperature",
    "unit": "unit"
}

BOOL_MAP = {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}

def to_numeric_safe(val):
    """Safely convert values to float or numeric equivalent."""
    if val is None:
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).strip()
    if s == "":
        return np.nan
    low = s.lower()
    if low in BOOL_MAP:
        return float(BOOL_MAP[low])
    try:
        cleaned = s.replace(",", "").split()[0].replace("%", "")
        return float(cleaned)
    except Exception:
        return np.nan

def safe_encode_categorical(col, value, encoder):
    """Encode categorical safely."""
    try:
        return encoder.transform([str(value)])[0]
    except Exception:
        return encoder.transform([encoder.classes_[0]])[0]

# ---------------- Routes ----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Backend Active ✅", "model_loaded": True})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data received"}), 400

        # normalize keys
        data_norm = {}
        for k, v in data.items():
            key = k.strip()
            if key in RENAME_MAP:
                mapped = RENAME_MAP[key]
                data_norm[mapped] = v
            else:
                data_norm[key] = v

        # handle temperature conversion
        if ("unit" in data_norm and ("temperature" in data_norm or "Temperature" in data_norm)):
            tkey = "temperature" if "temperature" in data_norm else "Temperature"
            tval = to_numeric_safe(data_norm.get(tkey))
            unit = str(data_norm.get("unit", "C")).strip().upper()
            if unit in ["F", "°F", "FAHRENHEIT"]:
                temp_c = (tval - 32) * 5.0/9.0
            else:
                temp_c = tval
            data_norm["Temperature"] = round(float(temp_c), 2) if temp_c is not None else np.nan
        elif "Temperature" in data_norm:
            data_norm["Temperature"] = to_numeric_safe(data_norm["Temperature"])

        # build input
        input_dict = {feat: data_norm.get(feat, np.nan) for feat in feature_columns}

        categorical_cols = set(feature_encoders.keys())
        numeric_cols = [c for c in feature_columns if c not in categorical_cols]

        # create dataframe
        row = {}
        for col in numeric_cols:
            row[col] = to_numeric_safe(input_dict.get(col))
        for col in categorical_cols:
            val = input_dict.get(col)
            if pd.isna(val) or val is None or str(val).strip() == "":
                row[col] = feature_encoders[col].classes_[0]
            else:
                row[col] = str(val)

        df_input = pd.DataFrame([row], columns=feature_columns)

        # encode categoricals
        for col in categorical_cols:
            df_input[col] = df_input[col].apply(lambda x: safe_encode_categorical(col, x, feature_encoders[col]))

        # fill missing numerics
        for col in numeric_cols:
            if pd.isna(df_input.at[0, col]):
                df_input.at[0, col] = 0.0

        # scale and predict
        X_scaled = scaler.transform(df_input.values.astype(float))
        triage_pred = triage_model.predict(X_scaled)
        triage_prob = float(triage_model.predict_proba(X_scaled).max())
        triage_label = severity_encoder.inverse_transform(triage_pred.astype(int))[0]

        med_pred = med_model.predict(X_scaled)
        med_label = med_encoder.inverse_transform(med_pred.astype(int))[0]

        # Clinical override logic
        tlower = triage_label.lower()
        if "normal" in tlower:
            med_label = "No medication - rest & hydration advised"
        elif "mild" in tlower:
            med_label = "Paracetamol"
        elif "high" in tlower or "severe" in tlower:
            med_label = "Ibuprofen"

        # Save to DB
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO patient_history
                (name, age, gender, temperature, heart_rate, blood_pressure, days_fever, triage, medication, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data_norm.get("Name", "Unknown"),
                int(to_numeric_safe(data_norm.get("Age")) or 0),
                data_norm.get("Gender", "Unknown"),
                float(df_input["Temperature"].iloc[0]) if "Temperature" in df_input.columns else 0.0,
                float(df_input["Heart_Rate"].iloc[0]) if "Heart_Rate" in df_input.columns else 0.0,
                str(data_norm.get("Blood_Pressure", "Unknown")),
                int(to_numeric_safe(data_norm.get("Days_Fever")) or 0),
                triage_label,
                med_label,
                round(triage_prob, 3),
                datetime.utcnow().isoformat() + "Z"
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print("⚠️ DB insert error:", e)

        importances = {feat: float(imp) for feat, imp in zip(feature_columns, triage_model.feature_importances_)}

        response = {
            "input": list(data_norm.values()),
            "predicted_triage": triage_label,
            "triage_confidence": round(triage_prob, 3),
            "recommended_medication": med_label,
            "feature_importances": importances,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        return jsonify(response), 200

    except Exception as e:
        print("❌ Prediction error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route("/history", methods=["GET"])
def history():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM patient_history ORDER BY id DESC", conn)
        conn.close()
        return jsonify(df.to_dict(orient="records")), 200
    except Exception as e:
        print("⚠️ Fetch history error:", e)
        return jsonify({"error": str(e)}), 400

@app.route("/delete_history", methods=["DELETE"])
def delete_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        # delete and reset autoincrement
        cur.execute("DELETE FROM patient_history")
        cur.execute("DELETE FROM sqlite_sequence WHERE name='patient_history'")
        conn.commit()
        conn.close()

        # verify deletion
        check_conn = sqlite3.connect(DB_PATH)
        count = check_conn.execute("SELECT COUNT(*) FROM patient_history").fetchone()[0]
        check_conn.close()

        if count == 0:
            return jsonify({"message": "✅ All patient records deleted successfully!"}), 200
        else:
            return jsonify({"error": "❌ Records could not be deleted, please retry"}), 500

    except Exception as e:
        print("⚠️ Delete error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

# ---------------- Run Server ----------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
