# trainer.py
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Paths (adjust if needed)
DATA_PATH = r"C:\Users\varsh\OneDrive\Desktop\feversense\Dataset\fever_dataset.csv"
MODEL_DIR = r"C:\Users\varsh\OneDrive\Desktop\feversense\backend\model"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_PATH)
print("Loaded dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Drop unwanted columns
DROP_COLS = ["BMI", "Humidity", "AQI"]
for c in DROP_COLS:
    if c in df.columns:
        df = df.drop(columns=[c])
print("After dropping columns, columns:", df.columns.tolist())

# 3. Targets
TARGET_SEVERITY = "Fever_Severity"
TARGET_MED = "Recommended_Medication"

# 4. Prepare features and target
# Use all remaining columns except targets
feature_cols = [c for c in df.columns if c not in (TARGET_SEVERITY, TARGET_MED)]
X = df[feature_cols].copy()
y_sev = df[TARGET_SEVERITY].astype(str).copy()
y_med = df[TARGET_MED].astype(str).copy()

# 5. Encode categorical features with LabelEncoder per-column
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print("Categorical feature columns to encode:", categorical_cols)

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# 6. Encode targets
sev_encoder = LabelEncoder()
y_sev_encoded = sev_encoder.fit_transform(y_sev)

med_encoder = LabelEncoder()
y_med_encoded = med_encoder.fit_transform(y_med)

# 7. Train-test split (stratify by severity to preserve class distribution)
X_train, X_test, ysev_train, ysev_test, ymed_train, ymed_test = train_test_split(
    X, y_sev_encoded, y_med_encoded, test_size=0.2, random_state=42, stratify=y_sev_encoded
)

# 8. Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Train models
triage_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
med_model = RandomForestClassifier(n_estimators=200, random_state=42)

triage_model.fit(X_train_scaled, ysev_train)
med_model.fit(X_train_scaled, ymed_train)

# 10. Evaluate
ysev_pred = triage_model.predict(X_test_scaled)
ymed_pred = med_model.predict(X_test_scaled)

print("\n--- Triage Model Report ---")
print(classification_report(ysev_test, ysev_pred, target_names=sev_encoder.classes_))
print("Triage Accuracy:", accuracy_score(ysev_test, ysev_pred))

print("\n--- Medication Model Report ---")
print(classification_report(ymed_test, ymed_pred, target_names=med_encoder.classes_))
print("Medication Accuracy:", accuracy_score(ymed_test, ymed_pred))

# 11. Save artifacts
joblib.dump(triage_model, os.path.join(MODEL_DIR, "fever_triage_model.pkl"))
joblib.dump(med_model, os.path.join(MODEL_DIR, "fever_medication_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "fever_scaler.pkl"))
joblib.dump(label_encoders, os.path.join(MODEL_DIR, "feature_encoders.pkl"))
joblib.dump(sev_encoder, os.path.join(MODEL_DIR, "severity_encoder.pkl"))
joblib.dump(med_encoder, os.path.join(MODEL_DIR, "med_encoder.pkl"))
joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_columns.pkl"))

print("\nSaved models and encoders to:", MODEL_DIR)
