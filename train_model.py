"""
train_model.py – Auto Insurance Fraud Detection
================================================
Trains a Random Forest classifier wrapped in a full sklearn Pipeline
(preprocessing + model) and saves to model/fraud_model.pkl.

The Pipeline handles all preprocessing internally so that app.py
only needs to pass a raw dict/DataFrame and call .predict().

Run:
    python train_model.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# ── 1. Load & clean data ────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "insurance_claims.csv")

df = pd.read_csv(DATA_PATH)

# Drop the empty trailing column
df = df.drop(columns=["_c39"], errors="ignore")

# Replace '?' with NaN then fill with mode (small number of affected rows)
df.replace("?", np.nan, inplace=True)
for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encode target
df["fraud_reported"] = df["fraud_reported"].map({"Y": 1, "N": 0})

# ── 2. Feature selection ────────────────────────────────────────────────────
# Drop high-cardinality ID / date / location columns that carry no signal
DROP_COLS = [
    "policy_number",
    "policy_bind_date",
    "incident_date",
    "incident_location",
    "insured_zip",
    "auto_model",          # ~39 categories; auto_make sufficiently represents make
    "fraud_reported",
]

feature_df = df.drop(columns=DROP_COLS)

NUMERIC_FEATURES = [
    "months_as_customer",
    "age",
    "policy_deductable",
    "policy_annual_premium",
    "umbrella_limit",
    "capital-gains",
    "capital-loss",
    "incident_hour_of_the_day",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses",
    "total_claim_amount",
    "injury_claim",
    "property_claim",
    "vehicle_claim",
    "auto_year",
]

CATEGORICAL_FEATURES = [
    "policy_state",
    "policy_csl",
    "insured_sex",
    "insured_education_level",
    "insured_occupation",
    "insured_hobbies",
    "insured_relationship",
    "incident_type",
    "collision_type",
    "incident_severity",
    "authorities_contacted",
    "incident_state",
    "incident_city",
    "property_damage",
    "police_report_available",
    "auto_make",
]

X = feature_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df["fraud_reported"]

# ── 3. Train / test split ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Preprocessing pipeline ───────────────────────────────────────────────
# OrdinalEncoder with handle_unknown so unseen categories don't crash
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            CATEGORICAL_FEATURES,
        ),
    ]
)

# ── 5. Build all model pipelines & compare ──────────────────────────────────
models = {
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(solver="liblinear", max_iter=1000),
    "Naive Bayes":         GaussianNB(),
    "SVM":                 SVC(),
}

print("\n" + "=" * 55)
print("  Model Comparison – 5-Fold Cross Validation (CV) & Accuracy")
print("=" * 55)

best_name  = None
best_score = 0.0
best_pipeline = None

for name, clf in models.items():
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred       = pipe.predict(X_test)
    test_acc     = accuracy_score(y_test, y_pred)
    cv_scores    = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    cv_mean      = cv_scores.mean()
    cv_std       = cv_scores.std()

    print(f"\n{name}")
    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  CV Accuracy   : {cv_mean:.4f} ± {cv_std:.4f}")

    if test_acc > best_score:
        best_score    = test_acc
        best_name     = name
        best_pipeline = pipe

print("\n" + "=" * 55)
print(f"  Best Model: {best_name}  (Test Accuracy = {best_score:.4f})")
print("=" * 55)

# Full classification report for best model
y_best_pred = best_pipeline.predict(X_test)
print(f"\nClassification Report – {best_name}:")
print(classification_report(y_test, y_best_pred, target_names=["Not Fraud", "Fraud"]))

# ── 6. Save the best pipeline ──────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(MODEL_DIR, "fraud_model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.pkl")

joblib.dump(best_pipeline, MODEL_PATH)

# Save metadata so app.py knows feature lists & categorical value maps
cat_value_map = {
    col: sorted(df[col].dropna().unique().tolist())
    for col in CATEGORICAL_FEATURES
}
metadata = {
    "numeric_features":     NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "cat_value_map":        cat_value_map,
    "best_model_name":      best_name,
    "test_accuracy":        round(best_score, 4),
}
joblib.dump(metadata, METADATA_PATH)

print(f"\n✅  Model saved  → {MODEL_PATH}")
print(f"✅  Metadata saved → {METADATA_PATH}")
