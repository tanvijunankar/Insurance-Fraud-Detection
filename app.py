"""
app.py – Auto Insurance Fraud Detection Flask Application
=========================================================
Run:  python app.py
URL:  http://127.0.0.1:5000
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, render_template

# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "model", "fraud_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model", "model_metadata.pkl")

# Load the trained pipeline (preprocessing + classifier) and metadata once
pipeline = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)

NUMERIC_FEATURES     = metadata["numeric_features"]
CATEGORICAL_FEATURES = metadata["categorical_features"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_NAME           = metadata["best_model_name"]
TEST_ACCURACY        = f"{metadata['test_accuracy'] * 100:.1f}%"


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Render the claim-input form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    1. Read all form fields.
    2. Build a single-row DataFrame matching the training feature schema.
    3. Pass to pipeline.predict() (pipeline handles scaling + encoding).
    4. Return result.html with the prediction.
    """
    form = request.form

    # Build input row: numeric fields cast to float, categoricals as str
    row = {}
    for feat in NUMERIC_FEATURES:
        raw = form.get(feat, "0").strip()
        try:
            row[feat] = float(raw)
        except ValueError:
            row[feat] = 0.0

    for feat in CATEGORICAL_FEATURES:
        row[feat] = form.get(feat, "").strip()

    input_df = pd.DataFrame([row], columns=ALL_FEATURES)

    # Predict
    prediction = int(pipeline.predict(input_df)[0])

    return render_template(
        "result.html",
        prediction=prediction,
        model_name=MODEL_NAME,
        accuracy=TEST_ACCURACY,
    )


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
