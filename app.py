"""
AI-Powered Loan Approval Prediction Web App
--------------------------------------------
Flask backend that serves the prediction UI and handles inference
requests using the RandomForestClassifier trained in the original
training script (loan_model.pkl + 3 LabelEncoder pickle files).
"""

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# ------------------------------------------------------------------
# Load trained model and label encoders at startup
# (these are the exact files produced by the training script)
# ------------------------------------------------------------------
MODEL_PATH = "loan_model.pkl"
EDU_ENCODER_PATH = "education_encoder.pkl"
EMP_ENCODER_PATH = "self_employed_encoder.pkl"
STATUS_ENCODER_PATH = "status_encoder.pkl"

model = None
le_education = None
le_self_employed = None
le_status = None
model_load_error = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(EDU_ENCODER_PATH, "rb") as f:
        le_education = pickle.load(f)
    with open(EMP_ENCODER_PATH, "rb") as f:
        le_self_employed = pickle.load(f)
    with open(STATUS_ENCODER_PATH, "rb") as f:
        le_status = pickle.load(f)
except FileNotFoundError as e:
    # Keep the app running so the page still renders; prediction
    # requests will report a clear error until the files are present.
    model_load_error = f"Model file not found: {e}"
except Exception as e:
    model_load_error = f"Error loading model: {e}"

# Column order the model was trained on (must match training script)
FIELD_ORDER = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

# Fields that must be validated as non-negative integers
INT_FIELDS = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]


@app.route("/")
def home():
    """Render the home page with the prediction form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle a loan prediction request.
    Accepts a JSON payload from the frontend, validates every field,
    runs the model, and returns the predicted status + confidence.
    """
    if model is None:
        return jsonify({
            "success": False,
            "error": model_load_error or "Model is not loaded on the server."
        }), 500

    try:
        data = request.get_json(force=True)

        # ---- 1. Check all required fields are present ----
        missing = [
            field for field in FIELD_ORDER
            if field not in data or str(data[field]).strip() == ""
        ]
        if missing:
            return jsonify({
                "success": False,
                "error": f"Missing required field(s): {', '.join(missing)}"
            }), 400

        # ---- 2. Validate & cast numeric fields ----
        values = {}
        for field in INT_FIELDS:
            try:
                num = int(float(data[field]))
            except (ValueError, TypeError):
                return jsonify({
                    "success": False,
                    "error": f"'{field}' must be a valid number."
                }), 400
            if num < 0:
                return jsonify({
                    "success": False,
                    "error": f"'{field}' cannot be negative."
                }), 400
            values[field] = num

        # ---- 3. Range check on CIBIL score ----
        if not (300 <= values["cibil_score"] <= 900):
            return jsonify({
                "success": False,
                "error": "CIBIL score must be between 300 and 900."
            }), 400

        # ---- 4. Validate categorical fields against known encoder classes ----
        education_input = str(data["education"]).strip()
        self_employed_input = str(data["self_employed"]).strip()

        if education_input not in le_education.classes_:
            return jsonify({
                "success": False,
                "error": f"Education must be one of: {list(le_education.classes_)}"
            }), 400

        if self_employed_input not in le_self_employed.classes_:
            return jsonify({
                "success": False,
                "error": f"Self Employed must be one of: {list(le_self_employed.classes_)}"
            }), 400

        education_encoded = le_education.transform([education_input])[0]
        self_employed_encoded = le_self_employed.transform([self_employed_input])[0]

        # ---- 5. Build the input dataframe in the exact training column order ----
        input_data = pd.DataFrame({
            "no_of_dependents": [values["no_of_dependents"]],
            "education": [education_encoded],
            "self_employed": [self_employed_encoded],
            "income_annum": [values["income_annum"]],
            "loan_amount": [values["loan_amount"]],
            "loan_term": [values["loan_term"]],
            "cibil_score": [values["cibil_score"]],
            "residential_assets_value": [values["residential_assets_value"]],
            "commercial_assets_value": [values["commercial_assets_value"]],
            "luxury_assets_value": [values["luxury_assets_value"]],
            "bank_asset_value": [values["bank_asset_value"]],
        })

        # ---- 6. Run prediction + confidence ----
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        confidence = round(float(max(probabilities)) * 100, 2)

        status_label = le_status.inverse_transform([prediction])[0]
        approved = str(status_label).strip().lower().startswith("approved")

        return jsonify({
            "success": True,
            "status": status_label,
            "approved": approved,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Unexpected server error: {str(e)}"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)