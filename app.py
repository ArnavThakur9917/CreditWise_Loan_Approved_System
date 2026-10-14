from pathlib import Path

import joblib
import pandas as pd

from flask import Flask, render_template, request


# =========================================================
# 1. APPLICATION CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "loan_approval_pipeline.joblib"

app = Flask(__name__)


# =========================================================
# 2. NUMERICAL INPUT FIELDS
# =========================================================
NUMERIC_FIELDS = [
    ("Applicant_Income", "Applicant monthly income", 200, 1998899, 10000, 1),
    ("Coapplicant_Income", "Co-applicant monthly income", 1, 99963, 2000, 1),
    ("Age", "Applicant age", 21, 59, 30, 1),
    ("Dependents", "Number of dependents", 0, 3, 0, 1),
    ("Credit_Score", "Credit score", 450, 799, 600, 1),
    ("Existing_Loans", "Existing loans", 0, 89, 0, 1),
    ("DTI_Ratio", "Debt-to-income ratio (0-1)", 0.1, 0.9, 0.3, 0.01),
    ("Savings", "Savings", 65, 199965, 5000, 1),
    ("Collateral_Value", "Collateral value", 36, 49954, 10000, 1),
    ("Loan_Amount", "Requested loan amount", 101, 399953, 15000, 1),
    ("Loan_Term", "Loan term (months)", 12, 84, 36, 1),
]


# =========================================================
# 3. CATEGORICAL INPUT FIELDS
# =========================================================

CATEGORICAL_FIELDS = [
    (
        "Employment_Status",
        "Employment status",
        ["Salaried", "Self-employed", "Contract", "Unemployed"],
    ),
    (
        "Marital_Status",
        "Marital status",
        ["Single", "Married"],
    ),
    (
        "Loan_Purpose",
        "Loan purpose",
        ["Personal", "Car", "Business", "Home", "Education"],
    ),
    (
        "Property_Area",
        "Property area",
        ["Urban", "Semiurban", "Rural"],
    ),
    (
        "Education_Level",
        "Education level",
        ["Graduate", "Not Graduate"],
    ),
    (
        "Gender",
        "Gender",
        ["Female", "Male"],
    ),
    (
        "Employer_Category",
        "Employer category",
        ["Private", "Government", "Unemployed", "MNC", "Business"],
    ),
]


# =========================================================
# 4. LOAD TRAINED MACHINE LEARNING MODEL
# =========================================================

def get_model_bundle():
    """
    Load the trained model pipeline.

    Returns:
        dict: Saved model bundle, if available.
        None: If the model file does not exist.
    """

    if not MODEL_PATH.exists():
        app.logger.error("Trained model file not found: %s", MODEL_PATH)
        return None

    try:
        bundle = joblib.load(MODEL_PATH)

        if not isinstance(bundle, dict):
            raise ValueError(
                "Unexpected model format. "
                "Check the output format in train_model.py."
            )

        if "pipeline" not in bundle:
            raise ValueError(
                "The saved model bundle does not contain 'pipeline'."
            )

        if "features" not in bundle:
            raise ValueError(
                "The saved model bundle does not contain 'features'."
            )

        return bundle

    except Exception:
        app.logger.exception("Failed to load the trained model.")
        raise


# =========================================================
# 5. HOME PAGE AND LOAN PREDICTION
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None
    values = {}

    accuracy = None

    try:
        bundle = get_model_bundle()
    except Exception:
        bundle = None
        error = (
            "The trained model could not be loaded. "
            "Check the model file and terminal error."
        )

    model_ready = bundle is not None

    if bundle is not None:
        accuracy = round(
            float(bundle.get("test_accuracy", 0)) * 100,
            1,
        )

    # -----------------------------------------------------
    # PROCESS FORM SUBMISSION
    # -----------------------------------------------------

    if request.method == "POST":

        values = request.form.to_dict()

        if bundle is None:

            error = (
                "The trained model is unavailable. "
                "Check models/loan_approval_pipeline.joblib "
                "and run the training script if necessary."
            )

        else:

            try:
                row = {}

                # -----------------------------------------
                # VALIDATE NUMERICAL INPUTS
                # -----------------------------------------

                for (
                    name,
                    label,
                    minimum,
                    maximum,
                    default,
                    step,
                ) in NUMERIC_FIELDS:

                    raw_value = request.form.get(
                        name,
                        str(default),
                    ).strip()

                    if not raw_value:
                        raise ValueError(
                            f"{label} is required."
                        )

                    value = float(raw_value)

                    if not pd.notna(value):
                        raise ValueError(
                            f"{label} must be a valid number."
                        )

                    if value < minimum or value > maximum:
                        raise ValueError(
                            f"{label} must be between "
                            f"{minimum} and {maximum}."
                        )

                    row[name] = value

                # -----------------------------------------
                # VALIDATE CATEGORICAL INPUTS
                # -----------------------------------------

                for name, label, options in CATEGORICAL_FIELDS:

                    value = request.form.get(
                        name,
                        options[0],
                    )

                    if value not in options:
                        raise ValueError(
                            f"Invalid value for {label}."
                        )

                    row[name] = value

                # -----------------------------------------
                # PREPARE MODEL INPUT
                # -----------------------------------------

                features = bundle["features"]

                missing_features = [
                    feature
                    for feature in features
                    if feature not in row
                ]

                if missing_features:
                    raise ValueError(
                        "The form does not provide all the "
                        "features required by the trained model: "
                        + ", ".join(missing_features)
                    )

                input_df = pd.DataFrame(
                    [row]
                )[features]

                pipeline = bundle["pipeline"]

                # -----------------------------------------
                # MAKE PREDICTION
                # -----------------------------------------

                prediction = int(
                    pipeline.predict(input_df)[0]
                )

                # -----------------------------------------
                # CALCULATE APPROVAL PROBABILITY
                # -----------------------------------------

                approved_probability = None

                if hasattr(pipeline, "predict_proba"):

                    probabilities = pipeline.predict_proba(
                        input_df
                    )[0]

                    classifier = pipeline.named_steps.get(
                        "classifier"
                    )

                    if classifier is not None:
                        classes = list(classifier.classes_)
                    else:
                        classes = list(
                            pipeline.classes_
                        )

                    if 1 in classes:
                        approved_probability = float(
                            probabilities[classes.index(1)]
                        )

                # -----------------------------------------
                # PREPARE RESULT FOR FRONTEND
                # -----------------------------------------

                result = {
                    "approved": prediction == 1,
                    "confidence": (
                        round(
                            approved_probability * 100,
                            1,
                        )
                        if approved_probability is not None
                        else None
                    ),
                    "accuracy": accuracy,
                }

            except (ValueError, TypeError) as exc:

                error = str(exc)

            except Exception:

                app.logger.exception(
                    "An error occurred during prediction."
                )

                error = (
                    "Prediction failed. Please check the "
                    "model pipeline and input feature names."
                )

    # -----------------------------------------------------
    # RENDER HTML PAGE
    # -----------------------------------------------------

    return render_template(
        "index.html",
        numeric_fields=NUMERIC_FIELDS,
        categorical_fields=CATEGORICAL_FIELDS,
        values=values,
        result=result,
        error=error,
        accuracy=accuracy,
        model_ready=model_ready,
    )


# =========================================================
# 6. HEALTH CHECK ENDPOINT
# =========================================================

@app.route("/health", methods=["GET"])
def health():

    return {
        "status": "ok",
        "model_ready": MODEL_PATH.exists(),
    }


# =========================================================
# 7. RUN FLASK APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=False,
        use_reloader=False,
        host="127.0.0.1",
        port=5000,
    )