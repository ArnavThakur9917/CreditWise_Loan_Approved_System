from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "loan_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "loan_approval_pipeline.joblib"

NUMERIC_FEATURES = [
    "Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
    "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings",
    "Collateral_Value", "Loan_Amount", "Loan_Term",
]
CATEGORICAL_FEATURES = [
    "Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area",
    "Education_Level", "Gender", "Employer_Category",
]
TARGET = "Loan_Approved"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    # Remove rows that are entirely empty and standardize target labels.
    df = df.dropna(how="all").copy()
    df[TARGET] = df[TARGET].astype("string").str.strip().str.title()
    df = df[df[TARGET].isin(["Yes", "No"])].copy()

    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print(f"Rows used: {len(df)}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, labels=[0, 1],
        target_names=["Rejected (No)", "Approved (Yes)"],
        zero_division=0,
    ))
    print("Confusion matrix (rows=true, columns=predicted; order No, Yes):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_mapping": {"Rejected": 0, "Approved": 1},
        "test_accuracy": float(accuracy_score(y_test, predictions)),
    }, MODEL_PATH)
    print(f"\nSaved trained pipeline to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
