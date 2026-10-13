from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "loan_approval_pipeline.joblib"

app = FastAPI(
    title="CreditWise Loan System API",
    description="Educational API for predicting historical loan approval outcomes.",
    version="1.0.0",
)

class Applicant(BaseModel):
    Applicant_Income: float = Field(ge=0, description="Monthly applicant income")
    Coapplicant_Income: float = Field(ge=0, description="Monthly co-applicant income")
    Employment_Status: Literal["Contract", "Salaried", "Self-employed", "Unemployed"]
    Age: int = Field(ge=18, le=100)
    Marital_Status: Literal["Married", "Single"]
    Dependents: int = Field(ge=0, le=20)
    Credit_Score: float = Field(ge=300, le=900)
    Existing_Loans: int = Field(ge=0, le=50)
    DTI_Ratio: float = Field(ge=0, le=10, description="Debt-to-income ratio; use the dataset's scale")
    Savings: float = Field(ge=0)
    Collateral_Value: float = Field(ge=0)
    Loan_Amount: float = Field(gt=0)
    Loan_Term: int = Field(gt=0, le=600, description="Loan duration in months")
    Loan_Purpose: Literal["Home", "Education", "Personal", "Business", "Car"]
    Property_Area: Literal["Urban", "Semiurban", "Rural"]
    Education_Level: Literal["Graduate", "Not Graduate"]
    Gender: Literal["Male", "Female"]
    Employer_Category: Literal["Business", "Government", "MNC", "Private", "Unemployed"]

def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Trained model not found. From the project root, run: python train_model.py"
        )
    return joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "CreditWise Loan System API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}

@app.post("/predict")
def predict(applicant: Applicant):
    try:
        bundle = load_model()
        row = pd.DataFrame([applicant.model_dump()])
        pipeline = bundle["pipeline"]
        prediction = int(pipeline.predict(row)[0])
        probability = pipeline.predict_proba(row)[0]
        classes = list(pipeline.named_steps["classifier"].classes_)
        approved_probability = float(probability[classes.index(1)]) if 1 in classes else 0.0
        return {
            "prediction": "Approved" if prediction == 1 else "Rejected",
            "approved_probability": round(approved_probability, 4),
            "model_test_accuracy": round(bundle.get("test_accuracy", 0.0), 4),
            "notice": "Educational estimate only. This is not a credit decision or financial advice.",
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
