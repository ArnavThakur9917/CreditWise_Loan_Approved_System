# CreditWise Loan System
### Machine Learning-based loan approval prediction — portfolio project

CreditWise is a demonstration project for predicting whether a loan application resembles historically **Approved** or **Rejected** applications. It uses a scikit-learn classification pipeline, a FastAPI backend, and a Streamlit frontend.

> **Important:** This is an educational prototype, not a real credit-decision system. Predictions can reflect historical bias and data quality problems. Do not use it to approve/deny real applicants. A real deployment requires legal/compliance review, fairness testing, explainability, security, monitoring, and human oversight.

## Features
- Data cleaning and preprocessing (missing values, categorical encoding, numerical imputation)
- Random Forest classifier with a held-out test split
- Model evaluation metrics printed during training
- FastAPI REST API (`/health`, `/predict`)
- Streamlit form for entering applicant details and viewing a prediction
- Clear setup steps for Windows PowerShell

## Project structure
```text
CreditWise_Loan_System/
├── backend/
│   └── main.py
├── frontend/
│   └── app.py
├── data/
│   └── loan_data.csv
├── models/                  # trained model is generated here
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset
`data/loan_data.csv` is the dataset supplied for this project. It contains applicant attributes and the `Loan_Approved` target. The training script removes fully empty rows, drops `Applicant_ID` because it is an identifier rather than a meaningful predictor, and trains on the remaining rows.

## 1. Requirements
- Python 3.10–3.12 recommended
- Git (only needed for GitHub upload)
- Internet access for installing Python packages

## 2. Setup on Windows (PowerShell)
Open PowerShell in the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If Python 3.12 is not installed, install Python 3.11 or 3.12 and use its launcher version instead.

## 3. Train the model
Run from the project root:

```powershell
python train_model.py
```

This creates `models/loan_approval_pipeline.joblib` and prints accuracy, a classification report, and a confusion matrix. Review these metrics; accuracy alone is not sufficient for evaluating a loan model.

## 4. Start the backend API
Open **Terminal 1** in VS Code, from the project root, with the virtual environment active:

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: open `http://127.0.0.1:8000/docs` in your browser. Health check: `http://127.0.0.1:8000/health`.

## 5. Start the frontend
Open **Terminal 2**, activate the same environment, and run:

```powershell
streamlit run frontend/app.py
```

The Streamlit page should open in your browser. Keep the backend terminal running while using the frontend.

If the API is hosted somewhere else, set the `CREDITWISE_API_URL` environment variable to the API base URL. For local development, the default is `http://127.0.0.1:8000`.

## API example
`POST /predict` JSON body:
```json
{
  "Applicant_Income": 25000,
  "Coapplicant_Income": 5000,
  "Employment_Status": "Salaried",
  "Age": 30,
  "Marital_Status": "Single",
  "Dependents": 0,
  "Credit_Score": 720,
  "Existing_Loans": 1,
  "DTI_Ratio": 0.25,
  "Savings": 50000,
  "Collateral_Value": 100000,
  "Loan_Amount": 150000,
  "Loan_Term": 120,
  "Loan_Purpose": "Home",
  "Property_Area": "Urban",
  "Education_Level": "Graduate",
  "Gender": "Male",
  "Employer_Category": "Private"
}
```

## Push to GitHub
1. Create a new **empty** GitHub repository named `CreditWise-Loan-System`.
2. Extract this ZIP and open a terminal in the extracted folder.
3. Run:
```powershell
git init
git add .
git commit -m "Build CreditWise loan prediction app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/CreditWise-Loan-System.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username. Do not upload `.venv/` or private applicant data. The included CSV is the dataset supplied for this educational project; verify that you have permission to publish it before pushing publicly.

## Suggested GitHub repository description
`An educational ML loan-approval prediction system with a scikit-learn pipeline, FastAPI backend, and Streamlit frontend.`

## Suggested LinkedIn post
I’ve built **CreditWise Loan System**, a machine-learning project that predicts loan approval outcomes from applicant and financial attributes.

**What I worked on**
- Data cleaning and preprocessing
- Categorical encoding and missing-value handling
- Training and evaluating a Random Forest classifier
- Building a REST API with FastAPI
- Creating an interactive Streamlit frontend

This project helped me practice the end-to-end ML workflow, from preparing data to serving model predictions through an application.

Note: This is an educational prototype, not a real financial decision system.

#MachineLearning #Python #FastAPI #Streamlit #DataScience #AI
