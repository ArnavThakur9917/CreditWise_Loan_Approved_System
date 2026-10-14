# CreditWise — Loan Approval Prediction System

A full-stack machine-learning demo with a Flask backend and responsive dark-blue frontend inspired by the clean layout of sentiment-analysis demos. It predicts a likely loan outcome using a Random Forest pipeline trained on the included dataset.

## Features
- Responsive form for 18 applicant features
- Random Forest classification with numeric imputation and categorical one-hot encoding
- Prediction result and estimated approval probability
- Held-out test accuracy shown after training
- Flask health endpoint at `/health`

## Project structure
```text
CreditWise_loan_approved/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── loan_data.csv
├── models/                 # model generated after training
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Run in VS Code (Windows PowerShell)
Open this folder in VS Code, then run these commands in the terminal:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
python app.py
```

If PowerShell blocks virtual-environment activation, use this for the current terminal only:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Then open **http://127.0.0.1:5000** in your browser. Keep the terminal running while you use the website. To stop the server, press `Ctrl+C`.

## Important notes
- Train the model before starting the web app. Training creates `models/loan_approval_pipeline.joblib`.
- The dataset and its labels determine what the model learns. Test accuracy is not a guarantee of real-world performance.
- This is an educational demo, not a real lending decision system. Do not use it to make actual credit decisions or submit sensitive personal information.
- The form shows the supported numeric ranges and categories from the included dataset. These are demo bounds, not lending rules.

## GitHub upload
Create a new empty repository on GitHub, then from this project folder run:
```powershell
git init
git add .
git commit -m "Build CreditWise loan approval prediction app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/CreditWise_loan_approved.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username. If Git says a remote already exists, inspect it with `git remote -v` before changing it.

## Suggested LinkedIn post

🚀 **Project: CreditWise — Loan Approval Prediction System**

I built a full-stack machine-learning web application that estimates loan approval outcomes from applicant and financial details.

**What I used:**
- Python, Pandas, Scikit-learn
- Random Forest classification
- Data preprocessing with imputation and one-hot encoding
- Flask backend and responsive HTML/CSS frontend

The app accepts applicant information through a web form and returns a model prediction with an estimated approval probability. I also included a training script and evaluation metrics to make the workflow reproducible.

This project helped me practice the end-to-end ML workflow—from preparing data and training a model to integrating it into a web application.

*Note: This is an educational project; predictions are not financial advice or official loan decisions.*

#MachineLearning #Python #Flask #ScikitLearn #AI #WebDevelopment #NIT Srinagar
