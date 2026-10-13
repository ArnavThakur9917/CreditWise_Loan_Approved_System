import os
import requests
import streamlit as st

API_URL = os.getenv("CREDITWISE_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="CreditWise | Loan Prediction", page_icon="🏦", layout="wide")

st.markdown("""
<style>
.main {background: #f7f9fc;}
.hero {padding: 1.2rem 1.5rem; border-radius: 16px; background: linear-gradient(110deg,#102a43,#246b8e); color: white;}
.hero h1 {color: white; margin-bottom: .25rem;}
.small-note {font-size: .9rem; color: #52606d;}
div[data-testid="stForm"] {background: white; padding: 1rem 1.2rem; border-radius: 14px; border: 1px solid #d9e2ec;}
</style>
<div class="hero">
<h1>🏦 CreditWise Loan System</h1>
<p>Machine-learning based loan approval prediction demo</p>
</div>
""", unsafe_allow_html=True)

st.write("")
st.markdown("Enter applicant details to get a prediction from the trained model.")
st.caption("Demo only: this output must not be used for actual lending decisions.")

with st.form("applicant_form"):
    st.subheader("Applicant profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        income = st.number_input("Applicant monthly income", min_value=0.0, value=25000.0, step=1000.0)
        co_income = st.number_input("Co-applicant monthly income", min_value=0.0, value=5000.0, step=1000.0)
        employment = st.selectbox("Employment status", ["Salaried", "Contract", "Self-employed", "Unemployed"])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        marital = st.selectbox("Marital status", ["Single", "Married"])
        dependents = st.number_input("Dependents", min_value=0, max_value=20, value=0)
    with c2:
        credit = st.number_input("Credit score", min_value=300.0, max_value=900.0, value=720.0, step=1.0)
        existing = st.number_input("Existing loans", min_value=0, max_value=50, value=1)
        dti = st.number_input("Debt-to-income ratio (dataset scale)", min_value=0.0, max_value=10.0, value=0.25, step=0.01)
        savings = st.number_input("Savings balance", min_value=0.0, value=50000.0, step=1000.0)
        collateral = st.number_input("Collateral value", min_value=0.0, value=100000.0, step=5000.0)
        loan_amount = st.number_input("Loan amount requested", min_value=1.0, value=150000.0, step=5000.0)
    with c3:
        term = st.number_input("Loan term (months)", min_value=1, max_value=600, value=120, step=12)
        purpose = st.selectbox("Loan purpose", ["Home", "Education", "Personal", "Business", "Car"])
        area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
        education = st.selectbox("Education level", ["Graduate", "Not Graduate"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        employer_category = st.selectbox("Employer category", ["Private", "Government", "MNC", "Business", "Unemployed"])

    submitted = st.form_submit_button("Predict loan outcome", type="primary", use_container_width=True)

if submitted:
    payload = {
        "Applicant_Income": income,
        "Coapplicant_Income": co_income,
        "Employment_Status": employment,
        "Age": age,
        "Marital_Status": marital,
        "Dependents": dependents,
        "Credit_Score": credit,
        "Existing_Loans": existing,
        "DTI_Ratio": dti,
        "Savings": savings,
        "Collateral_Value": collateral,
        "Loan_Amount": loan_amount,
        "Loan_Term": term,
        "Loan_Purpose": purpose,
        "Property_Area": area,
        "Education_Level": education,
        "Gender": gender,
        "Employer_Category": employer_category,
    }
    try:
        with st.spinner("Contacting the prediction API..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        if response.ok:
            result = response.json()
            if result["prediction"] == "Approved":
                st.success("Model prediction: APPROVED")
            else:
                st.error("Model prediction: REJECTED")
            st.metric("Estimated approval probability", f"{result['approved_probability'] * 100:.1f}%")
            st.progress(float(result["approved_probability"]))
            st.caption(f"Held-out test accuracy: {result['model_test_accuracy'] * 100:.1f}%")
            st.info(result["notice"])
        else:
            st.error(f"API error ({response.status_code}): {response.text}")
            st.caption("Check that the model is trained and the backend is running.")
    except requests.RequestException:
        st.error("Could not connect to the backend. Start the API in another terminal with:")
        st.code("uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000")

st.divider()
st.markdown("**Built with:** Python · scikit-learn · FastAPI · Streamlit")
st.markdown('<p class="small-note">Responsible AI note: a model trained on historical outcomes can inherit bias. Never rely on this prototype alone for real financial decisions.</p>', unsafe_allow_html=True)
