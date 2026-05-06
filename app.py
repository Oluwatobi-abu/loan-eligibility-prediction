import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ──────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Eligibility Checker",
    page_icon="🏦",
    layout="centered"
)

# ──────────────────────────────────────────────────────────
# STYLING
# ──────────────────────────────────────────────────────────
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
        .main { background-color: #f4f6fb; }

        .header-box {
            background: linear-gradient(135deg, #1a3c5e, #2d7dd2);
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 2rem;
            color: white;
        }
        .header-box h1 { font-size: 2rem; margin: 0; }
        .header-box p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 0.95rem; }

        .model-card {
            background: white;
            border: 2px solid #e0e8f0;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }
        .model-card.best {
            border-color: #2d7dd2;
            background: #f0f7ff;
        }
        .model-badge {
            display: inline-block;
            background: #2d7dd2;
            color: white;
            font-size: 0.7rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 20px;
            margin-left: 8px;
            vertical-align: middle;
        }
        .model-stat {
            font-size: 0.82rem;
            color: #555;
            margin-top: 4px;
        }

        .result-eligible {
            background: #e8f9f0;
            border-left: 6px solid #27ae60;
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-top: 1.5rem;
        }
        .result-not-eligible {
            background: #fdecea;
            border-left: 6px solid #e74c3c;
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-top: 1.5rem;
        }
        .result-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.3rem; }
        .result-sub   { font-size: 0.95rem; color: #555; }

        div[data-testid="stButton"] > button {
            background: linear-gradient(135deg, #1a3c5e, #2d7dd2);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 2.5rem;
            font-size: 1rem;
            font-weight: 600;
            width: 100%;
            margin-top: 1rem;
            cursor: pointer;
        }
        div[data-testid="stButton"] > button:hover { opacity: 0.88; }

        .section-label {
            font-weight: 600;
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #2d7dd2;
            margin: 1.5rem 0 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────
st.markdown("""
    <div class="header-box">
        <h1>🏦 Loan Eligibility Checker</h1>
        <p>Fill in the applicant's details and choose a model to check loan eligibility instantly.</p>
    </div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# LOAD MODELS & BEST MODEL NAME
# ──────────────────────────────────────────────────────────
MODEL_FILES = {
    'Logistic Regression': 'outputs/logistic_regression.pkl',
    'Random Forest':       'outputs/random_forest.pkl',
    'Gradient Boosting':   'outputs/gradient_boosting.pkl',
}

MODEL_STATS = {
    'Logistic Regression': {'val_acc': '86.2%', 'cv_mean': '80.3%'},
    'Random Forest':       {'val_acc': '82.9%', 'cv_mean': '78.5%'},
    'Gradient Boosting':   {'val_acc': '81.3%', 'cv_mean': '76.6%'},
}

@st.cache_resource
def load_all_models():
    loaded = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
    return loaded

@st.cache_resource
def load_best_name():
    path = 'outputs/best_model_name.pkl'
    if os.path.exists(path):
        return joblib.load(path)
    return 'Logistic Regression'

loaded_models = load_all_models()
best_name     = load_best_name()

if not loaded_models:
    st.error("⚠️ No models found. Please run `loan_prediction.py` first to generate the model files.")
    st.stop()

# ──────────────────────────────────────────────────────────
# MODEL SELECTOR
# ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Choose a Model</div>', unsafe_allow_html=True)

selected_model_name = st.radio(
    label="Select the model you want to use for prediction:",
    options=list(loaded_models.keys()),
    index=list(loaded_models.keys()).index(best_name),
    horizontal=True,
    label_visibility="collapsed"
)

# Show model info card
stats = MODEL_STATS[selected_model_name]
is_best = selected_model_name == best_name
badge = '<span class="model-badge">⭐ Best Model</span>' if is_best else ''
card_class = "model-card best" if is_best else "model-card"

st.markdown(f"""
    <div class="{card_class}">
        <strong>{selected_model_name}</strong>{badge}
        <div class="model-stat">
            Validation Accuracy: <strong>{stats['val_acc']}</strong> &nbsp;|&nbsp;
            CV Mean (5-Fold): <strong>{stats['cv_mean']}</strong>
        </div>
    </div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# APPLICANT FORM
# ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Personal Information</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    gender     = st.selectbox("Gender", ["Male", "Female"])
    married    = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
with col2:
    education     = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.markdown('<div class="section-label">Financial Details</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    applicant_income   = st.number_input("Applicant Monthly Income (₦)", min_value=0, value=50000, step=1000)
    coapplicant_income = st.number_input("Co-applicant Monthly Income (₦)", min_value=0, value=0, step=1000)
with col4:
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1, value=100, step=1)
    loan_term   = st.selectbox("Loan Term (months)", [360, 180, 120, 84, 60, 36, 240, 300, 480, 12])

st.markdown('<div class="section-label">Credit Information</div>', unsafe_allow_html=True)
credit_history = st.radio(
    "Does the applicant have a credit history?",
    ["Yes", "No"],
    horizontal=True
)

# ──────────────────────────────────────────────────────────
# FEATURE BUILDER
# ──────────────────────────────────────────────────────────
def build_features(gender, married, dependents, education, self_employed,
                   property_area, applicant_income, coapplicant_income,
                   loan_amount, loan_term, credit_history):

    gender_map     = {'Female': 0, 'Male': 1}
    married_map    = {'No': 0, 'Yes': 1}
    dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
    education_map  = {'Graduate': 0, 'Not Graduate': 1}
    self_emp_map   = {'No': 0, 'Yes': 1}
    property_map   = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}

    credit       = 1.0 if credit_history == "Yes" else 0.0
    total_income = applicant_income + coapplicant_income
    emi          = loan_amount / loan_term if loan_term > 0 else 0

    return pd.DataFrame([{
        'Gender':            gender_map[gender],
        'Married':           married_map[married],
        'Dependents':        dependents_map[dependents],
        'Education':         education_map[education],
        'Self_Employed':     self_emp_map[self_employed],
        'ApplicantIncome':   applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount':        loan_amount,
        'Loan_Amount_Term':  loan_term,
        'Credit_History':    credit,
        'Property_Area':     property_map[property_area],
        'TotalIncome':       total_income,
        'Log_LoanAmount':    np.log1p(loan_amount),
        'Log_TotalIncome':   np.log1p(total_income),
        'EMI':               emi,
        'BalanceIncome':     total_income - (emi * 1000),
    }])

# ──────────────────────────────────────────────────────────
# PREDICT
# ──────────────────────────────────────────────────────────
if st.button("Check Eligibility"):
    model      = loaded_models[selected_model_name]
    input_df   = build_features(
        gender, married, dependents, education, self_employed,
        property_area, applicant_income, coapplicant_income,
        loan_amount, loan_term, credit_history
    )
    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.markdown(f"""
            <div class="result-eligible">
                <div class="result-title">✅ Eligible for Loan</div>
                <div class="result-sub">
                    This applicant is likely to be <strong>approved</strong>
                    using <strong>{selected_model_name}</strong>.<br>
                    Approval Probability: <strong>{probability*100:.1f}%</strong>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-not-eligible">
                <div class="result-title">❌ Not Eligible for Loan</div>
                <div class="result-sub">
                    This applicant is likely to be <strong>rejected</strong>
                    using <strong>{selected_model_name}</strong>.<br>
                    Approval Probability: <strong>{probability*100:.1f}%</strong>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#aaa; font-size:0.8rem;'>Built with Streamlit · Bincom Academy | Data Science · Oluwatobi</center>",
    unsafe_allow_html=True
)
