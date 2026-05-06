# 🏦 Loan Eligibility Prediction App

An interactive web application that predicts whether a loan applicant is **eligible or not eligible** for a loan — instantly, based on their personal and financial details.

Built with **Streamlit** and **Scikit-learn**.

---

## 🌐 Live Demo

👉 **[Launch App on Streamlit Cloud](https://oluwatobi-loan-checker.streamlit.app/)**

---

## 💡 What the App Does

- Takes in an applicant's **personal info** (gender, marital status, dependents, education)
- Takes in their **financial details** (income, co-applicant income, loan amount, loan term)
- Takes in their **credit history** and **property area**
- Lets you **choose from 3 ML models** to run the prediction:
  - Logistic Regression ⭐ Best Model
  - Random Forest
  - Gradient Boosting
- Instantly outputs:
  - ✅ **Eligible for Loan** — with approval probability
  - ❌ **Not Eligible for Loan** — with approval probability

---

## 📁 Project Structure

```
loan-eligibility-prediction/
├── train.csv                     # Training dataset (614 records)
├── test.csv                      # Test dataset (367 records)
├── loan_prediction.py            # ML pipeline (EDA → preprocessing → modelling → saves models)
├── app.py                        # Streamlit web app — the main deliverable
├── requirements.txt              # Python dependencies for Streamlit Cloud deployment
├── outputs/
│   ├── logistic_regression.pkl   # Saved Logistic Regression model
│   ├── random_forest.pkl         # Saved Random Forest model
│   ├── gradient_boosting.pkl     # Saved Gradient Boosting model
│   ├── best_model_name.pkl       # Stores name of best performing model
│   ├── loan_predictions.csv      # Batch predictions on test set
│   ├── eda_plots.png             # EDA visualizations
│   └── model_evaluation.png      # Confusion matrix, ROC curves, feature importances
└── README.md
```

---

## 🧠 How the Model Was Built

### 1. Exploratory Data Analysis
- Loan status class distribution
- Income and loan amount distributions
- Approval rates by property area, education, and credit history

### 2. Data Preprocessing
- **Split before preprocessing** to prevent data leakage
- Missing values: categorical → **mode**, numerical → **median**
- **Label encoders fitted on training data only**, reused on validation and test
- Feature engineering:
  - `TotalIncome` = ApplicantIncome + CoapplicantIncome
  - `Log_LoanAmount` and `Log_TotalIncome` (log-transform to reduce skew)
  - `EMI` = LoanAmount / Loan_Amount_Term
  - `BalanceIncome` = TotalIncome − (EMI × 1000)

### 3. Models Compared

| Model | Validation Accuracy | CV Mean (5-Fold) |
|---|---|---|
| **Logistic Regression** ✅ | **86.2%** | **80.3%** |
| Random Forest | 82.9% | 78.5% |
| Gradient Boosting | 81.3% | 76.6% |

> **Best model: Logistic Regression** — selected based on 5-fold cross-validation mean accuracy.

### 4. Model Performance

| | Precision | Recall | F1-Score |
|---|---|---|---|
| Rejected (N) | 0.96 | 0.58 | 0.72 |
| Approved (Y) | 0.84 | 0.99 | 0.91 |
| **Accuracy** | | | **86%** |

---

## 🔍 Key Insight

**Credit History** was the strongest predictor of loan approval. Applicants without a credit history were overwhelmingly rejected regardless of income or loan amount.

---

## 🚀 How to Run Locally

### Step 1 — Clone the repository
```bash
git clone https://github.com/Oluwatobi-abu/loan-eligibility-prediction.git
cd loan-eligibility-prediction
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Train the models
Run this once to generate all model files inside `outputs/`:
```bash
python loan_prediction.py
```

### Step 4 — Launch the app
```bash
streamlit run app.py
```

The app opens in your browser automatically. Select a model, fill in the applicant's details, and click **Check Eligibility**.

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Streamlit | Web app interface |
| Scikit-learn | Model training and evaluation |
| Pandas & NumPy | Data manipulation and feature engineering |
| Matplotlib & Seaborn | EDA and evaluation visualizations |
| Joblib | Model saving and loading |
| Python 3 | Core language |

---

## 📊 Dataset

Sourced from the [Analytics Vidhya Loan Prediction Practice Problem](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/).

---

## 👤 Author

**Oluwatobi**
Data Science — Bincom Academy
GitHub: [@Oluwatobi-abu](https://github.com/Oluwatobi-abu)
