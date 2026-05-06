# ============================================================
#  LOAN ELIGIBILITY PREDICTION
#  Dataset: Analytics Vidhya Loan Prediction Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# ──────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

print("Train shape:", train.shape)
print("Test shape :", test.shape)
print("\nFirst 5 rows of training data:")
print(train.head())
print("\nMissing values in Train:")
print(train.isnull().sum())

# ──────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Loan Eligibility - EDA", fontsize=16, fontweight='bold')

train['Loan_Status'].value_counts().plot(kind='bar', ax=axes[0,0],
    color=['#2ecc71','#e74c3c'], edgecolor='black')
axes[0,0].set_title("Loan Status Distribution")
axes[0,0].set_xticklabels(['Approved (Y)', 'Rejected (N)'], rotation=0)

train['ApplicantIncome'].plot(kind='hist', bins=40, ax=axes[0,1],
    color='steelblue', edgecolor='black')
axes[0,1].set_title("Applicant Income Distribution")

train['LoanAmount'].plot(kind='hist', bins=40, ax=axes[0,2],
    color='darkorange', edgecolor='black')
axes[0,2].set_title("Loan Amount Distribution")

train['Credit_History'].value_counts().plot(kind='bar', ax=axes[1,0],
    color=['#3498db','#e67e22'], edgecolor='black')
axes[1,0].set_title("Credit History")
axes[1,0].set_xticklabels(['Has Credit (1)', 'No Credit (0)'], rotation=0)

pd.crosstab(train['Property_Area'], train['Loan_Status']).plot(
    kind='bar', ax=axes[1,1], color=['#e74c3c','#2ecc71'], edgecolor='black')
axes[1,1].set_title("Property Area vs Loan Status")
axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=0)

pd.crosstab(train['Education'], train['Loan_Status']).plot(
    kind='bar', ax=axes[1,2], color=['#e74c3c','#2ecc71'], edgecolor='black')
axes[1,2].set_title("Education vs Loan Status")
axes[1,2].set_xticklabels(axes[1,2].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'eda_plots.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ EDA plot saved.")

# ──────────────────────────────────────────────────────────
# 3. SPLIT FIRST — prevents data leakage
# ──────────────────────────────────────────────────────────
train_raw, val_raw = train_test_split(
    train, test_size=0.2, random_state=42,
    stratify=train['Loan_Status'])

print(f"\nRaw training set  : {len(train_raw)} rows")
print(f"Raw validation set: {len(val_raw)} rows")

# ──────────────────────────────────────────────────────────
# 4. FIT LABEL ENCODERS ON TRAINING DATA ONLY
# ──────────────────────────────────────────────────────────
CAT_COLS = ['Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'Property_Area']

encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    le.fit(train_raw[col].astype(str))
    encoders[col] = le

# ──────────────────────────────────────────────────────────
# 5. PREPROCESSING
# ──────────────────────────────────────────────────────────
def compute_impute_stats(df):
    stats = {}
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
        stats[col] = df[col].mode()[0]
    stats['LoanAmount'] = df['LoanAmount'].median()
    term_clean = df['Loan_Amount_Term'].replace(0, np.nan)
    stats['Loan_Amount_Term_median'] = term_clean.median()
    return stats


def preprocess(df, impute_stats, encoders, is_train=True):
    df = df.copy()
    df.drop('Loan_ID', axis=1, inplace=True)

    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
        df[col].fillna(impute_stats[col], inplace=True)
    df['LoanAmount'].fillna(impute_stats['LoanAmount'], inplace=True)
    df['Loan_Amount_Term'].replace(0, np.nan, inplace=True)
    df['Loan_Amount_Term'].fillna(impute_stats['Loan_Amount_Term_median'], inplace=True)

    df['TotalIncome']     = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Log_LoanAmount']  = np.log1p(df['LoanAmount'])
    df['Log_TotalIncome'] = np.log1p(df['TotalIncome'])
    df['EMI']             = df['LoanAmount'] / df['Loan_Amount_Term']
    df['BalanceIncome']   = df['TotalIncome'] - (df['EMI'] * 1000)

    df.fillna(df.median(numeric_only=True), inplace=True)

    if is_train:
        df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    for col in CAT_COLS:
        le = encoders[col]
        df[col] = df[col].astype(str).apply(
            lambda x: x if x in le.classes_ else le.classes_[0]
        )
        df[col] = le.transform(df[col])

    return df


impute_stats = compute_impute_stats(train_raw)

train_clean = preprocess(train_raw, impute_stats, encoders, is_train=True)
val_clean   = preprocess(val_raw,   impute_stats, encoders, is_train=True)
test_clean  = preprocess(test,      impute_stats, encoders, is_train=False)

# ──────────────────────────────────────────────────────────
# 6. PREPARE FEATURES
# ──────────────────────────────────────────────────────────
feature_cols = [c for c in train_clean.columns if c != 'Loan_Status']

X_train = train_clean[feature_cols]
y_train = train_clean['Loan_Status']
X_val   = val_clean[feature_cols]
y_val   = val_clean['Loan_Status']
X_all   = pd.concat([X_train, X_val])
y_all   = pd.concat([y_train, y_val])

# ──────────────────────────────────────────────────────────
# 7. MODEL TRAINING & EVALUATION
# ──────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*55)
print(f"{'Model':<25} {'Val Acc':>8} {'CV Mean':>8} {'CV Std':>8}")
print("="*55)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred    = model.predict(X_val)
    val_acc   = accuracy_score(y_val, y_pred)
    cv_scores = cross_val_score(model, X_all, y_all, cv=cv, scoring='accuracy')
    results[name] = {
        'model':   model,
        'val_acc': val_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std':  cv_scores.std(),
        'y_pred':  y_pred,
        'y_proba': model.predict_proba(X_val)[:,1]
    }
    print(f"{name:<25} {val_acc:>8.4f} {cv_scores.mean():>8.4f} {cv_scores.std():>8.4f}")

print("="*55)

# ──────────────────────────────────────────────────────────
# 8. BEST MODEL — DETAILED REPORT
# ──────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['cv_mean'])
best      = results[best_name]
print(f"\n Best Model: {best_name}")
print("\nClassification Report:")
print(classification_report(y_val, best['y_pred'],
                             target_names=['Rejected', 'Approved']))

# ──────────────────────────────────────────────────────────
# 9. PLOTS — Confusion Matrix + ROC + Feature Importance
# ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Model Evaluation - {best_name}", fontsize=14, fontweight='bold')

cm = confusion_matrix(y_val, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Rejected','Approved'],
            yticklabels=['Rejected','Approved'])
axes[0].set_title("Confusion Matrix")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_val, res['y_proba'])
    auc = roc_auc_score(y_val, res['y_proba'])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
axes[1].plot([0,1],[0,1],'k--')
axes[1].set_title("ROC Curves")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(fontsize=8)

rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
importances.sort_values().plot(kind='barh', ax=axes[2],
    color='steelblue', edgecolor='black')
axes[2].set_title("Feature Importances (Random Forest)")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_evaluation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Evaluation plot saved.")

# ──────────────────────────────────────────────────────────
# 10. SAVE ALL THREE MODELS
#     app.py loads whichever model the user selects
# ──────────────────────────────────────────────────────────
model_filenames = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Random Forest':       'random_forest.pkl',
    'Gradient Boosting':   'gradient_boosting.pkl',
}

for name, filename in model_filenames.items():
    joblib.dump(results[name]['model'], os.path.join(output_dir, filename))
    print(f"✅ Saved {name} → outputs/{filename}")

# Save best model name so app can highlight it
joblib.dump(best_name, os.path.join(output_dir, 'best_model_name.pkl'))
print(f"\n🏆 Best model: {best_name}")

# ──────────────────────────────────────────────────────────
# 11. GENERATE PREDICTIONS ON TEST SET (using best model)
# ──────────────────────────────────────────────────────────
best_model = results[best_name]['model']
X_test     = test_clean[feature_cols]
test_preds = best_model.predict(X_test)
test_proba = best_model.predict_proba(X_test)[:,1]

submission = pd.DataFrame({
    'Loan_ID':              test['Loan_ID'],
    'Loan_Status':          ['Y' if p == 1 else 'N' for p in test_preds],
    'Approval_Probability': test_proba.round(3)
})

submission.to_csv(os.path.join(output_dir, 'loan_predictions.csv'), index=False)
print("\n✅ Predictions saved to outputs/loan_predictions.csv")
print(f"\nPrediction Summary:")
print(submission['Loan_Status'].value_counts())
print("\nSample predictions:")
print(submission.head(10).to_string(index=False))
print("\n✅ All done!")
