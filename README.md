# 🏦 Banking Churn Prediction — World-Class ML System

A complete, production-grade machine learning project for predicting customer churn in a banking environment. This repository demonstrates a dual-track approach: a comprehensive research pipeline and a high-performance custom ML engine built from scratch.

[![CI](https://github.com/parthanallelu/Churn_predicition/actions/workflows/ci.yml/badge.svg)](https://github.com/parthanallelu/Churn_predicition/actions/workflows/ci.yml)

---

## 🚀 Key Features

| Component | Description |
|---|---|
| `churn_prediction_world_class.ipynb` | **Research Powerhouse**: Full pipeline featuring EDA, Stacking Ensembles (XGBoost + LightGBM + CatBoost + RF), Optuna hyperparameter tuning, and advanced model interpretation (SHAP, PDP). |
| `custom_train.py` | **Custom Engine**: Decision Tree + Bagging classifier implemented **from scratch** in pure Python. Demonstrates algorithmic depth without relying on Scikit-Learn's core fit/predict logic. |
| `app.py` | **Production API**: Flask-based REST service with input validation, rate limiting, and a beautiful glassmorphism dashboard. |

---

## 📊 Dataset & Schema

The system is trained on a modern **Banking Dataset** (`cell2celltrain.csv`) featuring 10,000 customers and 18 core features:

- **Demographics**: `age`, `income`, `salary_band`
- **Account Health**: `account_balance`, `credit_score`, `risk_score`
- **Engagement**: `tenure_months`, `monthly_transactions`, `digital_logins`, `atm_usage`, `branch_visits`
- **Sentiment**: `satisfaction_score`, `num_complaints`
- **Product Usage**: `num_products`, `has_loan`, `investment_products`, `late_payments`

---

## 🔬 Research & Performance — `churn_prediction_world_class.ipynb`

### Stacking Ensemble Results
Our world-class notebook implements a multi-stage stacking ensemble that significantly outperforms baseline models.

| Model | Test AUC | F1 Score |
|---|---|---|
| Logistic Regression (Baseline) | ~0.72 | ~0.55 |
| Random Forest | ~0.76 | ~0.62 |
| XGBoost / LightGBM (Tuned) | ~0.78 | ~0.65 |
| **Stacking Ensemble (Meta-LR)** | **0.7894** | **~0.68** |

### Insights derived from partial dependence analysis:
- **Tenure Sensitivity**: Customers in their first 6 months show 1.5x higher churn risk.
- **Digital Engagement**: Users with <2 `digital_logins` monthly are 40% more likely to exit.
- **Product Depth**: Having >3 `num_products` creates a strong "lock-in" effect reducing churn by 60%.

---

## 🛠️ Custom ML Engine — `custom_train.py`

A pure Python implementation of a **Bagging Classifier** utilizing balanced bootstrap sampling to handle class imbalance.

- **Algorithm**: CART-based Decision Trees with Gini Impurity.
- **Ensemble**: 30 Trees with deterministic feature/threshold sampling.
- **Current Performance**: **77.46% AUC** (Dashboard Accuracy).
- **Features**: Automatic median imputation and target encoding for categorical bands.

```bash
python custom_train.py   # Trains 30 trees, saves artifacts to .pkl
```

---

## 🌐 Flask Web App — `app.py`

### API Endpoint: `POST /predict`
Predict churn probability for a single customer profile with built-in validation.

```json
// Request
{
  "age": 35,
  "tenure_months": 24,
  "account_balance": 15000.50,
  "credit_score": 720,
  "num_products": 2,
  "digital_logins": 5,
  "num_complaints": 0
}

// Response
{
  "status": "success",
  "prediction": "No",
  "probability": "18.25%"
}
```

---

## 🚀 Quick Start

1. **Setup Environment**:
   ```bash
   python -m venv venv_stable
   venv_stable\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run Training**:
   ```bash
   python custom_train.py
   python generate_graphs.py
   ```

3. **Launch Dashboard**:
   ```bash
   python app.py
   ```
   Visit [http://localhost:8080](http://localhost:8080)

---

## 📈 Business Recommendations

1. **Digital Onboarding**: Incentivize digital mobile logins early in the tenure to increase engagement.
2. **Proactive Complaint Resolution**: Customers with >1 complaint should be flagged immediately for retention calls.
3. **Wealth Tiers**: Targeted loyalty programs for 'salary_band' 5-7 to protect high-balance accounts.

---

## 🏗️ Tech Stack
- **Languages**: Python 3.11+, JavaScript
- **ML Libraries**: XGBoost, LightGBM, CatBoost, Scikit-Learn, Optuna, Category Encoders
- **Web**: Flask 3.0, Gunicorn, Jinja2
- **Testing**: Pytest, Pytest-cov
- **Infrastructure**: Docker, GitHub Actions
