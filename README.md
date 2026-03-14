# Cell2Cell Telecom — Churn Prediction

A complete, production-grade machine learning project for telecom customer churn prediction — **two powerhouses in one repo.**

[![CI](https://github.com/parthanallelu/Churn_predicition/actions/workflows/ci.yml/badge.svg)](https://github.com/parthanallelu/Churn_predicition/actions/workflows/ci.yml)

---

## What's Inside

| Component | Description |
|---|---|
| `churn_prediction_world_class.ipynb` | Full ML research pipeline — EDA, feature engineering, XGBoost + LightGBM + Optuna tuning, ensemble, SHAP-style interpretability, business ROI |
| `custom_train.py` | Custom Decision Tree + Bagging classifier built **from scratch** — zero sklearn, zero pandas, pure Python |
| `app.py` | Production Flask web app serving the custom model via REST API |

---

## Dataset

**Cell2Cell Telecom** — 51,047 customers, 58 features, ~28.8% churn rate

---

## Notebook — `churn_prediction_world_class.ipynb`

### Pipeline

```
Raw Data (51K rows)
   ↓
Exploratory Data Analysis
  • Churn distribution, class imbalance
  • Feature distributions by churn (plotly + matplotlib)
  • Categorical churn rates, correlation heatmaps
   ↓
Feature Engineering  (+22 engineered features = 79 total)
  • Revenue & usage ratios
  • Call quality scores
  • Churn risk signals (care calls, retention refusals, revenue decline)
  • Credit & device features
   ↓
Preprocessing Pipeline
  • Binary encoding, label encoding, median imputation, RobustScaler
   ↓
Model Training
  • Logistic Regression (baseline)
  • Random Forest (300 trees)
  • XGBoost (early stopping)
  • LightGBM (early stopping)
   ↓
Hyperparameter Tuning — Optuna (50 trials × 2 models)
  • TPE Sampler, early stopping callbacks
   ↓
Ensemble Model
  • Weighted average (XGBoost + LightGBM + RF)
  • Weights = validation AUC
   ↓
Evaluation
  • ROC-AUC, F1, Avg Precision, Confusion matrices
  • ROC + Precision-Recall curves
  • Threshold optimization (F1 + business cost)
  • Calibration curves
   ↓
Interpretability
  • Built-in feature importance (XGBoost + LightGBM)
  • Permutation importance
  • Partial Dependence Plots (PDPs)
   ↓
Business Analysis
  • 4-tier customer risk segmentation
  • Lift & gains curves
  • ROI calculation vs random targeting
  • Actionable recommendations
```

### Results

| Model | Test AUC | F1 | Avg Precision |
|---|---|---|---|
| Logistic Regression | ~0.74 | ~0.53 | ~0.52 |
| Random Forest | ~0.82 | ~0.60 | ~0.63 |
| XGBoost (Tuned) | ~0.84 | ~0.62 | ~0.65 |
| LightGBM (Tuned) | ~0.84 | ~0.62 | ~0.65 |
| **Ensemble** | **~0.85** | **~0.63** | **~0.66** |

**Business impact:** Targeting the top 20% of customers by churn score captures ~55% of all churners — a **2.75× lift** over random targeting.

### Run the Notebook

```bash
pip install -r requirements.txt
jupyter notebook churn_prediction_world_class.ipynb
# Kernel → Restart & Run All
```

Full run takes ~15 minutes (dominated by Optuna tuning).

---

## Custom ML Engine — `custom_train.py`

Pure Python implementation of a Decision Tree + Bagging ensemble — **no ML libraries whatsoever.**

### Algorithm Details

- **Decision Tree** — CART with Gini impurity, configurable depth/min_samples
- **Bagging** — 30 trees trained on balanced bootstrap samples (1000 pos + 1000 neg each)
- **Encoding** — Custom `LabelEncoder` for categoricals, mean/mode imputation
- **Metrics** — Accuracy, Precision, Recall, F1, Confusion Matrix — all from scratch

```bash
python3 custom_train.py   # trains 30 trees, saves model artifacts
```

---

## Flask Web App — `app.py`

Production-ready REST API + interactive dashboard.

### Quick Start

```bash
# Option 1 — Python
pip install -r requirements.txt
python3 app.py
# Open http://localhost:8080

# Option 2 — Docker
docker-compose up --build
# Open http://localhost:8080
```

### API — `POST /predict`

```json
// Request
{
  "MonthlyMinutes": 500,
  "OverageMinutes": 10,
  "MonthlyRevenue": 55.0,
  "TotalRecurringCharge": 45.0,
  "IncomeGroup": 3,
  "AgeHH1": 35,
  "MonthsInService": 24,
  "CurrentEquipmentDays": 300,
  "RoamingCalls": 5,
  "HandsetPrice": "100",
  "CreditRating": "3-Good",
  "MaritalStatus": "Yes"
}

// Response
{
  "status": "success",
  "prediction": "Yes",
  "probability": "73.40%"
}
```

Rate limit: 30 requests/minute per IP.

### Run Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
Churn_predicition/
├── churn_prediction_world_class.ipynb   ← Full ML research notebook
├── cell2celltrain.csv                   ← Dataset (51K rows)
├── custom_train.py                      ← Custom ML engine (pure Python)
├── app.py                               ← Flask REST API
├── generate_graphs.py                   ← EDA chart generator
├── custom_model.pkl                     ← Trained model artifacts
├── custom_metrics.json                  ← Model performance metrics
├── custom_plot_data.json                ← Dashboard chart data
├── .github/workflows/ci.yml            ← CI/CD pipeline
├── docs/                               ← Algorithm documentation
├── static/                             ← CSS, JS, images
├── templates/                          ← HTML pages
├── tests/                              ← pytest suite (40+ tests)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Notebook ML | XGBoost, LightGBM, scikit-learn, Optuna |
| Notebook Viz | Plotly, Matplotlib, Seaborn |
| Custom ML | Pure Python (no ML libraries) |
| Web Backend | Flask 3.0, Gunicorn |
| Frontend | HTML5, CSS3 (Glassmorphism), Vanilla JS |
| Tests | pytest |
| CI/CD | GitHub Actions |
| Container | Docker + docker-compose |

---

## Key Business Insights

1. **Customer Care Calls** — >2 care calls = 2–3× higher churn rate. Flag proactively.
2. **Retention Refusals** — Called but refused all offers → extremely high churn probability.
3. **Revenue Decline** — >10% monthly drop is an early churn warning signal.
4. **Low Credit Rating** — Customers rated 6-VeryLow / 7-Lowest churn significantly more.
5. **New Customers** — First 12 months is the highest-risk window. Onboarding matters.
6. **Call Quality** — High dropped/blocked call rates strongly predict churn.

## Recommended Actions

1. Deploy model weekly → score all customers → contact top 20% for retention
2. Auto-flag customers with ≥3 care calls in 30 days
3. Create tailored offers for low credit-rating segments
4. Improve network quality in high drop-rate service areas
5. Build dedicated onboarding for new customers (0–12 months)
6. A/B test retention offer types for customers who previously refused
