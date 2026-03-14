# Customer Churn Prediction

A production-ready machine learning web application that predicts telecom customer churn using a **custom-built Bagging + Decision Tree ensemble** — zero sklearn, zero pandas, pure Python algorithms.

[![CI](https://github.com/parthanallelu/Churn_predicition/actions/workflows/ci.yml/badge.svg)](https://github.com/parthanallelu/Churn_predicition/actions/workflows/ci.yml)

---

## What It Does

- Takes 12 key customer features (usage, financials, profile) as input
- Runs them through a 30-tree Bagging ensemble of custom Decision Trees
- Returns a churn probability (0–100%) and a Yes/No prediction
- Displays interactive EDA visualizations and model metrics on the dashboard

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, Flask 3.0 |
| ML Engine | Custom Decision Tree + Bagging (pure Python) |
| Frontend | HTML5, CSS3 (Glassmorphism), Vanilla JS |
| Server | Gunicorn |
| Tests | pytest |
| CI/CD | GitHub Actions |
| Container | Docker + docker-compose |

---

## Quick Start

### Option 1 — Python (Local)

```bash
# 1. Clone
git clone https://github.com/parthanallelu/Churn_predicition.git
cd Churn_predicition

# 2. Install
pip install -r requirements.txt

# 3. Run (model artifacts already included)
python app.py
```

Open http://localhost:8080

### Option 2 — Docker

```bash
docker-compose up --build
```

Open http://localhost:8080

---

## Run Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Retrain the Model

> Requires `cell2celltrain.csv` in the project root (not committed — see dataset note).

```bash
python custom_train.py      # trains model, saves .pkl and .json artifacts
python generate_graphs.py   # regenerates EDA visualisation images
```

---

## API

### `POST /predict`

**Request:**
```json
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
```

**Response:**
```json
{
  "status": "success",
  "prediction": "Yes",
  "probability": "73.40%"
}
```

**Rate limit:** 30 requests/minute per IP.

---

## Project Structure

```
Churn_predicition/
├── .github/workflows/ci.yml   CI/CD pipeline
├── docs/                       Algorithm documentation
├── static/                     CSS, JS, chart images
├── templates/                  HTML pages
├── tests/                      pytest test suite
│   ├── test_model.py           30+ unit tests
│   └── test_api.py             10+ API tests
├── app.py                      Flask API server
├── custom_train.py             ML engine (Decision Tree + Bagging)
├── generate_graphs.py          EDA visualisation generator
├── Dockerfile                  Container definition
├── docker-compose.yml          One-command local run
└── requirements.txt            Pinned dependencies
```

---

## Documentation

Full algorithm documentation, pseudo-code, and architecture diagrams are in the [`/about`](http://localhost:8080/about) page of the running app, or in the `docs/` folder.

---

## Dataset

Cell2Cell Telecom dataset (~51,000 rows, 58 features). Not committed to this repo.
Place `cell2celltrain.csv` in the project root before running `custom_train.py`.
