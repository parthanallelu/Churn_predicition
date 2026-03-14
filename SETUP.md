# Setup Guide

## Prerequisites

- Python 3.9 or higher
- pip
- Git
- (Optional) Docker Desktop

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/parthanallelu/Churn_predicition.git
cd Churn_predicition

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# or: venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
python app.py
```

Visit http://localhost:8080

---

## Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/test_model.py --cov=custom_train --cov-report=term-missing
```

---

## Docker Setup

```bash
# Build and run in one command
docker-compose up --build

# Stop
docker-compose down
```

---

## Retrain the Model

> Only needed if you want to update the model with new data.

1. Place `cell2celltrain.csv` in the project root
2. Run:
```bash
python custom_train.py
python generate_graphs.py
```
This overwrites `custom_model.pkl`, `custom_encoders.pkl`, `custom_features.json`, `custom_metrics.json`, and all chart images.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: custom_model.pkl` | Run `python custom_train.py` first |
| `ModuleNotFoundError: flask_limiter` | Run `pip install -r requirements.txt` |
| Port 8080 already in use | Change port in `app.py` last line: `app.run(port=XXXX)` |
| Docker build fails | Make sure Docker Desktop is running |
