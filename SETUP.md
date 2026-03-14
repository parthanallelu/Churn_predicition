# 🛠️ Setup & Installation Guide

## 📌 Prerequisites

- **Python**: 3.11 or 3.12 (Recommended).
  - *Note: Python 3.14+ may require manual compilation of `matplotlib` or `xgboost` depending on OS binaries availability.*
- **Dataset**: Ensure `cell2celltrain.csv` (Banking version) is present in the root directory.
- **Virtual Environment**: Recommended to avoid dependency conflicts.

---

## 💻 Local Installation (Windows)

```powershell
# 1. Clone the repository
git clone https://github.com/parthanallelu/Churn_predicition.git
cd Churn_predicition

# 2. Create and activate a stable virtual environment
python -m venv venv_stable
.\venv_stable\Scripts\activate

# 3. Upgrade pip and install requirements
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. (Optional) Run the training pipeline to refresh artifacts
python custom_train.py
python generate_graphs.py

# 5. Start the production server
python app.py
```

Visit: [http://localhost:8080](http://localhost:8080)

---

## 🧪 Running Tests

Validate the custom ML engine and API integrity:

```powershell
# Run all tests with verbosity
pytest tests/ -v

# Run with coverage report
pytest --cov=custom_train --cov-report=term-missing
```

---

## 🐳 Docker Setup

For consistent deployment using containers:

```bash
# Build and launch
docker-compose up --build

# Stop services
docker-compose down
```

---

## 🔄 Retraining Documentation

To update the model with new banking records:
1. Update `cell2celltrain.csv` in the root folder.
2. Run `python custom_train.py`.
3. Run `python generate_graphs.py` to sync the dashboard visuals.
4. Restart `app.py`.

---

## ⚠️ Troubleshooting

| Issue | Resolution |
|---|---|
| `KeyError` in Notebook | Ensure `update_notebook.py` has been run to map legacy Telecom columns to the new Banking schema. |
| `ModuleNotFoundError` | Verify `venv_stable` is active and `pip install -r requirements.txt` completed without errors. |
| `Matplotlib` Build Error | On Windows, ensure C++ Build Tools are installed, or use a Python version < 3.14 to use pre-compiled wheels. |
| Port 8080 Busy | Identify the process using `netstat -ano | findstr :8080` or change the port in the last line of `app.py`. |
| Prediction mismatch | Ensure `custom_model.pkl` and `custom_features.json` come from the same training run. |
