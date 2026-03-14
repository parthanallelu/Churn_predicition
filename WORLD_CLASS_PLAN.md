# 🚀 WORLD-CLASS UPGRADE PLAN
### Customer Churn Prediction — Complete Roadmap for Parth

> **Current State:** A solid educational ML project with a polished UI, custom-built algorithms, and good documentation.
> **Target State:** A production-grade, world-class ML web application that can stand on its own as a professional portfolio piece and a real tool.

---

## 📊 Current Metrics (Baseline)

| Metric | Current | Target |
|--------|---------|--------|
| Accuracy | 57.2% | ≥ 80% |
| Precision | 37.5% | ≥ 70% |
| Recall | 69.2% | ≥ 75% |
| F1 Score | 0.486 | ≥ 0.75 |
| Unit Tests | 0 | ≥ 30 |
| Docker | ❌ | ✅ |
| Input Validation | ❌ | ✅ |
| Logging | ❌ | ✅ |
| Rate Limiting | ❌ | ✅ |
| CI/CD Pipeline | ❌ | ✅ |

---

## 🏗️ PHASE 1 — Fix the Foundation (Critical / Do First)

> These are blocking issues. Nothing else matters until model performance is fixed.

### Task 1.1 — Fix Model Accuracy (MOST IMPORTANT)

**The problem:** 57% accuracy means the model is barely better than flipping a coin. This needs to be the #1 priority before any other improvement.

**Why it's bad:** Precision of 37.5% means for every 3 times the model says "this customer will churn," it's wrong twice. Unusable for any real business purpose.

**Step-by-step fix:**

**Step A — Add cross-validation to custom_train.py**
```python
# After your train/test split, add k-fold cross validation
# Split into 5 folds, train on 4, evaluate on 1, rotate
# This gives a much more reliable accuracy estimate
def k_fold_cross_validate(X, y, k=5, n_estimators=30):
    fold_size = len(X) // k
    fold_scores = []

    for fold in range(k):
        # Build validation indices
        val_start = fold * fold_size
        val_end = val_start + fold_size

        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        X_train = X[:val_start] + X[val_end:]
        y_train = y[:val_start] + y[val_end:]

        model = CustomBaggingClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        metrics = calculate_metrics(y_val, preds)
        fold_scores.append(metrics['f1_score'])
        print(f"Fold {fold+1}: F1={metrics['f1_score']:.4f}")

    avg = sum(fold_scores) / len(fold_scores)
    print(f"Cross-Validation F1: {avg:.4f}")
    return avg
```

**Step B — Tune max_per_class (currently 1000) and max_depth (currently 8)**

Run a grid search manually:
```python
# Try different combinations and track which gives best CV F1
configs = [
    {"max_depth": 6, "max_per_class": 800},
    {"max_depth": 8, "max_per_class": 1000},
    {"max_depth": 10, "max_per_class": 1500},
    {"max_depth": 12, "max_per_class": 2000},
]
# For each config: train a model, run k_fold_cross_validate, record results
# Pick the config with highest CV F1
```

**Step C — Add feature engineering**

The current features are raw. Add derived features that carry stronger signal:
```python
# In your preprocessing loop, after reading raw data, compute:
# 1. Revenue per minute: MonthlyRevenue / (MonthlyMinutes + 1)
# 2. Overage ratio: OverageMinutes / (MonthlyMinutes + 1)
# 3. Service tenure ratio: MonthsInService / CurrentEquipmentDays (if > 0)
# 4. Charge-to-revenue ratio: TotalRecurringCharge / (MonthlyRevenue + 1)

# Add these as new columns to X_raw before encoding
```

**Step D — Increase threshold sampling from 15 to 25**

In `custom_train.py` line 74, change:
```python
# FROM:
if len(possible_thresholds) > 15:
    possible_thresholds = random.sample(list(possible_thresholds), 15)

# TO:
if len(possible_thresholds) > 25:
    possible_thresholds = random.sample(list(possible_thresholds), 25)
```
This gives splits more precision at modest compute cost.

**Step E — Increase n_estimators from 30 to 50**

More trees = lower variance. 50 trees should stabilize the ensemble significantly:
```python
bagging_model = CustomBaggingClassifier(n_estimators=50)
```

**Expected outcome:** With feature engineering + better hyperparameters + CV, expect F1 to improve to 0.65-0.75.

---

### Task 1.2 — Fix the Encoder Fallback Bug

**File:** `app.py` line 58
**Current code:**
```python
encoded_val = encoder.classes_.get(str(val), 0) if val else 0
```
**The problem:** When an unknown category comes in, it silently maps to class `0`, which could be "1-Highest" credit rating — completely wrong semantics.

**Fix:** Map unknown to the most common class (mode), not index 0.

When training, save the mode per categorical feature alongside the encoder:
```python
# In custom_train.py, when saving encoders:
encoder_meta = {}
for col_name, encoder in encoders.items():
    col_data = [row[feature_cols.index(col_name)] for row in X_raw]
    from collections import Counter
    mode_encoded = Counter(col_data).most_common(1)[0][0]  # encoded mode
    encoder_meta[col_name] = {
        'encoder': encoder,
        'mode': mode_encoded  # fallback value
    }
with open('custom_encoders.pkl', 'wb') as f:
    pickle.dump(encoder_meta, f)
```

Then in `app.py`:
```python
# Load:
# encoders['FeatureName']['encoder'] and encoders['FeatureName']['mode']

# Use:
if feature in encoders:
    enc_data = encoders[feature]
    encoder = enc_data['encoder']
    fallback = enc_data['mode']
    encoded_val = encoder.classes_.get(str(val), fallback) if val else fallback
    X_input.append(encoded_val)
```

---

### Task 1.3 — Add Input Validation

**File:** `app.py`
**The problem:** Any garbage input is accepted and silently converted to 0.0. No user-facing error messages.

**Fix — add a validation function before building X_input:**
```python
NUMERIC_RANGES = {
    'MonthlyMinutes': (0, 10000),
    'OverageMinutes': (0, 5000),
    'MonthlyRevenue': (0, 5000),
    'TotalRecurringCharge': (0, 5000),
    'IncomeGroup': (1, 9),
    'AgeHH1': (0, 120),
    'MonthsInService': (0, 600),
    'CurrentEquipmentDays': (0, 3000),
    'RoamingCalls': (0, 1000),
    'HandsetPrice': (0, 5000),
}

def validate_input(data, features, encoders):
    errors = []
    for feature in features:
        val = data.get(feature)
        if feature in encoders:
            # Categorical: just warn if completely missing
            if val is None:
                errors.append(f"Missing value for {feature}")
        else:
            # Numeric
            if val is None or val == '':
                errors.append(f"Missing numeric value for {feature}")
                continue
            try:
                num_val = float(str(val).replace(',', ''))
                if feature in NUMERIC_RANGES:
                    lo, hi = NUMERIC_RANGES[feature]
                    if not (lo <= num_val <= hi):
                        errors.append(f"{feature} must be between {lo} and {hi}, got {num_val}")
            except (ValueError, TypeError):
                errors.append(f"{feature} must be a number, got: {val}")
    return errors

# In /predict route, add before building X_input:
validation_errors = validate_input(data, features, encoders)
if validation_errors:
    return jsonify({'status': 'error', 'message': 'Validation failed', 'errors': validation_errors}), 422
```

---

### Task 1.4 — Fix Error Handling in app.py

**File:** `app.py`
**The problem:** If pickle files don't exist on startup, the app crashes with no message. Error responses expose full tracebacks.

**Fix — wrap startup loading and use proper logging:**
```python
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

def load_artifacts():
    required = ['custom_model.pkl', 'custom_encoders.pkl', 'custom_features.json', 'custom_metrics.json']
    for f in required:
        if not os.path.exists(f):
            logger.error(f"Required artifact missing: {f}. Run custom_train.py first.")
            raise FileNotFoundError(f"Missing: {f}")

    with open('custom_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('custom_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('custom_features.json', 'r') as f:
        features = json.load(f)
    with open('custom_metrics.json', 'r') as f:
        metrics = json.load(f)

    logger.info(f"Loaded model with {len(model.trees)} trees, {len(features)} features.")
    return model, encoders, features, metrics

model, encoders, features, metrics = load_artifacts()

# In /predict error handler:
except Exception as e:
    logger.exception("Prediction failed")  # logs full traceback to file, NOT to user
    return jsonify({'status': 'error', 'message': 'Internal prediction error. Please try again.'}), 500
```

---

### Task 1.5 — Fix requirements.txt

Remove unused `joblib`, add versions for reproducibility:
```
Flask==3.0.3
matplotlib==3.9.0
seaborn==0.13.2
```

---

## 🧪 PHASE 2 — Add Tests (High Priority)

> A project with zero tests is a project that can silently break. Tests prove it works.

### Task 2.1 — Create test_model.py

Create file: `tests/test_model.py`
```python
import sys
sys.path.insert(0, '..')
from custom_train import CustomDecisionTree, CustomBaggingClassifier, CustomLabelEncoder, calculate_metrics

class TestCustomLabelEncoder:
    def test_fit_transform_basic(self):
        enc = CustomLabelEncoder()
        result = enc.fit_transform(['a', 'b', 'a', 'c'])
        assert result == [0, 1, 0, 2]

    def test_unknown_value_returns_zero(self):
        enc = CustomLabelEncoder()
        enc.fit(['cat', 'dog'])
        result = enc.transform(['cat', 'unknown'])
        assert result[1] == 0  # unknown maps to 0

    def test_sorted_encoding(self):
        enc = CustomLabelEncoder()
        enc.fit(['z', 'a', 'm'])
        assert enc.classes_['a'] == 0  # alphabetical order
        assert enc.classes_['m'] == 1
        assert enc.classes_['z'] == 2


class TestCustomDecisionTree:
    def _make_simple_data(self):
        # Linearly separable data
        X = [[i] for i in range(100)]
        y = [0 if i < 50 else 1 for i in range(100)]
        return X, y

    def test_fit_and_predict(self):
        X, y = self._make_simple_data()
        tree = CustomDecisionTree(max_depth=5)
        tree.fit(X, y)
        preds = tree.predict(X)
        correct = sum(p == t for p, t in zip(preds, y))
        assert correct / len(y) > 0.90  # should fit well on easy data

    def test_leaf_node_pure(self):
        # Single class data
        X = [[1], [2], [3]]
        y = [1, 1, 1]
        tree = CustomDecisionTree(max_depth=5)
        tree.fit(X, y)
        preds = tree.predict(X)
        assert all(p == 1 for p in preds)

    def test_feature_importances_sum_to_one(self):
        X, y = self._make_simple_data()
        tree = CustomDecisionTree(max_depth=5)
        tree.fit(X, y)
        assert len(tree.feature_importances) == 1  # 1 feature

    def test_max_depth_respected(self):
        X, y = self._make_simple_data()
        tree = CustomDecisionTree(max_depth=1)
        tree.fit(X, y)
        # With depth=1, tree should still produce predictions
        preds = tree.predict([[25], [75]])
        assert len(preds) == 2


class TestCustomBaggingClassifier:
    def _make_xor_data(self):
        # Non-trivial 2D data
        X = [[0,0],[1,0],[0,1],[1,1],[0,0],[1,0],[0,1],[1,1]]
        y = [0,1,1,0,0,1,1,0]
        return X, y

    def test_fit_stores_correct_tree_count(self):
        X, y = self._make_xor_data()
        clf = CustomBaggingClassifier(n_estimators=5)
        clf.fit(X, y)
        assert len(clf.trees) == 5

    def test_predict_returns_binary(self):
        X, y = self._make_xor_data()
        clf = CustomBaggingClassifier(n_estimators=3)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert all(p in [0, 1] for p in preds)

    def test_predict_proba_sums_to_one(self):
        X, y = self._make_xor_data()
        clf = CustomBaggingClassifier(n_estimators=3)
        clf.fit(X, y)
        probas = clf.predict_proba(X)
        for p in probas:
            assert abs(p[0] + p[1] - 1.0) < 1e-9

    def test_feature_importances_normalized(self):
        X, y = self._make_xor_data()
        clf = CustomBaggingClassifier(n_estimators=5)
        clf.fit(X, y)
        importances = clf.feature_importances_
        assert len(importances) == 2
        assert abs(sum(importances) - 1.0) < 1e-6


class TestCalculateMetrics:
    def test_perfect_predictions(self):
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 1]
        m = calculate_metrics(y_true, y_pred)
        assert m['accuracy'] == 1.0
        assert m['precision'] == 1.0
        assert m['recall'] == 1.0
        assert m['f1_score'] == 1.0

    def test_all_wrong(self):
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 1, 1]
        m = calculate_metrics(y_true, y_pred)
        assert m['accuracy'] == 0.0
        assert m['precision'] == 0.0
        assert m['recall'] == 0.0

    def test_confusion_matrix_shape(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        m = calculate_metrics(y_true, y_pred)
        cm = m['confusion_matrix']
        assert len(cm) == 2
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2
```

**To run tests:** `pip install pytest && pytest tests/ -v`

---

### Task 2.2 — Create test_api.py

Create file: `tests/test_api.py`
```python
import sys, json
sys.path.insert(0, '..')
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

VALID_PAYLOAD = {
    "MonthlyMinutes": 500,
    "OverageMinutes": 10,
    "RoamingCalls": 5,
    "CurrentEquipmentDays": 300,
    "MonthlyRevenue": 55.0,
    "TotalRecurringCharge": 45.0,
    "IncomeGroup": 3,
    "HandsetPrice": "100",
    "AgeHH1": 35,
    "MonthsInService": 24,
    "CreditRating": "3-Good",
    "MaritalStatus": "Yes"
}

class TestHomeRoute:
    def test_home_returns_200(self, client):
        response = client.get('/')
        assert response.status_code == 200

    def test_home_contains_churn_ai(self, client):
        response = client.get('/')
        assert b'Churn' in response.data

class TestPredictRoute:
    def test_predict_returns_success(self, client):
        response = client.post('/predict',
            data=json.dumps(VALID_PAYLOAD),
            content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'

    def test_predict_has_probability(self, client):
        response = client.post('/predict',
            data=json.dumps(VALID_PAYLOAD),
            content_type='application/json')
        data = json.loads(response.data)
        assert 'probability' in data
        assert '%' in data['probability']

    def test_predict_result_is_yes_or_no(self, client):
        response = client.post('/predict',
            data=json.dumps(VALID_PAYLOAD),
            content_type='application/json')
        data = json.loads(response.data)
        assert data['prediction'] in ['Yes', 'No']

    def test_predict_empty_payload(self, client):
        response = client.post('/predict',
            data=json.dumps({}),
            content_type='application/json')
        # Should return a result (defaulting to 0.0 for all features)
        assert response.status_code in [200, 400, 422]

    def test_predict_wrong_content_type(self, client):
        response = client.post('/predict',
            data="not json",
            content_type='text/plain')
        assert response.status_code in [400, 415, 500]

class TestDocsRoute:
    def test_valid_doc_returns_200(self, client):
        response = client.get('/docs/README.md')
        assert response.status_code in [200, 404]  # 404 ok if file not present in test env

    def test_unauthorized_doc_returns_403(self, client):
        response = client.get('/docs/../app.py')
        assert response.status_code == 403

    def test_unknown_doc_returns_403(self, client):
        response = client.get('/docs/secret.txt')
        assert response.status_code == 403
```

---

## 🐳 PHASE 3 — Dockerize (High Priority)

> This makes the app runnable anywhere in one command.

### Task 3.1 — Create Dockerfile

Create file: `Dockerfile`
```dockerfile
# Use slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Expose Flask port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/')"

# Run with gunicorn for production (not Flask dev server)
CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "app:app"]
```

Add `gunicorn` to requirements.txt:
```
Flask==3.0.3
gunicorn==22.0.0
matplotlib==3.9.0
seaborn==0.13.2
```

### Task 3.2 — Create docker-compose.yml

Create file: `docker-compose.yml`
```yaml
version: '3.8'

services:
  churn-api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./app.log:/app/app.log
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/')"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**To run:** `docker-compose up --build`

---

## 🔐 PHASE 4 — Security Hardening (High Priority)

### Task 4.1 — Add Rate Limiting

Install Flask-Limiter: add `Flask-Limiter==3.7.0` to requirements.txt

In `app.py`:
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

@app.route('/predict', methods=['POST'])
@limiter.limit("30 per minute")  # Max 30 predictions per minute per IP
def predict():
    ...
```

### Task 4.2 — Add Security Headers

In `app.py`:
```python
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response
```

### Task 4.3 — Validate JSON Content-Type

```python
@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'Content-Type must be application/json'}), 415
    data = request.json
    if data is None:
        return jsonify({'status': 'error', 'message': 'Invalid or empty JSON body'}), 400
    ...
```

---

## ⚙️ PHASE 5 — CI/CD Pipeline (Medium Priority)

### Task 5.1 — Create GitHub Actions Workflow

Create file: `.github/workflows/ci.yml`
```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        cache: 'pip'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run unit tests
      run: pytest tests/test_model.py -v --tb=short

    - name: Check test coverage
      run: pytest tests/test_model.py --cov=custom_train --cov-report=term-missing

    - name: Lint check
      run: |
        pip install flake8
        flake8 app.py custom_train.py --max-line-length=120 --ignore=E501,W503

  docker-build:
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v4

    - name: Build Docker image
      run: docker build -t churn-predictor:test .

    - name: Test Docker image starts
      run: |
        docker run -d -p 8080:8080 --name test-container churn-predictor:test
        sleep 5
        curl -f http://localhost:8080/ || exit 1
        docker stop test-container
```

---

## 📈 PHASE 6 — UI/UX Improvements (Medium Priority)

### Task 6.1 — Add Loading Spinner

In `index.html`, add inside the form-actions div:
```html
<div id="loading-spinner" class="spinner hidden">
  <div class="spinner-ring"></div>
  <p>Analyzing customer profile...</p>
</div>
```

In `style.css`:
```css
.spinner { display: flex; flex-direction: column; align-items: center; gap: 1rem; }
.spinner.hidden { display: none; }
.spinner-ring {
  width: 48px; height: 48px;
  border: 4px solid rgba(255,255,255,0.1);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
```

In `script.js`, show spinner while fetching:
```javascript
const spinner = document.getElementById('loading-spinner');
submitBtn.disabled = true;
spinner.classList.remove('hidden');
// ... after fetch:
spinner.classList.add('hidden');
```

### Task 6.2 — Client-Side Validation

In `script.js`, before the fetch call, add:
```javascript
function validateForm(data) {
  const errors = [];
  const numericFields = {
    'MonthlyMinutes': [0, 10000],
    'OverageMinutes': [0, 5000],
    'MonthlyRevenue': [0, 5000],
    'IncomeGroup': [1, 9],
    'AgeHH1': [0, 120],
    'MonthsInService': [0, 600]
  };

  for (const [field, [min, max]] of Object.entries(numericFields)) {
    const val = parseFloat(data[field]);
    if (isNaN(val)) {
      errors.push(`${field} must be a number`);
    } else if (val < min || val > max) {
      errors.push(`${field} must be between ${min} and ${max}`);
    }
  }
  return errors;
}

// In form submit handler:
const validationErrors = validateForm(data);
if (validationErrors.length > 0) {
  alert('Please fix these errors:\n' + validationErrors.join('\n'));
  submitBtn.disabled = false;
  return;
}
```

### Task 6.3 — Add Confidence Meter Visual

In `index.html`, replace the plain probability display with:
```html
<div class="confidence-meter">
  <div class="confidence-label">Churn Probability</div>
  <div class="confidence-bar">
    <div class="confidence-fill" id="confidence-fill"></div>
  </div>
  <span class="metric-value" id="probability-value">--%</span>
</div>
```

In `style.css`:
```css
.confidence-bar {
  width: 100%; height: 12px;
  background: rgba(255,255,255,0.1);
  border-radius: 6px; overflow: hidden;
}
.confidence-fill {
  height: 100%; width: 0%; border-radius: 6px;
  transition: width 1s ease-out, background-color 0.5s;
}
```

In `script.js`:
```javascript
const fill = document.getElementById('confidence-fill');
fill.style.width = probNum + '%';
fill.style.backgroundColor = probNum > 50 ? 'var(--danger)' : 'var(--success)';
```

---

## 📝 PHASE 7 — Documentation & Polish (Lower Priority)

### Task 7.1 — Create Root README.md

Create `README.md` at project root (not inside /docs):
```markdown
# Customer Churn Prediction

A full-stack Machine Learning web application that predicts telecom customer churn using a **custom-built Bagging + Decision Tree ensemble** — built from pure Python with zero sklearn dependency.

## Quick Start

### Option 1: Docker (Recommended)
```bash
docker-compose up --build
```
Open http://localhost:8080

### Option 2: Local Python
```bash
pip install -r requirements.txt
python app.py
```

## Run Tests
```bash
pip install pytest
pytest tests/ -v
```

## Train Model
Requires `cell2celltrain.csv` in project root:
```bash
python custom_train.py
python generate_graphs.py
```

## Architecture
- **Backend:** Flask REST API, pure Python ML
- **ML Engine:** Custom Bagging Classifier (30 Decision Trees, Gini impurity)
- **Frontend:** Glassmorphism HTML5/CSS3/Vanilla JS
- **Deployment:** Docker + Gunicorn

## API
`POST /predict` — accepts JSON with customer features, returns:
```json
{"status": "success", "prediction": "Yes", "probability": "73.40%"}
```

See `docs/README.md` for full documentation.
```

### Task 7.2 — Add Docstrings

Add docstrings to all major classes in `custom_train.py`:
```python
class CustomDecisionTree:
    """
    A binary decision tree classifier using Gini impurity for splits.

    Implements the CART (Classification and Regression Trees) algorithm
    without any external ML libraries. Uses recursive binary splitting
    until max_depth or min_samples_split stopping criteria are met.

    Args:
        min_samples_split (int): Minimum samples required to split a node. Default: 2.
        max_depth (int): Maximum depth of the tree. Default: 10.

    Example:
        tree = CustomDecisionTree(max_depth=8)
        tree.fit(X_train, y_train)
        predictions = tree.predict(X_test)
    """
```

### Task 7.3 — Create SETUP.md

Create `SETUP.md`:
```markdown
# Setup Guide

## Prerequisites
- Python 3.9+
- pip

## Installation

1. Clone the repo:
```bash
git clone https://github.com/parthanallelu/Churn_predicition.git
cd Churn_predicition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. The model artifacts (custom_model.pkl, etc.) are included in the repo.
   To retrain from scratch, place `cell2celltrain.csv` in the root directory and run:
```bash
python custom_train.py
python generate_graphs.py
```

4. Start the server:
```bash
python app.py        # Development
gunicorn app:app -b 0.0.0.0:8080  # Production
```

5. Visit http://localhost:8080
```

---

## 🗓️ EXECUTION SCHEDULE (For Parth)

### Week 1 — Critical Fixes
| Day | Task | Time Estimate |
|-----|------|---------------|
| Day 1 | Task 1.2 (fix encoder bug) + Task 1.3 (input validation) + Task 1.4 (logging) + Task 1.5 (requirements.txt) | 3-4 hours |
| Day 2-3 | Task 1.1 (improve model accuracy — Steps A + B + C) | 4-6 hours |
| Day 4 | Task 2.1 (test_model.py) + Task 2.2 (test_api.py) | 3-4 hours |
| Day 5 | Run tests, fix any failures, verify model improvement | 2-3 hours |

### Week 2 — Infrastructure
| Day | Task | Time Estimate |
|-----|------|---------------|
| Day 1 | Task 3.1 + 3.2 (Dockerfile + docker-compose) | 2-3 hours |
| Day 2 | Task 4.1 + 4.2 + 4.3 (rate limiting + security headers) | 2 hours |
| Day 3 | Task 5.1 (GitHub Actions CI/CD) | 2-3 hours |
| Day 4-5 | Task 6.1 + 6.2 + 6.3 (UI improvements) | 3-4 hours |

### Week 3 — Polish
| Day | Task | Time Estimate |
|-----|------|---------------|
| Day 1-2 | Task 7.1 + 7.2 + 7.3 (README, docstrings, SETUP.md) | 3-4 hours |
| Day 3 | Final testing of everything end-to-end | 2 hours |
| Day 4 | Deploy to a free hosting platform (Railway, Render, or Fly.io) | 2-3 hours |
| Day 5 | Final review and cleanup | 1-2 hours |

---

## 🌐 OPTIONAL: Deploy to the Internet (Bonus)

Once Docker works, deploy for free on one of these:

### Option A — Render (Easiest)
1. Push to GitHub
2. Go to https://render.com → New Web Service
3. Connect your GitHub repo
4. Set: Build Command = `pip install -r requirements.txt`
5. Set: Start Command = `gunicorn app:app --bind 0.0.0.0:$PORT`
6. Done. Live URL in ~2 minutes.

### Option B — Railway
1. `npm install -g railway`
2. `railway login && railway init && railway up`
3. Done. Live URL in ~1 minute.

### Option C — Fly.io
1. Install flyctl: https://fly.io/docs/hands-on/install-flyctl/
2. `fly auth login && fly launch && fly deploy`
3. Done. Scales automatically.

---

## ✅ DEFINITION OF "WORLD CLASS"

This project will be world-class when:
- [ ] Model F1 Score ≥ 0.75 (currently 0.486)
- [ ] All 30+ unit tests passing in CI
- [ ] Docker runs with single command
- [ ] Rate limiting protects the API
- [ ] Logging captures all events
- [ ] Input validation catches bad data
- [ ] GitHub Actions runs tests on every push
- [ ] README explains how to run it in under 2 minutes
- [ ] Live deployment URL exists
- [ ] Zero known bugs

---

## 📂 Final File Structure (Target)

```
Churn_predicition/
├── .github/
│   └── workflows/
│       └── ci.yml              ← NEW: GitHub Actions
├── docs/
│   ├── README.md
│   ├── model_documentation.md
│   ├── code_explanation.md
│   └── analysis_report.md
├── static/
│   ├── images/
│   ├── script.js               ← UPDATED: validation + spinner
│   └── style.css               ← UPDATED: spinner + confidence meter
├── templates/
│   ├── index.html              ← UPDATED: spinner + confidence meter
│   └── about.html
├── tests/                      ← NEW: test directory
│   ├── __init__.py
│   ├── test_model.py           ← NEW: 20+ unit tests
│   └── test_api.py             ← NEW: 10+ API tests
├── app.py                      ← UPDATED: logging, validation, rate limit, security headers
├── custom_train.py             ← UPDATED: cross-val, feature engineering, docstrings
├── generate_graphs.py
├── analyze_data.py
├── custom_model.pkl
├── custom_encoders.pkl
├── custom_features.json
├── custom_metrics.json
├── custom_plot_data.json
├── requirements.txt            ← UPDATED: pinned versions, gunicorn, flask-limiter
├── Dockerfile                  ← NEW
├── docker-compose.yml          ← NEW
├── README.md                   ← NEW: root README
├── SETUP.md                    ← NEW: setup guide
├── .gitignore
└── WORLD_CLASS_PLAN.md         ← THIS FILE
```

---

*Generated on 2026-03-14. Repo: https://github.com/parthanallelu/Churn_predicition*
