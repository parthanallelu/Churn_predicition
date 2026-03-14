import os
import json
import pickle
import logging
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from custom_train import CustomLabelEncoder, Node, CustomDecisionTree, CustomBaggingClassifier

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Rate limiting ─────────────────────────────────────────────────────────────
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# ── Load model artifacts ──────────────────────────────────────────────────────
def load_artifacts():
    required = ['custom_model.pkl', 'custom_encoders.pkl', 'custom_features.json', 'custom_metrics.json']
    for fname in required:
        if not os.path.exists(fname):
            logger.error(f"Required artifact missing: {fname}. Run custom_train.py first.")
            raise FileNotFoundError(f"Missing required file: {fname}")

    with open('custom_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('custom_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('custom_features.json', 'r') as f:
        features = json.load(f)
    with open('custom_metrics.json', 'r') as f:
        metrics = json.load(f)

    logger.info(f"Model loaded: {len(model.trees)} trees, {len(features)} features.")
    return model, encoders, features, metrics

model, encoders, features, metrics = load_artifacts()

# ── Input validation config ───────────────────────────────────────────────────
NUMERIC_RANGES = {
    'MonthlyMinutes':        (0, 10000),
    'OverageMinutes':        (0, 5000),
    'MonthlyRevenue':        (0, 5000),
    'TotalRecurringCharge':  (0, 5000),
    'IncomeGroup':           (1, 9),
    'AgeHH1':                (0, 120),
    'MonthsInService':       (0, 600),
    'CurrentEquipmentDays':  (0, 3000),
    'RoamingCalls':          (0, 1000),
    'HandsetPrice':          (0, 5000),
}

def validate_input(data):
    errors = []
    for feature in features:
        val = data.get(feature)
        if feature not in encoders:
            # Numeric field
            if val is None or val == '':
                continue  # will default to 0.0, not a hard error
            try:
                num_val = float(str(val).replace(',', '').strip())
                if feature in NUMERIC_RANGES:
                    lo, hi = NUMERIC_RANGES[feature]
                    if not (lo <= num_val <= hi):
                        errors.append(f"{feature} must be between {lo} and {hi} (got {num_val})")
            except (ValueError, TypeError):
                errors.append(f"{feature} must be a number (got: {val})")
    return errors

# ── Security headers ──────────────────────────────────────────────────────────
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html', metrics=metrics)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/docs/<filename>')
def serve_doc(filename):
    allowed_docs = [
        'README.md', 'model_documentation.md',
        'code_explanation.md', 'analysis_report.md', 'internal_model_review.md'
    ]
    if filename not in allowed_docs:
        return "Unauthorized access.", 403
    try:
        filepath = os.path.join('docs', filename)
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "File not generated yet.", 404

@app.route('/predict', methods=['POST'])
@limiter.limit("30 per minute")
def predict():
    # Enforce JSON content type
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'Content-Type must be application/json'}), 415

    data = request.json
    if data is None:
        return jsonify({'status': 'error', 'message': 'Empty or invalid JSON body'}), 400

    logger.info(f"Prediction request received for {len(data)} fields.")

    # Validate input ranges
    validation_errors = validate_input(data)
    if validation_errors:
        logger.warning(f"Validation failed: {validation_errors}")
        return jsonify({'status': 'error', 'message': 'Validation failed', 'errors': validation_errors}), 422

    try:
        X_input = []
        for feature in features:
            val = data.get(feature)

            if feature in encoders:
                # Categorical — use mode as fallback instead of index 0
                encoder = encoders[feature]
                if val and str(val) in encoder.classes_:
                    encoded_val = encoder.classes_[str(val)]
                else:
                    # Fall back to mode (most common class index)
                    encoded_val = max(encoder.classes_.values(),
                                      key=list(encoder.classes_.values()).count)
                X_input.append(encoded_val)
            else:
                # Numeric
                try:
                    if isinstance(val, str):
                        val = val.replace(',', '').strip()
                    X_input.append(float(val) if val not in (None, '') else 0.0)
                except (ValueError, TypeError):
                    X_input.append(0.0)

        prediction = model.predict([X_input])[0]
        prediction_proba = model.predict_proba([X_input])[0]
        churn_prob = prediction_proba[1] * 100
        result = "Yes" if prediction == 1 else "No"

        logger.info(f"Prediction: {result} ({churn_prob:.2f}%)")
        return jsonify({
            'prediction': result,
            'probability': f"{churn_prob:.2f}%",
            'status': 'success'
        })

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({'status': 'error', 'message': 'Internal prediction error. Please try again.'}), 500

if __name__ == '__main__':
    app.run(port=8080, debug=False)
