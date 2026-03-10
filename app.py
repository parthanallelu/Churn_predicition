from flask import Flask, request, jsonify, render_template
import pickle
import json
from custom_train import CustomLabelEncoder, Node, CustomDecisionTree, CustomBaggingClassifier

app = Flask(__name__)

# Load custom model artifacts
with open('custom_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('custom_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

with open('custom_features.json', 'r') as f:
    features = json.load(f)

with open('custom_metrics.json', 'r') as f:
    metrics = json.load(f)

@app.route('/')
def home():
    return render_template('index.html', metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received Data:", data)
        
        # Build strict array aligned to custom_features
        X_input = []
        for feature in features:
            val = data.get(feature)
            
            # Numeric types
            if feature in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
                try:
                    val = float(val) if val else 0.0
                except (ValueError, TypeError):
                    val = 0.0
                X_input.append(val)
            else:
                # Categorical strings -> pass to custom Label Encoder
                if feature in encoders:
                    encoder = encoders[feature]
                    encoded_val = encoder.classes_.get(str(val), 0)
                    X_input.append(encoded_val)
                else:
                    X_input.append(0) # Emergency fallback
                    
        print("Array mapping:", X_input)

        # Make prediction passing 2D array
        prediction = model.predict([X_input])[0]
        prediction_proba = model.predict_proba([X_input])[0]
        
        # Churn probability (Class index 1)
        churn_prob = prediction_proba[1] * 100
        
        result = "Yes" if prediction == 1 else "No"
        
        return jsonify({
            'prediction': result,
            'probability': f"{churn_prob:.2f}%",
            'status': 'success'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(port=8080, debug=False)
