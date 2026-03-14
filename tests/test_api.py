"""
Integration tests for the Flask API routes in app.py.

Run with:  pytest tests/test_api.py -v

Note: Requires model artifacts (custom_model.pkl etc.) to be present.
      Run custom_train.py first if they are missing.
"""
import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client


# A realistic payload covering the key fields the form submits
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
    "MaritalStatus": "Yes",
}


# ── Home route ────────────────────────────────────────────────────────────────

class TestHomeRoute:
    def test_returns_200(self, client):
        res = client.get('/')
        assert res.status_code == 200

    def test_html_content_type(self, client):
        res = client.get('/')
        assert b'html' in res.content_type.encode() or b'text' in res.data[:50]

    def test_contains_brand(self, client):
        res = client.get('/')
        assert b'Churn' in res.data


# ── About route ───────────────────────────────────────────────────────────────

class TestAboutRoute:
    def test_returns_200(self, client):
        res = client.get('/about')
        assert res.status_code == 200


# ── Docs route ────────────────────────────────────────────────────────────────

class TestDocsRoute:
    def test_allowed_doc(self, client):
        res = client.get('/docs/README.md')
        assert res.status_code in (200, 404)  # 404 ok in test env if file absent

    def test_path_traversal_blocked(self, client):
        res = client.get('/docs/../app.py')
        assert res.status_code == 403

    def test_unknown_file_blocked(self, client):
        res = client.get('/docs/secret.txt')
        assert res.status_code == 403

    def test_env_file_blocked(self, client):
        res = client.get('/docs/.env')
        assert res.status_code == 403


# ── Predict route ─────────────────────────────────────────────────────────────

class TestPredictRoute:
    def test_valid_payload_returns_success(self, client):
        res = client.post('/predict',
                          data=json.dumps(VALID_PAYLOAD),
                          content_type='application/json')
        assert res.status_code == 200
        data = res.get_json()
        assert data['status'] == 'success'

    def test_prediction_is_yes_or_no(self, client):
        res = client.post('/predict',
                          data=json.dumps(VALID_PAYLOAD),
                          content_type='application/json')
        data = res.get_json()
        assert data['prediction'] in ('Yes', 'No')

    def test_probability_has_percent_sign(self, client):
        res = client.post('/predict',
                          data=json.dumps(VALID_PAYLOAD),
                          content_type='application/json')
        data = res.get_json()
        assert '%' in data['probability']

    def test_probability_is_numeric(self, client):
        res = client.post('/predict',
                          data=json.dumps(VALID_PAYLOAD),
                          content_type='application/json')
        data = res.get_json()
        prob = float(data['probability'].replace('%', ''))
        assert 0.0 <= prob <= 100.0

    def test_wrong_content_type_returns_415(self, client):
        res = client.post('/predict',
                          data='not json',
                          content_type='text/plain')
        assert res.status_code == 415

    def test_empty_json_body_returns_400(self, client):
        res = client.post('/predict',
                          data='',
                          content_type='application/json')
        assert res.status_code == 400

    def test_out_of_range_value_returns_422(self, client):
        bad_payload = dict(VALID_PAYLOAD)
        bad_payload['IncomeGroup'] = 999  # valid range is 1-9
        res = client.post('/predict',
                          data=json.dumps(bad_payload),
                          content_type='application/json')
        assert res.status_code == 422
        data = res.get_json()
        assert data['status'] == 'error'
        assert 'errors' in data

    def test_empty_payload_handled_gracefully(self, client):
        res = client.post('/predict',
                          data=json.dumps({}),
                          content_type='application/json')
        # Empty payload: all features default to 0 — should still return a prediction
        assert res.status_code in (200, 422, 400)

    def test_response_has_status_field(self, client):
        res = client.post('/predict',
                          data=json.dumps(VALID_PAYLOAD),
                          content_type='application/json')
        data = res.get_json()
        assert 'status' in data

    def test_security_headers_present(self, client):
        res = client.get('/')
        assert 'X-Content-Type-Options' in res.headers
        assert res.headers['X-Content-Type-Options'] == 'nosniff'
        assert 'X-Frame-Options' in res.headers
