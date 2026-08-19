"""
Comprehensive Test Suite for Step 16:
Generic Prediction API (POST /predictions).

Tests:
1. Valid authenticated request (Diabetes) -> 200 OK & valid PredictionResponse
2. Valid authenticated request (Heart Disease) -> 200 OK & valid PredictionResponse
3. Missing JWT -> 401/403 Rejected
4. Invalid JWT -> 401 Rejected
5. Unknown disease ID -> 404 Not Found
6. Missing required feature data -> 422 Unprocessable Entity
7. Invalid feature types (e.g. non-numeric string for float) -> 422 Unprocessable Entity
8. Category mismatch (Tabular request to Image disease) -> 422 Unprocessable Entity
9. List active disease models (GET /predictions/diseases) -> 200 OK
10. Architectural symmetry: One single endpoint handles multiple distinct diseases
"""

import sys
from pathlib import Path
import pytest

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mongomock
from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.database.connection import get_database
from app.main import app
from app.services.user_service import UserService


# Mock database for user auth verification
mock_client = mongomock.MongoClient()
mock_db = mock_client["test_medical_api_db"]
app.dependency_overrides[get_database] = lambda: mock_db


@pytest.fixture(scope="module")
def api_client():
    return TestClient(app)


@pytest.fixture(scope="module")
def auth_headers():
    user_service = UserService(mock_db)
    user = user_service.create_user(
        name="Test Patient",
        email="patient@example.com",
        password_hash="mock_hash_patient"
    )
    user_id = str(user["_id"])
    token = create_access_token(user_id)
    return {"Authorization": f"Bearer {token}"}


def test_authenticated_diabetes_prediction(api_client, auth_headers):
    """1. Test valid authenticated prediction for Diabetes model."""
    payload = {
        "disease_id": "diabetes",
        "inputs": {
            "Pregnancies": 2,
            "Glucose": 160.0,
            "BloodPressure": 80.0,
            "BMI": 33.5,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50,
        }
    }
    response = api_client.post("/predictions", json=payload, headers=auth_headers)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert data["disease_id"] == "diabetes"
    assert data["disease_display_name"] == "Diabetes Risk Assessment"
    assert "prediction_label" in data
    assert "is_positive" in data
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0
    assert data["decision_threshold"] == 0.4
    assert data["model_version"] == "v1"
    assert "metadata" in data
    assert "disclaimer" in data
    assert "timestamp" in data
    print(f"  PASS: 1. Authenticated Diabetes prediction: prob={data['probability']}, label='{data['prediction_label']}'")


def test_authenticated_heart_disease_prediction(api_client, auth_headers):
    """2. Test valid authenticated prediction for Heart Disease model."""
    payload = {
        "disease_id": "heart_disease",
        "inputs": {
            "age": 58,
            "sex": 1,
            "chest_pain_type": 2,
            "resting_bp": 140.0,
            "cholestoral": 240.0,
            "fasting_blood_sugar": 0,
            "restecg": 1,
            "max_hr": 160.0,
            "exang": 0,
            "oldpeak": 1.4,
            "slope": 2,
            "num_major_vessels": 0,
            "thal": 2,
        }
    }
    response = api_client.post("/predictions", json=payload, headers=auth_headers)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert data["disease_id"] == "heart_disease"
    assert data["disease_display_name"] == "Heart Disease Risk Assessment"
    assert isinstance(data["probability"], float)
    print(f"  PASS: 2. Authenticated Heart Disease prediction: prob={data['probability']}, label='{data['prediction_label']}'")


def test_missing_jwt_rejected(api_client):
    """3. Test prediction endpoint rejects requests without Authorization header."""
    payload = {
        "disease_id": "diabetes",
        "inputs": {"Glucose": 100}
    }
    response = api_client.post("/predictions", json=payload)
    assert response.status_code in [401, 403]
    print("  PASS: 3. Missing JWT rejected with 401/403")


def test_invalid_jwt_rejected(api_client):
    """4. Test prediction endpoint rejects invalid JWT token."""
    payload = {
        "disease_id": "diabetes",
        "inputs": {"Glucose": 100}
    }
    bad_headers = {"Authorization": "Bearer invalid_token_xyz_123"}
    response = api_client.post("/predictions", json=payload, headers=bad_headers)
    assert response.status_code == 401
    print("  PASS: 4. Invalid JWT rejected with 401")


def test_unknown_disease_id_returns_404(api_client, auth_headers):
    """5. Test prediction endpoint returns 404 for unknown disease."""
    payload = {
        "disease_id": "non_existent_disease_abc",
        "inputs": {"x": 1}
    }
    response = api_client.post("/predictions", json=payload, headers=auth_headers)
    assert response.status_code == 404
    assert "not registered" in response.json()["detail"]
    print("  PASS: 5. Unknown disease ID returns 404 Not Found")


def test_missing_required_features_returns_422(api_client, auth_headers):
    """6. Test prediction endpoint returns 422 for missing required features."""
    payload = {
        "disease_id": "diabetes",
        "inputs": {"Glucose": 140}  # Missing other 5 required features
    }
    response = api_client.post("/predictions", json=payload, headers=auth_headers)
    assert response.status_code == 422
    assert "Validation failed" in response.json()["detail"] or "missing" in response.json()["detail"].lower()
    print("  PASS: 6. Missing required features returns 422 Unprocessable Entity")


def test_invalid_feature_types_returns_422(api_client, auth_headers):
    """7. Test prediction endpoint returns 422 for invalid feature types."""
    payload = {
        "disease_id": "diabetes",
        "inputs": {
            "Pregnancies": "invalid_int_string",
            "Glucose": 160.0,
            "BloodPressure": 80.0,
            "BMI": 33.5,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50,
        }
    }
    response = api_client.post("/predictions", json=payload, headers=auth_headers)
    assert response.status_code == 422
    print("  PASS: 7. Invalid feature types return 422 Unprocessable Entity")


def test_category_mismatch_returns_422(api_client, auth_headers):
    """8. Test tabular request to image disease (Pneumonia) returns 422."""
    payload = {
        "disease_id": "pneumonia",
        "inputs": {"pixel_val": 100}
    }
    response = api_client.post("/predictions", json=payload, headers=auth_headers)
    assert response.status_code == 422
    assert "image" in response.json()["detail"].lower()
    print("  PASS: 8. Tabular request to Image model returns 422 Unprocessable Entity")


def test_list_active_diseases_endpoint(api_client, auth_headers):
    """9. Test GET /predictions/diseases returns public metadata."""
    response = api_client.get("/predictions/diseases", headers=auth_headers)
    assert response.status_code == 200

    diseases = response.json()
    assert isinstance(diseases, list)
    assert len(diseases) >= 3

    disease_ids = {d["id"] for d in diseases}
    assert "diabetes" in disease_ids
    assert "heart_disease" in disease_ids
    assert "pneumonia" in disease_ids

    # Validate structure
    first_disease = diseases[0]
    assert "display_name" in first_disease
    assert "category" in first_disease
    assert "clinical_purpose" in first_disease
    print("  PASS: 9. GET /predictions/diseases returns registered models metadata")


def test_architectural_symmetry_single_endpoint(api_client, auth_headers):
    """10. Verify single endpoint handles multiple distinct diseases uniformly."""
    diabetes_payload = {
        "disease_id": "diabetes",
        "inputs": {
            "Pregnancies": 1,
            "Glucose": 105.0,
            "BloodPressure": 75.0,
            "BMI": 23.5,
            "DiabetesPedigreeFunction": 0.25,
            "Age": 30,
        }
    }
    heart_payload = {
        "disease_id": "heart_disease",
        "inputs": {
            "age": 45,
            "sex": 0,
            "chest_pain_type": 1,
            "resting_bp": 120.0,
            "cholestoral": 200.0,
            "fasting_blood_sugar": 0,
            "restecg": 0,
            "max_hr": 175.0,
            "exang": 0,
            "oldpeak": 0.0,
            "slope": 1,
            "num_major_vessels": 0,
            "thal": 2,
        }
    }

    res_d = api_client.post("/predictions", json=diabetes_payload, headers=auth_headers)
    res_h = api_client.post("/predictions", json=heart_payload, headers=auth_headers)

    assert res_d.status_code == 200
    assert res_h.status_code == 200
    assert res_d.json()["disease_id"] == "diabetes"
    assert res_h.json()["disease_id"] == "heart_disease"
    assert res_d.json().keys() == res_h.json().keys(), "Response schemas must have identical keys"

    print("  PASS: 10. Single generic endpoint dynamically executes both models with identical schema symmetry")


if __name__ == "__main__":
    print("\n=== Generic Prediction API Verification Tests ===\n")
    client = TestClient(app)
    
    # Generate mock headers
    u_svc = UserService(mock_db)
    u = u_svc.create_user(name="P", email="p@ex.com", password_hash="h")
    headers = {"Authorization": f"Bearer {create_access_token(str(u['_id']))}"}

    tests = [
        ("1. Authenticated Diabetes Prediction", lambda: test_authenticated_diabetes_prediction(client, headers)),
        ("2. Authenticated Heart Disease Prediction", lambda: test_authenticated_heart_disease_prediction(client, headers)),
        ("3. Missing JWT Rejection", lambda: test_missing_jwt_rejected(client)),
        ("4. Invalid JWT Rejection", lambda: test_invalid_jwt_rejected(client)),
        ("5. Unknown Disease 404", lambda: test_unknown_disease_id_returns_404(client, headers)),
        ("6. Missing Features 422", lambda: test_missing_required_features_returns_422(client, headers)),
        ("7. Invalid Feature Types 422", lambda: test_invalid_feature_types_returns_422(client, headers)),
        ("8. Category Mismatch 422", lambda: test_category_mismatch_returns_422(client, headers)),
        ("9. List Active Diseases Endpoint", lambda: test_list_active_diseases_endpoint(client, headers)),
        ("10. Single Generic Endpoint Symmetry", lambda: test_architectural_symmetry_single_endpoint(client, headers)),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} - {e}")
            failed += 1

    print(f"\n{'=' * 55}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 55}\n")
    sys.exit(1 if failed else 0)
