"""
Comprehensive Verification Test Suite for Step 20:
Register the First Real Disease Module (Diabetes).

Tests:
1. Diabetes discovery and registration in DiseaseRegistry
2. DiabetesPredictor resolution and binding
3. Model artifact loading (diabetes_model_v1.joblib)
4. High-risk diabetes inference with 0.40 threshold
5. Low-risk diabetes inference with 0.40 threshold
6. End-to-end generic API prediction (POST /predictions) for high risk
7. End-to-end generic API prediction (POST /predictions) for low risk
8. Missing/invalid inputs rejected through generic API (422)
9. Metadata discovery via GET /predictions/diseases
"""

import sys
import uuid
from pathlib import Path
import pytest

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mongomock
from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.database.connection import get_database
from app.main import app
from app.ml.disease_registry import disease_registry
from app.ml.tabular_models.diabetes.predictor import DiabetesPredictor
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import PredictionService
from app.services.user_service import UserService


# Mock database for authentication
mock_client = mongomock.MongoClient()
mock_db = mock_client["test_diabetes_db"]
app.dependency_overrides[get_database] = lambda: mock_db


@pytest.fixture
def auth_headers():
    client = TestClient(app)
    suffix = uuid.uuid4().hex[:8]
    reg = client.post("/auth/register", json={
        "name": f"DiabetesPatient {suffix}",
        "email": f"diabetes_{suffix}@example.com",
        "password": "Password123!"
    })
    token = create_access_token(reg.json()["id"])
    return {"Authorization": f"Bearer {token}"}


def test_diabetes_discovery_and_predictor_binding():
    """1 & 2. Verify Diabetes is discovered and bound to DiabetesPredictor."""
    assert disease_registry.has_disease("diabetes") is True
    config = disease_registry.get("diabetes")
    assert config is not None
    assert config.id == "diabetes"
    assert config.artifact_filename == "diabetes_model_v1.joblib"
    assert config.decision_threshold == 0.40
    assert len(config.tabular_features) == 6

    predictor = disease_registry.get_predictor("diabetes")
    assert isinstance(predictor, DiabetesPredictor)
    print("  PASS: 1 & 2. Diabetes configuration and DiabetesPredictor registered successfully")


def test_diabetes_model_inference_high_and_low_risk():
    """3, 4, 5. Verify model loading, inference, and 0.40 threshold application."""
    predictor = disease_registry.get_predictor("diabetes")

    # High-Risk Sample
    high_risk_input = {
        "Pregnancies": 6,
        "Glucose": 170.0,
        "BloodPressure": 88.0,
        "BMI": 36.5,
        "DiabetesPedigreeFunction": 0.85,
        "Age": 52,
    }
    res_high = predictor.predict(high_risk_input)
    assert res_high.is_positive is True
    assert res_high.prediction_label == "High Risk of Diabetes"
    assert res_high.probability >= 0.40
    assert res_high.decision_threshold == 0.40
    assert res_high.disease_id == "diabetes"
    assert res_high.model_version == "v1"

    # Low-Risk Sample
    low_risk_input = {
        "Pregnancies": 0,
        "Glucose": 78.0,
        "BloodPressure": 65.0,
        "BMI": 19.5,
        "DiabetesPedigreeFunction": 0.15,
        "Age": 21,
    }
    res_low = predictor.predict(low_risk_input)
    assert res_low.is_positive is False
    assert res_low.prediction_label == "Low Risk of Diabetes"
    assert res_low.probability < 0.40
    assert res_low.decision_threshold == 0.40

    print(f"  PASS: 3, 4, 5. High-risk (prob={res_high.probability}) and Low-risk (prob={res_low.probability}) inference verified")


def test_generic_api_diabetes_high_risk(auth_headers):
    """6. Verify generic POST /predictions endpoint executes Diabetes prediction for high-risk case."""
    client = TestClient(app)
    payload = {
        "disease_id": "diabetes",
        "inputs": {
            "Pregnancies": 5,
            "Glucose": 165.0,
            "BloodPressure": 85.0,
            "BMI": 34.0,
            "DiabetesPedigreeFunction": 0.70,
            "Age": 48,
        }
    }

    response = client.post("/predictions", json=payload, headers=auth_headers)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert data["disease_id"] == "diabetes"
    assert data["disease_display_name"] == "Diabetes Risk Assessment"
    assert data["prediction_label"] == "High Risk of Diabetes"
    assert data["is_positive"] is True
    assert data["probability"] >= 0.40
    assert data["decision_threshold"] == 0.40
    assert data["model_version"] == "v1"
    assert "disclaimer" in data
    assert "timestamp" in data

    print(f"  PASS: 6. Generic API POST /predictions returned High Risk: prob={data['probability']}")


def test_generic_api_diabetes_low_risk(auth_headers):
    """7. Verify generic POST /predictions endpoint executes Diabetes prediction for low-risk case."""
    client = TestClient(app)
    payload = {
        "disease_id": "diabetes",
        "inputs": {
            "Pregnancies": 1,
            "Glucose": 82.0,
            "BloodPressure": 68.0,
            "BMI": 21.0,
            "DiabetesPedigreeFunction": 0.20,
            "Age": 24,
        }
    }

    response = client.post("/predictions", json=payload, headers=auth_headers)
    assert response.status_code == 200

    data = response.json()
    assert data["disease_id"] == "diabetes"
    assert data["prediction_label"] == "Low Risk of Diabetes"
    assert data["is_positive"] is False
    assert data["probability"] < 0.40

    print(f"  PASS: 7. Generic API POST /predictions returned Low Risk: prob={data['probability']}")


def test_generic_api_invalid_diabetes_inputs(auth_headers):
    """8. Verify invalid inputs for Diabetes are rejected by validation layer with 422."""
    client = TestClient(app)

    # Missing Glucose
    missing_glucose = {
        "disease_id": "diabetes",
        "inputs": {
            "Pregnancies": 1,
            "BloodPressure": 70.0,
            "BMI": 22.0,
            "DiabetesPedigreeFunction": 0.3,
            "Age": 25,
        }
    }
    resp1 = client.post("/predictions", json=missing_glucose, headers=auth_headers)
    assert resp1.status_code == 422
    assert "Missing required feature 'Glucose'" in resp1.json()["detail"]

    # Out-of-bounds BMI (> 70.0)
    invalid_bmi = {
        "disease_id": "diabetes",
        "inputs": {
            "Pregnancies": 1,
            "Glucose": 100.0,
            "BloodPressure": 70.0,
            "BMI": 150.0,  # Invalid
            "DiabetesPedigreeFunction": 0.3,
            "Age": 25,
        }
    }
    resp2 = client.post("/predictions", json=invalid_bmi, headers=auth_headers)
    assert resp2.status_code == 422
    assert "exceeds the maximum allowed limit" in resp2.json()["detail"]

    print("  PASS: 8. Invalid/incomplete Diabetes inputs rejected with 422")


def test_diabetes_metadata_discovery(auth_headers):
    """9. Verify Diabetes model metadata is listed via GET /predictions/diseases."""
    client = TestClient(app)
    response = client.get("/predictions/diseases", headers=auth_headers)
    assert response.status_code == 200

    diseases = response.json()
    diabetes_info = next((d for d in diseases if d["id"] == "diabetes"), None)
    assert diabetes_info is not None
    assert diabetes_info["display_name"] == "Diabetes Risk Assessment"
    assert diabetes_info["category"] == "tabular"
    assert len(diabetes_info["tabular_features"]) == 6
    assert diabetes_info["supports_probability"] is True
    assert diabetes_info["metrics"]["roc_auc"] == 0.81

    print("  PASS: 9. Diabetes metadata discoverable via GET /predictions/diseases")


if __name__ == "__main__":
    print("\n=== Diabetes Module Integration & API Tests ===\n")
    tests = [
        ("1 & 2. Discovery & Predictor Binding", test_diabetes_discovery_and_predictor_binding),
        ("3, 4, 5. Model Loading & Inference", test_diabetes_model_inference_high_and_low_risk),
        ("6. Generic API High-Risk Prediction", test_generic_api_diabetes_high_risk),
        ("7. Generic API Low-Risk Prediction", test_generic_api_diabetes_low_risk),
        ("8. Invalid Input Rejection (422)", test_generic_api_invalid_diabetes_inputs),
        ("9. Metadata Discovery API", test_diabetes_metadata_discovery),
    ]

    passed = 0
    failed = 0
    headers = {
        "Authorization": f"Bearer {create_access_token(str(UserService(mock_db).create_user('DUser', 'du@ex.com', 'p')['_id']))}"
    }

    for name, fn in tests:
        try:
            if "API" in name or "Generic" in name or "Metadata" in name or "Input" in name:
                fn(headers)
            else:
                fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} - {e}")
            failed += 1

    print(f"\n{'=' * 55}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 55}\n")
    sys.exit(1 if failed else 0)
