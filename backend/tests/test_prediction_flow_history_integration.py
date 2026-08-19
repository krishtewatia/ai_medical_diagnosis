"""
Comprehensive Integration Test Suite for Step 23.4:
Connect Prediction Flow to MongoDB Prediction History.

Tests:
1. Diabetes prediction automatically creates valid history record in MongoDB
2. Heart Disease prediction creates valid 13-feature history record in MongoDB
3. Failed validation / inference does NOT create any history record
4. User isolation: Predictions made by User A and User B are accurately tagged with their respective user_id
"""

import sys
from pathlib import Path
from bson import ObjectId
from fastapi.testclient import TestClient
import mongomock
import pytest

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.dependencies import get_current_user
from app.api.prediction import get_prediction_service
from app.main import app
from app.ml.disease_registry import disease_registry
from app.services.prediction_history_service import PredictionHistoryService
from app.services.prediction_service import PredictionService


@pytest.fixture
def mock_history_service():
    mock_db = mongomock.MongoClient()["test_prediction_history_flow_db"]
    return PredictionHistoryService(db=mock_db)


@pytest.fixture
def client_with_mock_db(mock_history_service):
    # Override get_prediction_service to use the mock history service
    app.dependency_overrides[get_prediction_service] = lambda: PredictionService(
        registry=disease_registry,
        history_service=mock_history_service
    )
    yield TestClient(app), mock_history_service
    app.dependency_overrides.clear()


def test_diabetes_prediction_saves_history(client_with_mock_db):
    """Test 1: Diabetes prediction creates a valid history document in MongoDB."""
    client, history_service = client_with_mock_db
    user_id = str(ObjectId())

    # Override current_user
    app.dependency_overrides[get_current_user] = lambda: {
        "_id": ObjectId(user_id),
        "email": "user_a@example.com",
        "full_name": "User Alpha"
    }

    diabetes_payload = {
        "disease_id": "diabetes",
        "inputs": {
            "Pregnancies": 3,
            "Glucose": 160.0,
            "BloodPressure": 80.0,
            "BMI": 33.5,
            "DiabetesPedigreeFunction": 0.62,
            "Age": 45,
        }
    }

    response = client.post("/predictions", json=diabetes_payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    res_data = response.json()
    assert res_data["disease_id"] == "diabetes"
    assert "High Risk" in res_data["prediction_label"]

    # Verify document in predictions collection
    records = history_service.get_user_predictions(user_id=user_id)
    assert len(records) == 1
    doc = records[0]

    assert doc["user_id"] == user_id
    assert doc["disease"] == "diabetes"
    assert doc["disease_display_name"] == "Diabetes Risk Assessment"
    assert doc["input_type"] == "tabular"
    assert doc["model"]["version"] == "v1"
    assert doc["model"]["threshold"] == 0.40

    # Verify input_data snapshot has the 6 evaluated model features
    assert doc["input_data"]["Pregnancies"] == 3
    assert doc["input_data"]["Glucose"] == 160.0
    assert doc["input_data"]["BMI"] == 33.5
    assert doc["input_data"]["DiabetesPedigreeFunction"] == 0.62
    assert doc["input_data"]["Age"] == 45

    # Verify result snapshot
    assert doc["result"]["prediction"] == res_data["prediction_label"]
    assert doc["result"]["is_positive"] is True
    assert doc["result"]["probability"] == res_data["probability"]
    assert "latency_ms" in doc["metadata"]

    print("  PASS: 1. Diabetes prediction saved complete MongoDB history document")


def test_heart_disease_prediction_saves_history(client_with_mock_db):
    """Test 2: Heart Disease prediction creates a valid 13-feature history record."""
    client, history_service = client_with_mock_db
    user_id = str(ObjectId())

    app.dependency_overrides[get_current_user] = lambda: {
        "_id": ObjectId(user_id),
        "email": "user_heart@example.com",
        "full_name": "Heart Patient"
    }

    heart_payload = {
        "disease_id": "heart_disease",
        "inputs": {
            "age": 58,
            "sex": 1,
            "cp": 2,
            "trestbps": 140.0,
            "chol": 211.0,
            "fbs": 1,
            "restecg": 0,
            "thalach": 165.0,
            "exang": 0,
            "oldpeak": 0.0,
            "slope": 2,
            "ca": 0,
            "thal": 2,
        }
    }

    response = client.post("/predictions", json=heart_payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    res_data = response.json()
    assert res_data["disease_id"] == "heart_disease"

    records = history_service.get_user_predictions(user_id=user_id, disease="heart_disease")
    assert len(records) == 1
    doc = records[0]

    assert doc["disease"] == "heart_disease"
    assert doc["model"]["model_type"] == "XGBoost"
    assert doc["model"]["threshold"] == 0.40
    # Evaluated input_data mapped to canonical 13 features
    assert len(doc["input_data"]) == 13
    assert doc["input_data"]["chest_pain_type"] == 2
    assert doc["input_data"]["resting_bp"] == 140.0

    print("  PASS: 2. Heart Disease prediction saved complete 13-feature history document")


def test_failed_prediction_does_not_save_history(client_with_mock_db):
    """Test 3: Invalid request rejected with 422 and does NOT write any record to history."""
    client, history_service = client_with_mock_db
    user_id = str(ObjectId())

    app.dependency_overrides[get_current_user] = lambda: {
        "_id": ObjectId(user_id),
        "email": "user_fail@example.com",
    }

    # Missing Glucose and BMI
    invalid_payload = {
        "disease_id": "diabetes",
        "inputs": {"Pregnancies": 1}
    }

    response = client.post("/predictions", json=invalid_payload)
    assert response.status_code == 422

    # Assert 0 records created in database
    records = history_service.get_user_predictions(user_id=user_id)
    assert len(records) == 0
    assert history_service.count_user_predictions(user_id=user_id) == 0

    print("  PASS: 3. Validation rejection (422) prevented history persistence (0 documents saved)")


def test_user_isolation_history_persistence(client_with_mock_db):
    """Test 4: User A and User B predictions are isolated and tagged with respective user_ids."""
    client, history_service = client_with_mock_db
    user_a = str(ObjectId())
    user_b = str(ObjectId())

    # User A prediction
    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_a), "email": "a@ex.com"}
    payload_a = {
        "disease_id": "diabetes",
        "inputs": {
            "Pregnancies": 1,
            "Glucose": 100.0,
            "BloodPressure": 70.0,
            "BMI": 24.0,
            "DiabetesPedigreeFunction": 0.25,
            "Age": 28,
        }
    }
    resp_a = client.post("/predictions", json=payload_a)
    assert resp_a.status_code == 200, f"Expected 200, got {resp_a.status_code}: {resp_a.text}"

    # User B prediction
    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_b), "email": "b@ex.com"}
    payload_b = {
        "disease_id": "heart_disease",
        "inputs": {
            "age": 50,
            "sex": 0,
            "cp": 0,
            "trestbps": 120.0,
            "chol": 200.0,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150.0,
            "exang": 0,
            "oldpeak": 0.0,
            "slope": 1,
            "ca": 0,
            "thal": 2,
        }
    }
    resp_b = client.post("/predictions", json=payload_b)
    assert resp_b.status_code == 200, f"Expected 200, got {resp_b.status_code}: {resp_b.text}"

    # User A history contains only User A
    records_a = history_service.get_user_predictions(user_id=user_a)
    assert len(records_a) == 1
    assert records_a[0]["user_id"] == user_a
    assert records_a[0]["disease"] == "diabetes"

    # User B history contains only User B
    records_b = history_service.get_user_predictions(user_id=user_b)
    assert len(records_b) == 1
    assert records_b[0]["user_id"] == user_b
    assert records_b[0]["disease"] == "heart_disease"

    print("  PASS: 4. User isolation verified: User A and User B predictions strictly separated")


if __name__ == "__main__":
    print("\n=== Prediction History Flow Integration Tests ===\n")
    mock_hs = mongomock.MongoClient()["test_runner_flow_db"]
    hs = PredictionHistoryService(db=mock_hs)
    app.dependency_overrides[get_prediction_service] = lambda: PredictionService(
        registry=disease_registry,
        history_service=hs
    )
    c = TestClient(app)

    tests = [
        ("1. Diabetes History Persistence", lambda: test_diabetes_prediction_saves_history((c, hs))),
        ("2. Heart Disease History Persistence", lambda: test_heart_disease_prediction_saves_history((c, hs))),
        ("3. Failed Prediction History Block", lambda: test_failed_prediction_does_not_save_history((c, hs))),
        ("4. User Isolation", lambda: test_user_isolation_history_persistence((c, hs))),
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

    app.dependency_overrides.clear()

    print(f"\n{'=' * 55}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 55}\n")
    sys.exit(1 if failed else 0)
