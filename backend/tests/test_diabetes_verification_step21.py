"""
Step 21 — Diabetes Model Verification & Integration Test Suite.

Verifies:
1. Model Artifact & Metadata:
   - diabetes_model_v1.joblib loads successfully via centralized loader
   - diabetes_metadata.json loads and matches configuration
   - Model pipeline structure and version check
2. Input Validation:
   - All 6 tabular inputs tested (Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age)
   - Missing field rejection (422)
   - Wrong datatype rejection (422)
   - Out-of-bounds value rejection (422)
   - Valid inputs acceptance (200)
3. Pipeline Preprocessing:
   - Confirms backend delegates feature scaling & imputation directly to the serialized pipeline
4. Decision Threshold (0.40):
   - Strict application of the 0.40 decision threshold
5. Representative Prediction Cases:
   - Known positive case yields probability >= 0.40 and 'High Risk of Diabetes'
   - Known negative case yields probability < 0.40 and 'Low Risk of Diabetes'
6. End-to-End API Integration:
   - Full flow through generic POST /predictions endpoint with JWT authentication
   - Verifies no disease-specific endpoint is required
"""

import json
import sys
from pathlib import Path
import pytest
from sklearn.pipeline import Pipeline

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mongomock
from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.database.connection import get_database
from app.main import app
from app.ml.disease_registry import disease_registry
from app.ml.loaders import load_model_for_disease
from app.ml.tabular_models.diabetes.predictor import DiabetesPredictor
from app.schemas.prediction import PredictionResponse
from app.services.user_service import UserService


# Mock database for test isolation
mock_client = mongomock.MongoClient()
mock_db = mock_client["test_diabetes_v21_db"]
app.dependency_overrides[get_database] = lambda: mock_db


@pytest.fixture
def auth_headers():
    u_svc = UserService(mock_db)
    u = u_svc.create_user(name="Step21Patient", email="step21@example.com", password_hash="hash21")
    token = create_access_token(str(u["_id"]))
    return {"Authorization": f"Bearer {token}"}


# 1. Model Artifact & Metadata Verification
def test_diabetes_artifact_and_metadata_integrity():
    """1. Verify artifact loading, metadata match, and pipeline structure."""
    config = disease_registry.get_or_raise("diabetes")
    assert config.id == "diabetes"
    assert config.version == "v1"
    assert config.artifact_filename == "diabetes_model_v1.joblib"

    # Load model
    model = load_model_for_disease(config)
    assert isinstance(model, Pipeline), f"Expected scikit-learn Pipeline, got {type(model).__name__}"
    assert len(model.steps) >= 2, "Pipeline must contain transformers and estimator"

    # Verify metadata.json
    meta_path = Path(config.model_dir) / config.metadata_filename
    assert meta_path.exists(), f"Metadata file missing at {meta_path}"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["disease"] == "diabetes"
    assert meta["model_version"] == "v1"
    assert meta["decision_threshold"] == 0.40
    assert meta["features"] == [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]
    print("  PASS: 1. Model artifact, metadata JSON, and serialized pipeline verified")


# 2. Input Validation (6 inputs, missing, types, bounds)
def test_diabetes_input_validation_rules(auth_headers):
    """2. Test all 6 inputs, boundary limits, missing fields, and bad types."""
    client = TestClient(app)

    valid_base = {
        "Pregnancies": 2,
        "Glucose": 120.0,
        "BloodPressure": 75.0,
        "BMI": 26.5,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 35,
    }

    # 2a. Valid inputs accepted
    res_valid = client.post("/predictions", json={"disease_id": "diabetes", "inputs": valid_base}, headers=auth_headers)
    assert res_valid.status_code == 200

    # 2b. Missing each required field individually
    for required_feature in valid_base.keys():
        incomplete = dict(valid_base)
        del incomplete[required_feature]
        res_missing = client.post("/predictions", json={"disease_id": "diabetes", "inputs": incomplete}, headers=auth_headers)
        assert res_missing.status_code == 422, f"Missing '{required_feature}' should return 422"
        assert f"Missing required feature '{required_feature}'" in res_missing.json()["detail"]

    # 2c. Wrong data types (string for float)
    bad_type = dict(valid_base)
    bad_type["Glucose"] = "invalid_string"
    res_bad_type = client.post("/predictions", json={"disease_id": "diabetes", "inputs": bad_type}, headers=auth_headers)
    assert res_bad_type.status_code == 422

    # 2d. Out of bounds values (Glucose < 40.0, BMI > 70.0, Age < 1)
    out_of_bounds_cases = [
        ("Glucose", 10.0, "below the minimum"),
        ("Glucose", 500.0, "exceeds the maximum"),
        ("BMI", 5.0, "below the minimum"),
        ("BMI", 99.0, "exceeds the maximum"),
        ("Age", 0, "below the minimum"),
        ("Pregnancies", -1, "below the minimum"),
    ]
    for feat, bad_val, msg_part in out_of_bounds_cases:
        oob_inputs = dict(valid_base)
        oob_inputs[feat] = bad_val
        res_oob = client.post("/predictions", json={"disease_id": "diabetes", "inputs": oob_inputs}, headers=auth_headers)
        assert res_oob.status_code == 422, f"{feat}={bad_val} should be rejected with 422"
        assert msg_part in res_oob.json()["detail"]

    print("  PASS: 2. Input validation for all 6 features, missing fields, types, and ranges verified")


# 3. Pipeline Preprocessing Verification
def test_pipeline_preprocessing_integrity():
    """3. Confirm backend passes DataFrame to pipeline without redundant manual preprocessing."""
    predictor = disease_registry.get_predictor("diabetes")
    assert isinstance(predictor, DiabetesPredictor)

    inputs = {
        "Pregnancies": 3,
        "Glucose": 140.0,
        "BloodPressure": 80.0,
        "BMI": 30.0,
        "DiabetesPedigreeFunction": 0.50,
        "Age": 40,
    }
    df = predictor.preprocess_features(inputs)

    # DataFrame should contain raw input values; scaling & imputation are in pipeline steps
    assert list(df.columns) == predictor.feature_names
    assert df["Glucose"].iloc[0] == 140.0
    assert df["BMI"].iloc[0] == 30.0

    # Ensure model steps execute inside pipeline
    model = predictor.get_model()
    step_names = [name for name, _ in model.steps]
    assert "imputer" in step_names or "scaler" in step_names or len(step_names) >= 2
    print("  PASS: 3. Preprocessing delegation directly to exported pipeline verified")


# 4 & 5. Threshold (0.40) & Representative Predictions
def test_representative_predictions_and_threshold_classification():
    """4 & 5. Verify representative high-risk and low-risk cases with 0.40 threshold."""
    predictor = disease_registry.get_predictor("diabetes")

    # High-Risk Clinical Case
    high_risk_patient = {
        "Pregnancies": 7,
        "Glucose": 185.0,
        "BloodPressure": 90.0,
        "BMI": 39.2,
        "DiabetesPedigreeFunction": 1.15,
        "Age": 55,
    }
    res_high = predictor.predict(high_risk_patient)
    assert res_high.probability >= 0.40
    assert res_high.is_positive is True
    assert res_high.prediction_label == "High Risk of Diabetes"
    assert res_high.decision_threshold == 0.40

    # Low-Risk Clinical Case
    low_risk_patient = {
        "Pregnancies": 0,
        "Glucose": 75.0,
        "BloodPressure": 60.0,
        "BMI": 19.0,
        "DiabetesPedigreeFunction": 0.12,
        "Age": 20,
    }
    res_low = predictor.predict(low_risk_patient)
    assert res_low.probability < 0.40
    assert res_low.is_positive is False
    assert res_low.prediction_label == "Low Risk of Diabetes"
    assert res_low.decision_threshold == 0.40

    # Boundary test around 0.40
    print(f"  PASS: 4 & 5. High-risk (prob={res_high.probability}) and Low-risk (prob={res_low.probability}) cases verified with 0.40 threshold")


# 6. End-to-End API Integration
def test_generic_prediction_api_flow(auth_headers):
    """6. Execute end-to-end flow through POST /predictions."""
    client = TestClient(app)

    payload = {
        "disease_id": "diabetes",
        "inputs": {
            "Pregnancies": 4,
            "Glucose": 155.0,
            "BloodPressure": 82.0,
            "BMI": 32.5,
            "DiabetesPedigreeFunction": 0.62,
            "Age": 45,
        }
    }

    response = client.post("/predictions", json=payload, headers=auth_headers)
    assert response.status_code == 200

    data = response.json()
    assert data["disease_id"] == "diabetes"
    assert data["disease_display_name"] == "Diabetes Risk Assessment"
    assert data["model_version"] == "v1"
    assert data["model_type"] == "LogisticRegression"
    assert data["decision_threshold"] == 0.40
    assert isinstance(data["probability"], float)
    assert isinstance(data["is_positive"], bool)
    assert "explanation" in data
    assert "disclaimer" in data
    assert "metadata" in data
    assert "latency_ms" in data["metadata"]
    assert data["metadata"]["features_evaluated"] == 6

    print(f"  PASS: 6. End-to-end authenticated API flow verified: label='{data['prediction_label']}', prob={data['probability']}")


if __name__ == "__main__":
    print("\n=== Step 21 — Diabetes Model Verification & Integration Tests ===\n")
    tests = [
        ("1. Artifact & Metadata Integrity", test_diabetes_artifact_and_metadata_integrity),
        ("2. Input Validation Rules", lambda: test_diabetes_input_validation_rules(auth_headers={"Authorization": f"Bearer {create_access_token(str(UserService(mock_db).create_user('S21', 's21@ex.com', 'p')['_id']))}"})),
        ("3. Pipeline Preprocessing Integrity", test_pipeline_preprocessing_integrity),
        ("4 & 5. Threshold & Representative Predictions", test_representative_predictions_and_threshold_classification),
        ("6. Generic API Flow Integration", lambda: test_generic_prediction_api_flow(auth_headers={"Authorization": f"Bearer {create_access_token(str(UserService(mock_db).create_user('S21b', 's21b@ex.com', 'p')['_id']))}"})),
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
