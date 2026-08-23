"""
Comprehensive Verification Test Suite for Step 22:
Integrate Heart Disease Module into Generic Prediction Architecture.

Tests:
1. Artifact & Metadata Integrity (heart_disease_model.pkl, heart_disease_metadata.json, XGBoost Pipeline)
2. Registry Discovery & HeartDiseasePredictor resolution
3. All 13 Input Features Validation:
   - Valid inputs accepted (both canonical names and clinical aliases cp, trestbps, chol, fbs, thalach, ca)
   - Missing feature rejected (422)
   - Wrong data type rejected (422)
   - Disallowed categorical value rejected (422)
   - Out-of-bounds numeric value rejected (422)
4. Decision Threshold (0.40) & Representative Predictions:
   - Known positive case yields prob >= 0.40 and 'High Risk of Heart Disease'
   - Known negative case yields prob < 0.40 and 'Low Risk of Heart Disease'
5. Generic API Integration (POST /predictions):
   - Authenticated flow returns 200 with standard PredictionResponse
   - Unauthenticated request rejected (401)
   - Coexists cleanly with Diabetes in the registry
"""

import json
import sys
import uuid
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
from app.ml.tabular_models.heart_disease.predictor import HeartDiseasePredictor
from app.schemas.prediction import PredictionResponse
from app.services.user_service import UserService


# Mock database for authentication
mock_client = mongomock.MongoClient()
mock_db = mock_client["test_heart_disease_db"]
app.dependency_overrides[get_database] = lambda: mock_db


@pytest.fixture
def auth_headers():
    client = TestClient(app)
    suffix = uuid.uuid4().hex[:8]
    reg = client.post("/auth/register", json={
        "name": f"CardioPatient {suffix}",
        "email": f"cardio_{suffix}@example.com",
        "password": "Password123!"
    })
    token = create_access_token(reg.json()["id"])
    return {"Authorization": f"Bearer {token}"}


def test_heart_disease_artifact_and_metadata():
    """1. Verify Heart Disease artifact loading, metadata integrity, and XGBoost pipeline."""
    disease_registry.reload()
    config = disease_registry.get_or_raise("heart_disease")
    assert config.id == "heart_disease"
    assert config.version == "v1"
    assert config.artifact_filename == "heart_disease_model.pkl"
    assert config.decision_threshold == 0.40
    assert len(config.tabular_features) == 13

    # Load model artifact
    model = load_model_for_disease(config)
    assert isinstance(model, Pipeline)
    assert len(model.steps) >= 2

    # Verify metadata.json
    meta_path = Path(config.model_dir) / config.metadata_filename
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["disease"] == "heart_disease"
    assert meta["model_version"] == "v1"
    assert meta["model_type"] == "XGBoost"
    assert meta["decision_threshold"] == 0.40
    assert len(meta["features"]) == 13
    print("  PASS: 1. Heart Disease artifact, metadata, and XGBoost pipeline verified")


def test_heart_disease_registry_and_predictor_binding():
    """2. Verify Heart Disease discovery and binding to HeartDiseasePredictor."""
    assert disease_registry.has_disease("heart_disease") is True
    assert disease_registry.has_disease("diabetes") is True  # Both coexisting!

    predictor = disease_registry.get_predictor("heart_disease")
    assert isinstance(predictor, HeartDiseasePredictor)
    assert len(predictor.feature_names) == 13
    print("  PASS: 2. Heart Disease discovered and bound to HeartDiseasePredictor")


def test_13_features_validation_and_aliases(auth_headers):
    """3. Test 13 features validation, alias support (cp, trestbps, etc.), missing, types, bounds."""
    client = TestClient(app)

    valid_canonical = {
        "age": 58,
        "sex": 1,
        "chest_pain_type": 2,
        "resting_bp": 140.0,
        "cholestoral": 211.0,
        "fasting_blood_sugar": 1,
        "restecg": 0,
        "max_hr": 165.0,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 2,
        "num_major_vessels": 0,
        "thal": 2,
    }

    # 3a. Canonical 13 features accepted
    res_canon = client.post("/predictions", json={"disease_id": "heart_disease", "inputs": valid_canonical}, headers=auth_headers)
    assert res_canon.status_code == 200

    # 3b. Clinical abbreviations / aliases (cp, trestbps, chol, fbs, thalach, ca) accepted
    valid_aliases = {
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
    res_alias = client.post("/predictions", json={"disease_id": "heart_disease", "inputs": valid_aliases}, headers=auth_headers)
    assert res_alias.status_code == 200

    # 3c. Missing required feature
    incomplete = dict(valid_canonical)
    del incomplete["chest_pain_type"]
    res_missing = client.post("/predictions", json={"disease_id": "heart_disease", "inputs": incomplete}, headers=auth_headers)
    assert res_missing.status_code == 422
    assert "Missing required feature 'chest_pain_type'" in res_missing.json()["detail"]

    # 3d. Wrong datatype
    bad_type = dict(valid_canonical)
    bad_type["resting_bp"] = "high_bp_text"
    res_type = client.post("/predictions", json={"disease_id": "heart_disease", "inputs": bad_type}, headers=auth_headers)
    assert res_type.status_code == 422

    # 3e. Invalid categorical value (sex=5, chest_pain_type=9, thal=8)
    bad_cat = dict(valid_canonical)
    bad_cat["sex"] = 5
    res_cat = client.post("/predictions", json={"disease_id": "heart_disease", "inputs": bad_cat}, headers=auth_headers)
    assert res_cat.status_code == 422
    assert "not allowed" in res_cat.json()["detail"].lower()

    # 3f. Out of bounds numeric values (resting_bp < 60 or > 240, chol < 80 or > 650)
    bad_bp = dict(valid_canonical)
    bad_bp["resting_bp"] = 500.0
    res_bp = client.post("/predictions", json={"disease_id": "heart_disease", "inputs": bad_bp}, headers=auth_headers)
    assert res_bp.status_code == 422
    assert "exceeds the maximum" in res_bp.json()["detail"]

    print("  PASS: 3. 13-feature input validation (canonical, clinical aliases, missing, types, bounds) verified")


def test_representative_predictions_and_threshold_classification():
    """4. Test representative positive and negative cases with 0.40 decision threshold."""
    predictor = disease_registry.get_predictor("heart_disease")

    # High-Risk Cardiovascular Case (prob=0.9495 >= 0.40)
    high_risk_patient = {
        "age": 58,
        "sex": 1,
        "chest_pain_type": 2,
        "resting_bp": 140.0,
        "cholestoral": 211.0,
        "fasting_blood_sugar": 1,
        "restecg": 0,
        "max_hr": 165.0,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 2,
        "num_major_vessels": 0,
        "thal": 2,
    }
    res_high = predictor.predict(high_risk_patient)
    assert res_high.probability >= 0.40
    assert res_high.is_positive is True
    assert res_high.prediction_label == "High Risk of Heart Disease"
    assert res_high.decision_threshold == 0.40
    assert res_high.model_type == "XGBoost"

    # Low-Risk Cardiovascular Case (prob=0.0619 < 0.40)
    low_risk_patient = {
        "age": 67,
        "sex": 1,
        "chest_pain_type": 0,
        "resting_bp": 160.0,
        "cholestoral": 286.0,
        "fasting_blood_sugar": 0,
        "restecg": 0,
        "max_hr": 108.0,
        "exang": 1,
        "oldpeak": 1.5,
        "slope": 1,
        "num_major_vessels": 3,
        "thal": 2,
    }
    res_low = predictor.predict(low_risk_patient)
    assert res_low.probability < 0.40
    assert res_low.is_positive is False
    assert res_low.prediction_label == "Low Risk of Heart Disease"
    assert res_low.decision_threshold == 0.40

    print(f"  PASS: 4. Representative High-risk (prob={res_high.probability}) and Low-risk (prob={res_low.probability}) cases verified")


def test_generic_api_heart_disease_e2e(auth_headers):
    """5. Verify complete authenticated flow through generic POST /predictions."""
    client = TestClient(app)

    # High-Risk payload
    payload = {
        "disease_id": "heart_disease",
        "inputs": {
            "age": 58,
            "sex": 1,
            "chest_pain_type": 2,
            "resting_bp": 140.0,
            "cholestoral": 211.0,
            "fasting_blood_sugar": 1,
            "restecg": 0,
            "max_hr": 165.0,
            "exang": 0,
            "oldpeak": 0.0,
            "slope": 2,
            "num_major_vessels": 0,
            "thal": 2,
        }
    }

    # 5a. Authenticated request succeeds
    response = client.post("/predictions", json=payload, headers=auth_headers)
    assert response.status_code == 200

    data = response.json()
    assert data["disease_id"] == "heart_disease"
    assert data["disease_display_name"] == "Heart Disease Risk Assessment"
    assert data["model_type"] == "XGBoost"
    assert data["model_version"] == "v1"
    assert data["decision_threshold"] == 0.40
    assert data["is_positive"] is True
    assert data["probability"] >= 0.40
    assert data["prediction_label"] == "High Risk of Heart Disease"
    assert data["metadata"]["features_evaluated"] == 13
    assert "disclaimer" in data

    # 5b. Unauthenticated request rejected (401)
    unauth_resp = client.post("/predictions", json=payload)
    assert unauth_resp.status_code in [401, 403]

    print(f"  PASS: 5. End-to-end Heart Disease screening via generic API verified (prob={data['probability']})")


if __name__ == "__main__":
    print("\n=== Step 22 — Heart Disease Integration & API Tests ===\n")
    tests = [
        ("1. Artifact & Metadata Verification", test_heart_disease_artifact_and_metadata),
        ("2. Registry & Predictor Resolution", test_heart_disease_registry_and_predictor_binding),
        ("3. 13-Features Validation & Aliases", lambda: test_13_features_validation_and_aliases(auth_headers={"Authorization": f"Bearer {create_access_token(str(UserService(mock_db).create_user('CPat', 'cpat@ex.com', 'p')['_id']))}"})),
        ("4. Decision Threshold & Predictions", test_representative_predictions_and_threshold_classification),
        ("5. Generic API End-to-End Flow", lambda: test_generic_api_heart_disease_e2e(auth_headers={"Authorization": f"Bearer {create_access_token(str(UserService(mock_db).create_user('CPat2', 'cpat2@ex.com', 'p')['_id']))}"})),
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
