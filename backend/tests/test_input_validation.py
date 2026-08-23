"""
Comprehensive Test Suite for Step 18:
Request & Input Validation Layer (DiseaseInputValidator).

Tests:
1. Unknown disease ID rejected
2. Missing required tabular fields rejected
3. Wrong data types rejected (e.g. non-numeric string or boolean for numeric)
4. Invalid categorical values rejected (e.g. sex=5, chest_pain_type=9)
5. Out-of-bounds numeric values rejected (e.g. Glucose=-10 or Glucose=5000)
6. Valid inputs pass validation and type conversion
7. Invalid input NEVER reaches the predictor/model (Predictor Spy test)
8. Image validation (Empty bytes, oversized files, unsupported formats)
9. FastAPI HTTP validation integration (422 Unprocessable Entity error formats)
"""

import sys
import uuid
from pathlib import Path
import pytest
from pydantic import ValidationError

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mongomock
from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.database.connection import get_database
from app.main import app
from app.ml.base_predictor import BaseTabularPredictor
from app.ml.disease_registry import DiseaseNotFoundError, DiseaseRegistry, disease_registry
from app.ml.validator import DiseaseInputValidator, PredictionValidationError
from app.schemas.disease_config import DiseaseCategory, DiseaseConfig, TabularFeatureSpec
from app.schemas.prediction import PredictionRequest, PredictionResult
from app.services.prediction_service import PredictionService
from app.services.user_service import UserService


# Mock database for user auth verification
mock_client = mongomock.MongoClient()
mock_db = mock_client["test_validation_db"]
app.dependency_overrides[get_database] = lambda: mock_db


def test_missing_and_unknown_disease_id():
    """1. Verify missing & unknown disease handling in validation flow."""
    service = PredictionService(registry=disease_registry)

    # Missing in schema
    with pytest.raises(ValidationError):
        PredictionRequest(inputs={"Glucose": 100})

    # Unknown in registry
    with pytest.raises(DiseaseNotFoundError):
        service.predict_tabular(PredictionRequest(disease_id="non_existent_disease", inputs={"a": 1}))

    print("  PASS: 1. Missing and unknown disease IDs rejected cleanly")


def test_missing_required_tabular_fields():
    """2. Verify rejection when required features are omitted."""
    config = disease_registry.get("diabetes")

    # Missing Glucose, BMI, Age, etc.
    incomplete_inputs = {"Pregnancies": 1}
    with pytest.raises(PredictionValidationError) as exc:
        DiseaseInputValidator.validate_tabular_inputs(config, incomplete_inputs)
    assert "Missing required feature" in str(exc.value)
    print("  PASS: 2. Missing required fields rejected with detailed field error")


def test_wrong_data_types():
    """3. Verify rejection of invalid types (e.g. non-numeric string or boolean for numeric)."""
    config = disease_registry.get("diabetes")

    # String for numeric float
    with pytest.raises(PredictionValidationError) as exc1:
        DiseaseInputValidator.validate_tabular_inputs(
            config,
            {
                "Pregnancies": 1,
                "Glucose": "NOT_A_NUMBER",
                "BloodPressure": 80.0,
                "BMI": 25.0,
                "DiabetesPedigreeFunction": 0.5,
                "Age": 30,
            }
        )
    assert "numeric" in str(exc1.value).lower() or "float" in str(exc1.value).lower()

    # Boolean passed for integer
    with pytest.raises(PredictionValidationError) as exc2:
        DiseaseInputValidator.validate_tabular_inputs(
            config,
            {
                "Pregnancies": True,  # Invalid bool
                "Glucose": 120.0,
                "BloodPressure": 80.0,
                "BMI": 25.0,
                "DiabetesPedigreeFunction": 0.5,
                "Age": 30,
            }
        )
    assert "boolean" in str(exc2.value).lower()
    print("  PASS: 3. Wrong data types and boolean values rejected")


def test_invalid_categorical_values():
    """4. Verify rejection when categorical features receive disallowed values."""
    config = disease_registry.get("heart_disease")

    # sex allowed: [0, 1] -> pass 5
    with pytest.raises(PredictionValidationError) as exc1:
        DiseaseInputValidator.validate_tabular_inputs(
            config,
            {
                "age": 50,
                "sex": 5,  # Invalid
                "chest_pain_type": 1,
                "resting_bp": 120.0,
                "cholestoral": 200.0,
                "fasting_blood_sugar": 0,
                "restecg": 0,
                "max_hr": 150.0,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 1,
                "num_major_vessels": 0,
                "thal": 2,
            }
        )
    assert "not allowed" in str(exc1.value).lower() or "permitted" in str(exc1.value).lower()

    # chest_pain_type allowed: [0, 1, 2, 3] -> pass 9
    with pytest.raises(PredictionValidationError) as exc2:
        DiseaseInputValidator.validate_tabular_inputs(
            config,
            {
                "age": 50,
                "sex": 1,
                "chest_pain_type": 9,  # Invalid
                "resting_bp": 120.0,
                "cholestoral": 200.0,
                "fasting_blood_sugar": 0,
                "restecg": 0,
                "max_hr": 150.0,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 1,
                "num_major_vessels": 0,
                "thal": 2,
            }
        )
    assert "not allowed" in str(exc2.value).lower()
    print("  PASS: 4. Invalid categorical values rejected against allowed_values")


def test_out_of_bounds_numeric_values():
    """5. Verify rejection of values violating min_value or max_value constraints."""
    config = disease_registry.get("diabetes")

    # Glucose min is 40.0 -> pass 10.0
    with pytest.raises(PredictionValidationError) as exc_low:
        DiseaseInputValidator.validate_tabular_inputs(
            config,
            {
                "Pregnancies": 1,
                "Glucose": 10.0,  # Below min (40.0)
                "BloodPressure": 80.0,
                "BMI": 25.0,
                "DiabetesPedigreeFunction": 0.5,
                "Age": 30,
            }
        )
    assert "below the minimum" in str(exc_low.value).lower()

    # Glucose max is 300.0 -> pass 500.0
    with pytest.raises(PredictionValidationError) as exc_high:
        DiseaseInputValidator.validate_tabular_inputs(
            config,
            {
                "Pregnancies": 1,
                "Glucose": 500.0,  # Above max (300.0)
                "BloodPressure": 80.0,
                "BMI": 25.0,
                "DiabetesPedigreeFunction": 0.5,
                "Age": 30,
            }
        )
    assert "exceeds the maximum" in str(exc_high.value).lower()
    print("  PASS: 5. Boundary limit violations (min/max) rejected")


def test_valid_inputs_pass_and_sanitize():
    """6. Verify valid inputs pass validation and are properly coerced."""
    config = disease_registry.get("diabetes")
    raw_inputs = {
        "Pregnancies": "2",  # String int coerced to int
        "Glucose": "145.5",  # String float coerced to float
        "BloodPressure": 80,
        "BMI": 28.4,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 40,
    }

    sanitized = DiseaseInputValidator.validate_tabular_inputs(config, raw_inputs)
    assert sanitized["Pregnancies"] == 2
    assert isinstance(sanitized["Pregnancies"], int)
    assert sanitized["Glucose"] == 145.5
    assert isinstance(sanitized["Glucose"], float)
    print("  PASS: 6. Valid inputs successfully validated and sanitized")


def test_invalid_input_never_reaches_predictor():
    """7. SPY TEST: Ensure predictor.predict() is NEVER called if input is invalid."""
    call_tracker = {"predict_invoked": False}

    custom_reg = DiseaseRegistry(auto_load=False)
    test_config = DiseaseConfig(
        id="spy_test_disease",
        version="v1",
        display_name="Spy Test Disease",
        category=DiseaseCategory.TABULAR,
        input_type="form",
        short_description="Test disease",
        framework="scikit-learn",
        model_type="Mock",
        artifact_filename="dummy.joblib",
        tabular_features=[
            TabularFeatureSpec(name="score", display_name="Score", data_type="float", required=True, min_value=0.0, max_value=100.0)
        ],
        positive_label="Positive",
        negative_label="Negative",
    )

    class SpyPredictor(BaseTabularPredictor):
        def load_model(self):
            return None
        def preprocess_features(self, features):
            return features
        def predict(self, input_data):
            call_tracker["predict_invoked"] = True
            return PredictionResult(
                disease_id="spy_test_disease",
                disease_display_name="Spy Test",
                model_version="v1",
                model_type="Mock",
                prediction_label="Positive",
                is_positive=True,
            )

    custom_reg.register(test_config, predictor_cls=SpyPredictor)
    service = PredictionService(registry=custom_reg)

    # Call with INVALID score (> 100.0)
    invalid_req = PredictionRequest(disease_id="spy_test_disease", inputs={"score": 999.0})
    with pytest.raises(PredictionValidationError):
        service.predict_tabular(invalid_req)

    # Predictor must NOT have been called!
    assert call_tracker["predict_invoked"] is False, "Predictor must NEVER be invoked when validation fails!"
    print("  PASS: 7. Spy test verified: Invalid input is halted before reaching predictor")


def test_image_validation_layer():
    """8. Verify image validation layer (empty bytes, size limit, allowed formats)."""
    config = disease_registry.get("pneumonia")

    # Empty bytes
    with pytest.raises(PredictionValidationError):
        DiseaseInputValidator.validate_image_input(config, b"")

    # Oversized payload (> max_size_bytes)
    oversized = b"x" * (config.image_spec.max_size_bytes + 1024)
    with pytest.raises(PredictionValidationError) as exc_size:
        DiseaseInputValidator.validate_image_input(config, oversized)
    assert "exceeds" in str(exc_size.value).lower()

    # Unsupported format
    with pytest.raises(PredictionValidationError) as exc_fmt:
        DiseaseInputValidator.validate_image_input(config, b"valid_fake_bytes", content_type="image/bmp")
    assert "unsupported" in str(exc_fmt.value).lower()

    print("  PASS: 8. Image input validation (size, emptyness, format) verified")


def test_api_422_integration():
    """9. Verify FastAPI returns 422 Unprocessable Entity for invalid domain values."""
    client = TestClient(app)
    suffix = uuid.uuid4().hex[:8]
    reg = client.post("/auth/register", json={
        "name": f"ValPatient {suffix}",
        "email": f"val_{suffix}@example.com",
        "password": "Password123!"
    })
    headers = {"Authorization": f"Bearer {create_access_token(reg.json()['id'])}"}

    # Diabetes with Glucose below min limit (Glucose=5.0)
    payload = {
        "disease_id": "diabetes",
        "inputs": {
            "Pregnancies": 1,
            "Glucose": 5.0,  # Below min limit (40.0)
            "BloodPressure": 70.0,
            "BMI": 22.0,
            "DiabetesPedigreeFunction": 0.3,
            "Age": 25,
        }
    }

    response = client.post("/predictions", json=payload, headers=headers)
    assert response.status_code == 422
    assert "below the minimum allowed limit" in response.json()["detail"]
    print("  PASS: 9. API returns 422 with descriptive error detail for domain violations")


if __name__ == "__main__":
    print("\n=== Request & Input Validation Verification Tests ===\n")
    tests = [
        ("1. Missing & Unknown Disease ID", test_missing_and_unknown_disease_id),
        ("2. Missing Required Tabular Fields", test_missing_required_tabular_fields),
        ("3. Wrong Data Types & Boolean Rejection", test_wrong_data_types),
        ("4. Invalid Categorical Values", test_invalid_categorical_values),
        ("5. Out-of-bounds Numeric Values", test_out_of_bounds_numeric_values),
        ("6. Valid Inputs Coercion & Sanitization", test_valid_inputs_pass_and_sanitize),
        ("7. Spy Test: Invalid Input Never Reaches Predictor", test_invalid_input_never_reaches_predictor),
        ("8. Image Validation (Size, Format, Payload)", test_image_validation_layer),
        ("9. API 422 HTTP Error Integration", test_api_422_integration),
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
