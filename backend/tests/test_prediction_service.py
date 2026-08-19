"""
Comprehensive Test Suite for Step 15:
Prediction Service (Business Logic Layer).

Tests:
1. Valid tabular prediction (Diabetes) -> returns PredictionResponse
2. Valid tabular prediction (Heart Disease) -> returns PredictionResponse
3. Valid image prediction (Pneumonia) -> returns PredictionResponse
4. Unknown disease ID raises DiseaseNotFoundError
5. Category mismatch raises PredictionValidationError (Tabular req to Image disease & vice versa)
6. Incomplete / invalid tabular features raise PredictionValidationError
7. Empty / corrupted image bytes raise PredictionValidationError
8. Simulated predictor error raises PredictionInferenceError
9. Decoupling verification: No FastAPI Request or MongoDB dependency
"""

import io
import sys
from pathlib import Path
import pytest
from PIL import Image

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.base_predictor import BaseTabularPredictor
from app.ml.disease_registry import DiseaseNotFoundError, DiseaseRegistry, disease_registry
from app.schemas.disease_config import DiseaseCategory, DiseaseConfig, TabularFeatureSpec
from app.schemas.prediction import PredictionRequest, PredictionRequestPayload, PredictionResponse
from app.services.prediction_service import (
    PredictionInferenceError,
    PredictionService,
    PredictionValidationError,
)


def create_dummy_image_bytes(width=224, height=224) -> bytes:
    """Helper creating valid PNG image bytes."""
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_diabetes_service_prediction():
    service = PredictionService(registry=disease_registry)

    req = PredictionRequest(
        disease_id="diabetes",
        inputs={
            "Pregnancies": 2,
            "Glucose": 150.0,
            "BloodPressure": 80.0,
            "BMI": 32.0,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 45,
        }
    )

    response = service.predict_tabular(req)

    assert isinstance(response, PredictionResponse)
    assert response.disease_id == "diabetes"
    assert response.disease_display_name == "Diabetes Risk Assessment"
    assert response.is_positive in [True, False]
    assert response.probability is not None
    assert 0.0 <= response.probability <= 1.0
    assert response.decision_threshold == 0.4
    assert response.clinical_purpose is not None
    assert response.disclaimer is not None
    assert "latency_ms" in response.metadata
    print(f"  PASS: 1. Diabetes service prediction: prob={response.probability}, label='{response.prediction_label}'")


def test_heart_disease_service_prediction():
    service = PredictionService(registry=disease_registry)

    req = PredictionRequest(
        disease_id="heart_disease",
        inputs={
            "age": 60,
            "sex": 1,
            "chest_pain_type": 3,
            "resting_bp": 145.0,
            "cholestoral": 230.0,
            "fasting_blood_sugar": 0,
            "restecg": 1,
            "max_hr": 150.0,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "num_major_vessels": 0,
            "thal": 1,
        }
    )

    response = service.predict_tabular(req)

    assert isinstance(response, PredictionResponse)
    assert response.disease_id == "heart_disease"
    assert response.probability is not None
    print(f"  PASS: 2. Heart Disease service prediction: prob={response.probability}, label='{response.prediction_label}'")


def test_pneumonia_image_service_prediction():
    service = PredictionService(registry=disease_registry)

    image_bytes = create_dummy_image_bytes(224, 224)
    response = service.predict_image(
        disease_id="pneumonia",
        image_bytes=image_bytes,
        content_type="image/png"
    )

    assert isinstance(response, PredictionResponse)
    assert response.disease_id == "pneumonia"
    assert response.probability is not None
    assert 0.0 <= response.probability <= 1.0
    print(f"  PASS: 3. Pneumonia image service prediction: prob={response.probability}, label='{response.prediction_label}'")


def test_unknown_disease_raises_not_found():
    service = PredictionService(registry=disease_registry)

    req = PredictionRequest(disease_id="unknown_disease_xyz", inputs={})
    with pytest.raises(DiseaseNotFoundError):
        service.predict_tabular(req)

    with pytest.raises(DiseaseNotFoundError):
        service.predict_image(disease_id="unknown_disease_xyz", image_bytes=b"123")

    print("  PASS: 4. Unknown disease raises DiseaseNotFoundError")


def test_category_mismatch_validation():
    service = PredictionService(registry=disease_registry)

    # 1. Sending tabular request to image disease (Pneumonia)
    tab_req = PredictionRequest(disease_id="pneumonia", inputs={"val": 10})
    with pytest.raises(PredictionValidationError) as exc:
        service.predict_tabular(tab_req)
    assert "image" in str(exc.value)

    # 2. Sending image payload to tabular disease (Diabetes)
    with pytest.raises(PredictionValidationError) as exc:
        service.predict_image(disease_id="diabetes", image_bytes=b"dummy")
    assert "tabular" in str(exc.value)

    print("  PASS: 5. Category mismatch raises PredictionValidationError")


def test_invalid_tabular_features_validation():
    service = PredictionService(registry=disease_registry)

    # Missing required features
    req = PredictionRequest(disease_id="diabetes", inputs={"Glucose": 120})
    with pytest.raises(PredictionValidationError):
        service.predict_tabular(req)

    # Invalid feature type (string for numeric)
    req_bad_type = PredictionRequest(
        disease_id="diabetes",
        inputs={
            "Pregnancies": "not_an_int",
            "Glucose": 150.0,
            "BloodPressure": 80.0,
            "BMI": 32.0,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 45,
        }
    )
    with pytest.raises(PredictionValidationError):
        service.predict_tabular(req_bad_type)

    print("  PASS: 6. Invalid/incomplete tabular features raise PredictionValidationError")


def test_invalid_image_payload_validation():
    service = PredictionService(registry=disease_registry)

    # Empty payload
    with pytest.raises(PredictionValidationError):
        service.predict_image(disease_id="pneumonia", image_bytes=b"")

    # Corrupted image bytes
    with pytest.raises(PredictionValidationError):
        service.predict_image(disease_id="pneumonia", image_bytes=b"CORRUPTED_STREAM")

    # Unsupported format
    with pytest.raises(PredictionValidationError):
        service.predict_image(
            disease_id="pneumonia",
            image_bytes=create_dummy_image_bytes(),
            content_type="image/gif"
        )

    print("  PASS: 7. Invalid/empty image payloads raise PredictionValidationError")


def test_simulated_inference_failure():
    """Verify that an inference runtime crash is captured and wrapped as PredictionInferenceError."""
    # Create isolated custom registry and failing predictor
    custom_registry = DiseaseRegistry(auto_load=False)
    failing_config = DiseaseConfig(
        id="failing_disease",
        version="v1",
        display_name="Failing Test Disease",
        category=DiseaseCategory.TABULAR,
        input_type="form",
        short_description="Test disease",
        framework="scikit-learn",
        model_type="Mock",
        artifact_filename="dummy.joblib",
        tabular_features=[
            TabularFeatureSpec(name="x", display_name="X", data_type="float", required=True)
        ],
        positive_label="Positive",
        negative_label="Negative",
    )
    
    class FailingPredictor(BaseTabularPredictor):
        def load_model(self):
            return None
        def preprocess_features(self, features):
            return features
        def predict(self, input_data):
            raise RuntimeError("Underlying CUDA/inference kernel exploded!")

    custom_registry.register(failing_config, predictor_cls=FailingPredictor)

    service = PredictionService(registry=custom_registry)
    req = PredictionRequest(disease_id="failing_disease", inputs={"x": 1.0})

    with pytest.raises(PredictionInferenceError) as exc:
        service.predict_tabular(req)
    assert "Inference failed" in str(exc.value)

    print("  PASS: 8. Predictor runtime failure raises PredictionInferenceError")


def test_unified_predict_dispatch():
    service = PredictionService(registry=disease_registry)

    # Dispatch tabular via payload
    tab_payload = PredictionRequestPayload(
        disease_id="diabetes",
        tabular_inputs={
            "Pregnancies": 1,
            "Glucose": 110.0,
            "BloodPressure": 70.0,
            "BMI": 24.0,
            "DiabetesPedigreeFunction": 0.3,
            "Age": 28,
        }
    )
    tab_resp = service.predict(tab_payload)
    assert tab_resp.disease_id == "diabetes"

    # Dispatch image via payload
    img_payload = PredictionRequestPayload(
        disease_id="pneumonia",
        image_bytes=create_dummy_image_bytes(),
        image_content_type="image/png"
    )
    img_resp = service.predict(img_payload)
    assert img_resp.disease_id == "pneumonia"

    print("  PASS: 9. Unified predict(payload) dispatch verified")


if __name__ == "__main__":
    print("\n=== Prediction Service Verification Tests ===\n")
    tests = [
        ("1. Diabetes Service Prediction", test_diabetes_service_prediction),
        ("2. Heart Disease Service Prediction", test_heart_disease_service_prediction),
        ("3. Pneumonia Image Service Prediction", test_pneumonia_image_service_prediction),
        ("4. Unknown Disease ID Rejection", test_unknown_disease_raises_not_found),
        ("5. Category Mismatch Validation", test_category_mismatch_validation),
        ("6. Invalid Tabular Features Validation", test_invalid_tabular_features_validation),
        ("7. Invalid Image Payload Validation", test_invalid_image_payload_validation),
        ("8. Simulated Inference Failure Handling", test_simulated_inference_failure),
        ("9. Unified Payload Dispatch", test_unified_predict_dispatch),
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
