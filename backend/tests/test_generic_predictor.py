"""
Comprehensive Test Suite for Step 13:
Generic Predictor Layer (Tabular & Image).

Tests:
1. Tabular prediction with Diabetes model (.joblib)
2. Tabular prediction with Heart Disease model (.pkl)
3. Decision threshold affects outcome (High vs Low risk classification)
4. Missing / invalid tabular feature handling
5. Image prediction with Pneumonia model (.keras) using synthetic X-ray bytes
6. Corrupted / empty image byte handling
7. Generic factory dispatch (get_generic_predictor)
8. PredictionResult schema contract & telemetry verification
"""

import io
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from app.ml.disease_registry import disease_registry
from app.ml.predictor import (
    GenericImagePredictor,
    GenericTabularPredictor,
    InferenceError,
    InputValidationError,
    get_generic_predictor,
)
from app.schemas.prediction import PredictionResult


def create_dummy_image_bytes(width=224, height=224, color=(128, 128, 128)) -> bytes:
    """Helper to generate valid PNG image bytes for image predictor testing."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_diabetes_tabular_prediction():
    config = disease_registry.get("diabetes")
    predictor = get_generic_predictor(config)

    assert isinstance(predictor, GenericTabularPredictor)

    # 6 features for diabetes: Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age
    sample_input = {
        "Pregnancies": 2,
        "Glucose": 160.0,
        "BloodPressure": 80.0,
        "BMI": 33.5,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    }

    result = predictor.predict(sample_input)

    assert isinstance(result, PredictionResult)
    assert result.disease_id == "diabetes"
    assert result.disease_display_name == "Diabetes Risk Assessment"
    assert result.probability is not None
    assert 0.0 <= result.probability <= 1.0
    assert result.decision_threshold == config.decision_threshold
    assert result.prediction_label in [config.positive_label, config.negative_label]
    assert "latency_ms" in result.metadata
    print(f"  PASS: 1. Diabetes prediction: prob={result.probability}, label='{result.prediction_label}'")


def test_heart_disease_tabular_prediction():
    config = disease_registry.get("heart_disease")
    predictor = get_generic_predictor(config)

    assert isinstance(predictor, GenericTabularPredictor)

    # 13 features matching heart disease config
    sample_input = {
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

    result = predictor.predict(sample_input)

    assert isinstance(result, PredictionResult)
    assert result.disease_id == "heart_disease"
    assert result.probability is not None
    assert 0.0 <= result.probability <= 1.0
    print(f"  PASS: 2. Heart Disease prediction: prob={result.probability}, label='{result.prediction_label}'")


def test_decision_threshold_evaluation():
    config = disease_registry.get("diabetes")
    predictor = GenericTabularPredictor(config)

    # Test apply_threshold directly
    is_pos_high, label_high = predictor.apply_threshold(0.80)
    assert is_pos_high is True
    assert label_high == config.positive_label

    is_pos_low, label_low = predictor.apply_threshold(0.10)
    assert is_pos_low is False
    assert label_low == config.negative_label
    print("  PASS: 3. Decision threshold evaluation verified")


def test_tabular_missing_features_rejection():
    config = disease_registry.get("diabetes")
    predictor = GenericTabularPredictor(config)

    incomplete_input = {"Glucose": 120.0}  # Missing required features

    try:
        predictor.predict(incomplete_input)
        assert False, "Should raise InputValidationError/ValueError on missing required features"
    except (InputValidationError, ValueError):
        print("  PASS: 4. Incomplete tabular features rejected cleanly")


def test_pneumonia_image_prediction():
    config = disease_registry.get("pneumonia")
    predictor = get_generic_predictor(config)

    assert isinstance(predictor, GenericImagePredictor)

    image_bytes = create_dummy_image_bytes(224, 224)
    result = predictor.predict(image_bytes)

    assert isinstance(result, PredictionResult)
    assert result.disease_id == "pneumonia"
    assert result.probability is not None
    assert 0.0 <= result.probability <= 1.0
    assert result.decision_threshold == config.decision_threshold
    assert result.prediction_label in [config.positive_label, config.negative_label]
    assert result.metadata.get("input_shape") == [1, 224, 224, 3]
    print(f"  PASS: 5. Pneumonia image prediction: prob={result.probability}, label='{result.prediction_label}'")


def test_image_invalid_payload_rejection():
    config = disease_registry.get("pneumonia")
    predictor = GenericImagePredictor(config)

    # Test empty bytes
    try:
        predictor.predict(b"")
        assert False, "Should reject empty image bytes"
    except (InputValidationError, ValueError):
        print("  PASS: 6a. Empty image byte payload rejected")

    # Test corrupted bytes
    try:
        predictor.predict(b"NOT_A_VALID_IMAGE_DATA_STREAM")
        assert False, "Should reject corrupted image bytes"
    except (InputValidationError, ValueError):
        print("  PASS: 6b. Corrupted image byte payload rejected")


def test_factory_dispatch():
    tab_config = disease_registry.get("diabetes")
    img_config = disease_registry.get("pneumonia")

    assert isinstance(get_generic_predictor(tab_config), GenericTabularPredictor)
    assert isinstance(get_generic_predictor(img_config), GenericImagePredictor)
    print("  PASS: 7. get_generic_predictor dispatch verified for both categories")


if __name__ == "__main__":
    print("\n=== Generic Predictor Layer Verification Tests ===\n")
    tests = [
        ("1. Diabetes Tabular Prediction", test_diabetes_tabular_prediction),
        ("2. Heart Disease Tabular Prediction", test_heart_disease_tabular_prediction),
        ("3. Decision Threshold Evaluation", test_decision_threshold_evaluation),
        ("4. Missing Tabular Features Rejection", test_tabular_missing_features_rejection),
        ("5. Pneumonia Image Prediction", test_pneumonia_image_prediction),
        ("6. Invalid Image Payload Rejection", test_image_invalid_payload_rejection),
        ("7. Predictor Factory Dispatch", test_factory_dispatch),
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
