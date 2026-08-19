"""
Verification Test Suite for Step 14:
Generic Prediction Schemas (PredictionRequest & PredictionResponse).

Tests:
1. Valid generic prediction request is accepted
2. Missing or empty disease ID is rejected
3. Invalid request structure is rejected
4. Valid prediction response is accepted
5. Optional probability/confidence can be absent when unsupported
6. Invalid probability out of bounds (<0 or >1) is rejected
7. Threshold and metadata telemetry representation
8. Conversion from PredictionResult via from_result factory
9. Strict schema separation (no hardcoded disease fields or DB fields)
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
import pytest
from pydantic import ValidationError

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.prediction import (
    PredictionRequest,
    PredictionRequestPayload,
    PredictionResponse,
    PredictionResult,
)


def test_valid_prediction_request():
    """Verify that a valid generic PredictionRequest is accepted."""
    req = PredictionRequest(
        disease_id="diabetes",
        inputs={"Glucose": 140, "BMI": 28.5, "Age": 45}
    )
    assert req.disease_id == "diabetes"
    assert req.inputs["Glucose"] == 140
    print("  PASS: 1. Valid generic PredictionRequest accepted")


def test_missing_and_empty_disease_id_rejected():
    """Verify that missing or whitespace-only disease_id is rejected."""
    # Missing disease_id
    with pytest.raises(ValidationError):
        PredictionRequest(inputs={"Glucose": 140})

    # Empty string disease_id
    with pytest.raises(ValidationError):
        PredictionRequest(disease_id="", inputs={"Glucose": 140})

    # Whitespace disease_id
    with pytest.raises(ValidationError):
        PredictionRequest(disease_id="   ", inputs={"Glucose": 140})
    print("  PASS: 2. Missing/empty disease_id rejected")


def test_invalid_request_structure():
    """Verify that malformed request payloads are rejected."""
    # Inputs not a dict
    with pytest.raises(ValidationError):
        PredictionRequest(disease_id="diabetes", inputs="not-a-dict")

    # None inputs defaults to empty dict
    req_default = PredictionRequest(disease_id="DIABETES")
    assert req_default.disease_id == "diabetes"  # Normalized to lower
    assert req_default.inputs == {}
    print("  PASS: 3. Invalid request structure rejected, normalization verified")


def test_valid_prediction_response():
    """Verify that a complete PredictionResponse passes validation."""
    resp = PredictionResponse(
        disease_id="heart_disease",
        disease_display_name="Heart Disease Risk Assessment",
        prediction_label="High Risk of Heart Disease",
        is_positive=True,
        probability=0.885,
        decision_threshold=0.50,
        model_version="v1",
        model_type="XGBoost",
        metadata={"latency_ms": 12.4},
    )
    assert resp.disease_id == "heart_disease"
    assert resp.is_positive is True
    assert resp.probability == 0.885
    assert resp.decision_threshold == 0.50
    assert resp.metadata["latency_ms"] == 12.4
    assert isinstance(resp.timestamp, datetime)
    print("  PASS: 4. Valid PredictionResponse accepted")


def test_probability_optional_when_unsupported():
    """Verify that probability can be None for deterministic/rule-based models."""
    resp = PredictionResponse(
        disease_id="ckd",
        disease_display_name="Chronic Kidney Disease",
        prediction_label="Elevated Risk",
        is_positive=True,
        probability=None,
        decision_threshold=0.50,
        model_version="v1",
        model_type="DecisionTree",
    )
    assert resp.probability is None
    print("  PASS: 5. PredictionResponse allows None probability")


def test_probability_out_of_bounds_rejected():
    """Verify that probability values outside [0.0, 1.0] are rejected."""
    with pytest.raises(ValidationError):
        PredictionResponse(
            disease_id="diabetes",
            disease_display_name="Diabetes",
            prediction_label="High Risk",
            is_positive=True,
            probability=1.5,  # Invalid (> 1.0)
            model_version="v1",
            model_type="LogisticRegression",
        )

    with pytest.raises(ValidationError):
        PredictionResponse(
            disease_id="diabetes",
            disease_display_name="Diabetes",
            prediction_label="High Risk",
            is_positive=True,
            probability=-0.1,  # Invalid (< 0.0)
            model_version="v1",
            model_type="LogisticRegression",
        )
    print("  PASS: 6. Out-of-bounds probability rejected")


def test_response_from_result_factory():
    """Verify factory method converting PredictionResult to PredictionResponse."""
    internal_res = PredictionResult(
        disease_id="pneumonia",
        disease_display_name="Pneumonia Detection",
        model_version="v1",
        model_type="DenseNet121",
        prediction_label="Lung Opacity Detected",
        is_positive=True,
        probability=0.742,
        decision_threshold=0.50,
        metadata={"input_shape": [1, 224, 224, 3], "latency_ms": 45.2},
    )

    api_response = PredictionResponse.from_result(
        result=internal_res,
        clinical_purpose="Chest X-Ray screening tool",
        disclaimer="Not a final clinical diagnosis"
    )

    assert api_response.disease_id == "pneumonia"
    assert api_response.probability == 0.742
    assert api_response.clinical_purpose == "Chest X-Ray screening tool"
    assert api_response.disclaimer == "Not a final clinical diagnosis"
    assert api_response.metadata["input_shape"] == [1, 224, 224, 3]
    print("  PASS: 7. from_result factory serialization verified")


def test_schema_decoupling_and_separation():
    """Verify that generic schemas contain no disease-hardcoded fields or DB fields."""
    req_fields = set(PredictionRequest.model_fields.keys())
    resp_fields = set(PredictionResponse.model_fields.keys())

    # Ensure no database internal keys
    assert "_id" not in req_fields and "_id" not in resp_fields
    assert "user_id" not in req_fields and "user_id" not in resp_fields

    # Ensure no disease-specific hardcoded keys
    for field in ["Glucose", "BMI", "age", "chest_pain_type", "Pregnancies"]:
        assert field not in req_fields
        assert field not in resp_fields

    print("  PASS: 8. Strict schema separation and generic contract verified")


if __name__ == "__main__":
    print("\n=== Prediction Schemas Verification Tests ===\n")
    tests = [
        ("1. Valid Prediction Request", test_valid_prediction_request),
        ("2. Missing/Empty Disease ID Rejection", test_missing_and_empty_disease_id_rejected),
        ("3. Invalid Request Structure Rejection", test_invalid_request_structure),
        ("4. Valid Prediction Response", test_valid_prediction_response),
        ("5. Optional Probability Field", test_probability_optional_when_unsupported),
        ("6. Out-of-bounds Probability Rejection", test_probability_out_of_bounds_rejected),
        ("7. from_result Factory Serialization", test_response_from_result_factory),
        ("8. Schema Decoupling & Separation", test_schema_decoupling_and_separation),
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
