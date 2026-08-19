"""
Comprehensive Schema Test Suite for Step 23.2:
Prediction History Schemas (PredictionHistoryCreate, PredictionHistoryResponse, PredictionModelInfo, PredictionResultRecord).

Tests:
1. Valid Diabetes history record
2. Valid Heart Disease history record
3. Image-modality history record with file metadata
4. Probability boundary validation (0.0, 1.0, <0 rejected, >1 rejected)
5. Confidence validation (None accepted, 0.0-1.0 accepted, out-of-bounds rejected)
6. Input modality validation (tabular, image accepted; invalid rejected)
7. Missing required fields rejected
8. Valid timestamps and ObjectId string conversions
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
import pytest
from pydantic import ValidationError

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.prediction_history import (
    PredictionHistoryCreate,
    PredictionHistoryResponse,
    PredictionModelInfo,
    PredictionResultRecord,
)


def test_valid_diabetes_history_record():
    """1. Verify valid complete Diabetes history record."""
    create_schema = PredictionHistoryCreate(
        user_id="66be50000000000000000001",
        disease="diabetes",
        disease_display_name="Diabetes Risk Assessment",
        input_type="tabular",
        model=PredictionModelInfo(
            version="v1",
            model_type="LogisticRegression",
            threshold=0.40,
        ),
        input_data={
            "Pregnancies": 2,
            "Glucose": 145.0,
            "BloodPressure": 80.0,
            "BMI": 28.5,
            "DiabetesPedigreeFunction": 0.55,
            "Age": 42,
        },
        result=PredictionResultRecord(
            prediction="High Risk of Diabetes",
            is_positive=True,
            probability=0.825,
            confidence=None,
        ),
        explanation="Model evaluation resulted in elevated risk.",
        metadata={"source": "api", "latency_ms": 1.45, "features_evaluated": 6},
    )

    assert create_schema.disease == "diabetes"
    assert create_schema.input_type == "tabular"
    assert create_schema.result.probability == 0.825

    # Simulate response representation
    response_schema = PredictionHistoryResponse(
        id="66bf1a34e123456789abcdef",
        user_id=create_schema.user_id,
        disease=create_schema.disease,
        disease_display_name=create_schema.disease_display_name,
        input_type=create_schema.input_type,
        model=create_schema.model,
        input_data=create_schema.input_data,
        result=create_schema.result,
        explanation=create_schema.explanation,
        created_at=create_schema.created_at,
        metadata=create_schema.metadata,
    )
    assert response_schema.id == "66bf1a34e123456789abcdef"
    assert isinstance(response_schema.created_at, datetime)
    print("  PASS: 1. Valid complete Diabetes history record verified")


def test_valid_heart_disease_history_record():
    """2. Verify valid complete Heart Disease history record."""
    rec = PredictionHistoryCreate(
        user_id="66be50000000000000000002",
        disease="heart_disease",
        disease_display_name="Heart Disease Risk Assessment",
        input_type="tabular",
        model=PredictionModelInfo(
            version="v1",
            model_type="XGBoost",
            threshold=0.40,
        ),
        input_data={
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
        },
        result=PredictionResultRecord(
            prediction="High Risk of Heart Disease",
            is_positive=True,
            probability=0.9495,
            confidence=None,
        ),
        explanation="High cardiovascular risk detected.",
        metadata={"source": "api", "latency_ms": 2.1, "features_evaluated": 13},
    )

    assert rec.disease == "heart_disease"
    assert rec.model.model_type == "XGBoost"
    assert rec.result.is_positive is True
    print("  PASS: 2. Valid complete Heart Disease history record verified")


def test_valid_image_history_record():
    """3. Verify image-type history record with file reference metadata."""
    img_rec = PredictionHistoryCreate(
        user_id="66be50000000000000000003",
        disease="pneumonia",
        disease_display_name="Pneumonia Chest X-ray Screening",
        input_type="image",
        model=PredictionModelInfo(
            version="v1",
            model_type="DenseNet121",
            threshold=0.30,
        ),
        input_data={
            "filename": "chest_xray_frontal.png",
            "content_type": "image/png",
            "file_size_bytes": 1048576,
            "image_dimensions": [224, 224],
            "storage_ref": "uploads/pneumonia/user_123/scan_abc.png",
        },
        result=PredictionResultRecord(
            prediction="Lung Opacity Detected",
            is_positive=True,
            probability=0.892,
            confidence=None,
        ),
        explanation="Deep learning model detected patterns consistent with lung opacity.",
        metadata={"source": "api", "latency_ms": 32.5},
    )

    assert img_rec.input_type == "image"
    assert img_rec.input_data["filename"] == "chest_xray_frontal.png"
    assert "storage_ref" in img_rec.input_data
    print("  PASS: 3. Image-type history record with file metadata verified")


def test_probability_boundaries():
    """4. Verify probability boundary rules (0.0, 1.0, <0 rejected, >1 rejected)."""
    # 0.0 valid
    r_zero = PredictionResultRecord(prediction="Neg", is_positive=False, probability=0.0)
    assert r_zero.probability == 0.0

    # 1.0 valid
    r_one = PredictionResultRecord(prediction="Pos", is_positive=True, probability=1.0)
    assert r_one.probability == 1.0

    # None valid
    r_none = PredictionResultRecord(prediction="Pos", is_positive=True, probability=None)
    assert r_none.probability is None

    # < 0.0 invalid
    with pytest.raises(ValidationError):
        PredictionResultRecord(prediction="Neg", is_positive=False, probability=-0.01)

    # > 1.0 invalid
    with pytest.raises(ValidationError):
        PredictionResultRecord(prediction="Pos", is_positive=True, probability=1.01)

    print("  PASS: 4. Probability boundary checks (0.0, 1.0, null, negatives, >1) verified")


def test_confidence_boundaries_and_optionality():
    """5. Verify confidence field validation and optional nullability."""
    # None valid
    c_none = PredictionResultRecord(prediction="P", is_positive=True, confidence=None)
    assert c_none.confidence is None

    # 0.75 valid
    c_val = PredictionResultRecord(prediction="P", is_positive=True, confidence=0.75)
    assert c_val.confidence == 0.75

    # Out of bounds invalid
    with pytest.raises(ValidationError):
        PredictionResultRecord(prediction="P", is_positive=True, confidence=-0.5)

    with pytest.raises(ValidationError):
        PredictionResultRecord(prediction="P", is_positive=True, confidence=1.5)

    print("  PASS: 5. Confidence optionality and boundary validation verified")


def test_input_modality_validation():
    """6. Verify input_type allows only 'tabular' or 'image'."""
    # Valid
    m_tab = PredictionModelInfo(version="v1", model_type="LR")
    res = PredictionResultRecord(prediction="P", is_positive=True)

    PredictionHistoryCreate(
        user_id="u1", disease="d", disease_display_name="D", input_type="tabular",
        model=m_tab, input_data={}, result=res
    )
    PredictionHistoryCreate(
        user_id="u1", disease="d", disease_display_name="D", input_type="image",
        model=m_tab, input_data={}, result=res
    )

    # Invalid modality
    with pytest.raises(ValidationError):
        PredictionHistoryCreate(
            user_id="u1", disease="d", disease_display_name="D", input_type="audio",
            model=m_tab, input_data={}, result=res
        )

    print("  PASS: 6. Input modality validation ('tabular'/'image' only) verified")


def test_missing_required_fields_rejected():
    """7. Verify missing required fields cause validation failures."""
    m_tab = PredictionModelInfo(version="v1", model_type="LR")
    res = PredictionResultRecord(prediction="P", is_positive=True)

    # Missing user_id
    with pytest.raises(ValidationError):
        PredictionHistoryCreate(
            disease="diabetes", disease_display_name="Diabetes", input_type="tabular",
            model=m_tab, input_data={}, result=res
        )

    # Missing model
    with pytest.raises(ValidationError):
        PredictionHistoryCreate(
            user_id="u1", disease="diabetes", disease_display_name="Diabetes", input_type="tabular",
            input_data={}, result=res
        )

    # Missing result
    with pytest.raises(ValidationError):
        PredictionHistoryCreate(
            user_id="u1", disease="diabetes", disease_display_name="Diabetes", input_type="tabular",
            model=m_tab, input_data={}
        )

    print("  PASS: 7. Missing required fields rejected by Pydantic schema")


def test_timestamps_and_id_serialization():
    """8. Verify JSON dump and ISO timestamp handling."""
    now = datetime.now(timezone.utc)
    resp = PredictionHistoryResponse(
        id="66bf1a34e123456789abcdef",
        user_id="66be50000000000000000001",
        disease="diabetes",
        disease_display_name="Diabetes Risk Assessment",
        input_type="tabular",
        model=PredictionModelInfo(version="v1", model_type="LR", threshold=0.40),
        input_data={"Glucose": 140},
        result=PredictionResultRecord(prediction="High Risk", is_positive=True, probability=0.85),
        created_at=now,
    )

    dump = resp.model_dump(mode="json")
    assert dump["id"] == "66bf1a34e123456789abcdef"
    assert isinstance(dump["created_at"], str)
    assert dump["result"]["probability"] == 0.85
    print("  PASS: 8. Timestamps and schema serialization verified")


if __name__ == "__main__":
    print("\n=== Prediction History Schemas Verification Tests ===\n")
    tests = [
        ("1. Valid Diabetes Record", test_valid_diabetes_history_record),
        ("2. Valid Heart Disease Record", test_valid_heart_disease_history_record),
        ("3. Valid Image Record", test_valid_image_history_record),
        ("4. Probability Boundary Checks", test_probability_boundaries),
        ("5. Confidence Boundary Checks", test_confidence_boundaries_and_optionality),
        ("6. Input Modality Validation", test_input_modality_validation),
        ("7. Missing Required Fields", test_missing_required_fields_rejected),
        ("8. Timestamp & Serialization", test_timestamps_and_id_serialization),
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
