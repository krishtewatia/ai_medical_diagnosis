"""
Verification Test Suite for Step 17:
Standardized Prediction Response Format Consistency across all Diseases.

Tests:
1. Tabular binary prediction response format
2. Response with probability score
3. Response when probability is unavailable (None)
4. Multi-disease response uniformity (Diabetes, Heart Disease, CKD, Pneumonia, Brain Tumor)
5. Model versioning representation (v1, v2.0, 2026.1)
6. Decision threshold consistency
7. Medical safety disclaimer & limitation separation
8. JSON contract symmetry for frontend consumption
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
import pytest
from pydantic import ValidationError

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.prediction import PredictionResponse, PredictionResult


def test_tabular_binary_response_format():
    """1. Verify standard format for tabular binary prediction."""
    resp = PredictionResponse(
        disease_id="diabetes",
        disease_display_name="Diabetes Risk Assessment",
        prediction_label="High Risk of Diabetes",
        is_positive=True,
        probability=0.825,
        decision_threshold=0.40,
        model_version="v1",
        model_type="LogisticRegression",
        explanation="Elevated glucose and BMI contribute to high risk score.",
        clinical_purpose="Screening research prototype",
        disclaimer="Not a definitive diagnosis",
        limitations="Trained on female adult cohort",
        metadata={"latency_ms": 5.2},
    )

    data = resp.model_dump()
    assert data["disease_id"] == "diabetes"
    assert data["is_positive"] is True
    assert data["probability"] == 0.825
    assert data["decision_threshold"] == 0.40
    assert "explanation" in data
    assert "disclaimer" in data
    assert "limitations" in data
    assert isinstance(resp.timestamp, datetime)
    print("  PASS: 1. Tabular binary prediction response format verified")


def test_response_with_and_without_probability():
    """2 & 3. Verify format handles both probabilistic and non-probabilistic models."""
    # Model supporting probability
    prob_resp = PredictionResponse(
        disease_id="heart_disease",
        disease_display_name="Heart Disease",
        prediction_label="High Risk",
        is_positive=True,
        probability=0.915,
        model_version="v1",
        model_type="XGBoost",
    )
    assert prob_resp.probability == 0.915

    # Model without probability
    non_prob_resp = PredictionResponse(
        disease_id="rule_based_screener",
        disease_display_name="Rule Based Screener",
        prediction_label="Normal",
        is_positive=False,
        probability=None,
        model_version="v1",
        model_type="DeterministicRuleEngine",
    )
    assert non_prob_resp.probability is None
    print("  PASS: 2 & 3. Both probabilistic and non-probabilistic responses verified")


def test_multi_disease_format_uniformity():
    """4. Verify response uniformity across diverse diseases (Tabular & Image)."""
    diseases = [
        ("diabetes", "Diabetes Risk Assessment", "LogisticRegression", "High Risk of Diabetes", True, 0.78),
        ("heart_disease", "Heart Disease Assessment", "XGBoost", "Low Risk of Heart Disease", False, 0.22),
        ("ckd", "Chronic Kidney Disease", "RandomForest", "Elevated Risk of CKD", True, 0.65),
        ("pneumonia", "Pneumonia Detection", "DenseNet121", "Lung Opacity Detected", True, 0.89),
        ("brain_tumor", "Brain Tumor Detection", "ResNet50", "No Neoplasm Detected", False, 0.08),
    ]

    expected_keys = {
        "disease_id",
        "disease_display_name",
        "prediction_label",
        "is_positive",
        "probability",
        "decision_threshold",
        "model_version",
        "model_type",
        "explanation",
        "clinical_purpose",
        "disclaimer",
        "limitations",
        "metadata",
        "timestamp",
    }

    for d_id, name, m_type, label, pos, prob in diseases:
        resp = PredictionResponse(
            disease_id=d_id,
            disease_display_name=name,
            prediction_label=label,
            is_positive=pos,
            probability=prob,
            model_version="v1.0",
            model_type=m_type,
        )
        assert set(resp.model_dump().keys()) == expected_keys
        assert resp.disease_id == d_id
        assert resp.is_positive == pos

    print("  PASS: 4. Format uniformity verified across 5 distinct disease domains")


def test_model_versioning_and_thresholds():
    """5 & 6. Verify version strings and varying decision thresholds."""
    resp1 = PredictionResponse(
        disease_id="diabetes",
        disease_display_name="Diabetes",
        prediction_label="Positive",
        is_positive=True,
        probability=0.45,
        decision_threshold=0.40,  # Lower threshold marks 0.45 as positive
        model_version="v2.1-rc1",
        model_type="Ensemble",
    )
    assert resp1.model_version == "v2.1-rc1"
    assert resp1.decision_threshold == 0.40

    resp2 = PredictionResponse(
        disease_id="diabetes",
        disease_display_name="Diabetes",
        prediction_label="Negative",
        is_positive=False,
        probability=0.45,
        decision_threshold=0.60,  # Higher threshold marks 0.45 as negative
        model_version="v2.1-rc1",
        model_type="Ensemble",
    )
    assert resp2.decision_threshold == 0.60
    print("  PASS: 5 & 6. Model versions and threshold representations verified")


def test_medical_safety_distinction():
    """7. Verify explicit separation of ML output from medical diagnosis."""
    res = PredictionResult(
        disease_id="pneumonia",
        disease_display_name="Pneumonia Detection",
        model_version="v1",
        model_type="DenseNet121",
        prediction_label="Lung Opacity Detected",
        is_positive=True,
        probability=0.88,
        decision_threshold=0.50,
        metadata={"latency_ms": 35.0}
    )

    resp = PredictionResponse.from_result(
        result=res,
        clinical_purpose="Research Screening Prototype",
        disclaimer="AI-generated output. Not a definitive clinical diagnosis. Consult a qualified medical practitioner.",
        limitations="Performance depends on frontal chest radiography quality."
    )

    # Verify classification vs safety notice separation
    assert resp.prediction_label == "Lung Opacity Detected"
    assert "Not a definitive clinical diagnosis" in resp.disclaimer
    assert "Research Screening" in resp.clinical_purpose
    assert "radiography quality" in resp.limitations
    assert resp.metadata["latency_ms"] == 35.0
    print("  PASS: 7. Medical safety distinction and disclaimer integrity verified")


def test_json_contract_symmetry_for_frontend():
    """8. Verify JSON serialization symmetry for consistent frontend ingestion."""
    resp = PredictionResponse(
        disease_id="diabetes",
        disease_display_name="Diabetes Screening",
        prediction_label="High Risk of Diabetes",
        is_positive=True,
        probability=0.75,
        decision_threshold=0.50,
        model_version="v1",
        model_type="LogisticRegression",
    )

    json_dict = resp.model_dump(mode="json")
    assert isinstance(json_dict["timestamp"], str)  # ISO format string for JSON
    assert isinstance(json_dict["probability"], float)
    assert isinstance(json_dict["is_positive"], bool)
    assert isinstance(json_dict["metadata"], dict)
    print("  PASS: 8. JSON contract symmetry for frontend consumption verified")


if __name__ == "__main__":
    print("\n=== Standard Prediction Response Format Verification Tests ===\n")
    tests = [
        ("1. Tabular Binary Response Format", test_tabular_binary_response_format),
        ("2. Probability Support & None Optionality", test_response_with_and_without_probability),
        ("3. Multi-disease Format Uniformity", test_multi_disease_format_uniformity),
        ("4. Versioning & Threshold Representation", test_model_versioning_and_thresholds),
        ("5. Medical Safety & Disclaimer Distinction", test_medical_safety_distinction),
        ("6. JSON Serialization Symmetry for Frontend", test_json_contract_symmetry_for_frontend),
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
