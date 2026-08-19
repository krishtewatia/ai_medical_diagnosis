"""
Verification Test Suite for Step 10:
Common Model / Predictor Interface & Contract.

Tests:
1. Abstract base class enforcement (cannot instantiate incomplete subclasses)
2. Category validation on predictor initialization
3. Concrete Tabular Predictor contract adherence (Diabetes / Heart Disease simulation)
4. Concrete Image Predictor contract adherence (Pneumonia simulation)
5. Decision threshold application logic
6. Standardized PredictionResult output consistency across both domains
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.base_predictor import (
    BaseImagePredictor,
    BasePredictor,
    BaseTabularPredictor,
)
from app.ml.disease_registry import disease_registry
from app.schemas.disease_config import DiseaseCategory, DiseaseConfig
from app.schemas.prediction import PredictionResult


# --- Mock Implementations to Verify the Contract ---

class MockDiabetesTabularPredictor(BaseTabularPredictor):
    def load_model(self) -> Any:
        return "mock_logistic_regression_model"

    def preprocess_features(self, features: Dict[str, Any]) -> list:
        self.validate_feature_names(features)
        return [features[f.name] for f in self.config.tabular_features]

    def predict(self, input_data: Dict[str, Any]) -> PredictionResult:
        processed = self.preprocess_features(input_data)
        glucose = float(input_data.get("Glucose", 100.0))
        simulated_prob = 0.85 if glucose > 140 else 0.15

        is_positive, label = self.apply_threshold(simulated_prob)

        return PredictionResult(
            disease_id=self.disease_id,
            disease_display_name=self.config.display_name,
            model_version=self.config.version,
            model_type=self.config.model_type,
            prediction_label=label,
            is_positive=is_positive,
            probability=simulated_prob,
            decision_threshold=self.config.decision_threshold,
            metadata={"simulated": True, "feature_count": len(processed)},
        )


class MockPneumoniaImagePredictor(BaseImagePredictor):
    def load_model(self) -> Any:
        return "mock_densenet121_model"

    def preprocess_image(self, image_bytes: bytes) -> tuple:
        if len(image_bytes) == 0:
            raise ValueError("Image bytes cannot be empty.")
        return (1, 224, 224, 3)

    def predict(self, input_data: bytes) -> PredictionResult:
        shape = self.preprocess_image(input_data)
        simulated_prob = 0.42  # Exceeds 0.30 threshold

        is_positive, label = self.apply_threshold(simulated_prob)

        return PredictionResult(
            disease_id=self.disease_id,
            disease_display_name=self.config.display_name,
            model_version=self.config.version,
            model_type=self.config.model_type,
            prediction_label=label,
            is_positive=is_positive,
            probability=simulated_prob,
            decision_threshold=self.config.decision_threshold,
            metadata={"simulated": True, "tensor_shape": list(shape)},
        )


def test_abstract_instantiation_blocked():
    diabetes_config = disease_registry.get("diabetes")
    try:
        BasePredictor(diabetes_config)
        assert False, "BasePredictor direct instantiation should fail"
    except TypeError:
        print("  PASS: 1. Direct instantiation of abstract BasePredictor is blocked")


def test_category_mismatch_validation():
    pneumonia_config = disease_registry.get("pneumonia")  # Image config
    try:
        MockDiabetesTabularPredictor(pneumonia_config)
        assert False, "Initializing Tabular predictor with image config should fail"
    except ValueError:
        print("  PASS: 2. Predictor category mismatch raises ValueError")


def test_tabular_predictor_flow():
    diabetes_config = disease_registry.get("diabetes")
    predictor = MockDiabetesTabularPredictor(diabetes_config)

    # 1. Test missing feature rejection
    incomplete_inputs = {"Glucose": 160.0, "BMI": 28.0}
    try:
        predictor.predict(incomplete_inputs)
        assert False, "Predictor should reject incomplete required features"
    except ValueError as e:
        assert "Missing required feature" in str(e)
        print("  PASS: 3a. Tabular predictor validates missing required features")

    # 2. Test high risk prediction (Glucose = 165 -> prob = 0.85 >= threshold 0.40)
    full_high_risk = {
        "Pregnancies": 2,
        "Glucose": 165.0,
        "BloodPressure": 80.0,
        "BMI": 32.0,
        "DiabetesPedigreeFunction": 0.65,
        "Age": 45,
    }
    result_high = predictor.predict(full_high_risk)
    assert isinstance(result_high, PredictionResult)
    assert result_high.is_positive is True
    assert result_high.prediction_label == "High Risk of Diabetes"
    assert result_high.probability == 0.85
    assert result_high.decision_threshold == 0.40
    print("  PASS: 3b. Tabular high-risk prediction produces valid PredictionResult")

    # 3. Test low risk prediction (Glucose = 95 -> prob = 0.15 < threshold 0.40)
    full_low_risk = {**full_high_risk, "Glucose": 95.0}
    result_low = predictor.predict(full_low_risk)
    assert result_low.is_positive is False
    assert result_low.prediction_label == "Low Risk of Diabetes"
    assert result_low.probability == 0.15
    print("  PASS: 3c. Tabular low-risk prediction correctly evaluates threshold")


def test_image_predictor_flow():
    pneumonia_config = disease_registry.get("pneumonia")
    predictor = MockPneumoniaImagePredictor(pneumonia_config)

    # 1. Test empty bytes rejection
    try:
        predictor.predict(b"")
        assert False, "Should reject empty image bytes"
    except ValueError:
        print("  PASS: 4a. Image predictor rejects empty byte payload")

    # 2. Test valid image inference (prob = 0.42 >= threshold 0.30)
    dummy_xray_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 500
    result = predictor.predict(dummy_xray_bytes)

    assert isinstance(result, PredictionResult)
    assert result.disease_id == "pneumonia"
    assert result.is_positive is True
    assert result.prediction_label == "Lung Opacity Detected"
    assert result.probability == 0.42
    assert result.decision_threshold == 0.30
    print("  PASS: 4b. Image prediction produces standardized PredictionResult")


def test_unified_contract_symmetry():
    diabetes_config = disease_registry.get("diabetes")
    pneumonia_config = disease_registry.get("pneumonia")

    tabular_pred = MockDiabetesTabularPredictor(diabetes_config)
    image_pred = MockPneumoniaImagePredictor(pneumonia_config)

    tab_res = tabular_pred.predict({
        "Pregnancies": 0, "Glucose": 100.0, "BloodPressure": 70.0,
        "BMI": 22.0, "DiabetesPedigreeFunction": 0.2, "Age": 25
    })
    img_res = image_pred.predict(b"fake_image_bytes")

    # Verify identical contract schema structure
    assert set(tab_res.model_dump().keys()) == set(img_res.model_dump().keys())
    print("  PASS: 5. Unified contract symmetry between Tabular and Image results verified")


if __name__ == "__main__":
    print("\n=== Common Predictor Interface & Contract Verification ===\n")
    tests = [
        ("1. Abstract Instantiation Check", test_abstract_instantiation_blocked),
        ("2. Category Mismatch Check", test_category_mismatch_validation),
        ("3. Tabular Predictor Flow", test_tabular_predictor_flow),
        ("4. Image Predictor Flow", test_image_predictor_flow),
        ("5. Contract Symmetry Check", test_unified_contract_symmetry),
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
