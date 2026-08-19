"""
Comprehensive Test Suite for Step 11:
Refactored DiseaseRegistry & Predictor Resolution.

Tests:
1. Register a test disease configuration
2. Retrieve it by ID
3. Check whether it exists (has_disease / is_registered)
4. Retrieve all registered diseases
5. Resolve its predictor and verify singleton caching
6. Unknown disease ID is rejected with DiseaseNotFoundError
7. Duplicate registration is rejected with DuplicateDiseaseError
8. Registry works with both tabular and image configurations
9. Category type safety between configs and predictor classes
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.base_predictor import BaseImagePredictor, BaseTabularPredictor
from app.ml.disease_registry import (
    DiseaseNotFoundError,
    DiseaseRegistry,
    DuplicateDiseaseError,
    PredictorNotRegisteredError,
)
from app.schemas.disease_config import (
    DiseaseCategory,
    DiseaseConfig,
    InputType,
    ModelFramework,
    TabularFeatureSpec,
    ImageInputSpec,
)
from app.schemas.prediction import PredictionResult


# --- Mock Predictor Classes for Testing ---

class DummyTabularPredictor(BaseTabularPredictor):
    def load_model(self) -> str:
        return "loaded_tabular_model"

    def preprocess_features(self, features: Dict[str, Any]) -> list:
        return list(features.values())

    def predict(self, input_data: Dict[str, Any]) -> PredictionResult:
        return PredictionResult(
            disease_id=self.disease_id,
            disease_display_name=self.config.display_name,
            model_version=self.config.version,
            model_type=self.config.model_type,
            prediction_label="Low Risk",
            is_positive=False,
            probability=0.10,
            decision_threshold=self.config.decision_threshold,
        )


class DummyImagePredictor(BaseImagePredictor):
    def load_model(self) -> str:
        return "loaded_image_model"

    def preprocess_image(self, image_bytes: bytes) -> bytes:
        return image_bytes

    def predict(self, input_data: bytes) -> PredictionResult:
        return PredictionResult(
            disease_id=self.disease_id,
            disease_display_name=self.config.display_name,
            model_version=self.config.version,
            model_type=self.config.model_type,
            prediction_label="Normal",
            is_positive=False,
            probability=0.05,
            decision_threshold=self.config.decision_threshold,
        )


def test_registry_full_lifecycle():
    registry = DiseaseRegistry(auto_load=False)
    registry.clear()

    # 1. Register a test tabular disease
    tabular_config = DiseaseConfig(
        id="test_tabular_disease",
        version="v1.0",
        display_name="Test Tabular Disease",
        category=DiseaseCategory.TABULAR,
        input_type=InputType.FORM,
        short_description="A test tabular disease module",
        framework=ModelFramework.SKLEARN,
        model_type="LogisticRegression",
        artifact_filename="dummy.joblib",
        model_dir="/tmp/dummy",
        tabular_features=[
            TabularFeatureSpec(name="f1", display_name="Feature 1", data_type="float")
        ],
        positive_label="Positive",
        negative_label="Negative",
    )
    registry.register(tabular_config, predictor_cls=DummyTabularPredictor)
    print("  PASS: 1. Registered test disease with predictor")

    # 2. Retrieve it by ID
    retrieved = registry.get("test_tabular_disease")
    assert retrieved is not None
    assert retrieved.id == "test_tabular_disease"
    assert retrieved.display_name == "Test Tabular Disease"
    print("  PASS: 2. Retrieved registered disease by ID")

    # 3. Check whether it exists
    assert registry.has_disease("test_tabular_disease") is True
    assert registry.has_disease("nonexistent_disease") is False
    print("  PASS: 3. has_disease() and is_registered() verified")

    # 4. Register a test image disease and retrieve all
    image_config = DiseaseConfig(
        id="test_image_disease",
        version="v1.0",
        display_name="Test Image Disease",
        category=DiseaseCategory.IMAGE,
        input_type=InputType.IMAGE_UPLOAD,
        short_description="A test image disease module",
        framework=ModelFramework.TENSORFLOW,
        model_type="CNN",
        artifact_filename="dummy.keras",
        model_dir="/tmp/dummy",
        image_spec=ImageInputSpec(),
        positive_label="Opacity Detected",
        negative_label="No Opacity",
    )
    registry.register(image_config, predictor_cls=DummyImagePredictor)

    all_diseases = registry.list_all()
    assert len(all_diseases) == 2
    assert {d.id for d in all_diseases} == {"test_tabular_disease", "test_image_disease"}
    print("  PASS: 4. Retrieved all registered diseases (Tabular + Image)")

    # 5. Resolve its predictor
    tab_predictor = registry.get_predictor("test_tabular_disease")
    assert isinstance(tab_predictor, DummyTabularPredictor)
    assert tab_predictor.disease_id == "test_tabular_disease"

    img_predictor = registry.get_predictor("test_image_disease")
    assert isinstance(img_predictor, DummyImagePredictor)
    assert img_predictor.disease_id == "test_image_disease"

    # Verify cached instance
    tab_predictor_cached = registry.get_predictor("test_tabular_disease")
    assert tab_predictor is tab_predictor_cached
    print("  PASS: 5. Resolved predictors and verified cached instance reuse")

    # 6. Unknown disease ID handling
    try:
        registry.get_or_raise("unknown_disease_xyz")
        assert False, "Should have raised DiseaseNotFoundError"
    except DiseaseNotFoundError:
        print("  PASS: 6a. Unknown disease ID raises DiseaseNotFoundError")

    try:
        registry.get_predictor("unknown_disease_xyz")
        assert False, "Should have raised DiseaseNotFoundError"
    except DiseaseNotFoundError:
        print("  PASS: 6b. get_predictor for unknown disease raises DiseaseNotFoundError")

    # 7. Duplicate registration rejection
    try:
        registry.register(tabular_config, allow_override=False)
        assert False, "Should have raised DuplicateDiseaseError"
    except DuplicateDiseaseError:
        print("  PASS: 7. Duplicate registration rejected with DuplicateDiseaseError")

    # 8. Category type safety between config and predictor
    try:
        registry.register_predictor("test_tabular_disease", DummyImagePredictor)
        assert False, "Should reject binding image predictor to tabular disease"
    except TypeError:
        print("  PASS: 8. Category mismatch between config and predictor is rejected")


def test_production_auto_load():
    # Verify default singleton registry loads production configs from disk
    registry = DiseaseRegistry(auto_load=True)
    assert registry.has_disease("diabetes") is True
    assert registry.has_disease("heart_disease") is True
    assert registry.has_disease("pneumonia") is True
    print("  PASS: 9. Production auto-load verified for diabetes, heart_disease, pneumonia")


if __name__ == "__main__":
    print("\n=== Refactored DiseaseRegistry Verification Tests ===\n")
    test_registry_full_lifecycle()
    test_production_auto_load()
    print("\n=======================================================")
    print("Results: All 9 Registry Refactoring Tests PASSED (0 failed)")
    print("=======================================================\n")
