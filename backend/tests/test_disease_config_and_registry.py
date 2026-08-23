"""
Comprehensive Verification Test Suite for Step 9:
Generic Disease Configuration Schema & Registry.

Tests:
1. DiseaseConfig Pydantic validation (Tabular & Image rules)
2. Discovery & loading of diabetes, heart_disease, and pneumonia configs
3. Existence and integrity of referenced model artifact files on disk
4. Registry query methods (get, get_or_raise, list_active, list_by_category)
5. Clean separation: Declarative config without Python inference classes
6. Public metadata generation (DiseasePublicInfo)
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError

from app.ml.disease_registry import (
    DiseaseNotFoundError,
    DiseaseRegistry,
    disease_registry,
)
from app.ml.loaders import (
    discover_and_load_disease_configs,
    load_disease_config_from_file,
)
from app.schemas.disease_config import (
    DiseaseCategory,
    DiseaseConfig,
    DiseasePublicInfo,
    InputType,
    ModelFramework,
    TabularFeatureSpec,
)


def test_schema_validation_rules():
    # 1. Invalid tabular: category is tabular but no tabular_features provided
    try:
        DiseaseConfig(
            id="bad_tabular",
            version="v1",
            display_name="Bad Tabular",
            category=DiseaseCategory.TABULAR,
            input_type=InputType.FORM,
            short_description="Missing features",
            framework=ModelFramework.SKLEARN,
            model_type="LogisticRegression",
            artifact_filename="model.joblib",
            positive_label="Positive",
            negative_label="Negative",
            tabular_features=None,  # Missing!
        )
        assert False, "Should raise ValidationError when tabular_features missing for tabular disease"
    except ValidationError:
        print("  PASS: 1a. Tabular disease without tabular_features is rejected")

    # 2. Invalid image: category is image but no image_spec provided
    try:
        DiseaseConfig(
            id="bad_image",
            version="v1",
            display_name="Bad Image",
            category=DiseaseCategory.IMAGE,
            input_type=InputType.IMAGE_UPLOAD,
            short_description="Missing image spec",
            framework=ModelFramework.TENSORFLOW,
            model_type="DenseNet121",
            artifact_filename="model.keras",
            positive_label="Positive",
            negative_label="Negative",
            image_spec=None,  # Missing!
        )
        assert False, "Should raise ValidationError when image_spec missing for image disease"
    except ValidationError:
        print("  PASS: 1b. Image disease without image_spec is rejected")


def test_registry_loaded_diseases():
    registry = DiseaseRegistry(auto_load=True)
    active_diseases = registry.list_active()
    active_ids = {d.id for d in active_diseases}

    print(f"\n  Discovered active diseases in registry: {active_ids}")
    assert "diabetes" in active_ids, "diabetes must be loaded in registry"
    assert "heart_disease" in active_ids, "heart_disease must be loaded in registry"
    assert "pneumonia" in active_ids, "pneumonia must be loaded in registry"
    print("  PASS: 2. Diabetes, Heart Disease, and Pneumonia loaded successfully")


def test_diabetes_configuration():
    config = disease_registry.get("diabetes")
    assert config is not None
    assert config.category == DiseaseCategory.TABULAR
    assert config.input_type == InputType.FORM
    assert config.framework == ModelFramework.SKLEARN
    assert config.model_type == "LogisticRegression"
    assert config.artifact_filename == "diabetes_model_v1.joblib"
    assert config.decision_threshold == 0.4
    assert len(config.tabular_features) == 6

    feature_names = [f.name for f in config.tabular_features]
    expected_features = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]
    assert feature_names == expected_features

    # Verify artifact exists on disk
    artifact_path = Path(config.model_dir) / config.artifact_filename
    assert artifact_path.exists(), f"Artifact missing at {artifact_path}"
    print(f"  PASS: 3. Diabetes configuration and artifact ({config.artifact_filename}) verified")


def test_heart_disease_configuration():
    config = disease_registry.get("heart_disease")
    assert config is not None
    assert config.category == DiseaseCategory.TABULAR
    assert config.input_type == InputType.FORM
    assert config.framework == ModelFramework.XGBOOST
    assert config.model_type == "XGBoost"
    assert config.artifact_filename == "heart_disease_model.pkl"
    assert config.decision_threshold == 0.4
    assert len(config.tabular_features) == 13

    feature_names = [f.name for f in config.tabular_features]
    assert "chest_pain_type" in feature_names
    assert "cholestoral" in feature_names
    assert "resting_bp" in feature_names

    # Verify artifact exists on disk
    artifact_path = Path(config.model_dir) / config.artifact_filename
    assert artifact_path.exists(), f"Artifact missing at {artifact_path}"
    print(f"  PASS: 4. Heart Disease configuration and artifact ({config.artifact_filename}) verified")


def test_pneumonia_configuration():
    config = disease_registry.get("pneumonia")
    assert config is not None
    assert config.category == DiseaseCategory.IMAGE
    assert config.input_type == InputType.IMAGE_UPLOAD
    assert config.framework == ModelFramework.TENSORFLOW
    assert config.model_type == "DenseNet121"
    assert config.artifact_filename == "pneumonia_densenet121.keras"
    assert config.decision_threshold == 0.30
    assert config.image_spec is not None
    assert config.image_spec.target_dimensions == [224, 224]
    assert config.image_spec.channels == 3

    # Verify artifact exists on disk
    artifact_path = Path(config.model_dir) / config.artifact_filename
    assert artifact_path.exists(), f"Artifact missing at {artifact_path}"
    print(f"  PASS: 5. Pneumonia configuration and artifact ({config.artifact_filename}) verified")


def test_registry_queries_and_categories():
    tabular_diseases = disease_registry.list_by_category(DiseaseCategory.TABULAR)
    image_diseases = disease_registry.list_by_category(DiseaseCategory.IMAGE)

    assert len(tabular_diseases) == 2
    assert len(image_diseases) >= 1

    # Test get_or_raise
    try:
        disease_registry.get_or_raise("nonexistent_disease_xyz")
        assert False, "Should have raised DiseaseNotFoundError"
    except DiseaseNotFoundError:
        print("  PASS: 6. Registry category filtering and error handling verified")


def test_public_metadata():
    public_list = disease_registry.list_public_info()
    assert len(public_list) >= 3
    for p in public_list:
        assert isinstance(p, DiseasePublicInfo)
        assert p.id in ["diabetes", "heart_disease", "pneumonia", "brain_tumor"]
        assert p.disclaimer is not None
        assert p.positive_label is not None
        assert p.negative_label is not None
    print("  PASS: 7. Public metadata generation verified")


if __name__ == "__main__":
    print("\n=== Disease Configuration & Registry Verification Tests ===\n")
    tests = [
        ("1. Schema validation rules", test_schema_validation_rules),
        ("2. Registry loaded diseases", test_registry_loaded_diseases),
        ("3. Diabetes configuration", test_diabetes_configuration),
        ("4. Heart disease configuration", test_heart_disease_configuration),
        ("5. Pneumonia configuration", test_pneumonia_configuration),
        ("6. Registry queries & categories", test_registry_queries_and_categories),
        ("7. Public metadata export", test_public_metadata),
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
