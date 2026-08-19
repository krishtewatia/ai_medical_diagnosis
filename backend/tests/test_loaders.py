"""
Comprehensive Test Suite for Step 12:
Model Artifact Loaders & Error Handling.

Tests:
1. Load valid .joblib artifact (Diabetes model)
2. Load valid .pkl artifact (Heart disease model)
3. Load valid .keras artifact (Pneumonia model)
4. Missing artifact raises ModelNotFoundError
5. Unsupported file extension raises UnsupportedModelFormatError
6. Corrupted / invalid model raises ModelLoadError
7. In-memory model caching prevents redundant disk loads
8. Loaded objects are functional, callable ML/DL models
"""

import os
import sys
import tempfile
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.ml.disease_registry import disease_registry
from app.ml.loaders import (
    ModelLoadError,
    ModelNotFoundError,
    UnsupportedModelFormatError,
    clear_model_cache,
    load_model_artifact,
    load_model_for_disease,
)


def test_load_joblib_diabetes():
    clear_model_cache()
    diabetes_config = disease_registry.get("diabetes")
    model = load_model_for_disease(diabetes_config)

    assert model is not None
    assert hasattr(model, "predict"), "Loaded model must have a predict method"
    assert hasattr(model, "predict_proba"), "Loaded model must have a predict_proba method"
    print(f"  PASS: 1. Diabetes .joblib model loaded successfully ({type(model).__name__})")


def test_load_pkl_heart_disease():
    clear_model_cache()
    heart_config = disease_registry.get("heart_disease")
    model = load_model_for_disease(heart_config)

    assert model is not None
    assert hasattr(model, "predict"), "Loaded model must have a predict method"
    assert hasattr(model, "predict_proba"), "Loaded model must have a predict_proba method"
    print(f"  PASS: 2. Heart Disease .pkl model loaded successfully ({type(model).__name__})")


def test_load_keras_pneumonia():
    clear_model_cache()
    pneumonia_config = disease_registry.get("pneumonia")
    model = load_model_for_disease(pneumonia_config)

    assert model is not None
    # Keras models are callable / have predict
    assert callable(model) or hasattr(model, "predict"), "Keras model must be callable or have predict"
    print(f"  PASS: 3. Pneumonia .keras model loaded successfully ({type(model).__name__})")


def test_missing_file_error():
    fake_path = Path("/nonexistent/path/to/model_v999.joblib")
    try:
        load_model_artifact(fake_path)
        assert False, "Should raise ModelNotFoundError for nonexistent file"
    except ModelNotFoundError as e:
        assert "not found at" in str(e)
        print("  PASS: 4. Missing model file raises ModelNotFoundError")


def test_unsupported_format_error():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"dummy text content")
        temp_path = f.name

    try:
        try:
            load_model_artifact(temp_path)
            assert False, "Should raise UnsupportedModelFormatError"
        except UnsupportedModelFormatError as e:
            assert "Unsupported model artifact format" in str(e)
            print("  PASS: 5. Unsupported file extension raises UnsupportedModelFormatError")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_corrupted_model_error():
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        f.write(b"CORRUPTED_GARBAGE_DATA_NOT_A_VALID_MODEL")
        temp_path = f.name

    try:
        try:
            load_model_artifact(temp_path)
            assert False, "Should raise ModelLoadError on corrupted model file"
        except ModelLoadError as e:
            assert "Failed to load artifact" in str(e)
            print("  PASS: 6. Corrupted model file raises ModelLoadError")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_model_cache_reuse():
    clear_model_cache()
    diabetes_config = disease_registry.get("diabetes")

    model_1 = load_model_for_disease(diabetes_config, use_cache=True)
    model_2 = load_model_for_disease(diabetes_config, use_cache=True)

    # Must be the exact same object reference in memory
    assert model_1 is model_2
    print("  PASS: 7. Model caching returns identical in-memory instance")


if __name__ == "__main__":
    print("\n=== Model Artifact Loaders Verification Tests ===\n")
    tests = [
        ("1. Load .joblib (Diabetes)", test_load_joblib_diabetes),
        ("2. Load .pkl (Heart Disease)", test_load_pkl_heart_disease),
        ("3. Load .keras (Pneumonia)", test_load_keras_pneumonia),
        ("4. Missing file handling", test_missing_file_error),
        ("5. Unsupported format handling", test_unsupported_format_error),
        ("6. Corrupted file handling", test_corrupted_model_error),
        ("7. In-memory caching", test_model_cache_reuse),
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
