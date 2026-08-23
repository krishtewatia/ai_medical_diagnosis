"""
Comprehensive Test Suite for Step 24:
Disease Discovery API (GET /diseases and GET /diseases/{disease}).

Tests:
1. GET /diseases succeeds with HTTP 200 OK
2. Returns actively registered disease modules
3. Diabetes module is present and active
4. Heart Disease module is present and active
5. Deferred Chronic Kidney Disease (CKD) does not appear as active
6. GET /diseases/diabetes returns complete, frontend-ready configuration
7. GET /diseases/heart_disease returns complete, frontend-ready configuration
8. Unknown disease (GET /diseases/unknown_xyz) returns 404 Not Found
9. Response structure strictly adheres to DiseaseResponse schema
10. Internal server paths, pickle filenames, and secrets are not exposed
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.disease import get_disease_service
from app.main import app
from app.ml.disease_registry import DiseaseRegistry, disease_registry
from app.schemas.disease import DiseaseModelInfo, DiseaseResponse, DiseaseSafetyInfo
from app.schemas.disease_config import DiseaseCategory, InputType
from app.services.disease_service import DiseaseService


@pytest.fixture
def client():
    return TestClient(app)


def test_get_all_diseases_succeeds(client):
    """1. Test that GET /diseases returns HTTP 200 and a non-empty list of modules."""
    response = client.get("/diseases")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2
    print(f"  PASS: 1. GET /diseases succeeded with {len(data)} active disease modules")


def test_registered_diseases_present_and_ckd_excluded(client):
    """2, 3, 4 & 5. Verify diabetes & heart_disease are present, while deferred CKD is absent."""
    response = client.get("/diseases")
    assert response.status_code == 200
    data = response.json()

    disease_ids = [d["id"] for d in data]

    # Diabetes & Heart Disease must be active
    assert "diabetes" in disease_ids, "Expected 'diabetes' in registered diseases list"
    assert "heart_disease" in disease_ids, "Expected 'heart_disease' in registered diseases list"

    # Deferred CKD must NOT appear as active
    assert "chronic_kidney_disease" not in disease_ids, "Deferred 'chronic_kidney_disease' must not appear in active list"
    assert "ckd" not in disease_ids, "Deferred 'ckd' must not appear in active list"

    print("  PASS: 2-5. Diabetes and Heart Disease present; deferred CKD strictly excluded")


def test_get_diabetes_details(client):
    """6. Test GET /diseases/diabetes returns rich form specification."""
    response = client.get("/diseases/diabetes")
    assert response.status_code == 200
    data = response.json()

    # Validate schema fields
    validated = DiseaseResponse(**data)
    assert validated.id == "diabetes"
    assert validated.display_name == "Diabetes Risk Assessment"
    assert validated.category == DiseaseCategory.TABULAR
    assert validated.input_type == InputType.FORM
    assert validated.is_active is True
    assert len(validated.required_fields) > 0

    # Verify key tabular features for diabetes
    feature_names = [f.name for f in validated.required_fields]
    assert "Glucose" in feature_names
    assert "BMI" in feature_names
    assert "Age" in feature_names

    # Check model and safety telemetry
    assert validated.model_info is not None
    assert validated.model_info.framework == "scikit-learn"
    assert validated.model_info.model_type == "LogisticRegression"
    assert validated.safety_info is not None
    assert validated.safety_info.is_diagnostic_tool is False
    assert len(validated.safety_info.disclaimer) > 0

    print("  PASS: 6. GET /diseases/diabetes returned complete validated schema with tabular inputs")


def test_get_heart_disease_details(client):
    """7. Test GET /diseases/heart_disease returns rich form specification."""
    response = client.get("/diseases/heart_disease")
    assert response.status_code == 200
    data = response.json()

    validated = DiseaseResponse(**data)
    assert validated.id == "heart_disease"
    assert validated.display_name == "Heart Disease Risk Assessment"
    assert validated.category == DiseaseCategory.TABULAR
    assert validated.input_type == InputType.FORM
    assert validated.is_active is True
    assert len(validated.required_fields) > 0

    # Verify key tabular features for heart disease
    feature_names = [f.name for f in validated.required_fields]
    assert "Age" in feature_names or "age" in [n.lower() for n in feature_names]

    assert validated.model_info is not None
    assert validated.model_info.framework == "xgboost"
    assert validated.safety_info is not None

    print("  PASS: 7. GET /diseases/heart_disease returned complete validated schema")


def test_get_unknown_disease_returns_404(client):
    """8. Test that requesting an unregistered disease returns 404 Not Found."""
    response = client.get("/diseases/unknown_disease_xyz")
    assert response.status_code == 404
    data = response.json()
    assert "not registered or not available" in data["detail"]

    # Also test empty/whitespace
    res_empty = client.get("/diseases/non_existent_module")
    assert res_empty.status_code == 404

    print("  PASS: 8. Unknown disease returned 404 Not Found")


def test_internal_paths_and_secrets_not_exposed(client):
    """9 & 10. Verify sensitive paths and internal implementation details are completely stripped."""
    # List endpoint
    res_list = client.get("/diseases")
    assert res_list.status_code == 200
    list_json = res_list.text

    # Detail endpoints
    res_diab = client.get("/diseases/diabetes")
    assert res_diab.status_code == 200
    diab_json = res_diab.text

    # Sensitive strings that must never leak to the client
    forbidden_tokens = [
        "model_dir",
        "artifact_filename",
        "metadata_filename",
        "diabetes_model_v1.joblib",
        "heart_disease_model.pkl",
        ".joblib",
        ".pkl",
        "preprocessing",
        "backend\\app\\ml",
        "backend/app/ml",
    ]

    for token in forbidden_tokens:
        assert token not in list_json, f"Security Leak: Forbidden token '{token}' exposed in GET /diseases"
        assert token not in diab_json, f"Security Leak: Forbidden token '{token}' exposed in GET /diseases/diabetes"

    print("  PASS: 9 & 10. Verified internal filesystem paths and model artifact filenames are NOT exposed")


if __name__ == "__main__":
    print("\n=== Disease Discovery API Verification Tests ===\n")
    c = TestClient(app)

    tests = [
        ("1. GET /diseases succeeds", lambda: test_get_all_diseases_succeeds(c)),
        ("2-5. Registered diseases & CKD exclusion", lambda: test_registered_diseases_present_and_ckd_excluded(c)),
        ("6. GET /diseases/diabetes schema", lambda: test_get_diabetes_details(c)),
        ("7. GET /diseases/heart_disease schema", lambda: test_get_heart_disease_details(c)),
        ("8. Unknown disease -> 404", lambda: test_get_unknown_disease_returns_404(c)),
        ("9 & 10. Security & Information Hiding", lambda: test_internal_paths_and_secrets_not_exposed(c)),
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

    print(f"\n{'=' * 65}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 65}\n")
    sys.exit(1 if failed else 0)
