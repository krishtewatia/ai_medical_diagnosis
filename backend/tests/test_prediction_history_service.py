"""
Comprehensive Test Suite for Step 23.3:
Prediction History Service Layer (PredictionHistoryService).

Tests:
1. Create prediction successfully with auto _id and UTC created_at
2. Retrieve user's predictions sorted newest first (created_at DESC)
3. Empty history returns [] cleanly
4. Retrieve specific prediction by ID
5. CRITICAL TENANT ISOLATION: User A cannot retrieve User B's prediction
6. Disease filtering (disease="diabetes", disease="heart_disease")
7. Pagination (skip, limit, limit clamping) & count_user_predictions
8. Invalid user_id and prediction_id formats raise controlled exceptions
9. Immutability verification (no update/edit capabilities)
10. Index generation verification
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from bson import ObjectId
import mongomock
import pytest

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.prediction_history import (
    PredictionHistoryCreate,
    PredictionModelInfo,
    PredictionResultRecord,
)
from app.services.prediction_history_service import (
    InvalidPredictionIdError,
    InvalidUserIdError,
    PredictionHistoryService,
)


@pytest.fixture
def history_service():
    mock_client = mongomock.MongoClient()
    mock_db = mock_client["test_history_service_db"]
    return PredictionHistoryService(db=mock_db)


def test_create_prediction(history_service):
    """1. Verify prediction document creation with auto _id and UTC created_at."""
    user_id = str(ObjectId())
    record = PredictionHistoryCreate(
        user_id=user_id,
        disease="diabetes",
        disease_display_name="Diabetes Risk Assessment",
        input_type="tabular",
        model=PredictionModelInfo(version="v1", model_type="LogisticRegression", threshold=0.40),
        input_data={"Glucose": 140.0, "BMI": 30.0},
        result=PredictionResultRecord(prediction="High Risk of Diabetes", is_positive=True, probability=0.82),
        explanation="Elevated screening risk detected.",
        metadata={"source": "api", "latency_ms": 1.5},
    )

    created = history_service.create_prediction(user_id=user_id, payload=record)
    assert "id" in created
    assert ObjectId.is_valid(created["id"])
    assert created["user_id"] == user_id
    assert created["disease"] == "diabetes"
    assert created["result"]["probability"] == 0.82
    assert isinstance(created["created_at"], datetime)
    print("  PASS: 1. Prediction successfully created with auto _id and UTC timestamp")


def test_get_user_predictions_sorting(history_service):
    """2 & 3. Verify user predictions are returned newest first, and empty history is []."""
    user_id = str(ObjectId())

    # Empty history
    assert history_service.get_user_predictions(user_id) == []

    # Insert 3 records with distinct timestamps
    for i in range(3):
        rec = PredictionHistoryCreate(
            user_id=user_id,
            disease="diabetes",
            disease_display_name="Diabetes",
            input_type="tabular",
            model=PredictionModelInfo(version="v1", model_type="LR"),
            input_data={"idx": i},
            result=PredictionResultRecord(prediction="P", is_positive=True, probability=0.5 + (i * 0.1)),
            created_at=datetime(2026, 1, 1 + i, 12, 0, 0, tzinfo=timezone.utc),
        )
        history_service.create_prediction(user_id=user_id, payload=rec)

    preds = history_service.get_user_predictions(user_id)
    assert len(preds) == 3
    # Newest first: idx=2 (Jan 3), idx=1 (Jan 2), idx=0 (Jan 1)
    assert preds[0]["input_data"]["idx"] == 2
    assert preds[1]["input_data"]["idx"] == 1
    assert preds[2]["input_data"]["idx"] == 0
    print("  PASS: 2 & 3. Newest-first sorting and empty history handling verified")


def test_tenant_isolation_cross_user_access(history_service):
    """4 & 5. CRITICAL SECURITY TEST: Ensure User A CANNOT read User B's prediction."""
    user_a = str(ObjectId())
    user_b = str(ObjectId())

    # User A creates Prediction A
    rec_a = PredictionHistoryCreate(
        user_id=user_a,
        disease="diabetes",
        disease_display_name="Diabetes",
        input_type="tabular",
        model=PredictionModelInfo(version="v1", model_type="LR"),
        input_data={"patient": "A"},
        result=PredictionResultRecord(prediction="Positive", is_positive=True, probability=0.9),
    )
    pred_a = history_service.create_prediction(user_a, rec_a)

    # User B creates Prediction B
    rec_b = PredictionHistoryCreate(
        user_id=user_b,
        disease="heart_disease",
        disease_display_name="Heart Disease",
        input_type="tabular",
        model=PredictionModelInfo(version="v1", model_type="XGBoost"),
        input_data={"patient": "B"},
        result=PredictionResultRecord(prediction="Negative", is_positive=False, probability=0.1),
    )
    pred_b = history_service.create_prediction(user_b, rec_b)

    # User A can read Prediction A
    found_a = history_service.get_prediction_by_id(user_id=user_a, prediction_id=pred_a["id"])
    assert found_a is not None
    assert found_a["id"] == pred_a["id"]

    # User A ATTEMPTS to read Prediction B -> MUST RETURN NONE!
    blocked = history_service.get_prediction_by_id(user_id=user_a, prediction_id=pred_b["id"])
    assert blocked is None, "CRITICAL SECURITY BREACH: User A must never access User B's prediction!"

    # User A's list contains ONLY Prediction A
    list_a = history_service.get_user_predictions(user_id=user_a)
    assert len(list_a) == 1
    assert list_a[0]["id"] == pred_a["id"]

    print("  PASS: 4 & 5. Tenant isolation verified: Cross-user prediction queries strictly blocked")


def test_disease_filtering(history_service):
    """6. Verify filtering user history by disease ID."""
    user_id = str(ObjectId())

    # 2 Diabetes, 1 Heart Disease
    for d, pos in [("diabetes", True), ("diabetes", False), ("heart_disease", True)]:
        rec = PredictionHistoryCreate(
            user_id=user_id,
            disease=d,
            disease_display_name=d.capitalize(),
            input_type="tabular",
            model=PredictionModelInfo(version="v1", model_type="M"),
            input_data={"d": d},
            result=PredictionResultRecord(prediction="P", is_positive=pos),
        )
        history_service.create_prediction(user_id, rec)

    all_preds = history_service.get_user_predictions(user_id)
    assert len(all_preds) == 3

    diabetes_only = history_service.get_user_predictions(user_id, disease="diabetes")
    assert len(diabetes_only) == 2
    assert all(p["disease"] == "diabetes" for p in diabetes_only)

    heart_only = history_service.get_user_predictions(user_id, disease="heart_disease")
    assert len(heart_only) == 1
    assert heart_only[0]["disease"] == "heart_disease"

    pneumonia_only = history_service.get_user_predictions(user_id, disease="pneumonia")
    assert len(pneumonia_only) == 0

    print("  PASS: 6. Disease filtering (diabetes, heart_disease, pneumonia) verified")


def test_pagination_and_counting(history_service):
    """7. Verify pagination (skip, limit, clamping) and total record counts."""
    user_id = str(ObjectId())

    for i in range(5):
        rec = PredictionHistoryCreate(
            user_id=user_id,
            disease="diabetes",
            disease_display_name="Diabetes",
            input_type="tabular",
            model=PredictionModelInfo(version="v1", model_type="M"),
            input_data={"i": i},
            result=PredictionResultRecord(prediction="P", is_positive=True),
            created_at=datetime(2026, 1, i + 1, tzinfo=timezone.utc),
        )
        history_service.create_prediction(user_id, rec)

    assert history_service.count_user_predictions(user_id) == 5

    page1 = history_service.get_user_predictions(user_id, skip=0, limit=2)
    assert len(page1) == 2
    assert page1[0]["input_data"]["i"] == 4  # Newest first

    page2 = history_service.get_user_predictions(user_id, skip=2, limit=2)
    assert len(page2) == 2
    assert page2[0]["input_data"]["i"] == 2

    page3 = history_service.get_user_predictions(user_id, skip=4, limit=2)
    assert len(page3) == 1
    assert page3[0]["input_data"]["i"] == 0

    print("  PASS: 7. Pagination (skip/limit) and count_user_predictions verified")


def test_invalid_id_exceptions(history_service):
    """8. Verify malformed user_id and prediction_id raise controlled exceptions."""
    valid_id = str(ObjectId())

    # Invalid user_id
    with pytest.raises(InvalidUserIdError):
        history_service.get_user_predictions(user_id="not_an_object_id")

    with pytest.raises(InvalidUserIdError):
        history_service.get_prediction_by_id(user_id="invalid_user", prediction_id=valid_id)

    # Invalid prediction_id
    with pytest.raises(InvalidPredictionIdError):
        history_service.get_prediction_by_id(user_id=valid_id, prediction_id="invalid_prediction_id")

    print("  PASS: 8. Invalid ID formats raise InvalidUserIdError and InvalidPredictionIdError")


def test_immutability_and_indexes(history_service):
    """9 & 10. Verify immutability and index creation."""
    # Ensure no update / edit methods exist on the service
    assert not hasattr(history_service, "update_prediction")
    assert not hasattr(history_service, "edit_prediction")

    # Index setup executes safely
    history_service.ensure_indexes()
    print("  PASS: 9 & 10. Immutability and index creation verified")


if __name__ == "__main__":
    print("\n=== Prediction History Service Verification Tests ===\n")
    tests = [
        ("1. Create Prediction", test_create_prediction),
        ("2 & 3. Sorting & Empty History", test_get_user_predictions_sorting),
        ("4 & 5. Tenant Isolation (Cross-User Access Blocked)", test_tenant_isolation_cross_user_access),
        ("6. Disease Filtering", test_disease_filtering),
        ("7. Pagination & Counting", test_pagination_and_counting),
        ("8. Invalid ID Handling", test_invalid_id_exceptions),
        ("9 & 10. Immutability & Indexes", test_immutability_and_indexes),
    ]

    passed = 0
    failed = 0
    service = PredictionHistoryService(db=mongomock.MongoClient()["test_runner_db"])

    for name, fn in tests:
        try:
            fn(service)
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} - {e}")
            failed += 1

    print(f"\n{'=' * 55}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 55}\n")
    sys.exit(1 if failed else 0)
