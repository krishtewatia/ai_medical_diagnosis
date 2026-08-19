"""
Comprehensive Test Suite for Step 23.5:
GET /history Endpoint.

Tests:
1. Authenticated user with history retrieves formatted predictions
2. Authenticated user with NO history receives empty list []
3. Missing JWT credentials rejected with 401/403
4. Invalid JWT rejected with 401
5. CRITICAL SECURITY / TENANT ISOLATION: User A sees only User A's history; User B sees only User B's history
6. Pagination (skip, limit, total count) operates accurately
7. Newest records are returned first (created_at DESC)
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from bson import ObjectId
from fastapi.testclient import TestClient
import mongomock
import pytest

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.dependencies import get_current_user
from app.api.history import get_history_service
from app.core.security import create_access_token
from app.main import app
from app.schemas.prediction_history import (
    PredictionHistoryCreate,
    PredictionModelInfo,
    PredictionResultRecord,
)
from app.services.prediction_history_service import PredictionHistoryService


@pytest.fixture
def mock_history_service():
    mock_db = mongomock.MongoClient()["test_history_api_db"]
    return PredictionHistoryService(db=mock_db)


@pytest.fixture
def client_with_service(mock_history_service):
    app.dependency_overrides[get_history_service] = lambda: mock_history_service
    yield TestClient(app), mock_history_service
    app.dependency_overrides.clear()


def test_authenticated_user_retrieves_history(client_with_service):
    """1. Verify authenticated user retrieves structured history records."""
    client, history_service = client_with_service
    user_id = str(ObjectId())

    # Pre-populate history
    rec = PredictionHistoryCreate(
        user_id=user_id,
        disease="diabetes",
        disease_display_name="Diabetes Risk Assessment",
        input_type="tabular",
        model=PredictionModelInfo(version="v1", model_type="LogisticRegression", threshold=0.40),
        input_data={"Glucose": 150.0, "BMI": 31.0},
        result=PredictionResultRecord(prediction="High Risk of Diabetes", is_positive=True, probability=0.85),
        explanation="Elevated risk detected.",
        metadata={"source": "api", "latency_ms": 1.8},
    )
    history_service.create_prediction(user_id=user_id, payload=rec)

    # Set auth override
    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_id), "email": "user@example.com"}

    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()

    assert "items" in data
    assert data["total"] == 1
    assert data["limit"] == 20
    assert data["skip"] == 0

    item = data["items"][0]
    assert item["disease"] == "diabetes"
    assert item["disease_display_name"] == "Diabetes Risk Assessment"
    assert item["model"]["version"] == "v1"
    assert item["result"]["prediction"] == "High Risk of Diabetes"
    assert item["result"]["probability"] == 0.85
    assert "id" in item

    print("  PASS: 1. Authenticated user successfully retrieved formatted history")


def test_user_with_no_history_receives_empty_list(client_with_service):
    """2. Verify user with no predictions receives empty items list."""
    client, _ = client_with_service
    user_id = str(ObjectId())

    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_id), "email": "empty@example.com"}

    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()

    assert data["items"] == []
    assert data["total"] == 0
    print("  PASS: 2. Authenticated user with no history received empty items list")


def test_missing_or_invalid_auth_rejected(client_with_service):
    """3 & 4. Verify unauthenticated or malformed requests are rejected."""
    client, _ = client_with_service
    # Clear any user overrides to test real auth dependency
    if get_current_user in app.dependency_overrides:
        del app.dependency_overrides[get_current_user]

    # No token
    res_no_auth = client.get("/history")
    assert res_no_auth.status_code in (401, 403)

    # Invalid token
    res_bad_auth = client.get("/history", headers={"Authorization": "Bearer invalid_token_123"})
    assert res_bad_auth.status_code == 401

    print("  PASS: 3 & 4. Missing / Invalid JWT rejected with 401/403")


def test_critical_security_tenant_isolation(client_with_service):
    """5. CRITICAL SECURITY TEST: Ensure User A sees only User A's history and User B sees only User B's."""
    client, history_service = client_with_service
    user_a = str(ObjectId())
    user_b = str(ObjectId())

    # User A creates Diabetes prediction
    rec_a = PredictionHistoryCreate(
        user_id=user_a,
        disease="diabetes",
        disease_display_name="Diabetes Assessment",
        input_type="tabular",
        model=PredictionModelInfo(version="v1", model_type="LR"),
        input_data={"patient": "User A Record"},
        result=PredictionResultRecord(prediction="Diabetes High Risk", is_positive=True, probability=0.91),
    )
    history_service.create_prediction(user_a, rec_a)

    # User B creates Heart Disease prediction
    rec_b = PredictionHistoryCreate(
        user_id=user_b,
        disease="heart_disease",
        disease_display_name="Heart Assessment",
        input_type="tabular",
        model=PredictionModelInfo(version="v1", model_type="XGBoost"),
        input_data={"patient": "User B Record"},
        result=PredictionResultRecord(prediction="Heart Disease High Risk", is_positive=True, probability=0.88),
    )
    history_service.create_prediction(user_b, rec_b)

    # User A queries GET /history
    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_a), "email": "a@ex.com"}
    res_a = client.get("/history")
    assert res_a.status_code == 200
    data_a = res_a.json()
    assert data_a["total"] == 1
    assert data_a["items"][0]["disease"] == "diabetes"
    assert data_a["items"][0]["input_data"]["patient"] == "User A Record"

    # User B queries GET /history
    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_b), "email": "b@ex.com"}
    res_b = client.get("/history")
    assert res_b.status_code == 200
    data_b = res_b.json()
    assert data_b["total"] == 1
    assert data_b["items"][0]["disease"] == "heart_disease"
    assert data_b["items"][0]["input_data"]["patient"] == "User B Record"

    print("  PASS: 5. Tenant isolation verified: User A and User B cannot see each other's predictions")


def test_pagination_and_sorting(client_with_service):
    """6 & 7. Verify pagination parameters and newest-first chronological sorting."""
    client, history_service = client_with_service
    user_id = str(ObjectId())
    base_time = datetime.now(timezone.utc)

    # Create 5 records over distinct times
    for i in range(5):
        rec = PredictionHistoryCreate(
            user_id=user_id,
            disease="diabetes",
            disease_display_name="Diabetes",
            input_type="tabular",
            model=PredictionModelInfo(version="v1", model_type="LR"),
            input_data={"seq": i},
            result=PredictionResultRecord(prediction="P", is_positive=True),
            created_at=base_time + timedelta(minutes=i),
        )
        history_service.create_prediction(user_id, rec)

    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_id), "email": "page@ex.com"}

    # Page 1: limit=2, skip=0 (expect seq=4, seq=3)
    p1 = client.get("/history?limit=2&skip=0").json()
    assert p1["total"] == 5
    assert len(p1["items"]) == 2
    assert p1["items"][0]["input_data"]["seq"] == 4
    assert p1["items"][1]["input_data"]["seq"] == 3

    # Page 2: limit=2, skip=2 (expect seq=2, seq=1)
    p2 = client.get("/history?limit=2&skip=2").json()
    assert len(p2["items"]) == 2
    assert p2["items"][0]["input_data"]["seq"] == 2
    assert p2["items"][1]["input_data"]["seq"] == 1

    # Page 3: limit=2, skip=4 (expect seq=0)
    p3 = client.get("/history?limit=2&skip=4").json()
    assert len(p3["items"]) == 1
    assert p3["items"][0]["input_data"]["seq"] == 0

    print("  PASS: 6 & 7. Pagination and newest-first sorting verified")


if __name__ == "__main__":
    print("\n=== GET /history API Verification Tests ===\n")
    mock_hs = mongomock.MongoClient()["test_history_api_runner_db"]
    hs = PredictionHistoryService(db=mock_hs)
    app.dependency_overrides[get_history_service] = lambda: hs
    c = TestClient(app)

    tests = [
        ("1. User Retrieves History", lambda: test_authenticated_user_retrieves_history((c, hs))),
        ("2. Empty History List", lambda: test_user_with_no_history_receives_empty_list((c, hs))),
        ("3 & 4. Auth Protection", lambda: test_missing_or_invalid_auth_rejected((c, hs))),
        ("5. Security & Isolation", lambda: test_critical_security_tenant_isolation((c, hs))),
        ("6 & 7. Pagination & Sorting", lambda: test_pagination_and_sorting((c, hs))),
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

    app.dependency_overrides.clear()

    print(f"\n{'=' * 55}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 55}\n")
    sys.exit(1 if failed else 0)
