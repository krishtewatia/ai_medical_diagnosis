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


def test_get_prediction_by_id_success(client_with_service):
    """8. Verify authenticated user retrieves a specific prediction by valid ID."""
    client, history_service = client_with_service
    user_id = str(ObjectId())

    rec = PredictionHistoryCreate(
        user_id=user_id,
        disease="diabetes",
        disease_display_name="Diabetes Risk Assessment",
        input_type="tabular",
        model=PredictionModelInfo(version="v1", model_type="LogisticRegression", threshold=0.40),
        input_data={"Glucose": 160.0, "BMI": 32.5, "Age": 45},
        result=PredictionResultRecord(prediction="High Risk of Diabetes", is_positive=True, probability=0.89, confidence=0.92),
        explanation="Elevated fasting plasma glucose and BMI indicate high screening risk.",
        metadata={"source": "api", "latency_ms": 2.1},
    )
    created = history_service.create_prediction(user_id=user_id, payload=rec)
    prediction_id = created["id"]

    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_id), "email": "user@example.com"}

    response = client.get(f"/history/{prediction_id}")
    assert response.status_code == 200
    data = response.json()

    assert data["id"] == prediction_id
    assert data["user_id"] == user_id
    assert data["disease"] == "diabetes"
    assert data["disease_display_name"] == "Diabetes Risk Assessment"
    assert data["input_type"] == "tabular"
    assert data["model"]["version"] == "v1"
    assert data["model"]["model_type"] == "LogisticRegression"
    assert data["model"]["threshold"] == 0.40
    assert data["input_data"] == {"Glucose": 160.0, "BMI": 32.5, "Age": 45}
    assert data["result"]["prediction"] == "High Risk of Diabetes"
    assert data["result"]["is_positive"] is True
    assert data["result"]["probability"] == 0.89
    assert data["result"]["confidence"] == 0.92
    assert data["explanation"] == "Elevated fasting plasma glucose and BMI indicate high screening risk."
    assert "created_at" in data
    assert data["metadata"]["source"] == "api"

    print("  PASS: 8. GET /history/{prediction_id} successfully retrieved complete structured document")


def test_get_prediction_by_id_nonexistent_returns_404(client_with_service):
    """9. Verify valid ObjectId format but nonexistent record returns 404."""
    client, _ = client_with_service
    user_id = str(ObjectId())
    non_existent_pred_id = str(ObjectId())

    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_id), "email": "user@example.com"}

    response = client.get(f"/history/{non_existent_pred_id}")
    assert response.status_code == 404
    data = response.json()
    assert data["detail"] == "Prediction record not found."

    print("  PASS: 9. Nonexistent prediction ID returned 404 Not Found")


def test_get_prediction_by_id_malformed_returns_400(client_with_service):
    """10. Verify malformed prediction ID returns 400 Bad Request."""
    client, _ = client_with_service
    user_id = str(ObjectId())

    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_id), "email": "user@example.com"}

    response = client.get("/history/invalid-non-hex-id-123")
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Invalid prediction ID format."

    print("  PASS: 10. Malformed prediction ID returned 400 Bad Request")


def test_get_prediction_by_id_auth_missing_or_invalid(client_with_service):
    """11. Verify missing or invalid JWT is rejected when querying /history/{prediction_id}."""
    client, _ = client_with_service
    pred_id = str(ObjectId())

    if get_current_user in app.dependency_overrides:
        del app.dependency_overrides[get_current_user]

    # No auth header
    res_no_auth = client.get(f"/history/{pred_id}")
    assert res_no_auth.status_code in (401, 403)

    # Invalid token
    res_bad_auth = client.get(f"/history/{pred_id}", headers={"Authorization": "Bearer bad_token_xyz"})
    assert res_bad_auth.status_code == 401

    print("  PASS: 11. Missing / Invalid JWT rejected on GET /history/{prediction_id}")


def test_get_prediction_by_id_cross_user_isolation(client_with_service):
    """12. CRITICAL SECURITY TEST: User A requesting User B's prediction must return 404."""
    client, history_service = client_with_service
    user_a = str(ObjectId())
    user_b = str(ObjectId())

    # User A creates Prediction A
    rec_a = PredictionHistoryCreate(
        user_id=user_a,
        disease="diabetes",
        disease_display_name="Diabetes Assessment",
        input_type="tabular",
        model=PredictionModelInfo(version="v1", model_type="LR"),
        input_data={"secret_patient_data": "User A Private Info"},
        result=PredictionResultRecord(prediction="Diabetes High Risk", is_positive=True, probability=0.95),
    )
    pred_a = history_service.create_prediction(user_a, rec_a)

    # User B creates Prediction B
    rec_b = PredictionHistoryCreate(
        user_id=user_b,
        disease="heart_disease",
        disease_display_name="Heart Disease Assessment",
        input_type="tabular",
        model=PredictionModelInfo(version="v1", model_type="XGBoost"),
        input_data={"secret_patient_data": "User B Private Info"},
        result=PredictionResultRecord(prediction="Heart Disease Elevated Risk", is_positive=True, probability=0.88),
    )
    pred_b = history_service.create_prediction(user_b, rec_b)

    # User A attempts to access User B's prediction
    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_a), "email": "a@ex.com"}
    res_a_reading_b = client.get(f"/history/{pred_b['id']}")
    assert res_a_reading_b.status_code == 404, "SECURITY BREACH: User A must not access User B's prediction!"
    assert res_a_reading_b.json()["detail"] == "Prediction record not found."

    # User B attempts to access User A's prediction
    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_b), "email": "b@ex.com"}
    res_b_reading_a = client.get(f"/history/{pred_a['id']}")
    assert res_b_reading_a.status_code == 404, "SECURITY BREACH: User B must not access User A's prediction!"
    assert res_b_reading_a.json()["detail"] == "Prediction record not found."

    # User A successfully accesses User A's prediction
    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_a), "email": "a@ex.com"}
    res_a_reading_a = client.get(f"/history/{pred_a['id']}")
    assert res_a_reading_a.status_code == 200
    assert res_a_reading_a.json()["id"] == pred_a["id"]

    # User B successfully accesses User B's prediction
    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_b), "email": "b@ex.com"}
    res_b_reading_b = client.get(f"/history/{pred_b['id']}")
    assert res_b_reading_b.status_code == 200
    assert res_b_reading_b.json()["id"] == pred_b["id"]

    print("  PASS: 12. Strict tenant isolation verified: Cross-user prediction lookups returned 404 Not Found")


def test_history_disease_filtering_diabetes_and_heart(client_with_service):
    """13. Verify filtering user history by registered disease IDs."""
    client, history_service = client_with_service
    user_id = str(ObjectId())

    # Create 2 Diabetes and 1 Heart Disease prediction
    history_service.create_prediction(
        user_id=user_id,
        payload=PredictionHistoryCreate(
            user_id=user_id,
            disease="diabetes",
            disease_display_name="Diabetes Assessment 1",
            input_type="tabular",
            model=PredictionModelInfo(version="v1", model_type="LR"),
            input_data={"Glucose": 140.0},
            result=PredictionResultRecord(prediction="Diabetes Risk", is_positive=True, probability=0.8),
        )
    )
    history_service.create_prediction(
        user_id=user_id,
        payload=PredictionHistoryCreate(
            user_id=user_id,
            disease="diabetes",
            disease_display_name="Diabetes Assessment 2",
            input_type="tabular",
            model=PredictionModelInfo(version="v1", model_type="LR"),
            input_data={"Glucose": 180.0},
            result=PredictionResultRecord(prediction="Diabetes High Risk", is_positive=True, probability=0.95),
        )
    )
    history_service.create_prediction(
        user_id=user_id,
        payload=PredictionHistoryCreate(
            user_id=user_id,
            disease="heart_disease",
            disease_display_name="Heart Disease Assessment",
            input_type="tabular",
            model=PredictionModelInfo(version="v1", model_type="XGBoost"),
            input_data={"Cholesterol": 240.0},
            result=PredictionResultRecord(prediction="Heart Disease Risk", is_positive=True, probability=0.75),
        )
    )

    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_id), "email": "filter@example.com"}

    # No filter -> returns all 3
    res_all = client.get("/history")
    assert res_all.status_code == 200
    assert res_all.json()["total"] == 3
    assert len(res_all.json()["items"]) == 3

    # Filter disease=diabetes -> returns 2
    res_diab = client.get("/history?disease=diabetes")
    assert res_diab.status_code == 200
    data_diab = res_diab.json()
    assert data_diab["total"] == 2
    assert len(data_diab["items"]) == 2
    assert all(item["disease"] == "diabetes" for item in data_diab["items"])

    # Filter disease=heart_disease -> returns 1
    res_heart = client.get("/history?disease=heart_disease")
    assert res_heart.status_code == 200
    data_heart = res_heart.json()
    assert data_heart["total"] == 1
    assert len(data_heart["items"]) == 1
    assert data_heart["items"][0]["disease"] == "heart_disease"

    print("  PASS: 13. Disease filtering for diabetes, heart_disease, and no-filter verified")


def test_history_disease_filtering_unknown_disease(client_with_service):
    """14. Verify unknown disease query parameter receives appropriate validation error (404)."""
    client, _ = client_with_service
    user_id = str(ObjectId())

    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_id), "email": "user@example.com"}

    response = client.get("/history?disease=unknown_nonexistent_disease_xyz")
    assert response.status_code == 404
    data = response.json()
    assert "not registered or not available" in data["detail"]

    print("  PASS: 14. Unknown disease query param returned 404 Not Found error")


def test_history_disease_filtering_pagination_and_sorting(client_with_service):
    """15. Verify pagination and newest-first sorting operate seamlessly with disease filter."""
    client, history_service = client_with_service
    user_id = str(ObjectId())
    base_time = datetime.now(timezone.utc)

    # Insert 4 Diabetes records and 2 Heart Disease records with distinct times
    for i in range(4):
        history_service.create_prediction(
            user_id=user_id,
            payload=PredictionHistoryCreate(
                user_id=user_id,
                disease="diabetes",
                disease_display_name="Diabetes",
                input_type="tabular",
                model=PredictionModelInfo(version="v1", model_type="LR"),
                input_data={"seq": i},
                result=PredictionResultRecord(prediction="P", is_positive=True),
                created_at=base_time + timedelta(minutes=i),
            )
        )
    for j in range(2):
        history_service.create_prediction(
            user_id=user_id,
            payload=PredictionHistoryCreate(
                user_id=user_id,
                disease="heart_disease",
                disease_display_name="Heart Disease",
                input_type="tabular",
                model=PredictionModelInfo(version="v1", model_type="XGBoost"),
                input_data={"heart_seq": j},
                result=PredictionResultRecord(prediction="P", is_positive=True),
                created_at=base_time + timedelta(minutes=10 + j),
            )
        )

    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_id), "email": "page_filter@example.com"}

    # Page 1 of Diabetes: limit=2, skip=0 (expect seq=3, seq=2, total=4)
    p1 = client.get("/history?disease=diabetes&limit=2&skip=0").json()
    assert p1["total"] == 4
    assert len(p1["items"]) == 2
    assert p1["items"][0]["input_data"]["seq"] == 3
    assert p1["items"][1]["input_data"]["seq"] == 2

    # Page 2 of Diabetes: limit=2, skip=2 (expect seq=1, seq=0, total=4)
    p2 = client.get("/history?disease=diabetes&limit=2&skip=2").json()
    assert p2["total"] == 4
    assert len(p2["items"]) == 2
    assert p2["items"][0]["input_data"]["seq"] == 1
    assert p2["items"][1]["input_data"]["seq"] == 0

    print("  PASS: 15. Pagination and sorting with disease filter verified")


def test_history_disease_filtering_user_isolation(client_with_service):
    """16. CRITICAL SECURITY TEST: Ensure filtered queries strictly maintain user isolation."""
    client, history_service = client_with_service
    user_a = str(ObjectId())
    user_b = str(ObjectId())

    # User A: 1 Diabetes, 1 Heart Disease
    history_service.create_prediction(
        user_id=user_a,
        payload=PredictionHistoryCreate(
            user_id=user_a,
            disease="diabetes",
            disease_display_name="Diabetes Assessment",
            input_type="tabular",
            model=PredictionModelInfo(version="v1", model_type="LR"),
            input_data={"patient": "User A Diabetes"},
            result=PredictionResultRecord(prediction="Positive", is_positive=True),
        )
    )
    history_service.create_prediction(
        user_id=user_a,
        payload=PredictionHistoryCreate(
            user_id=user_a,
            disease="heart_disease",
            disease_display_name="Heart Assessment",
            input_type="tabular",
            model=PredictionModelInfo(version="v1", model_type="XGBoost"),
            input_data={"patient": "User A Heart"},
            result=PredictionResultRecord(prediction="Positive", is_positive=True),
        )
    )

    # User B: 1 Diabetes
    history_service.create_prediction(
        user_id=user_b,
        payload=PredictionHistoryCreate(
            user_id=user_b,
            disease="diabetes",
            disease_display_name="Diabetes Assessment",
            input_type="tabular",
            model=PredictionModelInfo(version="v1", model_type="LR"),
            input_data={"patient": "User B Diabetes"},
            result=PredictionResultRecord(prediction="Positive", is_positive=True),
        )
    )

    # User A queries disease=diabetes -> should get only User A's Diabetes record
    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_a), "email": "a@ex.com"}
    res_a_diab = client.get("/history?disease=diabetes")
    assert res_a_diab.status_code == 200
    data_a_diab = res_a_diab.json()
    assert data_a_diab["total"] == 1
    assert data_a_diab["items"][0]["input_data"]["patient"] == "User A Diabetes"

    # User B queries disease=diabetes -> should get only User B's Diabetes record
    app.dependency_overrides[get_current_user] = lambda: {"_id": ObjectId(user_b), "email": "b@ex.com"}
    res_b_diab = client.get("/history?disease=diabetes")
    assert res_b_diab.status_code == 200
    data_b_diab = res_b_diab.json()
    assert data_b_diab["total"] == 1
    assert data_b_diab["items"][0]["input_data"]["patient"] == "User B Diabetes"

    # User B queries disease=heart_disease -> should get 0 records
    res_b_heart = client.get("/history?disease=heart_disease")
    assert res_b_heart.status_code == 200
    data_b_heart = res_b_heart.json()
    assert data_b_heart["total"] == 0
    assert data_b_heart["items"] == []

    print("  PASS: 16. User isolation strictly enforced on filtered disease history queries")


if __name__ == "__main__":
    print("\n=== GET /history and GET /history/{prediction_id} API Verification Tests ===\n")
    mock_hs = mongomock.MongoClient()["test_history_api_runner_db"]
    hs = PredictionHistoryService(db=mock_hs)
    app.dependency_overrides[get_history_service] = lambda: hs
    c = TestClient(app)

    tests = [
        ("1. User Retrieves History List", lambda: test_authenticated_user_retrieves_history((c, hs))),
        ("2. Empty History List", lambda: test_user_with_no_history_receives_empty_list((c, hs))),
        ("3 & 4. Auth Protection on List", lambda: test_missing_or_invalid_auth_rejected((c, hs))),
        ("5. Security & Isolation on List", lambda: test_critical_security_tenant_isolation((c, hs))),
        ("6 & 7. Pagination & Sorting", lambda: test_pagination_and_sorting((c, hs))),
        ("8. User Retrieves Prediction By ID", lambda: test_get_prediction_by_id_success((c, hs))),
        ("9. Nonexistent Prediction ID (404)", lambda: test_get_prediction_by_id_nonexistent_returns_404((c, hs))),
        ("10. Malformed Prediction ID (400)", lambda: test_get_prediction_by_id_malformed_returns_400((c, hs))),
        ("11. Auth Protection on Detail Endpoint", lambda: test_get_prediction_by_id_auth_missing_or_invalid((c, hs))),
        ("12. Tenant Isolation Cross-User Access Blocked (404)", lambda: test_get_prediction_by_id_cross_user_isolation((c, hs))),
        ("13. Disease Filtering (Diabetes & Heart)", lambda: test_history_disease_filtering_diabetes_and_heart((c, hs))),
        ("14. Unknown Disease Query Param (404)", lambda: test_history_disease_filtering_unknown_disease((c, hs))),
        ("15. Disease Filtering Pagination & Sorting", lambda: test_history_disease_filtering_pagination_and_sorting((c, hs))),
        ("16. Filtered Query Tenant Isolation", lambda: test_history_disease_filtering_user_isolation((c, hs))),
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

    print(f"\n{'=' * 65}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 65}\n")
    sys.exit(1 if failed else 0)


