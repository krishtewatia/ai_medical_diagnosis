"""
Step 23.8 Final Verification Script:
Comprehensive Authorization, Authentication, and User-Isolation Verification
for the Prediction History System.

Tests & Verifications:
1. Multi-tenant User Setup (User A & User B with multiple prediction records)
2. GET /history Scoping (User A sees only A's records; User B sees only B's records)
3. GET /history?disease=... Filtering & Scoping (User A filtering diabetes sees only A's diabetes)
4. GET /history/{prediction_id} Cross-tenant Access Blocked (User A cannot access User B's record -> 404)
5. Direct MongoDB Database Ownership Inspection (Every record has valid user_id ObjectId bound to creator)
6. Authentication Failure Matrix across all history routes (No token, invalid, expired, valid)
7. Pagination + Disease Filtering + Tenant Isolation Combined Matrix
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from bson import ObjectId
import jwt
import mongomock
import pytest
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.dependencies import get_current_user
from app.api.history import get_disease_registry, get_history_service
from app.core.config import settings
from app.core.security import create_access_token
from app.database.connection import get_database
from app.main import app
from app.ml.disease_registry import disease_registry
from app.schemas.prediction_history import (
    PredictionHistoryCreate,
    PredictionModelInfo,
    PredictionResultRecord,
)
from app.services.prediction_history_service import PredictionHistoryService
from app.services.user_service import UserService


def run_comprehensive_history_security_verification():
    print("\n" + "=" * 70)
    print(" STEP 23.8: PREDICTION HISTORY FINAL AUTHORIZATION & ISOLATION AUDIT")
    print("=" * 70 + "\n")

    # 1. Setup Mock Database & Services
    mock_client = mongomock.MongoClient()
    mock_db = mock_client["security_audit_history_db"]
    user_service = UserService(mock_db)
    history_service = PredictionHistoryService(db=mock_db)
    history_service.ensure_indexes()

    # Configure FastAPI dependency overrides
    app.dependency_overrides[get_database] = lambda: mock_db
    app.dependency_overrides[get_history_service] = lambda: history_service
    app.dependency_overrides[get_disease_registry] = lambda: disease_registry
    if get_current_user in app.dependency_overrides:
        del app.dependency_overrides[get_current_user]

    client = TestClient(app)
    results = []

    # ---------------------------------------------------------
    # Scenario 1: Provision Two Separate Users & Predictions
    # ---------------------------------------------------------
    print("--- 1. PROVISIONING TEST USERS AND PREDICTIONS ---")
    user_a = user_service.create_user(
        name="User A",
        email="user_a@medical.org",
        password_hash="argon2_mock_hash_a"
    )
    user_b = user_service.create_user(
        name="User B",
        email="user_b@medical.org",
        password_hash="argon2_mock_hash_b"
    )

    user_a_id = str(user_a["_id"])
    user_b_id = str(user_b["_id"])
    token_a = create_access_token(user_a_id)
    token_b = create_access_token(user_b_id)
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # User A: Diabetes & Heart Disease
    pred_a_diabetes = history_service.create_prediction(
        user_id=user_a_id,
        payload=PredictionHistoryCreate(
            user_id=user_a_id,
            disease="diabetes",
            disease_display_name="Diabetes Risk Assessment",
            input_type="tabular",
            model=PredictionModelInfo(version="v1", model_type="LogisticRegression", threshold=0.40),
            input_data={"Glucose": 155.0, "BMI": 31.0, "patient": "User A Record"},
            result=PredictionResultRecord(prediction="High Risk", is_positive=True, probability=0.85),
            explanation="Elevated glucose.",
            created_at=datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        )
    )
    pred_a_heart = history_service.create_prediction(
        user_id=user_a_id,
        payload=PredictionHistoryCreate(
            user_id=user_a_id,
            disease="heart_disease",
            disease_display_name="Heart Disease Assessment",
            input_type="tabular",
            model=PredictionModelInfo(version="v1", model_type="XGBoost", threshold=0.45),
            input_data={"Cholesterol": 260.0, "patient": "User A Record"},
            result=PredictionResultRecord(prediction="Elevated Risk", is_positive=True, probability=0.78),
            explanation="Elevated cholesterol.",
            created_at=datetime(2026, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )
    )

    # User B: Diabetes
    pred_b_diabetes = history_service.create_prediction(
        user_id=user_b_id,
        payload=PredictionHistoryCreate(
            user_id=user_b_id,
            disease="diabetes",
            disease_display_name="Diabetes Risk Assessment",
            input_type="tabular",
            model=PredictionModelInfo(version="v1", model_type="LogisticRegression", threshold=0.40),
            input_data={"Glucose": 170.0, "BMI": 34.0, "patient": "User B Record"},
            result=PredictionResultRecord(prediction="High Risk", is_positive=True, probability=0.92),
            explanation="Very high glucose.",
            created_at=datetime(2026, 1, 3, 10, 0, 0, tzinfo=timezone.utc),
        )
    )

    print(f"  User A (ID: {user_a_id}): Created Diabetes ({pred_a_diabetes['id']}) & Heart ({pred_a_heart['id']})")
    print(f"  User B (ID: {user_b_id}): Created Diabetes ({pred_b_diabetes['id']})\n")

    # ---------------------------------------------------------
    # Scenario 2: Test GET /history Scoping
    # ---------------------------------------------------------
    print("--- 2. VERIFY GET /history SCOPING ---")
    res_a_history = client.get("/history", headers=headers_a)
    assert res_a_history.status_code == 200
    data_a = res_a_history.json()
    user_a_ids = [item["id"] for item in data_a["items"]]
    pass_2a = (
        data_a["total"] == 2
        and pred_a_diabetes["id"] in user_a_ids
        and pred_a_heart["id"] in user_a_ids
        and pred_b_diabetes["id"] not in user_a_ids
    )
    print(f"  User A GET /history -> total={data_a['total']}, IDs={user_a_ids} -> [{'PASS' if pass_2a else 'FAIL'}]")
    results.append(("2. User A History Scoping", pass_2a))

    res_b_history = client.get("/history", headers=headers_b)
    assert res_b_history.status_code == 200
    data_b = res_b_history.json()
    user_b_ids = [item["id"] for item in data_b["items"]]
    pass_2b = (
        data_b["total"] == 1
        and pred_b_diabetes["id"] in user_b_ids
        and pred_a_diabetes["id"] not in user_b_ids
        and pred_a_heart["id"] not in user_b_ids
    )
    print(f"  User B GET /history -> total={data_b['total']}, IDs={user_b_ids} -> [{'PASS' if pass_2b else 'FAIL'}]\n")
    results.append(("2. User B History Scoping", pass_2b))

    # ---------------------------------------------------------
    # Scenario 3: Test Disease Filtering
    # ---------------------------------------------------------
    print("--- 3. VERIFY DISEASE FILTERING ---")
    # User A requests diabetes
    res_a_diab = client.get("/history?disease=diabetes", headers=headers_a)
    assert res_a_diab.status_code == 200
    data_a_diab = res_a_diab.json()
    pass_3a = (
        data_a_diab["total"] == 1
        and data_a_diab["items"][0]["id"] == pred_a_diabetes["id"]
        and data_a_diab["items"][0]["disease"] == "diabetes"
    )
    print(f"  User A GET /history?disease=diabetes -> total={data_a_diab['total']}, disease={data_a_diab['items'][0]['disease']} -> [{'PASS' if pass_3a else 'FAIL'}]")
    results.append(("3. User A Diabetes Filter", pass_3a))

    # User A requests heart_disease
    res_a_heart = client.get("/history?disease=heart_disease", headers=headers_a)
    assert res_a_heart.status_code == 200
    data_a_heart = res_a_heart.json()
    pass_3b = (
        data_a_heart["total"] == 1
        and data_a_heart["items"][0]["id"] == pred_a_heart["id"]
        and data_a_heart["items"][0]["disease"] == "heart_disease"
    )
    print(f"  User A GET /history?disease=heart_disease -> total={data_a_heart['total']}, disease={data_a_heart['items'][0]['disease']} -> [{'PASS' if pass_3b else 'FAIL'}]")
    results.append(("3. User A Heart Filter", pass_3b))

    # User B requests heart_disease (has 0)
    res_b_heart = client.get("/history?disease=heart_disease", headers=headers_b)
    assert res_b_heart.status_code == 200
    data_b_heart = res_b_heart.json()
    pass_3c = data_b_heart["total"] == 0 and data_b_heart["items"] == []
    print(f"  User B GET /history?disease=heart_disease -> total={data_b_heart['total']}, items={data_b_heart['items']} -> [{'PASS' if pass_3c else 'FAIL'}]\n")
    results.append(("3. User B Heart Filter Empty", pass_3c))

    # ---------------------------------------------------------
    # Scenario 4: Individual Prediction Access & Cross-Tenant Block
    # ---------------------------------------------------------
    print("--- 4. VERIFY INDIVIDUAL PREDICTION ACCESS & CROSS-TENANT BLOCK ---")
    # User A attempts to access User B's prediction
    res_a_access_b = client.get(f"/history/{pred_b_diabetes['id']}", headers=headers_a)
    pass_4a = res_a_access_b.status_code == 404
    print(f"  User A GET /history/{pred_b_diabetes['id']} (User B's doc) -> HTTP {res_a_access_b.status_code} [{'PASS (Blocked with 404)' if pass_4a else 'FAIL'}]")
    results.append(("4. User A blocked from User B doc", pass_4a))

    # User B attempts to access User A's predictions
    res_b_access_a1 = client.get(f"/history/{pred_a_diabetes['id']}", headers=headers_b)
    pass_4b = res_b_access_a1.status_code == 404
    print(f"  User B GET /history/{pred_a_diabetes['id']} (User A's doc 1) -> HTTP {res_b_access_a1.status_code} [{'PASS (Blocked with 404)' if pass_4b else 'FAIL'}]")
    results.append(("4. User B blocked from User A doc 1", pass_4b))

    res_b_access_a2 = client.get(f"/history/{pred_a_heart['id']}", headers=headers_b)
    pass_4c = res_b_access_a2.status_code == 404
    print(f"  User B GET /history/{pred_a_heart['id']} (User A's doc 2) -> HTTP {res_b_access_a2.status_code} [{'PASS (Blocked with 404)' if pass_4c else 'FAIL'}]")
    results.append(("4. User B blocked from User A doc 2", pass_4c))

    # Legitimate ownership access
    res_a_access_a = client.get(f"/history/{pred_a_diabetes['id']}", headers=headers_a)
    pass_4d = res_a_access_a.status_code == 200 and res_a_access_a.json()["id"] == pred_a_diabetes["id"]
    print(f"  User A GET /history/{pred_a_diabetes['id']} (Own doc) -> HTTP {res_a_access_a.status_code} [{'PASS' if pass_4d else 'FAIL'}]")
    results.append(("4. User A access own doc", pass_4d))

    res_b_access_b = client.get(f"/history/{pred_b_diabetes['id']}", headers=headers_b)
    pass_4e = res_b_access_b.status_code == 200 and res_b_access_b.json()["id"] == pred_b_diabetes["id"]
    print(f"  User B GET /history/{pred_b_diabetes['id']} (Own doc) -> HTTP {res_b_access_b.status_code} [{'PASS' if pass_4e else 'FAIL'}]\n")
    results.append(("4. User B access own doc", pass_4e))

    # ---------------------------------------------------------
    # Scenario 5: Direct Database Ownership Inspection
    # ---------------------------------------------------------
    print("--- 5. DIRECT MONGODB DATABASE OWNERSHIP AUDIT ---")
    raw_docs = list(mock_db["predictions"].find())
    print(f"  Total raw documents in 'predictions' collection: {len(raw_docs)}")
    db_pass = True
    for doc in raw_docs:
        doc_id = str(doc["_id"])
        user_id_field = doc.get("user_id")
        is_valid_user_id = isinstance(user_id_field, ObjectId) and (str(user_id_field) in (user_a_id, user_b_id))
        has_created_at = isinstance(doc.get("created_at"), datetime)
        if not (is_valid_user_id and has_created_at):
            db_pass = False
            print(f"  [SECURITY ALERT] Invalid doc: {doc_id}, user_id={user_id_field}")

    print(f"  All documents have immutable BSON ObjectId user_id strictly bound to creator: [{'PASS' if db_pass else 'FAIL'}]\n")
    results.append(("5. DB user_id Ownership Audit", db_pass))

    # ---------------------------------------------------------
    # Scenario 6: Authentication Failure Matrix across all Routes
    # ---------------------------------------------------------
    print("--- 6. AUTHENTICATION FAILURE MATRIX ---")
    expired_payload = {"sub": user_a_id, "exp": datetime.now(timezone.utc) - timedelta(minutes=15)}
    expired_token = jwt.encode(expired_payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

    endpoints_to_test = [
        ("GET /history", "/history"),
        ("GET /history/{id}", f"/history/{pred_a_diabetes['id']}"),
        ("GET /history?disease=diabetes", "/history?disease=diabetes"),
    ]

    for label, url in endpoints_to_test:
        # No JWT
        r_none = client.get(url)
        pass_none = r_none.status_code in (401, 403)

        # Invalid JWT
        r_bad = client.get(url, headers={"Authorization": "Bearer malformed_token_123"})
        pass_bad = r_bad.status_code == 401

        # Expired JWT
        r_exp = client.get(url, headers={"Authorization": f"Bearer {expired_token}"})
        pass_exp = r_exp.status_code == 401

        # Valid JWT
        r_val = client.get(url, headers=headers_a)
        pass_val = r_val.status_code == 200

        all_auth = pass_none and pass_bad and pass_exp and pass_val
        print(f"  {label:<30} -> NoToken:{r_none.status_code} Bad:{r_bad.status_code} Exp:{r_exp.status_code} Valid:{r_val.status_code} [{'PASS' if all_auth else 'FAIL'}]")
        results.append((f"6. Auth: {label}", all_auth))

    print()

    # ---------------------------------------------------------
    # Scenario 7: Combined Pagination + Disease Filter + Tenant Isolation
    # ---------------------------------------------------------
    print("--- 7. COMBINED PAGINATION + FILTER + ISOLATION ---")
    base_time = datetime.now(timezone.utc)
    # Add 4 more diabetes predictions for User A (total 5)
    for i in range(1, 5):
        history_service.create_prediction(
            user_id=user_a_id,
            payload=PredictionHistoryCreate(
                user_id=user_a_id,
                disease="diabetes",
                disease_display_name="Diabetes Assessment",
                input_type="tabular",
                model=PredictionModelInfo(version="v1", model_type="LR"),
                input_data={"seq": i, "user": "A"},
                result=PredictionResultRecord(prediction="P", is_positive=True),
                created_at=base_time + timedelta(minutes=i),
            )
        )

    # Add 3 more diabetes predictions for User B (total 4)
    for j in range(1, 4):
        history_service.create_prediction(
            user_id=user_b_id,
            payload=PredictionHistoryCreate(
                user_id=user_b_id,
                disease="diabetes",
                disease_display_name="Diabetes Assessment",
                input_type="tabular",
                model=PredictionModelInfo(version="v1", model_type="LR"),
                input_data={"seq": j, "user": "B"},
                result=PredictionResultRecord(prediction="P", is_positive=True),
                created_at=base_time + timedelta(minutes=10 + j),
            )
        )

    # User A queries page 1 (limit=2, skip=0)
    p1_a = client.get("/history?disease=diabetes&limit=2&skip=0", headers=headers_a).json()
    # User A queries page 2 (limit=2, skip=2)
    p2_a = client.get("/history?disease=diabetes&limit=2&skip=2", headers=headers_a).json()
    # User A queries page 3 (limit=2, skip=4)
    p3_a = client.get("/history?disease=diabetes&limit=2&skip=4", headers=headers_a).json()

    # User B queries page 1 (limit=2, skip=0)
    p1_b = client.get("/history?disease=diabetes&limit=2&skip=0", headers=headers_b).json()

    pass_7 = (
        p1_a["total"] == 5
        and len(p1_a["items"]) == 2
        and p2_a["total"] == 5
        and len(p2_a["items"]) == 2
        and p3_a["total"] == 5
        and len(p3_a["items"]) == 1
        and all(item["user_id"] == user_a_id for item in p1_a["items"] + p2_a["items"] + p3_a["items"])
        and p1_b["total"] == 4
        and len(p1_b["items"]) == 2
        and all(item["user_id"] == user_b_id for item in p1_b["items"])
    )
    print(f"  User A Total Diabetes: {p1_a['total']} across 3 pages (2, 2, 1 items)")
    print(f"  User B Total Diabetes: {p1_b['total']} across page 1 (2 items)")
    print(f"  Combined Pagination & Isolation: [{'PASS' if pass_7 else 'FAIL'}]\n")
    results.append(("7. Combined Pagination + Filter + Isolation", pass_7))

    # ---------------------------------------------------------
    # Final Summary Matrix
    # ---------------------------------------------------------
    print("=" * 70)
    print(" FINAL VERIFICATION RESULTS SUMMARY:")
    print("-" * 70)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {name:<50} : [{status}]")
    print("=" * 70)

    app.dependency_overrides.clear()

    if all_passed:
        print("\n>>> ALL STAGE 6 HISTORY SECURITY & ISOLATION CHECKS PASSED SUCCESSFULLY! <<<\n")
    else:
        print("\n>>> CRITICAL: SECURITY CHECKS FAILED! <<<\n")

    return all_passed


# Pytest bridge to include in automated test suites
def test_verify_history_authorization_and_isolation():
    assert run_comprehensive_history_security_verification() is True


if __name__ == "__main__":
    success = run_comprehensive_history_security_verification()
    sys.exit(0 if success else 1)
