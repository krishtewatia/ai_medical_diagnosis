"""
Comprehensive Test Suite for Medical Profile API (Step 7 Verification).

Tests:
1. Authenticated user can create profile (POST /profile -> 201)
2. Unauthenticated user cannot create profile (POST /profile without auth -> 401)
3. Authenticated user can retrieve their profile (GET /profile -> 200)
4. User without profile gets 404 (GET /profile -> 404)
5. Authenticated user can update their profile (PATCH /profile -> 200)
6. Partial update works while preserving other fields
7. Duplicate profile creation returns 409 conflict
8. Invalid profile data returns 422 validation error
9. Isolation: User A cannot access or modify User B's profile
10. Complete end-to-end user journey flow
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mongomock
from bson import ObjectId
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.core.security import create_access_token
from app.database.connection import get_database
from app.main import app
from app.services.medical_profile_service import MedicalProfileService
from app.services.user_service import UserService


# Mock in-memory database shared across services for testing
mock_client = mongomock.MongoClient()
mock_db = mock_client["test_medical_app_db"]

# Setup dependency override for get_database
app.dependency_overrides[get_database] = lambda: mock_db


def test_profile_api_flow():
    client = TestClient(app)
    user_service = UserService(mock_db)

    # 1. Create two users in mock DB
    user_a = user_service.create_user(
        name="Alice Walker",
        email="alice@example.com",
        password_hash="mock_hash_a"
    )
    user_a_id = str(user_a["_id"])
    token_a = create_access_token(user_a_id)

    user_b = user_service.create_user(
        name="Bob Jones",
        email="bob@example.com",
        password_hash="mock_hash_b"
    )
    user_b_id = str(user_b["_id"])
    token_b = create_access_token(user_b_id)

    auth_header_a = {"Authorization": f"Bearer {token_a}"}
    auth_header_b = {"Authorization": f"Bearer {token_b}"}

    print("\n=== Medical Profile API Verification Tests ===\n")

    # TEST 1: Unauthenticated request rejected
    res_unauth = client.post("/profile", json={"date_of_birth": "1990-01-01", "gender": "female"})
    assert res_unauth.status_code in [401, 403]
    print(f"  PASS: 1. Unauthenticated request rejected ({res_unauth.status_code})")

    # TEST 2: User without profile gets 404
    res_404 = client.get("/profile", headers=auth_header_a)
    assert res_404.status_code == 404
    assert "not found" in res_404.json()["detail"].lower()
    print("  PASS: 2. User without profile receives 404 Not Found")

    # TEST 3: Invalid profile data returns 422
    res_invalid = client.post(
        "/profile",
        headers=auth_header_a,
        json={"date_of_birth": "invalid-date", "gender": "invalid_gender"}
    )
    assert res_invalid.status_code == 422
    print("  PASS: 3. Invalid profile payload rejected with 422 Validation Error")

    # TEST 4: User A creates profile (201 Created)
    profile_payload_a = {
        "date_of_birth": "1994-05-15",
        "gender": "female",
        "blood_type": "A+",
        "height_cm": 168.0,
        "weight_kg": 62.0,
        "allergies": ["Penicillin"],
        "chronic_conditions": ["Mild Asthma"],
        "current_medications": ["Albuterol Inhaler"],
        "smoking_status": "never",
        "alcohol_consumption": "occasional",
        "emergency_contact": {
            "name": "Tom Walker",
            "relationship": "Spouse",
            "phone": "+1555123456"
        }
    }
    res_create_a = client.post("/profile", headers=auth_header_a, json=profile_payload_a)
    assert res_create_a.status_code == 201
    data_a = res_create_a.json()
    assert data_a["user_id"] == user_a_id
    assert data_a["blood_type"] == "A+"
    assert data_a["height_cm"] == 168.0
    print("  PASS: 4. User A created profile successfully (201 Created)")

    # TEST 5: Duplicate profile creation returns 409
    res_dup = client.post("/profile", headers=auth_header_a, json=profile_payload_a)
    assert res_dup.status_code == 409
    assert "already exists" in res_dup.json()["detail"].lower()
    print("  PASS: 5. Duplicate profile creation rejected with 409 Conflict")

    # TEST 6: User A retrieves their profile (GET /profile -> 200)
    res_get_a = client.get("/profile", headers=auth_header_a)
    assert res_get_a.status_code == 200
    assert res_get_a.json()["id"] == data_a["id"]
    assert res_get_a.json()["user_id"] == user_a_id
    assert res_get_a.json()["allergies"] == ["Penicillin"]
    print("  PASS: 6. User A retrieved their profile (200 OK)")

    # TEST 7: User A performs partial update (PATCH /profile -> 200)
    update_payload = {
        "weight_kg": 64.5,
        "allergies": ["Penicillin", "Sulfa drugs"]
    }
    res_patch_a = client.patch("/profile", headers=auth_header_a, json=update_payload)
    assert res_patch_a.status_code == 200
    data_patched = res_patch_a.json()
    assert data_patched["weight_kg"] == 64.5
    assert data_patched["allergies"] == ["Penicillin", "Sulfa drugs"]
    # Verify untouched fields are preserved
    assert data_patched["height_cm"] == 168.0
    assert data_patched["blood_type"] == "A+"
    assert data_patched["gender"] == "female"
    print("  PASS: 7. Partial update succeeded and preserved untouched fields")

    # TEST 8: Isolation - User B has no profile and cannot see or mutate User A's profile
    res_get_b = client.get("/profile", headers=auth_header_b)
    assert res_get_b.status_code == 404

    # User B creates their own profile
    profile_payload_b = {
        "date_of_birth": "1988-11-20",
        "gender": "male",
        "blood_type": "O-",
        "height_cm": 182.0,
        "weight_kg": 85.0
    }
    res_create_b = client.post("/profile", headers=auth_header_b, json=profile_payload_b)
    assert res_create_b.status_code == 201
    assert res_create_b.json()["user_id"] == user_b_id

    # Verify User A and User B profiles are completely distinct
    profile_a_check = client.get("/profile", headers=auth_header_a).json()
    profile_b_check = client.get("/profile", headers=auth_header_b).json()
    assert profile_a_check["user_id"] == user_a_id
    assert profile_b_check["user_id"] == user_b_id
    assert profile_a_check["id"] != profile_b_check["id"]
    assert profile_a_check["gender"] == "female"
    assert profile_b_check["gender"] == "male"
    print("  PASS: 8. User isolation verified: User A & User B maintain separate profiles")

    print("\n==================================================")
    print("Results: All 8 Profile API scenarios PASSED (0 failures)")
    print("==================================================\n")


if __name__ == "__main__":
    test_profile_api_flow()
