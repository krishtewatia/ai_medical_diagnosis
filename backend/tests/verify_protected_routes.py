"""
Step 3 Verification Script:
Verify Protected Routes Reject Missing/Invalid JWT

Demonstrates all required test conditions on GET /auth/me:
1. Request with no Authorization header -> rejected ([REJECTED])
2. Request with Authorization: Bearer invalid_token -> rejected ([REJECTED])
3. Request with an expired JWT -> rejected ([REJECTED])
4. Request with a JWT signed using the wrong secret -> rejected ([REJECTED])
5. Request with a valid JWT -> succeeds ([SUCCESS])
6. Valid JWT returns the correct authenticated user identity
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jwt
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.security import create_access_token
from app.main import app
from app.services.user_service import UserService


# Mock User Fixture
USER_ID = "64a1b2c3d4e5f6a7b8c9d0e1"
MOCK_USER = {
    "_id": USER_ID,
    "name": "Dr. Jane Doe",
    "email": "jane.doe@hospital.org",
    "password_hash": "$argon2id$v=19$m=65536,t=3,p=4$secret_hash_not_to_leak",
    "created_at": datetime.now(timezone.utc),
    "updated_at": datetime.now(timezone.utc),
}


def run_verification():
    client = TestClient(app)

    # Mock find_by_id on UserService
    def mock_find_by_id(self, user_id):
        if str(user_id) == USER_ID:
            return MOCK_USER
        return None

    UserService.find_by_id = mock_find_by_id

    print("\n========================================================")
    print(" STEP 3 - VERIFY PROTECTED ROUTE AUTHENTICATION (/auth/me)")
    print("========================================================\n")

    results = []

    # 1. No token
    res1 = client.get("/auth/me")
    is_rejected_1 = res1.status_code in [401, 403]
    status_icon_1 = "[REJECTED] (Expected)" if is_rejected_1 else "[UNEXPECTED SUCCESS]"
    print(f"1. No token:")
    print(f"   Request:  GET /auth/me (No Authorization header)")
    print(f"   Response: HTTP {res1.status_code} - {res1.json()}")
    print(f"   Result:   {status_icon_1}\n")
    results.append(("No token", is_rejected_1))

    # 2. Invalid token
    res2 = client.get("/auth/me", headers={"Authorization": "Bearer invalid_malformed_token"})
    is_rejected_2 = res2.status_code == 401
    status_icon_2 = "[REJECTED] (Expected)" if is_rejected_2 else "[UNEXPECTED SUCCESS]"
    print(f"2. Invalid token:")
    print(f"   Request:  GET /auth/me [Authorization: Bearer invalid_malformed_token]")
    print(f"   Response: HTTP {res2.status_code} - {res2.json()}")
    print(f"   Result:   {status_icon_2}\n")
    results.append(("Invalid token", is_rejected_2))

    # 3. Expired token
    expired_payload = {
        "sub": USER_ID,
        "exp": datetime.now(timezone.utc) - timedelta(minutes=10),
    }
    expired_jwt = jwt.encode(expired_payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    res3 = client.get("/auth/me", headers={"Authorization": f"Bearer {expired_jwt}"})
    is_rejected_3 = res3.status_code == 401
    status_icon_3 = "[REJECTED] (Expected)" if is_rejected_3 else "[UNEXPECTED SUCCESS]"
    print(f"3. Expired token:")
    print(f"   Request:  GET /auth/me [Authorization: Bearer <expired_token>]")
    print(f"   Response: HTTP {res3.status_code} - {res3.json()}")
    print(f"   Result:   {status_icon_3}\n")
    results.append(("Expired token", is_rejected_3))

    # 4. Wrong secret
    wrong_secret_payload = {
        "sub": USER_ID,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=60),
    }
    wrong_secret_jwt = jwt.encode(wrong_secret_payload, "completely_wrong_secret_key_12345678", algorithm="HS256")
    res4 = client.get("/auth/me", headers={"Authorization": f"Bearer {wrong_secret_jwt}"})
    is_rejected_4 = res4.status_code == 401
    status_icon_4 = "[REJECTED] (Expected)" if is_rejected_4 else "[UNEXPECTED SUCCESS]"
    print(f"4. Wrong secret:")
    print(f"   Request:  GET /auth/me [Authorization: Bearer <wrong_secret_signed_token>]")
    print(f"   Response: HTTP {res4.status_code} - {res4.json()}")
    print(f"   Result:   {status_icon_4}\n")
    results.append(("Wrong secret", is_rejected_4))

    # 5 & 6. Valid token -> succeeds and returns authenticated user
    valid_jwt = create_access_token(USER_ID)
    res5 = client.get("/auth/me", headers={"Authorization": f"Bearer {valid_jwt}"})
    body5 = res5.json()
    is_success_5 = (
        res5.status_code == 200
        and body5.get("id") == USER_ID
        and body5.get("email") == MOCK_USER["email"]
        and body5.get("name") == MOCK_USER["name"]
        and "password_hash" not in body5
    )
    status_icon_5 = "[SUCCESS] (Expected)" if is_success_5 else "[FAILED]"
    print(f"5. Valid token:")
    print(f"   Request:  GET /auth/me [Authorization: Bearer <valid_jwt>]")
    print(f"   Response: HTTP {res5.status_code} - {body5}")
    print(f"   Result:   {status_icon_5}\n")
    results.append(("Valid token", is_success_5))

    print("========================================================")
    print(" SUMMARY MATRIX:")
    print("--------------------------------------------------------")
    all_passed = True
    for scenario, passed in results:
        symbol = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {scenario:<16} : [{symbol}]")
    print("========================================================\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    run_verification()
