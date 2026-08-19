"""
Test suite for GET /auth/me endpoint verification.

Tests all required cases:
1. Valid JWT -> returns current user
2. No JWT -> rejected
3. Invalid JWT -> rejected
4. Expired JWT -> rejected
5. JWT belonging to nonexistent user -> rejected
6. Response does not expose password/hash or secrets
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

# Add the backend directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jwt
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.security import create_access_token
from app.main import app
from app.services.user_service import UserService


# Mock user in DB
FAKE_USER_ID = "64a1b2c3d4e5f6a7b8c9d0e1"
MOCK_USER_DOC = {
    "_id": FAKE_USER_ID,
    "name": "Test User",
    "email": "test@example.com",
    "password_hash": "$argon2id$v=19$m=65536,t=3,p=4$fakehash123",
    "created_at": datetime.now(timezone.utc),
    "updated_at": datetime.now(timezone.utc),
}


def test_auth_me_valid_jwt(monkeypatch):
    """1. Valid JWT -> returns current user (200 OK)"""
    client = TestClient(app)
    
    # Mock find_by_id to return MOCK_USER_DOC
    monkeypatch.setattr(
        UserService,
        "find_by_id",
        lambda self, uid: MOCK_USER_DOC if uid == FAKE_USER_ID else None
    )

    token = create_access_token(FAKE_USER_ID)
    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["id"] == FAKE_USER_ID
    assert data["name"] == "Test User"
    assert data["email"] == "test@example.com"
    print("  PASS: 1. Valid JWT returns current user")


def test_auth_me_no_jwt():
    """2. No JWT -> rejected with 401 or 403"""
    client = TestClient(app)
    response = client.get("/auth/me")
    assert response.status_code in [401, 403], f"Expected 401/403, got {response.status_code}"
    print(f"  PASS: 2. No JWT rejected with status {response.status_code}")


def test_auth_me_invalid_jwt():
    """3. Invalid JWT -> rejected with 401"""
    client = TestClient(app)
    response = client.get(
        "/auth/me",
        headers={"Authorization": "Bearer invalid.token.value"}
    )
    assert response.status_code == 401, f"Expected 401, got {response.status_code}"
    assert "invalid authentication token" in response.json()["detail"].lower()
    print("  PASS: 3. Invalid JWT rejected with 401")


def test_auth_me_expired_jwt():
    """4. Expired JWT -> rejected with 401"""
    client = TestClient(app)
    payload = {
        "sub": FAKE_USER_ID,
        "exp": datetime.now(timezone.utc) - timedelta(minutes=5),
    }
    expired_token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {expired_token}"}
    )
    assert response.status_code == 401, f"Expected 401, got {response.status_code}"
    assert "expired" in response.json()["detail"].lower()
    print("  PASS: 4. Expired JWT rejected with 401")


def test_auth_me_nonexistent_user(monkeypatch):
    """5. JWT belonging to nonexistent user -> rejected with 401"""
    client = TestClient(app)
    monkeypatch.setattr(UserService, "find_by_id", lambda self, uid: None)

    token = create_access_token("507f1f77bcf86cd799439011")
    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 401, f"Expected 401, got {response.status_code}"
    assert "user not found" in response.json()["detail"].lower()
    print("  PASS: 5. JWT for nonexistent user rejected with 401")


def test_auth_me_no_sensitive_data_exposed(monkeypatch):
    """6. Response does not expose password, hash, or internal fields"""
    client = TestClient(app)
    monkeypatch.setattr(UserService, "find_by_id", lambda self, uid: MOCK_USER_DOC)

    token = create_access_token(FAKE_USER_ID)
    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Strictly check only safe fields exist
    assert "password" not in data
    assert "password_hash" not in data
    assert set(data.keys()) == {"id", "name", "email"}
    print("  PASS: 6. Response strictly exposes only safe profile fields (id, name, email)")


if __name__ == "__main__":
    import pytest
    print("\n=== GET /auth/me Endpoint Verification Tests ===\n")
    
    class MonkeyPatch:
        def setattr(self, target, attr, value):
            setattr(target, attr, value)

    mp = MonkeyPatch()
    
    tests = [
        ("1. Valid JWT", lambda: test_auth_me_valid_jwt(mp)),
        ("2. No JWT", test_auth_me_no_jwt),
        ("3. Invalid JWT", test_auth_me_invalid_jwt),
        ("4. Expired JWT", test_auth_me_expired_jwt),
        ("5. Nonexistent user", lambda: test_auth_me_nonexistent_user(mp)),
        ("6. No sensitive data exposed", lambda: test_auth_me_no_sensitive_data_exposed(mp)),
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

    print(f"\n{'=' * 45}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 45}\n")
    sys.exit(1 if failed else 0)
