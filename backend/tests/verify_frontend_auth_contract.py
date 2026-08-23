import sys
import uuid
from fastapi.testclient import TestClient
from app.main import app
from app.api.dependencies import get_current_user

def test_frontend_auth_lifecycle():
    # Ensure real JWT authentication runs by clearing any test mock overrides for get_current_user
    if get_current_user in app.dependency_overrides:
        del app.dependency_overrides[get_current_user]

    client = TestClient(app)
    
    unique_suffix = uuid.uuid4().hex[:8]
    test_email = f"clinical_user_{unique_suffix}@aegishealth.org"
    test_password = "SecurePassword123!"
    test_name = f"Dr. Clinical User {unique_suffix}"

    print("\n--- 1. Testing Registration (POST /auth/register) ---")
    reg_payload = {
        "name": test_name,
        "email": test_email,
        "password": test_password
    }
    reg_res = client.post("/auth/register", json=reg_payload)
    assert reg_res.status_code == 201, f"Registration failed: {reg_res.status_code} {reg_res.text}"
    user_data = reg_res.json()
    assert user_data["email"] == test_email
    assert user_data["name"] == test_name
    assert "id" in user_data
    assert "password" not in user_data
    print(f"✅ User registered: id={user_data['id']}, email={user_data['email']}")

    print("\n--- 2. Testing Duplicate Email Conflict (409 Conflict) ---")
    dup_res = client.post("/auth/register", json=reg_payload)
    assert dup_res.status_code == 409, f"Expected 409 Conflict, got {dup_res.status_code}"
    print(f"✅ Duplicate email rejected properly: {dup_res.json()['detail']}")

    print("\n--- 3. Testing Login with Invalid Password (401 Unauthorized) ---")
    wrong_login_payload = {
        "email": test_email,
        "password": "WrongPassword999"
    }
    bad_login_res = client.post("/auth/login", json=wrong_login_payload)
    assert bad_login_res.status_code == 401
    print(f"✅ Invalid password rejected properly: {bad_login_res.json()['detail']}")

    print("\n--- 4. Testing Login with Valid Credentials (POST /auth/login) ---")
    valid_login_payload = {
        "email": test_email,
        "password": test_password
    }
    login_res = client.post("/auth/login", json=valid_login_payload)
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    access_token = token_data["access_token"]
    assert len(access_token) > 20
    print("✅ Login succeeded and JWT access token issued.")

    print("\n--- 5. Testing Current User Retrieval (GET /auth/me) with JWT ---")
    me_res = client.get("/auth/me", headers={"Authorization": f"Bearer {access_token}"})
    assert me_res.status_code == 200, f"Expected 200, got {me_res.status_code}: {me_res.text}"
    me_data = me_res.json()
    assert me_data["id"] == user_data["id"]
    assert me_data["email"] == test_email
    assert me_data["name"] == test_name
    print(f"✅ /auth/me successfully returned authenticated identity: {me_data['name']} ({me_data['email']})")

    print("\n--- 6. Testing Unauthorized Access (GET /auth/me without Token) ---")
    no_auth_res = client.get("/auth/me")
    assert no_auth_res.status_code == 401
    print(f"✅ Missing JWT token rejected: {no_auth_res.json()['detail']}")

    print("\n--- 7. Testing Invalid JWT Token ---")
    invalid_token_res = client.get("/auth/me", headers={"Authorization": "Bearer invalid.token.value"})
    assert invalid_token_res.status_code == 401
    print(f"✅ Invalid JWT token rejected: {invalid_token_res.json()['detail']}")

    print("\n🎉 ALL FRONTEND AUTH CONTRACT TESTS PASSED PERFECTLY!\n")
