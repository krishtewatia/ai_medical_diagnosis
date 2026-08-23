import uuid
import pytest
from fastapi.testclient import TestClient
from app.main import app

def test_full_step25_e2e_user_journey():
    client = TestClient(app)
    
    # 1. Register User
    print("\n--- 1. Testing User Registration ---")
    user_suffix = uuid.uuid4().hex[:8]
    email = f"dr_end2end_{user_suffix}@aegishealth.org"
    password = "SecurePassword123!"
    name = f"Dr. Alex Thorne {user_suffix}"

    reg_res = client.post("/auth/register", json={
        "name": name,
        "email": email,
        "password": password
    })
    assert reg_res.status_code == 201, f"Registration failed: {reg_res.text}"
    user_data = reg_res.json()
    assert user_data["email"] == email
    print(f"[OK] User registered: {name} ({email})")

    # 2. Login User & Obtain JWT
    print("\n--- 2. Testing User Login ---")
    login_res = client.post("/auth/login", json={
        "email": email,
        "password": password
    })
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token_data = login_res.json()
    token = token_data["access_token"]
    auth_headers = {"Authorization": f"Bearer {token}"}
    print("[OK] JWT successfully issued.")

    # 3. Dashboard Me Check
    print("\n--- 3. Testing GET /auth/me ---")
    me_res = client.get("/auth/me", headers=auth_headers)
    assert me_res.status_code == 200
    assert me_res.json()["name"] == name
    print(f"[OK] Session verified: {me_res.json()['name']}")

    # 4. View Available Diseases
    print("\n--- 4. Testing GET /diseases ---")
    diseases_res = client.get("/diseases")
    assert diseases_res.status_code == 200
    diseases = diseases_res.json()
    disease_ids = [d["id"] for d in diseases]
    assert "diabetes" in disease_ids
    assert "heart_disease" in disease_ids
    print(f"[OK] Active modules discovered: {disease_ids}")

    # 5. Select Diabetes & Retrieve Config
    print("\n--- 5. Testing GET /diseases/diabetes ---")
    dia_cfg_res = client.get("/diseases/diabetes")
    assert dia_cfg_res.status_code == 200
    dia_cfg = dia_cfg_res.json()
    assert dia_cfg["id"] == "diabetes"
    assert len(dia_cfg["required_fields"]) == 6
    print("[OK] Diabetes specification retrieved (6 features).")

    # 6. Run Diabetes Prediction
    print("\n--- 6. Testing POST /predictions (Diabetes) ---")
    dia_payload = {
        "disease_id": "diabetes",
        "inputs": {
            "Pregnancies": 3,
            "Glucose": 155.0,
            "BloodPressure": 82.0,
            "BMI": 33.1,
            "DiabetesPedigreeFunction": 0.52,
            "Age": 50
        }
    }
    dia_pred_res = client.post("/predictions", json=dia_payload, headers=auth_headers)
    assert dia_pred_res.status_code == 200, f"Prediction failed: {dia_pred_res.text}"
    dia_pred = dia_pred_res.json()
    assert dia_pred["disease_id"] == "diabetes"
    assert "prediction_label" in dia_pred
    print(f"[OK] Diabetes prediction outcome: {dia_pred['prediction_label']} (Probability: {dia_pred.get('probability')})")

    # 7. Select Heart Disease & Run Prediction
    print("\n--- 7. Testing POST /predictions (Heart Disease) ---")
    hd_payload = {
        "disease_id": "heart_disease",
        "inputs": {
            "age": 60,
            "sex": 1,
            "chest_pain_type": 2,
            "resting_bp": 140.0,
            "cholestoral": 260.0,
            "fasting_blood_sugar": 1,
            "restecg": 1,
            "max_hr": 145.0,
            "exang": 1,
            "oldpeak": 2.0,
            "slope": 2,
            "num_major_vessels": 1,
            "thal": 3
        }
    }
    hd_pred_res = client.post("/predictions", json=hd_payload, headers=auth_headers)
    assert hd_pred_res.status_code == 200, f"Prediction failed: {hd_pred_res.text}"
    hd_pred = hd_pred_res.json()
    assert hd_pred["disease_id"] == "heart_disease"
    assert "prediction_label" in hd_pred
    print(f"[OK] Heart Disease prediction outcome: {hd_pred['prediction_label']} (Probability: {hd_pred.get('probability')})")

    # 8. Query History
    print("\n--- 8. Testing GET /history ---")
    hist_res = client.get("/history", headers=auth_headers)
    assert hist_res.status_code == 200
    hist = hist_res.json()
    assert hist["total"] == 2
    assert len(hist["items"]) == 2
    latest_item = hist["items"][0]
    print(f"[OK] History retrieved {hist['total']} records. Latest: {latest_item['disease_display_name']}")

    # 9. Query History Detail by ID
    print("\n--- 9. Testing GET /history/{prediction_id} ---")
    detail_res = client.get(f"/history/{latest_item['id']}", headers=auth_headers)
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["id"] == latest_item["id"]
    assert detail["disease"] == latest_item["disease"]
    print(f"[OK] Audit detail retrieved for record: {detail['id']}")

    # 10. Filter History by Disease
    print("\n--- 10. Testing GET /history?disease=diabetes ---")
    dia_hist_res = client.get("/history?disease=diabetes", headers=auth_headers)
    assert dia_hist_res.status_code == 200
    dia_hist = dia_hist_res.json()
    assert dia_hist["total"] == 1
    assert dia_hist["items"][0]["disease"] == "diabetes"
    print("[OK] Filtered history successfully by disease='diabetes'")

    # 11. Create Medical Profile
    print("\n--- 11. Testing POST /profile ---")
    profile_create = {
        "date_of_birth": "1980-05-15",
        "gender": "male",
        "blood_type": "O+",
        "height_cm": 180.0,
        "weight_kg": 82.5,
        "allergies": ["Penicillin"],
        "chronic_conditions": ["Mild Hypertension"],
        "current_medications": ["Lisinopril 10mg"],
        "smoking_status": "former",
        "alcohol_consumption": "occasional",
        "emergency_contact": {
            "name": "Sarah Thorne",
            "relationship": "Spouse",
            "phone": "+1-555-9012"
        }
    }
    create_prof_res = client.post("/profile", json=profile_create, headers=auth_headers)
    assert create_prof_res.status_code == 201, f"Profile create failed: {create_prof_res.text}"
    prof = create_prof_res.json()
    assert prof["gender"] == "male"
    assert prof["height_cm"] == 180.0
    print("[OK] Medical Profile created successfully.")

    # 12. Update Medical Profile (Weight update)
    print("\n--- 12. Testing PATCH /profile ---")
    update_prof_res = client.patch("/profile", json={"weight_kg": 80.0}, headers=auth_headers)
    assert update_prof_res.status_code == 200, f"Profile update failed: {update_prof_res.text}"
    updated_prof = update_prof_res.json()
    assert updated_prof["weight_kg"] == 80.0
    assert updated_prof["height_cm"] == 180.0
    print(f"[OK] Medical Profile updated: weight_kg = {updated_prof['weight_kg']}")

    # 13. Verify GET /profile
    print("\n--- 13. Testing GET /profile ---")
    get_prof_res = client.get("/profile", headers=auth_headers)
    assert get_prof_res.status_code == 200
    assert get_prof_res.json()["weight_kg"] == 80.0
    print("[OK] Verified persistent profile state.")

    # 14. Unauthenticated Access Protection
    print("\n--- 14. Testing Unauthenticated Access Protection ---")
    unauth_res = client.get("/history")
    assert unauth_res.status_code in [401, 403]
    unauth_prof_res = client.get("/profile")
    assert unauth_prof_res.status_code in [401, 403]
    print("[OK] Unauthenticated requests properly rejected with HTTP 401.")

    print("\n[SUCCESS] COMPLETE END-TO-END STEP 25 USER JOURNEY PASSED 100%!\n")

if __name__ == "__main__":
    test_full_step25_e2e_user_journey()
