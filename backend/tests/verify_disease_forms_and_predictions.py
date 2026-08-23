import pytest
import uuid
from fastapi.testclient import TestClient
from app.main import app
from app.core.security import create_access_token

def test_dynamic_disease_form_contract_and_predictions():
    client = TestClient(app)
    
    # Create test user to obtain valid JWT
    user_suffix = uuid.uuid4().hex[:8]
    reg_payload = {
        "name": f"Doctor {user_suffix}",
        "email": f"dr_{user_suffix}@test.org",
        "password": "Password123!"
    }
    reg_res = client.post("/auth/register", json=reg_payload)
    assert reg_res.status_code == 201
    user_id = reg_res.json()["id"]
    token = create_access_token(user_id)
    auth_headers = {"Authorization": f"Bearer {token}"}

    # 1. Verify Diabetes Dynamic Config
    print("\n--- 1. Testing GET /diseases/diabetes ---")
    dia_res = client.get("/diseases/diabetes")
    assert dia_res.status_code == 200
    dia_cfg = dia_res.json()
    assert dia_cfg["id"] == "diabetes"
    assert dia_cfg["category"] == "tabular"
    assert len(dia_cfg["required_fields"]) == 6
    dia_field_names = [f["name"] for f in dia_cfg["required_fields"]]
    expected_dia_fields = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]
    assert dia_field_names == expected_dia_fields
    print(f"✅ Diabetes dynamically discovered 6 fields: {dia_field_names}")

    # 2. Verify Diabetes Prediction Execution
    print("\n--- 2. Testing POST /predictions (Diabetes) ---")
    dia_payload = {
        "disease_id": "diabetes",
        "inputs": {
            "Pregnancies": 2,
            "Glucose": 140.0,
            "BloodPressure": 80.0,
            "BMI": 32.5,
            "DiabetesPedigreeFunction": 0.65,
            "Age": 45
        }
    }
    dia_pred_res = client.post("/predictions", json=dia_payload, headers=auth_headers)
    assert dia_pred_res.status_code == 200, f"Failed: {dia_pred_res.text}"
    dia_pred = dia_pred_res.json()
    assert dia_pred["disease_id"] == "diabetes"
    assert "prediction_label" in dia_pred
    assert "is_positive" in dia_pred
    assert "probability" in dia_pred
    print(f"✅ Diabetes prediction executed: {dia_pred['prediction_label']} (Probability: {dia_pred['probability']:.3f})")

    # 3. Verify Heart Disease Dynamic Config
    print("\n--- 3. Testing GET /diseases/heart_disease ---")
    hd_res = client.get("/diseases/heart_disease")
    assert hd_res.status_code == 200
    hd_cfg = hd_res.json()
    assert hd_cfg["id"] == "heart_disease"
    assert hd_cfg["category"] == "tabular"
    assert len(hd_cfg["required_fields"]) == 13
    hd_field_names = [f["name"] for f in hd_cfg["required_fields"]]
    expected_hd_fields = [
        "age", "sex", "chest_pain_type", "resting_bp", "cholestoral",
        "fasting_blood_sugar", "restecg", "max_hr", "exang", "oldpeak",
        "slope", "num_major_vessels", "thal"
    ]
    assert hd_field_names == expected_hd_fields
    print(f"✅ Heart Disease dynamically discovered 13 fields: {hd_field_names}")

    # 4. Verify Heart Disease Prediction Execution
    print("\n--- 4. Testing POST /predictions (Heart Disease) ---")
    hd_payload = {
        "disease_id": "heart_disease",
        "inputs": {
            "age": 55,
            "sex": 1,
            "chest_pain_type": 1,
            "resting_bp": 130.0,
            "cholestoral": 240.0,
            "fasting_blood_sugar": 0,
            "restecg": 0,
            "max_hr": 160.0,
            "exang": 0,
            "oldpeak": 1.2,
            "slope": 1,
            "num_major_vessels": 0,
            "thal": 2
        }
    }
    hd_pred_res = client.post("/predictions", json=hd_payload, headers=auth_headers)
    assert hd_pred_res.status_code == 200, f"Failed: {hd_pred_res.text}"
    hd_pred = hd_pred_res.json()
    assert hd_pred["disease_id"] == "heart_disease"
    assert "prediction_label" in hd_pred
    assert "is_positive" in hd_pred
    assert "probability" in hd_pred
    print(f"✅ Heart disease prediction executed: {hd_pred['prediction_label']} (Probability: {hd_pred['probability']:.3f})")

    # 5. Verify Invalid Disease ID returns 404
    print("\n--- 5. Testing Unknown Disease ID ---")
    unknown_res = client.get("/diseases/non_existent_disease")
    assert unknown_res.status_code == 404
    print("✅ Unknown disease properly returned 404 Not Found")

    print("\n🎉 ALL DYNAMIC FORM & PREDICTION TESTS PASSED!\n")
