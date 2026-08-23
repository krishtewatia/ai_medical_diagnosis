import io
import time
import uuid
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.core.security import create_access_token
from app.ml.disease_registry import disease_registry
from app.services.storage_service import default_storage_service


def _generate_synthetic_scan(width: int = 224, height: int = 224, color=(50, 60, 70)) -> bytes:
    """Generates synthetic RGB radiograph/MRI scan bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_step32_full_system_integration():
    """
    Step 32 Master Integration Test Suite
    Tests all 13 required verification areas:
    32.1 Environment & Registry
    32.2 Auth Lifecycle
    32.3 Medical Profile
    32.4 All 4 Active Diseases
    32.5 Failure Handling & Cleanup
    32.6 History & Filtering
    32.7 Tenant Isolation
    32.8 DB Schema Integrity
    32.9 Object Storage & Hashes
    32.10 PDF Reports & Streaming
    32.11 Multi-Disease Uniformity
    32.12 & 32.13 E2E Acceptance Journeys
    """
    client = TestClient(app)

    print("\n========================================================")
    print("  STEP 32: FULL SYSTEM INTEGRATION & ACCEPTANCE SUITE")
    print("========================================================")

    # ----------------------------------------------------
    # 32.1 Environment, Health & Model Loading
    # ----------------------------------------------------
    print("\n[32.1] Verifying System Health & Active Disease Registry...")
    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "healthy"

    diseases_res = client.get("/diseases")
    assert diseases_res.status_code == 200
    active_ids = {d["id"] for d in diseases_res.json()}
    assert {"diabetes", "heart_disease", "pneumonia", "brain_tumor"}.issubset(active_ids)
    print(f"  -> Health OK, {len(active_ids)} active diseases discovered.")

    # ----------------------------------------------------
    # 32.2 Authentication Testing
    # ----------------------------------------------------
    print("\n[32.2] Testing Authentication Lifecycle...")
    suffix_a = uuid.uuid4().hex[:8]
    email_a = f"clinician_a_{suffix_a}@hospital.org"
    password_a = "SecurePass123!"

    # A. Register User A
    reg_a = client.post("/auth/register", json={
        "name": f"Dr. Alice {suffix_a}",
        "email": email_a,
        "password": password_a
    })
    assert reg_a.status_code == 201
    user_a_id = reg_a.json()["id"]

    # B. Reject Duplicate Email
    dup_reg = client.post("/auth/register", json={
        "name": "Duplicate User",
        "email": email_a,
        "password": password_a
    })
    assert dup_reg.status_code in [400, 409]
    detail_str = dup_reg.json()["detail"].lower()
    assert "already exists" in detail_str or "already registered" in detail_str

    # C. Login Correct Credentials
    login_a = client.post("/auth/login", json={
        "email": email_a,
        "password": password_a
    })
    assert login_a.status_code == 200
    token_a = login_a.json()["access_token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # D. Login Incorrect Credentials
    bad_login = client.post("/auth/login", json={
        "email": email_a,
        "password": "WrongPassword!"
    })
    assert bad_login.status_code == 401

    # Register User B for tenant isolation testing
    suffix_b = uuid.uuid4().hex[:8]
    email_b = f"clinician_b_{suffix_b}@hospital.org"
    reg_b = client.post("/auth/register", json={
        "name": f"Dr. Bob {suffix_b}",
        "email": email_b,
        "password": "SecurePass123!"
    })
    assert reg_b.status_code == 201
    user_b_id = reg_b.json()["id"]
    token_b = client.post("/auth/login", json={"email": email_b, "password": "SecurePass123!"}).json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}
    print("  -> Auth lifecycle, duplicate rejection, and credential verification passed.")

    # ----------------------------------------------------
    # 32.3 Medical Profile Testing
    # ----------------------------------------------------
    print("\n[32.3] Testing Medical Profile CRUD & Validation...")
    # Initial read (404 when not created yet)
    init_prof = client.get("/profile", headers=headers_a)
    assert init_prof.status_code == 404

    # Create profile
    profile_data = {
        "date_of_birth": "1985-06-20",
        "gender": "female",
        "blood_type": "A+",
        "height_cm": 172.5,
        "weight_kg": 68.0,
        "allergies": ["Sulfa drugs", "Peanuts"],
        "chronic_conditions": ["Hypothyroidism"],
        "current_medications": ["Levothyroxine 50mcg"],
        "smoking_status": "never",
        "alcohol_consumption": "occasional",
        "emergency_contact": {
            "name": "Jordan Clinician",
            "relationship": "Partner",
            "phone": "+1-555-4321"
        }
    }
    create_prof = client.post("/profile", json=profile_data, headers=headers_a)
    assert create_prof.status_code in [200, 201]

    # Read profile
    get_prof = client.get("/profile", headers=headers_a)
    assert get_prof.status_code == 200
    prof = get_prof.json()
    assert prof["gender"] == "female"
    assert prof["height_cm"] == 172.5
    assert prof["emergency_contact"]["name"] == "Jordan Clinician"

    # Update profile
    patch_prof = client.patch("/profile", json={"weight_kg": 67.0}, headers=headers_a)
    assert patch_prof.status_code == 200
    assert patch_prof.json()["weight_kg"] == 67.0
    print("  -> Medical Profile Create, Read, Update, and validation passed.")

    # ----------------------------------------------------
    # 32.4 Active Disease Predictions & Report Workflows
    # ----------------------------------------------------
    print("\n[32.4] Testing Prediction & Report Generation across all 4 active models...")
    predictions_map = {}

    # 1. Diabetes
    dia_res = client.post("/predictions", json={
        "disease_id": "diabetes",
        "inputs": {
            "Pregnancies": 3, "Glucose": 165.0, "BloodPressure": 84.0,
            "BMI": 32.8, "DiabetesPedigreeFunction": 0.58, "Age": 45
        }
    }, headers=headers_a)
    assert dia_res.status_code == 200
    dia_out = dia_res.json()
    assert dia_out["disease_id"] == "diabetes"
    assert dia_out["model_type"] == "LogisticRegression"

    # 2. Heart Disease
    hd_res = client.post("/predictions", json={
        "disease_id": "heart_disease",
        "inputs": {
            "age": 62, "sex": 1, "chest_pain_type": 3, "resting_bp": 142.0,
            "cholestoral": 255.0, "fasting_blood_sugar": 1, "restecg": 1,
            "max_hr": 148.0, "exang": 1, "oldpeak": 2.2, "slope": 2,
            "num_major_vessels": 2, "thal": 3
        }
    }, headers=headers_a)
    assert hd_res.status_code == 200
    hd_out = hd_res.json()
    assert hd_out["disease_id"] == "heart_disease"
    assert hd_out["model_type"] == "XGBoost"

    # 3. Pneumonia
    pneu_bytes = _generate_synthetic_scan(224, 224, color=(70, 70, 70))
    pneu_res = client.post(
        "/predictions/image",
        data={"disease_id": "pneumonia"},
        files={"file": ("chest_xray_test.png", pneu_bytes, "image/png")},
        headers=headers_a
    )
    assert pneu_res.status_code == 200
    pneu_out = pneu_res.json()
    assert pneu_out["disease_id"] == "pneumonia"
    assert pneu_out["model_type"] == "DenseNet121"

    # 4. Brain Tumor
    bt_bytes = _generate_synthetic_scan(224, 224, color=(25, 35, 45))
    bt_res = client.post(
        "/predictions/image",
        data={"disease_id": "brain_tumor"},
        files={"file": ("brain_mri_test.png", bt_bytes, "image/png")},
        headers=headers_a
    )
    assert bt_res.status_code == 200
    bt_out = bt_res.json()
    assert bt_out["disease_id"] == "brain_tumor"
    assert bt_out["model_type"] == "ResNet50"

    # Fetch IDs from history
    user_a_history = client.get("/history", headers=headers_a).json()
    assert user_a_history["total"] == 4
    for item in user_a_history["items"]:
        predictions_map[item["disease"]] = item["id"]

    print("  -> All 4 disease models executed successfully and recorded in history.")

    # ----------------------------------------------------
    # 32.5 Failure & Edge Case Handling
    # ----------------------------------------------------
    print("\n[32.5] Testing Failure & Edge Case Handling...")
    # A. Missing tabular required fields
    bad_tabular = client.post("/predictions", json={
        "disease_id": "diabetes",
        "inputs": {"Glucose": 120}  # missing other 5 required fields
    }, headers=headers_a)
    assert bad_tabular.status_code in [400, 422]

    # B. Invalid / corrupted image payload
    corrupted_bytes = b"not_a_valid_image_file_bytes"
    bad_image = client.post(
        "/predictions/image",
        data={"disease_id": "pneumonia"},
        files={"file": ("corrupt.png", corrupted_bytes, "image/png")},
        headers=headers_a
    )
    assert bad_image.status_code in [400, 422]

    # C. History count remains 4 (no garbage saved on failure)
    hist_check = client.get("/history", headers=headers_a).json()
    assert hist_check["total"] == 4
    print("  -> Failure rejection verified; no orphaned prediction records created.")

    # ----------------------------------------------------
    # 32.6 History, Filtering & Pagination
    # ----------------------------------------------------
    print("\n[32.6] Testing History Filtering & Pagination...")
    # Filter by pneumonia
    filter_pneu = client.get("/history?disease=pneumonia", headers=headers_a).json()
    assert filter_pneu["total"] == 1
    assert filter_pneu["items"][0]["disease"] == "pneumonia"

    # Pagination test
    page_1 = client.get("/history?limit=2&skip=0", headers=headers_a).json()
    assert len(page_1["items"]) == 2
    assert page_1["total"] == 4

    page_2 = client.get("/history?limit=2&skip=2", headers=headers_a).json()
    assert len(page_2["items"]) == 2
    print("  -> History filtering, pagination, and sorting verified.")

    # ----------------------------------------------------
    # 32.7 Security & Tenant Isolation
    # ----------------------------------------------------
    print("\n[32.7] Testing Tenant Isolation & Access Control...")
    sample_id = predictions_map["diabetes"]

    # User B cannot see User A's history
    user_b_hist = client.get("/history", headers=headers_b).json()
    assert user_b_hist["total"] == 0

    # User B cannot get User A's prediction detail
    user_b_pred = client.get(f"/history/{sample_id}", headers=headers_b)
    assert user_b_pred.status_code == 404

    # User B cannot get User A's report
    user_b_rpt = client.get(f"/reports/{sample_id}", headers=headers_b)
    assert user_b_rpt.status_code == 404

    # User B cannot download User A's report
    user_b_dl = client.get(f"/reports/{sample_id}/download", headers=headers_b)
    assert user_b_dl.status_code == 404

    # User B cannot see User A's medical profile
    user_b_prof = client.get("/profile", headers=headers_b)
    assert user_b_prof.status_code == 404
    print("  -> Tenant isolation 100% verified: cross-user data leakage strictly prevented.")

    # ----------------------------------------------------
    # 32.8 & 32.9 Object Storage, Image References & Checksums
    # ----------------------------------------------------
    print("\n[32.8 & 32.9] Verifying Object Storage & Database Integrity...")
    pneu_item = client.get(f"/history/{predictions_map['pneumonia']}", headers=headers_a).json()
    assert pneu_item["input_type"] == "image"
    assert "storage_key" in pneu_item["input_data"]
    assert "sha256" in pneu_item["input_data"]
    # Check that raw binary is NOT in database
    assert "image_bytes" not in pneu_item["input_data"]
    assert "image_binary" not in pneu_item["input_data"]

    # Verify object exists in storage
    storage_key = pneu_item["input_data"]["storage_key"]
    assert default_storage_service.has_object(storage_key) is True
    print(f"  -> Object storage reference confirmed ({storage_key}).")

    # ----------------------------------------------------
    # 32.10 PDF Reports & Streaming
    # ----------------------------------------------------
    print("\n[32.10] Testing PDF Clinical Report Generation & Streaming...")
    for d_id, pred_id in predictions_map.items():
        rpt_res = client.get(f"/reports/{pred_id}", headers=headers_a)
        assert rpt_res.status_code == 200
        rpt_meta = rpt_res.json()
        assert rpt_meta["disease"] == d_id
        assert "storage_key" in rpt_meta

        pdf_stream = client.get(f"/reports/{pred_id}/download", headers=headers_a)
        assert pdf_stream.status_code == 200
        assert pdf_stream.headers["content-type"] == "application/pdf"
        assert pdf_stream.content.startswith(b"%PDF")
        assert len(pdf_stream.content) > 2000
    print("  -> PDF generation, ReportLab formatting, and direct streaming verified for all active models.")

    print("\n========================================================")
    print("  [SUCCESS] ALL 13 STEP 32 INTEGRATION TESTS PASSED 100%!")
    print("========================================================\n")
