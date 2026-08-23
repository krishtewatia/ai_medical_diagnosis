import io
import uuid
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.core.security import create_access_token


def _generate_synthetic_image(width: int = 224, height: int = 224, color=(40, 50, 60)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_step29_report_generation_across_all_active_diseases():
    """Verify PDF report generation across Diabetes, Heart Disease, Pneumonia, and Brain Tumor."""
    client = TestClient(app)

    # 1. Register User A
    user_a_suffix = uuid.uuid4().hex[:8]
    email_a = f"dr_reports_{user_a_suffix}@clinic.org"
    reg_a = client.post("/auth/register", json={
        "name": f"Dr. Lead Clinician {user_a_suffix}",
        "email": email_a,
        "password": "Password123!"
    })
    assert reg_a.status_code == 201
    user_a_id = reg_a.json()["id"]
    token_a = create_access_token(user_a_id)
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # 2. Register User B (for cross-user attack testing)
    user_b_suffix = uuid.uuid4().hex[:8]
    reg_b = client.post("/auth/register", json={
        "name": f"Dr. Other Clinician {user_b_suffix}",
        "email": f"dr_other_{user_b_suffix}@clinic.org",
        "password": "Password123!"
    })
    assert reg_b.status_code == 201
    user_b_id = reg_b.json()["id"]
    token_b = create_access_token(user_b_id)
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # ==========================================
    # A. Diabetes Report
    # ==========================================
    print("\n--- Testing Diabetes Report Generation ---")
    dia_pred_res = client.post("/predictions", json={
        "disease_id": "diabetes",
        "inputs": {
            "Pregnancies": 2,
            "Glucose": 160.0,
            "BloodPressure": 80.0,
            "BMI": 34.0,
            "DiabetesPedigreeFunction": 0.60,
            "Age": 48
        }
    }, headers=headers_a)
    assert dia_pred_res.status_code == 200

    # Fetch prediction ID from history
    hist_a = client.get("/history?disease=diabetes", headers=headers_a).json()
    dia_pred_id = hist_a["items"][0]["id"]

    # Generate Report
    dia_rpt_res = client.get(f"/reports/{dia_pred_id}", headers=headers_a)
    assert dia_rpt_res.status_code == 200, f"Diabetes report failed: {dia_rpt_res.text}"
    dia_rpt = dia_rpt_res.json()
    assert dia_rpt["prediction_id"] == dia_pred_id
    assert dia_rpt["disease"] == "diabetes"
    assert dia_rpt["model"]["model_type"] == "LogisticRegression"
    assert "storage_key" in dia_rpt
    assert "download_url" in dia_rpt
    assert "disclaimer" in dia_rpt
    print(f"[OK] Diabetes Report Generated: {dia_rpt['report_id']}")

    # Download raw PDF stream
    dia_pdf_res = client.get(f"/reports/{dia_pred_id}/download", headers=headers_a)
    assert dia_pdf_res.status_code == 200
    assert dia_pdf_res.headers["content-type"] == "application/pdf"
    assert len(dia_pdf_res.content) > 1000
    assert dia_pdf_res.content[:4] == b"%PDF"
    print(f"[OK] Downloaded Diabetes PDF ({len(dia_pdf_res.content)} bytes)")

    # ==========================================
    # B. Heart Disease Report
    # ==========================================
    print("\n--- Testing Heart Disease Report Generation ---")
    hd_pred_res = client.post("/predictions", json={
        "disease_id": "heart_disease",
        "inputs": {
            "age": 55, "sex": 1, "chest_pain_type": 1, "resting_bp": 130.0,
            "cholestoral": 240.0, "fasting_blood_sugar": 0, "restecg": 0,
            "max_hr": 160.0, "exang": 0, "oldpeak": 1.0, "slope": 1,
            "num_major_vessels": 0, "thal": 2
        }
    }, headers=headers_a)
    assert hd_pred_res.status_code == 200

    hist_hd = client.get("/history?disease=heart_disease", headers=headers_a).json()
    hd_pred_id = hist_hd["items"][0]["id"]

    hd_rpt_res = client.get(f"/reports/{hd_pred_id}", headers=headers_a)
    assert hd_rpt_res.status_code == 200
    assert hd_rpt_res.json()["disease"] == "heart_disease"
    assert hd_rpt_res.json()["model"]["model_type"] == "XGBoost"
    print(f"[OK] Heart Disease Report Generated: {hd_rpt_res.json()['report_id']}")

    # ==========================================
    # C. Pneumonia Report
    # ==========================================
    print("\n--- Testing Pneumonia Report Generation ---")
    pneu_bytes = _generate_synthetic_image(224, 224)
    pneu_res = client.post("/predictions/image", data={"disease_id": "pneumonia"}, files={"file": ("chest.png", pneu_bytes, "image/png")}, headers=headers_a)
    assert pneu_res.status_code == 200

    hist_pneu = client.get("/history?disease=pneumonia", headers=headers_a).json()
    pneu_pred_id = hist_pneu["items"][0]["id"]

    pneu_rpt_res = client.get(f"/reports/{pneu_pred_id}", headers=headers_a)
    assert pneu_rpt_res.status_code == 200
    assert pneu_rpt_res.json()["disease"] == "pneumonia"
    assert pneu_rpt_res.json()["model"]["model_type"] == "DenseNet121"
    print(f"[OK] Pneumonia Report Generated: {pneu_rpt_res.json()['report_id']}")

    # ==========================================
    # D. Brain Tumor Report
    # ==========================================
    print("\n--- Testing Brain Tumor Report Generation ---")
    bt_bytes = _generate_synthetic_image(224, 224)
    bt_res = client.post("/predictions/image", data={"disease_id": "brain_tumor"}, files={"file": ("brain_mri.png", bt_bytes, "image/png")}, headers=headers_a)
    assert bt_res.status_code == 200

    hist_bt = client.get("/history?disease=brain_tumor", headers=headers_a).json()
    bt_pred_id = hist_bt["items"][0]["id"]

    bt_rpt_res = client.get(f"/reports/{bt_pred_id}", headers=headers_a)
    assert bt_rpt_res.status_code == 200
    assert bt_rpt_res.json()["disease"] == "brain_tumor"
    assert bt_rpt_res.json()["model"]["model_type"] == "ResNet50"
    print(f"[OK] Brain Tumor Report Generated: {bt_rpt_res.json()['report_id']}")

    # ==========================================
    # E. Security & Ownership Verification
    # ==========================================
    print("\n--- Testing Security & Ownership Enforcement ---")
    # User B attempts to access User A's report
    cross_user_res = client.get(f"/reports/{dia_pred_id}", headers=headers_b)
    assert cross_user_res.status_code == 404  # Not found for User B's scope
    print("[OK] User B cross-user report access properly blocked.")

    # User B attempts to download User A's PDF
    cross_user_dl = client.get(f"/reports/{dia_pred_id}/download", headers=headers_b)
    assert cross_user_dl.status_code == 404
    print("[OK] User B cross-user PDF download properly blocked.")

    # Unauthenticated access
    unauth_res = client.get(f"/reports/{dia_pred_id}")
    assert unauth_res.status_code in [401, 403]
    print("[OK] Unauthenticated report request rejected with 401.")

    # Invalid prediction ID
    invalid_id_res = client.get("/reports/000000000000000000000000", headers=headers_a)
    assert invalid_id_res.status_code == 404
    print("[OK] Non-existent prediction ID handled with 404.")

    print("\n[SUCCESS] ALL STEP 29 MEDICAL REPORT GENERATION TESTS PASSED 100%!\n")
