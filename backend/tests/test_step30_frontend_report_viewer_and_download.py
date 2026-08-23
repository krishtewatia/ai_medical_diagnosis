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


def test_step30_e2e_report_viewer_and_download_flow():
    """Verify complete Step 30 Report Viewer & Download API contracts across all active diseases."""
    client = TestClient(app)

    # 1. Register User A
    user_a_suffix = uuid.uuid4().hex[:8]
    email_a = f"dr_step30_{user_a_suffix}@clinic.org"
    reg_a = client.post("/auth/register", json={
        "name": f"Dr. Step30 Clinician {user_a_suffix}",
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
        "name": f"Dr. Intruder {user_b_suffix}",
        "email": f"intruder_{user_b_suffix}@clinic.org",
        "password": "Password123!"
    })
    assert reg_b.status_code == 201
    user_b_id = reg_b.json()["id"]
    token_b = create_access_token(user_b_id)
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # ----------------------------------------------------
    # 3. Test Active Diseases: Diabetes, Heart Disease, Pneumonia, Brain Tumor
    # ----------------------------------------------------
    active_predictions = {}

    # A. Diabetes
    dia_res = client.post("/predictions", json={
        "disease_id": "diabetes",
        "inputs": {
            "Pregnancies": 1, "Glucose": 140.0, "BloodPressure": 75.0,
            "BMI": 28.0, "DiabetesPedigreeFunction": 0.45, "Age": 42
        }
    }, headers=headers_a)
    assert dia_res.status_code == 200
    hist_dia = client.get("/history?disease=diabetes", headers=headers_a).json()
    active_predictions["diabetes"] = hist_dia["items"][0]["id"]

    # B. Heart Disease
    hd_res = client.post("/predictions", json={
        "disease_id": "heart_disease",
        "inputs": {
            "age": 60, "sex": 1, "chest_pain_type": 2, "resting_bp": 145.0,
            "cholestoral": 260.0, "fasting_blood_sugar": 1, "restecg": 1,
            "max_hr": 150.0, "exang": 1, "oldpeak": 2.0, "slope": 2,
            "num_major_vessels": 1, "thal": 3
        }
    }, headers=headers_a)
    assert hd_res.status_code == 200
    hist_hd = client.get("/history?disease=heart_disease", headers=headers_a).json()
    active_predictions["heart_disease"] = hist_hd["items"][0]["id"]

    # C. Pneumonia
    pneu_bytes = _generate_synthetic_image(224, 224)
    pneu_res = client.post("/predictions/image", data={"disease_id": "pneumonia"}, files={"file": ("chest.png", pneu_bytes, "image/png")}, headers=headers_a)
    assert pneu_res.status_code == 200
    hist_pneu = client.get("/history?disease=pneumonia", headers=headers_a).json()
    active_predictions["pneumonia"] = hist_pneu["items"][0]["id"]

    # D. Brain Tumor
    bt_bytes = _generate_synthetic_image(224, 224)
    bt_res = client.post("/predictions/image", data={"disease_id": "brain_tumor"}, files={"file": ("mri.png", bt_bytes, "image/png")}, headers=headers_a)
    assert bt_res.status_code == 200
    hist_bt = client.get("/history?disease=brain_tumor", headers=headers_a).json()
    active_predictions["brain_tumor"] = hist_bt["items"][0]["id"]

    print("\n--- Testing Report API and Download for All Active Diseases ---")
    for disease_id, pred_id in active_predictions.items():
        # 1. GET /reports/{prediction_id}
        rpt_res = client.get(f"/reports/{pred_id}", headers=headers_a)
        assert rpt_res.status_code == 200, f"Failed for {disease_id}: {rpt_res.text}"
        data = rpt_res.json()
        assert data["prediction_id"] == pred_id
        assert data["disease"] == disease_id
        assert "storage_key" in data
        assert "download_url" in data
        assert "prediction" in data
        assert "model" in data
        print(f"[OK] Report metadata retrieved for {disease_id}: {data['report_id']}")

        # 2. GET /reports/{prediction_id}/download
        dl_res = client.get(f"/reports/{pred_id}/download", headers=headers_a)
        assert dl_res.status_code == 200
        assert dl_res.headers["content-type"] == "application/pdf"
        assert dl_res.content[:4] == b"%PDF"
        assert len(dl_res.content) > 1000
        print(f"[OK] PDF binary stream validated for {disease_id} ({len(dl_res.content)} bytes)")

    # ----------------------------------------------------
    # 4. Strict Ownership & Tenant Isolation
    # ----------------------------------------------------
    print("\n--- Testing Ownership Enforcement & Security ---")
    sample_pred_id = active_predictions["diabetes"]

    # User B cannot access User A's report metadata
    forbidden_meta = client.get(f"/reports/{sample_pred_id}", headers=headers_b)
    assert forbidden_meta.status_code == 404
    print("[OK] User B metadata access blocked.")

    # User B cannot download User A's report PDF
    forbidden_dl = client.get(f"/reports/{sample_pred_id}/download", headers=headers_b)
    assert forbidden_dl.status_code == 404
    print("[OK] User B PDF download blocked.")

    # Unauthenticated request blocked
    unauth_req = client.get(f"/reports/{sample_pred_id}")
    assert unauth_req.status_code in [401, 403]
    print("[OK] Unauthenticated request blocked with 401.")

    print("\n[SUCCESS] ALL STEP 30 REPORT VIEWER & DOWNLOAD TESTS PASSED 100%!\n")
