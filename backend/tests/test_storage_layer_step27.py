import io
import time
import uuid
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.core.security import create_access_token
from app.services.storage_service import (
    StorageNotFoundError,
    StorageSecurityError,
    StorageService,
    default_storage_service,
)
from app.services.prediction_service import (
    PredictionInferenceError,
    PredictionService,
)
from app.ml.disease_registry import disease_registry


def _generate_synthetic_xray_bytes(width: int = 250, height: int = 250) -> bytes:
    """Generates synthetic grayscale X-ray bytes."""
    img = Image.new("L", (width, height), color=130)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_27_1_and_27_3_storage_service_crud_and_signed_urls(tmp_path):
    """Test StorageService upload, hash calculation, signed URL generation, and deletion."""
    test_storage = StorageService(driver="local", local_dir=str(tmp_path))
    
    img_bytes = _generate_synthetic_xray_bytes()
    user_id = uuid.uuid4().hex
    
    # 1. Upload
    meta = test_storage.upload_file(
        image_bytes=img_bytes,
        filename="test_patient_xray.png",
        content_type="image/png",
        disease_id="pneumonia",
        user_id=user_id
    )
    assert meta.file_name == "test_patient_xray.png"
    assert meta.content_type == "image/png"
    assert meta.file_size == len(img_bytes)
    assert user_id in meta.storage_key
    assert test_storage.has_object(meta.storage_key) is True
    print("\n[OK] 27.1 Uploaded image to object storage with SHA256:", meta.sha256)

    # 2. Retrieve bytes
    fetched_bytes = test_storage.get_object(meta.storage_key)
    assert fetched_bytes == img_bytes
    print("[OK] Retrieved exact matching image bytes from storage.")

    # 3. Signed URL generation & verification
    signed_url = test_storage.get_signed_url(meta.storage_key, expires_in=60)
    assert "signature=" in signed_url
    assert "expires=" in signed_url

    # Parse query params
    query_str = signed_url.split("?")[1]
    params = dict(q.split("=") for q in query_str.split("&"))
    expires = int(params["expires"])
    sig = params["signature"]

    # Verify valid signature
    assert test_storage.verify_signed_url(meta.storage_key, expires=expires, signature=sig) is True
    print("[OK] Verified HMAC signed media access URL.")

    # Test expired signature
    with pytest.raises(StorageSecurityError):
        test_storage.verify_signed_url(meta.storage_key, expires=int(time.time()) - 10, signature=sig)

    # Test tampered signature
    with pytest.raises(StorageSecurityError):
        test_storage.verify_signed_url(meta.storage_key, expires=expires, signature="tampered_signature")

    # 4. Delete
    deleted = test_storage.delete_object(meta.storage_key)
    assert deleted is True
    assert test_storage.has_object(meta.storage_key) is False
    print("[OK] Deleted storage object cleanly.")


def test_27_4_pneumonia_upload_with_history_metadata():
    """Verify Pneumonia prediction saves storage metadata reference to MongoDB and NOT raw binary."""
    client = TestClient(app)

    # Create test user
    user_suffix = uuid.uuid4().hex[:8]
    email = f"dr_storage_{user_suffix}@clinic.org"
    reg_res = client.post("/auth/register", json={
        "name": f"Dr. Imaging {user_suffix}",
        "email": email,
        "password": "Password123!"
    })
    assert reg_res.status_code == 201
    user_id = reg_res.json()["id"]
    token = create_access_token(user_id)
    auth_headers = {"Authorization": f"Bearer {token}"}

    # Upload X-ray
    img_bytes = _generate_synthetic_xray_bytes(280, 280)
    files = {"file": ("chest_scan.png", img_bytes, "image/png")}
    data = {"disease_id": "pneumonia"}

    res = client.post("/predictions/image", data=data, files=files, headers=auth_headers)
    assert res.status_code == 200, f"Failed: {res.text}"
    pred = res.json()
    assert pred["disease_id"] == "pneumonia"
    print("\n[OK] 27.4 Prediction succeeded with image storage pipeline.")

    # Check MongoDB history record
    hist_res = client.get("/history", headers=auth_headers)
    assert hist_res.status_code == 200
    hist = hist_res.json()
    assert hist["total"] >= 1
    latest = hist["items"][0]
    
    # Verify input_data metadata structure
    input_data = latest["input_data"]
    assert input_data["file_name"] == "chest_scan.png"
    assert input_data["content_type"] == "image/png"
    assert input_data["file_size"] == len(img_bytes)
    assert "storage_key" in input_data
    assert "uploaded_at" in input_data
    assert "sha256" in input_data

    # Verify NO raw image binary in MongoDB
    assert "image_binary" not in input_data
    assert "image_bytes" not in input_data
    assert "data" not in input_data
    print(f"[OK] 27.2 Verified stored metadata reference in history: {input_data['storage_key']}")


def test_27_4_cleanup_on_prediction_failure(tmp_path, monkeypatch):
    """Verify that if model inference fails, the uploaded object is automatically deleted."""
    test_storage = StorageService(driver="local", local_dir=str(tmp_path))
    
    # Mock predictor to simulate inference failure
    class BrokenPredictor:
        def predict(self, *args, **kwargs):
            raise RuntimeError("Simulated GPU out of memory inference failure.")

    service = PredictionService(
        registry=disease_registry,
        storage_service=test_storage
    )
    monkeypatch.setattr(service, "_resolve_predictor", lambda config: BrokenPredictor())

    img_bytes = _generate_synthetic_xray_bytes()
    user_id = uuid.uuid4().hex

    with pytest.raises(PredictionInferenceError):
        service.predict_image(
            disease_id="pneumonia",
            image_bytes=img_bytes,
            filename="failed_scan.png",
            user_id=user_id
        )

    # Verify that NO orphaned file was left behind in storage
    files_remaining = list(tmp_path.glob("**/*.*"))
    assert len(files_remaining) == 0
    print("\n[OK] 27.4 Verified automatic cleanup of uploaded object on inference failure.")


def test_27_5_storage_security_user_isolation():
    """Verify user A cannot request signed access URL for user B's storage key."""
    client = TestClient(app)

    # User A
    user_a_suffix = uuid.uuid4().hex[:8]
    reg_a = client.post("/auth/register", json={
        "name": f"User A {user_a_suffix}",
        "email": f"usera_{user_a_suffix}@clinic.org",
        "password": "Password123!"
    })
    token_a = create_access_token(reg_a.json()["id"])
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # User B
    user_b_suffix = uuid.uuid4().hex[:8]
    reg_b = client.post("/auth/register", json={
        "name": f"User B {user_b_suffix}",
        "email": f"userb_{user_b_suffix}@clinic.org",
        "password": "Password123!"
    })
    user_b_id = reg_b.json()["id"]

    # Construct storage key belonging to User B
    user_b_storage_key = f"medical_images/pneumonia/{user_b_id}/scan_12345.png"

    # User A attempts to request signed URL for User B's image
    attack_res = client.get(
        f"/storage/signed-url?storage_key={user_b_storage_key}",
        headers=headers_a
    )
    assert attack_res.status_code == 403
    print("\n[OK] 27.5 Security isolation verified: cross-user signed URL request rejected with 403 Forbidden.")

    print("\n[SUCCESS] ALL STEP 27 STORAGE TESTS PASSED 100%!\n")
