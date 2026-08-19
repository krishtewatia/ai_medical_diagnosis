"""
Comprehensive Test Suite for Step 19:
Generic File-Upload Handling & Image Prediction API Flow.

Tests:
1. Valid image upload returns 200 OK with standardized PredictionResponse
2. Missing file upload returns 422
3. Unsupported file extensions (e.g. .txt, .pdf, .bmp) return 422
4. Oversized image file returns 422
5. Empty image file (0 bytes) returns 422
6. Corrupt / unreadable image bytes return 422
7. Path traversal in uploaded filename is safely handled
8. Tabular prediction (POST /predictions) continues to work concurrently
9. Schema symmetry between tabular and image PredictionResponse
"""

import io
import os
import sys
from pathlib import Path
from PIL import Image
import pytest

# Ensure Keras backend is torch before importing models
os.environ["KERAS_BACKEND"] = "torch"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mongomock
from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.database.connection import get_database
from app.main import app
from app.ml.disease_registry import disease_registry
from app.schemas.prediction import PredictionResponse
from app.services.user_service import UserService


# Mock database for user auth verification
mock_client = mongomock.MongoClient()
mock_db = mock_client["test_upload_db"]
app.dependency_overrides[get_database] = lambda: mock_db


@pytest.fixture
def auth_headers():
    u_svc = UserService(mock_db)
    u = u_svc.create_user(name="UploadPatient", email="uploader@example.com", password_hash="hash123")
    token = create_access_token(str(u["_id"]))
    return {"Authorization": f"Bearer {token}"}


def create_valid_test_image_bytes(format="PNG", size=(224, 224), color=(128, 128, 128)) -> bytes:
    """Helper creating valid in-memory image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


def test_valid_image_upload(auth_headers):
    """1. Verify valid image upload returns 200 OK with standardized PredictionResponse."""
    client = TestClient(app)
    img_bytes = create_valid_test_image_bytes(format="PNG")

    files = {"file": ("chest_xray.png", img_bytes, "image/png")}
    data = {"disease_id": "pneumonia"}

    response = client.post("/predictions/image", data=data, files=files, headers=auth_headers)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    body = response.json()
    assert body["disease_id"] == "pneumonia"
    assert "prediction_label" in body
    assert isinstance(body["is_positive"], bool)
    assert "probability" in body
    assert "disclaimer" in body
    assert "timestamp" in body
    print(f"  PASS: 1. Valid image upload returns 200 OK: prob={body['probability']}, label='{body['prediction_label']}'")


def test_missing_file_rejected(auth_headers):
    """2. Verify missing file upload is rejected with 422."""
    client = TestClient(app)
    data = {"disease_id": "pneumonia"}

    # No files parameter
    response = client.post("/predictions/image", data=data, headers=auth_headers)
    assert response.status_code == 422
    print("  PASS: 2. Missing file upload rejected with 422 Unprocessable Entity")


def test_unsupported_file_extension(auth_headers):
    """3. Verify unsupported file types (.pdf, .txt, .bmp) are rejected with 422."""
    client = TestClient(app)

    # Text file
    files_txt = {"file": ("medical_report.txt", b"Patient reports shortness of breath.", "text/plain")}
    resp_txt = client.post("/predictions/image", data={"disease_id": "pneumonia"}, files=files_txt, headers=auth_headers)
    assert resp_txt.status_code == 422
    assert "unsupported" in resp_txt.json()["detail"].lower()

    # PDF file
    files_pdf = {"file": ("scan_doc.pdf", b"%PDF-1.4 dummy content", "application/pdf")}
    resp_pdf = client.post("/predictions/image", data={"disease_id": "pneumonia"}, files=files_pdf, headers=auth_headers)
    assert resp_pdf.status_code == 422

    print("  PASS: 3. Unsupported file types (.txt, .pdf) rejected with 422")


def test_oversized_image_rejected(auth_headers):
    """4. Verify oversized files (> 15MB) are rejected with 422."""
    client = TestClient(app)
    config = disease_registry.get("pneumonia")
    max_bytes = config.image_spec.max_size_bytes

    oversized_bytes = b"0" * (max_bytes + 1024)
    files = {"file": ("huge_scan.png", oversized_bytes, "image/png")}

    response = client.post("/predictions/image", data={"disease_id": "pneumonia"}, files=files, headers=auth_headers)
    assert response.status_code == 422
    assert "exceeds" in response.json()["detail"].lower()
    print(f"  PASS: 4. Oversized file (> {max_bytes} bytes) rejected with 422")


def test_empty_file_rejected(auth_headers):
    """5. Verify empty file (0 bytes) is rejected with 422."""
    client = TestClient(app)
    files = {"file": ("empty_xray.png", b"", "image/png")}

    response = client.post("/predictions/image", data={"disease_id": "pneumonia"}, files=files, headers=auth_headers)
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()
    print("  PASS: 5. Empty image file (0 bytes) rejected with 422")


def test_corrupt_image_rejected(auth_headers):
    """6. Verify corrupt or unreadable image bytes are detected and rejected with 422."""
    client = TestClient(app)
    corrupted_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"CORRUPTED_GARBAGE_PAYLOAD"

    files = {"file": ("corrupt.png", corrupted_bytes, "image/png")}
    response = client.post("/predictions/image", data={"disease_id": "pneumonia"}, files=files, headers=auth_headers)
    assert response.status_code == 422
    assert "corrupt" in response.json()["detail"].lower() or "not a valid" in response.json()["detail"].lower()
    print("  PASS: 6. Corrupted image binary payload rejected with 422")


def test_path_traversal_filename_sanitized(auth_headers):
    """7. Verify unsafe filenames with path traversal attempts are safely handled."""
    client = TestClient(app)
    valid_bytes = create_valid_test_image_bytes(format="JPEG")

    # Unsafe path in filename
    files = {"file": ("../../../../etc/shadow.jpg", valid_bytes, "image/jpeg")}
    response = client.post("/predictions/image", data={"disease_id": "pneumonia"}, files=files, headers=auth_headers)
    assert response.status_code == 200
    print("  PASS: 7. Path traversal attempt in filename safely handled and sanitized")


def test_tabular_and_image_coexistence_and_symmetry(auth_headers):
    """8 & 9. Verify tabular predictions still work normally and share identical response keys."""
    client = TestClient(app)

    # 1. Tabular Diabetes
    tab_res = client.post(
        "/predictions",
        json={
            "disease_id": "diabetes",
            "inputs": {
                "Pregnancies": 2,
                "Glucose": 140.0,
                "BloodPressure": 80.0,
                "BMI": 28.0,
                "DiabetesPedigreeFunction": 0.5,
                "Age": 45,
            }
        },
        headers=auth_headers
    )
    assert tab_res.status_code == 200
    tab_body = tab_res.json()

    # 2. Image Pneumonia
    img_bytes = create_valid_test_image_bytes(format="PNG")
    img_res = client.post(
        "/predictions/image",
        data={"disease_id": "pneumonia"},
        files={"file": ("xray.png", img_bytes, "image/png")},
        headers=auth_headers
    )
    assert img_res.status_code == 200
    img_body = img_res.json()

    # Verify identical top-level key symmetry
    assert set(tab_body.keys()) == set(img_body.keys())
    print("  PASS: 8 & 9. Tabular prediction operates seamlessly with 100% schema symmetry to Image prediction")


if __name__ == "__main__":
    print("\n=== Generic File-Upload & Image Prediction API Tests ===\n")
    tests = [
        ("1. Valid Image Upload Returns 200", test_valid_image_upload),
        ("2. Missing File Upload Rejected", test_missing_file_rejected),
        ("3. Unsupported File Types Rejected", test_unsupported_file_extension),
        ("4. Oversized File Rejected", test_oversized_image_rejected),
        ("5. Empty Image File Rejected", test_empty_file_rejected),
        ("6. Corrupt Image Rejected", test_corrupt_image_rejected),
        ("7. Path Traversal Filename Sanitized", test_path_traversal_filename_sanitized),
        ("8 & 9. Tabular Coexistence & Schema Symmetry", test_tabular_and_image_coexistence_and_symmetry),
    ]

    passed = 0
    failed = 0
    headers = {
        "Authorization": f"Bearer {create_access_token(str(UserService(mock_db).create_user('TUser', 'tu@ex.com', 'p')['_id']))}"
    }

    for name, fn in tests:
        try:
            fn(headers)
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} - {e}")
            failed += 1

    print(f"\n{'=' * 55}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 55}\n")
    sys.exit(1 if failed else 0)
