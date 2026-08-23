import io
import uuid
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.core.security import create_access_token
from app.ml.loaders import load_model_artifact
from app.ml.disease_registry import disease_registry
from app.ml.image_models.brain_tumor.predictor import BrainTumorPredictor


def _generate_synthetic_mri_bytes(width: int = 256, height: int = 256) -> bytes:
    """Generates synthetic RGB brain MRI scan bytes for testing."""
    img = Image.new("RGB", (width, height), color=(30, 40, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_28_1_verify_brain_tumor_artifact():
    """28.1 Verify Brain Tumor model artifact and loader."""
    config = disease_registry.get_or_raise("brain_tumor")
    assert config.id == "brain_tumor"
    assert config.artifact_filename == "mri_model.keras"
    assert config.model_type == "ResNet50"

    model = load_model_artifact(config.model_dir + "/" + config.artifact_filename)
    assert model is not None
    print("\n[OK] 28.1 Brain Tumor ResNet50 model artifact loaded successfully.")


def test_28_2_and_28_4_brain_tumor_discovery():
    """28.2 & 28.4 Verify GET /diseases and GET /diseases/brain_tumor."""
    client = TestClient(app)

    # 1. Discover all active diseases
    res = client.get("/diseases")
    assert res.status_code == 200
    diseases = res.json()
    disease_ids = [d["id"] for d in diseases]
    assert "brain_tumor" in disease_ids
    assert "pneumonia" in disease_ids
    assert "diabetes" in disease_ids
    assert "heart_disease" in disease_ids

    # 2. Get specific brain_tumor metadata
    bt_res = client.get("/diseases/brain_tumor")
    assert bt_res.status_code == 200
    bt_data = bt_res.json()
    assert bt_data["id"] == "brain_tumor"
    assert bt_data["category"] == "image"
    assert bt_data["input_type"] == "image_upload"
    assert "image_spec" in bt_data
    assert bt_data["image_spec"]["target_dimensions"] == [224, 224]
    assert bt_data["model_info"]["model_type"] == "ResNet50"
    assert bt_data["model_info"]["threshold"] == 0.50
    print("\n[OK] 28.2 & 28.4 Brain Tumor discovery endpoints verified.")


def test_28_3_and_28_5_brain_tumor_predictor_inference():
    """28.3 & 28.5 Verify BrainTumorPredictor preprocessing and inference."""
    config = disease_registry.get_or_raise("brain_tumor")
    predictor = BrainTumorPredictor(config)

    img_bytes = _generate_synthetic_mri_bytes(300, 300)
    result = predictor.predict(img_bytes)

    assert result.disease_id == "brain_tumor"
    assert result.model_type == "ResNet50"
    assert isinstance(result.is_positive, bool)
    assert result.probability is not None
    assert 0.0 <= result.probability <= 1.0
    assert result.decision_threshold == 0.50
    assert result.metadata["input_shape"] == [1, 224, 224, 3]
    assert "detected_subclass" in result.metadata
    assert "class_probabilities" in result.metadata
    print(f"\n[OK] 28.3 & 28.5 Brain Tumor inference: {result.prediction_label} (p={result.probability}, subclass={result.metadata['detected_subclass']})")


def test_28_6_and_28_7_api_prediction_and_history():
    """28.6 & 28.7 Verify end-to-end API upload and history recording."""
    client = TestClient(app)

    # 1. Create authenticated user
    user_suffix = uuid.uuid4().hex[:8]
    email = f"dr_neuro_{user_suffix}@hospital.org"
    reg_res = client.post("/auth/register", json={
        "name": f"Dr. Neurologist {user_suffix}",
        "email": email,
        "password": "Password123!"
    })
    assert reg_res.status_code == 201
    user_id = reg_res.json()["id"]
    token = create_access_token(user_id)
    auth_headers = {"Authorization": f"Bearer {token}"}

    # 2. Upload Brain MRI image to POST /predictions/image
    img_bytes = _generate_synthetic_mri_bytes(256, 256)
    files = {
        "file": ("brain_mri_scan.png", img_bytes, "image/png")
    }
    data = {
        "disease_id": "brain_tumor"
    }

    pred_res = client.post("/predictions/image", data=data, files=files, headers=auth_headers)
    assert pred_res.status_code == 200, f"API inference failed: {pred_res.text}"
    pred_data = pred_res.json()
    assert pred_data["disease_id"] == "brain_tumor"
    assert pred_data["model_type"] == "ResNet50"
    assert "prediction_label" in pred_data
    assert "probability" in pred_data
    assert "explanation" in pred_data
    assert "disclaimer" in pred_data
    print(f"\n[OK] 28.6 API Prediction response: {pred_data['prediction_label']} ({pred_data['probability']})")

    # 3. Verify history recording in GET /history
    hist_res = client.get("/history", headers=auth_headers)
    assert hist_res.status_code == 200
    hist = hist_res.json()
    assert hist["total"] == 1
    item = hist["items"][0]
    assert item["disease"] == "brain_tumor"
    assert item["input_type"] == "image"
    assert item["input_data"]["file_name"] == "brain_mri_scan.png"
    assert item["input_data"]["content_type"] == "image/png"
    assert "storage_key" in item["input_data"]
    assert "sha256" in item["input_data"]
    # Verify raw binary is NOT stored in MongoDB
    assert "image_binary" not in item["input_data"]
    assert "image_bytes" not in item["input_data"]
    print(f"\n[OK] 28.7 History persisted with storage reference: {item['input_data']['storage_key']}")

    # 4. Verify history filtering for brain_tumor
    filter_res = client.get("/history?disease=brain_tumor", headers=auth_headers)
    assert filter_res.status_code == 200
    assert filter_res.json()["total"] == 1

    # 5. Verify history detail endpoint
    detail_res = client.get(f"/history/{item['id']}", headers=auth_headers)
    assert detail_res.status_code == 200
    assert detail_res.json()["id"] == item["id"]
    print("\n[OK] 28.7 History detail endpoint verified.")

    print("\n[SUCCESS] ALL STEP 28 BRAIN TUMOR INTEGRATION TESTS PASSED 100%!\n")
