import io
import uuid
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app
from app.core.security import create_access_token
from app.ml.loaders import load_model_artifact
from app.ml.disease_registry import disease_registry
from app.ml.image_models.pneumonia.predictor import PneumoniaPredictor


def _generate_synthetic_xray_bytes(width: int = 300, height: int = 300) -> bytes:
    """Creates synthetic grayscale chest radiograph image bytes for testing."""
    img = Image.new("L", (width, height), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_26_1_verify_pneumonia_artifact():
    """Verify pneumonia artifact and loader."""
    config = disease_registry.get_or_raise("pneumonia")
    assert config.id == "pneumonia"
    assert config.artifact_filename == "pneumonia_densenet121.keras"
    
    # Load model directly
    model = load_model_artifact(config.model_dir + "/" + config.artifact_filename)
    assert model is not None
    print("\n[OK] 26.1 Pneumonia DenseNet121 model artifact loaded successfully.")


def test_26_2_and_26_4_pneumonia_disease_discovery():
    """Verify GET /diseases and GET /diseases/pneumonia."""
    client = TestClient(app)

    # 1. Discover all active diseases
    res = client.get("/diseases")
    assert res.status_code == 200
    diseases = res.json()
    disease_ids = [d["id"] for d in diseases]
    assert "pneumonia" in disease_ids
    assert "diabetes" in disease_ids
    assert "heart_disease" in disease_ids

    # 2. Get specific pneumonia metadata
    pneu_res = client.get("/diseases/pneumonia")
    assert pneu_res.status_code == 200
    pneu_data = pneu_res.json()
    assert pneu_data["id"] == "pneumonia"
    assert pneu_data["category"] == "image"
    assert pneu_data["input_type"] == "image_upload"
    assert "image_spec" in pneu_data
    assert pneu_data["image_spec"]["target_dimensions"] == [224, 224]
    assert pneu_data["positive_label"] == "Lung Opacity Detected"
    assert pneu_data["model_info"]["threshold"] == 0.30
    assert pneu_data["model_info"]["model_type"] == "DenseNet121"
    print("\n[OK] 26.2 & 26.4 Pneumonia discovery endpoints verified.")


def test_26_3_pneumonia_predictor_inference():
    """Verify PneumoniaPredictor preprocessing and inference directly."""
    config = disease_registry.get_or_raise("pneumonia")
    predictor = PneumoniaPredictor(config)
    
    img_bytes = _generate_synthetic_xray_bytes(250, 250)
    result = predictor.predict(img_bytes)

    assert result.disease_id == "pneumonia"
    assert result.model_type == "DenseNet121"
    assert result.prediction_label in ["Lung Opacity Detected", "No Lung Opacity Detected"]
    assert isinstance(result.is_positive, bool)
    assert result.probability is not None
    assert 0.0 <= result.probability <= 1.0
    assert result.decision_threshold == 0.30
    assert result.metadata["input_shape"] == [1, 224, 224, 3]
    print(f"\n[OK] 26.3 Predictor inference: {result.prediction_label} (p={result.probability})")


def test_26_6_and_26_7_api_prediction_and_history_integration():
    """Verify end-to-end API upload and history recording."""
    client = TestClient(app)

    # 1. Create authenticated user
    user_suffix = uuid.uuid4().hex[:8]
    email = f"dr_pneu_{user_suffix}@hospital.org"
    reg_res = client.post("/auth/register", json={
        "name": f"Dr. Radiologist {user_suffix}",
        "email": email,
        "password": "Password123!"
    })
    assert reg_res.status_code == 201
    user_id = reg_res.json()["id"]
    token = create_access_token(user_id)
    auth_headers = {"Authorization": f"Bearer {token}"}

    # 2. Upload chest X-ray image to POST /predictions/image
    img_bytes = _generate_synthetic_xray_bytes(300, 300)
    files = {
        "file": ("chest_xray_sample.png", img_bytes, "image/png")
    }
    data = {
        "disease_id": "pneumonia"
    }

    pred_res = client.post("/predictions/image", data=data, files=files, headers=auth_headers)
    assert pred_res.status_code == 200, f"API inference failed: {pred_res.text}"
    pred_data = pred_res.json()
    assert pred_data["disease_id"] == "pneumonia"
    assert pred_data["model_type"] == "DenseNet121"
    assert "prediction_label" in pred_data
    assert "probability" in pred_data
    assert "explanation" in pred_data
    assert "disclaimer" in pred_data
    print(f"\n[OK] 26.6 API Prediction response: {pred_data['prediction_label']} ({pred_data['probability']})")

    # 3. Verify history recording in GET /history
    hist_res = client.get("/history", headers=auth_headers)
    assert hist_res.status_code == 200
    hist = hist_res.json()
    assert hist["total"] == 1
    item = hist["items"][0]
    assert item["disease"] == "pneumonia"
    assert item["input_type"] == "image"
    assert item["input_data"]["file_name"] == "chest_xray_sample.png"
    assert item["input_data"]["content_type"] == "image/png"
    assert "file_size" in item["input_data"]
    assert "storage_key" in item["input_data"]
    # Verify raw binary is NOT stored in MongoDB
    assert "image_bytes" not in item["input_data"]
    print(f"\n[OK] 26.7 History persisted properly for record: {item['id']}")

    # 4. Verify history filtering for pneumonia
    filter_res = client.get("/history?disease=pneumonia", headers=auth_headers)
    assert filter_res.status_code == 200
    assert filter_res.json()["total"] == 1

    # 5. Verify history detail endpoint
    detail_res = client.get(f"/history/{item['id']}", headers=auth_headers)
    assert detail_res.status_code == 200
    assert detail_res.json()["id"] == item["id"]
    print("\n[OK] 26.7 History detail endpoint verified.")

    print("\n[SUCCESS] ALL STEP 26 PNEUMONIA INTEGRATION TESTS PASSED 100%!\n")
