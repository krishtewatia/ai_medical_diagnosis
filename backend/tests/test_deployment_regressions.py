import asyncio
import threading
import sys
import zipfile
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from app.api import database, storage
from app.api.dependencies import get_current_user
from app.api.prediction import get_prediction_service
from app.main import app
from app.services.report_service import ReportService
from app.services.storage_service import StorageService
from app.ml import loaders
from app.ml.disease_registry import DiseaseRegistry, disease_registry


def test_report_signed_url_serves_uploaded_pdf(tmp_path, monkeypatch):
    user_id = "a" * 24
    history = Mock()
    history.get_prediction_by_id.return_value = {
        "id": "b" * 24, "user_id": user_id,
        "disease": "diabetes", "disease_display_name": "Diabetes",
        "input_type": "tabular", "input_data": {"Age": 35},
        "model": {"version": "v1", "model_type": "LogisticRegression", "threshold": 0.4},
        "result": {"prediction": "Low Risk", "is_positive": False, "probability": 0.1},
        "created_at": datetime.now(timezone.utc),
    }
    store = StorageService(driver="local", local_dir=str(tmp_path))
    service = ReportService(history_service=history, storage_service=store)
    report = service.get_or_generate_report("b" * 24, user_id)
    assert report.storage_key.endswith(".pdf")
    assert store.get_object(report.storage_key).startswith(b"%PDF")
    monkeypatch.setattr(storage, "default_storage_service", store)
    with TestClient(app) as client:
        response = client.get(report.download_url)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content == store.get_object(report.storage_key)
        assert client.get(report.download_url.replace("signature=", "signature=invalid")).status_code == 403


def test_database_health_returns_failure_status(monkeypatch):
    with TestClient(app) as client:
        monkeypatch.setattr(database, "check_database_connection", lambda: False)
        response = client.get("/database/health")
        assert response.status_code == 503
        assert response.json()["database"] == "disconnected"
        monkeypatch.setattr(database, "check_database_connection", lambda: True)
        assert client.get("/database/health").status_code == 200


def test_image_inference_keeps_event_loop_responsive():
    started = threading.Event()
    release = threading.Event()

    def slow_inference(**kwargs):
        started.set()
        assert release.wait(5)
        raise ValueError("synthetic inference finished")

    previous = app.dependency_overrides.copy()
    app.dependency_overrides[get_current_user] = lambda: {"_id": "a" * 24}
    app.dependency_overrides[get_prediction_service] = lambda: SimpleNamespace(predict_image=slow_inference)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            upload = asyncio.create_task(client.post(
                "/predictions/image", data={"disease_id": "pneumonia"},
                files={"file": ("synthetic.png", b"synthetic", "image/png")},
            ))
            try:
                assert await asyncio.to_thread(started.wait, 2)
                health = await asyncio.wait_for(client.get("/health"), timeout=1)
                assert health.status_code == 200
            finally:
                release.set()
                await upload

    try:
        asyncio.run(exercise())
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)


def test_keras_archive_loads_weights_from_disk(tmp_path, monkeypatch):
    artifact = tmp_path / "sample.keras"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("config.json", "{}")
        archive.writestr("model.weights.h5", b"sample-weights")
    model = object()
    extracted = []

    def load(directory, compile):
        assert compile is False
        folder = Path(directory)
        assert folder.is_dir()
        assert (folder / "model.weights.h5").read_bytes() == b"sample-weights"
        extracted.append(folder)
        return model

    monkeypatch.setitem(sys.modules, "keras", SimpleNamespace(models=SimpleNamespace(load_model=load)))
    assert loaders._load_keras_artifact(artifact) is model
    assert not extracted[0].exists()


def test_switching_image_models_releases_previous_model(monkeypatch):
    registry = DiseaseRegistry(auto_load=False)
    for disease in ["pneumonia", "brain_tumor"]:
        registry.register(disease_registry.get_or_raise(disease))
    old_model, new_model = object(), object()
    registry._predictor_instances = {"pneumonia": old_model, "brain_tumor": new_model}
    config = registry.get_or_raise("pneumonia")
    key = str((Path(config.model_dir) / config.artifact_filename).resolve())
    monkeypatch.setitem(loaders._MODEL_CACHE, key, (0, old_model))
    assert registry.get_predictor("brain_tumor") is new_model
    assert "pneumonia" not in registry._predictor_instances
    assert key not in loaders._MODEL_CACHE
