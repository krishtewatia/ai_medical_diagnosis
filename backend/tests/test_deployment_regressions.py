import asyncio
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from fastapi.testclient import TestClient

from app.api import database, storage
from app.api.dependencies import get_current_user
from app.api.prediction import get_prediction_service
from app.main import app
from app.services.report_service import ReportService
from app.services.storage_service import StorageService


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
