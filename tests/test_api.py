import io
from typing import Dict

import pytest
from fastapi.testclient import TestClient

from app.main import app
from core.exceptions import DurianServiceException

client = TestClient(app, raise_server_exceptions=False)

VALID_API_KEY  = "test_super_secret_key"
ALL_CLASSES    = ["D101", "D13", "D197", "D198", "D2", "D200", "D24", "D88"]
TOP_CLASS      = "D101"
TOP_CONFIDENCE = 0.9231


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("API_KEY", VALID_API_KEY)


def _make_png_bytes(size: tuple = (100, 100)) -> bytes:
    from PIL import Image
    img = Image.new("RGB", size, color=(34, 85, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(size: tuple = (100, 100)) -> bytes:
    from PIL import Image
    img = Image.new("RGB", size, color=(34, 85, 20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_mock_response():
    from schemas.response import PredictionResponse, PredictionResult

    scores = {cls: 0.01 for cls in ALL_CLASSES}
    scores[TOP_CLASS] = TOP_CONFIDENCE
    total = sum(scores.values())
    scores = {k: round(v / total, 6) for k, v in scores.items()}

    return PredictionResponse(
        success=True,
        prediction=PredictionResult(
            class_name=TOP_CLASS,
            confidence_score=scores[TOP_CLASS],
        ),
        confidence_scores=scores,
        inference_time_ms=18.5,
        model_version="1.0.0",
    )


class TestHealthCheck:
    def test_health_returns_200(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_schema(self):
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "app_name" in data
        assert "version" in data
        assert isinstance(data["model_loaded"], bool)

    def test_health_status_value(self):
        data = client.get("/api/v1/health").json()
        assert data["status"] in ("healthy", "degraded")


class TestAuthentication:
    def test_missing_api_key(self):
        files = {"file": ("test.jpg", io.BytesIO(_make_jpeg_bytes()), "image/jpeg")}
        response = client.post("/api/v1/predict", files=files)
        assert response.status_code == 403
        assert "Missing" in response.json()["detail"]

    def test_invalid_api_key(self):
        files   = {"file": ("test.jpg", io.BytesIO(_make_jpeg_bytes()), "image/jpeg")}
        headers = {"X-API-Key": "wrong_key_12345"}
        response = client.post("/api/v1/predict", files=files, headers=headers)
        assert response.status_code == 403
        assert "Invalid" in response.json()["detail"]

    def test_valid_api_key_passes_auth(self, mocker):
        mocker.patch("app.api.routes.ImageProcessor.process", return_value="dummy")
        mocker.patch("app.api.routes.InferenceService.predict", return_value=_make_mock_response())
        files   = {"file": ("test.jpg", io.BytesIO(_make_jpeg_bytes()), "image/jpeg")}
        headers = {"X-API-Key": VALID_API_KEY}
        response = client.post("/api/v1/predict", files=files, headers=headers)
        assert response.status_code != 403


class TestInputValidation:
    def test_no_data_returns_400(self):
        headers  = {"X-API-Key": VALID_API_KEY}
        response = client.post("/api/v1/predict", headers=headers)
        assert response.status_code == 400

    def test_invalid_extension_returns_415(self):
        files   = {"file": ("document.txt", io.BytesIO(b"text content"), "text/plain")}
        headers = {"X-API-Key": VALID_API_KEY}
        response = client.post("/api/v1/predict", files=files, headers=headers)
        assert response.status_code == 415
        data = response.json()
        assert data.get("success") is False
        assert "UnsupportedFileTypeException" in (data.get("error") or data.get("detail", ""))

    def test_file_too_large_returns_413(self, mocker):
        mocker.patch.object(
            type(pytest.importorskip("core.config").settings),
            "max_file_size_bytes",
            new_callable=lambda: property(lambda self: 1)
        )
        files   = {"file": ("test.jpg", io.BytesIO(_make_jpeg_bytes()), "image/jpeg")}
        headers = {"X-API-Key": VALID_API_KEY}
        response = client.post("/api/v1/predict", files=files, headers=headers)
        assert response.status_code in (413, 422)

    def test_both_file_and_payload_returns_400(self):
        headers = {"X-API-Key": VALID_API_KEY}
        files   = {"file": ("test.jpg", io.BytesIO(_make_jpeg_bytes()), "image/jpeg")}
        response = client.post(
            "/api/v1/predict",
            headers=headers,
            files=files,
            data={"payload": '{"image_base64": "dGVzdA=="}'},
        )
        assert response.status_code in (400, 415, 422)


class TestPredictionSuccess:
    def test_predict_file_upload_success(self, mocker):
        mocker.patch("app.api.routes.ImageProcessor.process", return_value="dummy_tensor")
        mocker.patch("app.api.routes.InferenceService.predict", return_value=_make_mock_response())

        files   = {"file": ("durian.jpg", io.BytesIO(_make_jpeg_bytes()), "image/jpeg")}
        headers = {"X-API-Key": VALID_API_KEY}
        response = client.post("/api/v1/predict", files=files, headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "prediction" in data
        assert "confidence_scores" in data
        assert "inference_time_ms" in data

    def test_predict_returns_correct_class(self, mocker):
        mocker.patch("app.api.routes.ImageProcessor.process", return_value="dummy_tensor")
        mocker.patch("app.api.routes.InferenceService.predict", return_value=_make_mock_response())

        files   = {"file": ("durian.png", io.BytesIO(_make_png_bytes()), "image/png")}
        headers = {"X-API-Key": VALID_API_KEY}
        response = client.post("/api/v1/predict", files=files, headers=headers)

        data = response.json()
        assert data["prediction"]["class_name"] == TOP_CLASS
        assert data["prediction"]["class_name"] in ALL_CLASSES

    def test_predict_confidence_scores_all_8_classes(self, mocker):
        mocker.patch("app.api.routes.ImageProcessor.process", return_value="dummy_tensor")
        mocker.patch("app.api.routes.InferenceService.predict", return_value=_make_mock_response())

        files   = {"file": ("durian.jpg", io.BytesIO(_make_jpeg_bytes()), "image/jpeg")}
        headers = {"X-API-Key": VALID_API_KEY}
        response = client.post("/api/v1/predict", files=files, headers=headers)

        data   = response.json()
        scores = data["confidence_scores"]

        assert len(scores) == 8
        for cls in ALL_CLASSES:
            assert cls in scores, f"Kelas '{cls}' tidak ada di confidence_scores"
            assert 0.0 <= scores[cls] <= 1.0

    def test_predict_confidence_sum_near_one(self, mocker):
        mocker.patch("app.api.routes.ImageProcessor.process", return_value="dummy_tensor")
        mocker.patch("app.api.routes.InferenceService.predict", return_value=_make_mock_response())

        files   = {"file": ("durian.jpg", io.BytesIO(_make_jpeg_bytes()), "image/jpeg")}
        headers = {"X-API-Key": VALID_API_KEY}
        response = client.post("/api/v1/predict", files=files, headers=headers)

        scores     = response.json()["confidence_scores"]
        total_prob = sum(scores.values())
        assert abs(total_prob - 1.0) < 0.01, f"Sum probability = {total_prob:.4f}, harusnya ≈ 1.0"

    def test_predict_inference_time_positive(self, mocker):
        mocker.patch("app.api.routes.ImageProcessor.process", return_value="dummy_tensor")
        mocker.patch("app.api.routes.InferenceService.predict", return_value=_make_mock_response())

        files   = {"file": ("durian.jpg", io.BytesIO(_make_jpeg_bytes()), "image/jpeg")}
        headers = {"X-API-Key": VALID_API_KEY}
        response = client.post("/api/v1/predict", files=files, headers=headers)

        assert response.json()["inference_time_ms"] >= 0.0

    def test_predict_png_format(self, mocker):
        mocker.patch("app.api.routes.ImageProcessor.process", return_value="dummy_tensor")
        mocker.patch("app.api.routes.InferenceService.predict", return_value=_make_mock_response())

        files   = {"file": ("durian.png", io.BytesIO(_make_png_bytes()), "image/png")}
        headers = {"X-API-Key": VALID_API_KEY}
        response = client.post("/api/v1/predict", files=files, headers=headers)

        assert response.status_code == 200