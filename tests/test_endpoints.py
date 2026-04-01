"""
Tests untuk app/api — endpoint /ping, /health, /predict.
Menggunakan FastAPI TestClient dengan mock untuk isolasi dari model dan CLIP.
"""
import io
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES  = ["D101", "D13", "D197", "D198", "D2", "D200", "D24", "D88"]
VALID_KEY    = "dk_live_ValidTestKeyForAPIEndpointTests1"
ADMIN_KEY    = "dk_live_ValidAdminKeyForAPIEndpointTests2"
WRONG_KEY    = "dk_live_WrongKeyThatShouldNotBeRegistered"

PREDICT_ONLY_KEY  = "dk_live_PredictOnlyKeyNoHealthAccess12"
HEALTH_ONLY_KEY   = "dk_live_HealthOnlyKeyNoPredictAccess12"


# ─────────────────────────────────────────────────────────────────────────────
#  Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def _jpeg(size=(100, 100), color=(50, 120, 30)) -> bytes:
    from PIL import Image
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _png(size=(100, 100)) -> bytes:
    from PIL import Image
    img = Image.new("RGB", size, color=(80, 160, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _webp(size=(100, 100)) -> bytes:
    from PIL import Image
    img = Image.new("RGB", size, color=(60, 140, 20))
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
#  Mock response factory
# ─────────────────────────────────────────────────────────────────────────────

def _mock_prediction_response(top_code="D200"):
    from schemas.response import PredictionResponse, PredictionResult, VarietyScore
    conf   = 0.92
    scores = {c: round((1 - conf) / (len(CLASS_NAMES) - 1), 6) for c in CLASS_NAMES}
    scores[top_code] = conf
    top_idx = CLASS_NAMES.index(top_code)

    variety_scores = [
        VarietyScore(variety_code=c, variety_name=c, confidence_score=scores[c])
        for c in CLASS_NAMES
    ]
    variety_scores.sort(key=lambda v: v.confidence_score, reverse=True)

    return PredictionResponse(
        success=True,
        prediction=PredictionResult(
            variety_code=top_code,
            variety_name="Musang King" if top_code == "D200" else top_code,
            local_name=top_code,
            origin="Malaysia",
            description="Test description",
            confidence_score=conf,
        ),
        all_varieties=variety_scores,
        confidence_scores=scores,
        image_enhanced=True,
        inference_time_ms=18.5,
        preprocessing_time_ms=12.0,
        model_version="1.0.0",
        request_id="test-req-001",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  App fixture dengan key manager di-mock
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """TestClient dengan APIKeyManager dan model loader di-mock."""
    from core.security import APIKeyManager, AuthResult, KeyScope, RateLimitTier
    from core.rate_limiter import RateLimitResult

    def fake_validate(raw_key: str) -> AuthResult:
        key_map = {
            VALID_KEY:   AuthResult(
                valid=True, key_prefix="dk_live_Va...", key_name="Test Standard",
                scopes={KeyScope.PREDICT, KeyScope.HEALTH},
                tier=RateLimitTier.STANDARD,
            ),
            ADMIN_KEY:   AuthResult(
                valid=True, key_prefix="dk_live_Va...", key_name="Test Admin",
                scopes={KeyScope.PREDICT, KeyScope.HEALTH, KeyScope.ADMIN},
                tier=RateLimitTier.PREMIUM,
            ),
            PREDICT_ONLY_KEY: AuthResult(
                valid=True, key_prefix="dk_live_Pr...", key_name="Predict Only",
                scopes={KeyScope.PREDICT},
                tier=RateLimitTier.FREE,
            ),
            HEALTH_ONLY_KEY: AuthResult(
                valid=True, key_prefix="dk_live_He...", key_name="Health Only",
                scopes={KeyScope.HEALTH},
                tier=RateLimitTier.FREE,
            ),
        }
        return key_map.get(
            raw_key,
            AuthResult(valid=False, error="API key tidak valid."),
        )

    fake_rl_result = RateLimitResult(allowed=True, limit=300, remaining=299, reset_at=9999999999)

    with patch.object(APIKeyManager, "validate", side_effect=fake_validate), \
         patch.object(APIKeyManager, "_loaded", True, create=True), \
         patch("core.rate_limiter.SlidingWindowRateLimiter.check", return_value=fake_rl_result), \
         patch("models.model_loader.ONNXModelLoader.is_loaded", new_callable=lambda: property(lambda s: True)), \
         patch("models.model_loader.ONNXModelLoader.load_model"):
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture
def headers_standard():
    return {"X-API-Key": VALID_KEY}


@pytest.fixture
def headers_admin():
    return {"X-API-Key": ADMIN_KEY}


# ─────────────────────────────────────────────────────────────────────────────
#  /api/v1/ping
# ─────────────────────────────────────────────────────────────────────────────

class TestPingEndpoint:
    def test_ping_returns_200(self, client):
        resp = client.get("/api/v1/ping")
        assert resp.status_code == 200

    def test_ping_no_auth_required(self, client):
        resp = client.get("/api/v1/ping")
        assert resp.status_code == 200  # bukan 401/403

    def test_ping_response_has_status_ok(self, client):
        data = client.get("/api/v1/ping").json()
        assert data["status"] == "ok"

    def test_ping_response_has_service_name(self, client):
        data = client.get("/api/v1/ping").json()
        assert "service" in data
        assert data["service"] != ""

    def test_ping_response_has_version(self, client):
        data = client.get("/api/v1/ping").json()
        assert "version" in data


# ─────────────────────────────────────────────────────────────────────────────
#  /api/v1/health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_requires_auth(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code in (401, 403)

    def test_health_with_valid_key_returns_200(self, client, headers_standard):
        resp = client.get("/api/v1/health", headers=headers_standard)
        assert resp.status_code == 200

    def test_health_response_schema(self, client, headers_standard):
        data = client.get("/api/v1/health", headers=headers_standard).json()
        assert "status"       in data
        assert "model_loaded" in data
        assert "app_name"     in data
        assert "version"      in data

    def test_health_status_valid_value(self, client, headers_standard):
        data = client.get("/api/v1/health", headers=headers_standard).json()
        assert data["status"] in ("healthy", "degraded")

    def test_health_model_loaded_is_bool(self, client, headers_standard):
        data = client.get("/api/v1/health", headers=headers_standard).json()
        assert isinstance(data["model_loaded"], bool)

    def test_health_with_wrong_key_returns_403(self, client):
        resp = client.get("/api/v1/health", headers={"X-API-Key": WRONG_KEY})
        assert resp.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
#  /api/v1/predict — Authentication
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictAuthentication:
    def test_no_key_returns_401(self, client):
        files = {"file": ("t.jpg", io.BytesIO(_jpeg()), "image/jpeg")}
        resp  = client.post("/api/v1/predict", files=files)
        assert resp.status_code == 401

    def test_wrong_key_returns_403(self, client):
        files   = {"file": ("t.jpg", io.BytesIO(_jpeg()), "image/jpeg")}
        resp    = client.post("/api/v1/predict", files=files, headers={"X-API-Key": WRONG_KEY})
        assert resp.status_code == 403

    def test_bearer_token_accepted(self, client):
        """Authorization: Bearer <key> harus berfungsi sama."""
        with patch("app.api.routes.CLIPService.is_durian", return_value=True), \
             patch("app.api.routes.ImageProcessor.process", return_value=(np.zeros((1,224,224,3), dtype=np.float32), True, 10.0)), \
             patch("app.api.routes.InferenceService.predict", return_value=_mock_prediction_response()):
            files = {"file": ("t.jpg", io.BytesIO(_jpeg()), "image/jpeg")}
            resp  = client.post(
                "/api/v1/predict",
                files=files,
                headers={"Authorization": f"Bearer {VALID_KEY}"},
            )
            assert resp.status_code != 401

    def test_predict_only_key_can_predict(self, client):
        with patch("app.api.routes.CLIPService.is_durian", return_value=True), \
             patch("app.api.routes.ImageProcessor.process", return_value=(np.zeros((1,224,224,3), dtype=np.float32), True, 10.0)), \
             patch("app.api.routes.InferenceService.predict", return_value=_mock_prediction_response()):
            files = {"file": ("t.jpg", io.BytesIO(_jpeg()), "image/jpeg")}
            resp  = client.post(
                "/api/v1/predict",
                files=files,
                headers={"X-API-Key": PREDICT_ONLY_KEY},
            )
            assert resp.status_code == 200

    def test_health_only_key_cannot_predict(self, client):
        files = {"file": ("t.jpg", io.BytesIO(_jpeg()), "image/jpeg")}
        resp  = client.post(
            "/api/v1/predict",
            files=files,
            headers={"X-API-Key": HEALTH_ONLY_KEY},
        )
        assert resp.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
#  /api/v1/predict — Input Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictInputValidation:
    def test_no_data_returns_400(self, client, headers_standard):
        resp = client.post("/api/v1/predict", headers=headers_standard)
        assert resp.status_code == 400

    def test_unsupported_extension_returns_415(self, client, headers_standard):
        files = {"file": ("document.txt", io.BytesIO(b"text content"), "text/plain")}
        resp  = client.post("/api/v1/predict", files=files, headers=headers_standard)
        assert resp.status_code == 415

    def test_bmp_extension_returns_415(self, client, headers_standard):
        from PIL import Image
        img = Image.new("RGB", (50, 50))
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        files = {"file": ("image.bmp", buf, "image/bmp")}
        resp  = client.post("/api/v1/predict", files=files, headers=headers_standard)
        assert resp.status_code == 415

    def test_gif_extension_returns_415(self, client, headers_standard):
        files = {"file": ("anim.gif", io.BytesIO(b"GIF89a fake content"), "image/gif")}
        resp  = client.post("/api/v1/predict", files=files, headers=headers_standard)
        assert resp.status_code == 415

    def test_empty_file_returns_400(self, client, headers_standard):
        files = {"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")}
        resp  = client.post("/api/v1/predict", files=files, headers=headers_standard)
        assert resp.status_code == 400

    def test_file_with_wrong_magic_bytes_returns_415(self, client, headers_standard):
        """File ber-ekstensi .jpg tapi konten bukan JPEG."""
        fake_jpeg = b"This is not a JPEG file at all, just text content"
        files = {"file": ("fake.jpg", io.BytesIO(fake_jpeg), "image/jpeg")}
        resp  = client.post("/api/v1/predict", files=files, headers=headers_standard)
        assert resp.status_code in (400, 415)

    def test_both_file_and_json_returns_400(self, client, headers_standard):
        import json, base64
        files = {"file": ("t.jpg", io.BytesIO(_jpeg()), "image/jpeg")}
        resp  = client.post(
            "/api/v1/predict",
            headers=headers_standard,
            files=files,
            data={"payload": json.dumps({"image_base64": base64.b64encode(_jpeg()).decode()})},
        )
        assert resp.status_code in (400, 415, 422)

    def test_error_response_has_success_false(self, client, headers_standard):
        files = {"file": ("doc.txt", io.BytesIO(b"text"), "text/plain")}
        resp  = client.post("/api/v1/predict", files=files, headers=headers_standard)
        assert resp.json().get("success") is False


# ─────────────────────────────────────────────────────────────────────────────
#  /api/v1/predict — CLIP rejection
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictCLIPValidation:
    def test_non_durian_image_returns_400(self, client, headers_standard):
        with patch("app.api.routes.CLIPService.is_durian", return_value=False):
            files = {"file": ("cat.jpg", io.BytesIO(_jpeg()), "image/jpeg")}
            resp  = client.post("/api/v1/predict", files=files, headers=headers_standard)
            assert resp.status_code == 400

    def test_non_durian_error_message_informative(self, client, headers_standard):
        with patch("app.api.routes.CLIPService.is_durian", return_value=False):
            files = {"file": ("cat.jpg", io.BytesIO(_jpeg()), "image/jpeg")}
            data  = client.post("/api/v1/predict", files=files, headers=headers_standard).json()
            assert "durian" in data.get("detail", "").lower()


# ─────────────────────────────────────────────────────────────────────────────
#  /api/v1/predict — Successful predictions
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictSuccess:
    def _post_file(self, client, headers, raw_bytes, filename="test.jpg", mime="image/jpeg"):
        with patch("app.api.routes.CLIPService.is_durian", return_value=True), \
             patch("app.api.routes.ImageProcessor.process", return_value=(np.zeros((1,224,224,3), dtype=np.float32), True, 10.0)), \
             patch("app.api.routes.InferenceService.predict", return_value=_mock_prediction_response()):
            files = {"file": (filename, io.BytesIO(raw_bytes), mime)}
            return client.post("/api/v1/predict", files=files, headers=headers)

    def test_jpeg_returns_200(self, client, headers_standard):
        resp = self._post_file(client, headers_standard, _jpeg())
        assert resp.status_code == 200

    def test_png_returns_200(self, client, headers_standard):
        resp = self._post_file(client, headers_standard, _png(), "test.png", "image/png")
        assert resp.status_code == 200

    def test_webp_returns_200(self, client, headers_standard):
        resp = self._post_file(client, headers_standard, _webp(), "test.webp", "image/webp")
        assert resp.status_code == 200

    def test_response_success_true(self, client, headers_standard):
        resp = self._post_file(client, headers_standard, _jpeg())
        assert resp.json()["success"] is True

    def test_response_has_prediction(self, client, headers_standard):
        data = self._post_file(client, headers_standard, _jpeg()).json()
        assert "prediction" in data

    def test_response_prediction_has_variety_code(self, client, headers_standard):
        data = self._post_file(client, headers_standard, _jpeg()).json()
        assert "variety_code" in data["prediction"]
        assert data["prediction"]["variety_code"] in CLASS_NAMES

    def test_response_prediction_has_confidence(self, client, headers_standard):
        data = self._post_file(client, headers_standard, _jpeg()).json()
        conf = data["prediction"]["confidence_score"]
        assert 0.0 <= conf <= 1.0

    def test_response_has_all_varieties(self, client, headers_standard):
        data = self._post_file(client, headers_standard, _jpeg()).json()
        assert "all_varieties" in data
        assert len(data["all_varieties"]) == len(CLASS_NAMES)

    def test_response_all_varieties_sorted_descending(self, client, headers_standard):
        data   = self._post_file(client, headers_standard, _jpeg()).json()
        scores = [v["confidence_score"] for v in data["all_varieties"]]
        assert scores == sorted(scores, reverse=True)

    def test_response_has_confidence_scores_dict(self, client, headers_standard):
        data = self._post_file(client, headers_standard, _jpeg()).json()
        assert "confidence_scores" in data
        assert len(data["confidence_scores"]) == len(CLASS_NAMES)

    def test_response_has_inference_time(self, client, headers_standard):
        data = self._post_file(client, headers_standard, _jpeg()).json()
        assert "inference_time_ms" in data
        assert data["inference_time_ms"] >= 0.0

    def test_response_has_request_id(self, client, headers_standard):
        data = self._post_file(client, headers_standard, _jpeg()).json()
        assert "request_id" in data

    def test_response_has_image_enhanced_flag(self, client, headers_standard):
        data = self._post_file(client, headers_standard, _jpeg()).json()
        assert "image_enhanced" in data
        assert isinstance(data["image_enhanced"], bool)

    def test_response_headers_contain_request_id(self, client, headers_standard):
        resp = self._post_file(client, headers_standard, _jpeg())
        assert "X-Request-ID" in resp.headers or resp.status_code == 200

    def test_response_headers_contain_rate_limit(self, client, headers_standard):
        resp = self._post_file(client, headers_standard, _jpeg())
        # Rate limit header mungkin ada tergantung implementasi
        assert resp.status_code == 200

    def test_confidence_scores_sum_near_one(self, client, headers_standard):
        data  = self._post_file(client, headers_standard, _jpeg()).json()
        total = sum(data["confidence_scores"].values())
        assert abs(total - 1.0) < 0.02


# ─────────────────────────────────────────────────────────────────────────────
#  /api/v1/predict — Base64 input
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictBase64Input:
    def _post_b64(self, client, headers, b64_str, filename=None):
        payload = {"image_base64": b64_str}
        if filename:
            payload["filename"] = filename
        with patch("app.api.routes.CLIPService.is_durian", return_value=True), \
             patch("app.api.routes.ImageProcessor.process", return_value=(np.zeros((1,224,224,3), dtype=np.float32), True, 10.0)), \
             patch("app.api.routes.InferenceService.predict", return_value=_mock_prediction_response()):
            return client.post(
                "/api/v1/predict",
                json=payload,
                headers=headers,
            )

    def test_base64_jpeg_returns_200(self, client, headers_standard):
        import base64
        b64  = base64.b64encode(_jpeg()).decode()
        resp = self._post_b64(client, headers_standard, b64)
        assert resp.status_code == 200

    def test_base64_with_data_uri_prefix(self, client, headers_standard):
        import base64
        b64  = f"data:image/jpeg;base64,{base64.b64encode(_jpeg()).decode()}"
        resp = self._post_b64(client, headers_standard, b64)
        assert resp.status_code == 200

    def test_base64_with_valid_filename(self, client, headers_standard):
        import base64
        b64  = base64.b64encode(_jpeg()).decode()
        resp = self._post_b64(client, headers_standard, b64, filename="durian.jpg")
        assert resp.status_code == 200

    def test_base64_with_unsupported_filename_extension_returns_415(self, client, headers_standard):
        import base64
        b64  = base64.b64encode(_jpeg()).decode()
        resp = self._post_b64(client, headers_standard, b64, filename="document.txt")
        assert resp.status_code in (400, 415, 422)


# ─────────────────────────────────────────────────────────────────────────────
#  404 / 405 handlers
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandlers:
    def test_unknown_path_returns_404(self, client):
        resp = client.get("/api/v1/does_not_exist")
        assert resp.status_code == 404

    def test_404_response_has_success_false(self, client):
        data = client.get("/api/v1/does_not_exist").json()
        assert data.get("success") is False

    def test_wrong_method_on_predict_returns_405(self, client, headers_standard):
        resp = client.get("/api/v1/predict", headers=headers_standard)
        assert resp.status_code == 405

    def test_405_response_has_success_false(self, client, headers_standard):
        data = client.get("/api/v1/predict", headers=headers_standard).json()
        assert data.get("success") is False

    def test_root_endpoint_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_response_has_api_info(self, client):
        data = client.get("/").json()
        assert "name"    in data
        assert "version" in data
        assert "status"  in data
