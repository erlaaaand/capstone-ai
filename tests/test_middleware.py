"""
Tests untuk core/middleware.py — AuditLogger, security headers, payload limit.
"""
import logging
import time
from unittest.mock import patch, MagicMock

import pytest
from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from core.middleware import (
    AuditLogger,
    PayloadSizeLimitMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Starlette test apps
# ─────────────────────────────────────────────────────────────────────────────

def _make_app_with_middleware(*middlewares):
    """Buat app Starlette minimal dengan middleware tertentu."""
    async def homepage(request):
        return JSONResponse({"ok": True})

    routes = [Route("/", homepage), Route("/test", homepage, methods=["POST"])]
    app    = Starlette(routes=routes)
    for mw_class, kwargs in reversed(middlewares):
        app.add_middleware(mw_class, **kwargs)
    return app


# ─────────────────────────────────────────────────────────────────────────────
#  SecurityHeadersMiddleware
# ─────────────────────────────────────────────────────────────────────────────

class TestSecurityHeadersMiddleware:
    @pytest.fixture
    def client(self):
        app = _make_app_with_middleware((SecurityHeadersMiddleware, {}))
        return TestClient(app)

    def test_hsts_header_present(self, client):
        resp = client.get("/")
        assert "Strict-Transport-Security" in resp.headers

    def test_x_content_type_options_nosniff(self, client):
        resp = client.get("/")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options_deny(self, client):
        resp = client.get("/")
        assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_cache_control_no_store(self, client):
        resp = client.get("/")
        assert "no-store" in resp.headers.get("Cache-Control", "")

    def test_server_header_custom_value(self, client):
        resp = client.get("/")
        assert resp.headers.get("Server") == "Durian-API"

    def test_x_api_version_header_present(self, client):
        resp = client.get("/")
        assert "X-API-Version" in resp.headers

    def test_content_security_policy_present(self, client):
        resp = client.get("/")
        assert "Content-Security-Policy" in resp.headers

    def test_referrer_policy_present(self, client):
        resp = client.get("/")
        assert "Referrer-Policy" in resp.headers

    def test_permissions_policy_present(self, client):
        resp = client.get("/")
        assert "Permissions-Policy" in resp.headers


# ─────────────────────────────────────────────────────────────────────────────
#  RequestLoggingMiddleware
# ─────────────────────────────────────────────────────────────────────────────

class TestRequestLoggingMiddleware:
    @pytest.fixture
    def client(self):
        app = _make_app_with_middleware(
            (RequestLoggingMiddleware, {}),
        )
        return TestClient(app)

    def test_request_id_added_to_state(self, client):
        """request_id di-inject ke request.state oleh middleware."""
        request_ids = []

        async def capture_id(request: Request):
            request_ids.append(getattr(request.state, "request_id", None))
            return JSONResponse({"ok": True})

        from starlette.applications import Starlette
        from starlette.routing import Route
        capture_app = Starlette(routes=[Route("/", capture_id)])
        capture_app.add_middleware(RequestLoggingMiddleware)
        c    = TestClient(capture_app)
        resp = c.get("/")
        assert resp.status_code == 200
        assert len(request_ids) == 1
        assert request_ids[0] is not None

    def test_response_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_post_request_passes_through(self, client):
        resp = client.post("/test", json={"data": "test"})
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
#  PayloadSizeLimitMiddleware
# ─────────────────────────────────────────────────────────────────────────────

class TestPayloadSizeLimitMiddleware:
    def _app(self, max_bytes: int) -> TestClient:
        app = _make_app_with_middleware(
            (PayloadSizeLimitMiddleware, {"max_bytes": max_bytes}),
        )
        return TestClient(app, raise_server_exceptions=False)

    def test_small_payload_allowed(self):
        client = self._app(max_bytes=10 * 1024 * 1024)
        resp   = client.post("/test", content=b"small payload", headers={"Content-Length": "13"})
        assert resp.status_code == 200

    def test_oversized_payload_returns_413(self):
        client = self._app(max_bytes=10)
        big    = b"x" * 1000
        resp   = client.post("/test", content=big, headers={"Content-Length": str(len(big))})
        assert resp.status_code == 413

    def test_413_response_is_json(self):
        client = self._app(max_bytes=10)
        big    = b"x" * 1000
        resp   = client.post("/test", content=big, headers={"Content-Length": str(len(big))})
        data   = resp.json()
        assert data.get("success") is False
        assert "error" in data

    def test_missing_content_length_passes_through(self):
        """Tanpa Content-Length header, middleware tidak bisa cek ukuran di awal."""
        client = self._app(max_bytes=10)
        resp   = client.get("/")  # GET tidak punya body
        assert resp.status_code == 200

    def test_invalid_content_length_passes_through(self):
        client = self._app(max_bytes=100)
        resp   = client.post("/test", content=b"data", headers={"Content-Length": "not-a-number"})
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
#  AuditLogger
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditLogger:
    def test_auth_success_logs_info(self):
        with patch.object(AuditLogger._audit_logger, "info") as mock_log:
            AuditLogger.auth_success("req-1", "dk_live_...", "App Name", "127.0.0.1", "/predict")
            mock_log.assert_called_once()
            msg = mock_log.call_args[0][0]
            assert "AUTH_SUCCESS" in msg
            assert "req-1" in msg

    def test_auth_failure_logs_warning(self):
        with patch.object(AuditLogger._audit_logger, "warning") as mock_log:
            AuditLogger.auth_failure("req-2", "Key tidak ada", "127.0.0.1", "/predict")
            mock_log.assert_called_once()
            msg = mock_log.call_args[0][0]
            assert "AUTH_FAILURE" in msg

    def test_auth_failure_includes_key_hint(self):
        with patch.object(AuditLogger._audit_logger, "warning") as mock_log:
            AuditLogger.auth_failure("req-3", "Invalid", "1.2.3.4", "/predict", key_hint="dk_live...")
            msg = mock_log.call_args[0][0]
            assert "dk_live..." in msg

    def test_rate_limit_exceeded_logs_warning(self):
        with patch.object(AuditLogger._audit_logger, "warning") as mock_log:
            AuditLogger.rate_limit_exceeded("req-4", "key:dk_live_...", 300, "127.0.0.1")
            mock_log.assert_called_once()
            msg = mock_log.call_args[0][0]
            assert "RATE_LIMIT_EXCEEDED" in msg
            assert "300" in msg

    def test_suspicious_file_logs_warning(self):
        with patch.object(AuditLogger._audit_logger, "warning") as mock_log:
            AuditLogger.suspicious_file("req-5", "Magic bytes mismatch", "fake.jpg", "127.0.0.1")
            mock_log.assert_called_once()
            msg = mock_log.call_args[0][0]
            assert "SUSPICIOUS_FILE" in msg
            assert "fake.jpg" in msg

    def test_deprecated_key_logs_warning(self):
        with patch.object(AuditLogger._audit_logger, "warning") as mock_log:
            AuditLogger.deprecated_key_used("req-6", "dk_live_old...", "127.0.0.1")
            mock_log.assert_called_once()
            msg = mock_log.call_args[0][0]
            assert "DEPRECATED_KEY" in msg
            assert "ROTATE_KEY_IMMEDIATELY" in msg

    def test_auth_success_includes_path(self):
        with patch.object(AuditLogger._audit_logger, "info") as mock_log:
            AuditLogger.auth_success("r", "k", "n", "ip", "/api/v1/predict")
            msg = mock_log.call_args[0][0]
            assert "/api/v1/predict" in msg

    def test_all_methods_do_not_raise(self):
        """Semua metode AuditLogger harus tidak raise exception apapun."""
        try:
            AuditLogger.auth_success("r", "k", "n", "ip", "/path")
            AuditLogger.auth_failure("r", "reason", "ip", "/path")
            AuditLogger.rate_limit_exceeded("r", "id", 100, "ip")
            AuditLogger.suspicious_file("r", "reason", "file.jpg", "ip")
            AuditLogger.deprecated_key_used("r", "k", "ip")
        except Exception as e:
            pytest.fail(f"AuditLogger method raised: {e}")
