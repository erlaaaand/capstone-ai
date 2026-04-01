"""
Tests untuk schemas/request.py dan schemas/response.py — validasi Pydantic.
"""
import base64

import pytest
from pydantic import ValidationError

from schemas.request import PredictionRequestBase64
from schemas.response import (
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    PredictionResult,
    VarietyScore,
)


# ─────────────────────────────────────────────────────────────────────────────
#  PredictionRequestBase64
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionRequestBase64:
    VALID_B64 = base64.b64encode(b"fake image bytes for test").decode()

    def test_valid_base64_accepted(self):
        req = PredictionRequestBase64(image_base64=self.VALID_B64)
        assert req.image_base64 == self.VALID_B64

    def test_data_uri_prefix_stripped(self):
        b64_with_prefix = f"data:image/jpeg;base64,{self.VALID_B64}"
        req = PredictionRequestBase64(image_base64=b64_with_prefix)
        assert not req.image_base64.startswith("data:")

    def test_empty_base64_raises_error(self):
        with pytest.raises(ValidationError):
            PredictionRequestBase64(image_base64="")

    def test_whitespace_only_base64_raises_error(self):
        with pytest.raises(ValidationError):
            PredictionRequestBase64(image_base64="   ")

    def test_filename_optional_none_by_default(self):
        req = PredictionRequestBase64(image_base64=self.VALID_B64)
        assert req.filename is None

    def test_filename_lowercased(self):
        req = PredictionRequestBase64(image_base64=self.VALID_B64, filename="DURIAN.JPG")
        assert req.filename == "durian.jpg"

    def test_filename_stripped_whitespace(self):
        req = PredictionRequestBase64(image_base64=self.VALID_B64, filename="  durian.jpg  ")
        assert req.filename == "durian.jpg"

    def test_none_filename_stays_none(self):
        req = PredictionRequestBase64(image_base64=self.VALID_B64, filename=None)
        assert req.filename is None


# ─────────────────────────────────────────────────────────────────────────────
#  PredictionResult
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionResult:
    def _valid(self, **kwargs) -> dict:
        base = dict(
            variety_code="D200",
            variety_name="Musang King",
            local_name="D200 / Musang King",
            origin="Malaysia",
            description="Test description",
            confidence_score=0.95,
        )
        base.update(kwargs)
        return base

    def test_valid_result_created(self):
        result = PredictionResult(**self._valid())
        assert result.variety_code == "D200"

    def test_confidence_zero_valid(self):
        result = PredictionResult(**self._valid(confidence_score=0.0))
        assert result.confidence_score == 0.0

    def test_confidence_one_valid(self):
        result = PredictionResult(**self._valid(confidence_score=1.0))
        assert result.confidence_score == 1.0

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError):
            PredictionResult(**self._valid(confidence_score=-0.1))

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            PredictionResult(**self._valid(confidence_score=1.1))

    def test_missing_required_field_raises(self):
        data = self._valid()
        del data["variety_code"]
        with pytest.raises(ValidationError):
            PredictionResult(**data)


# ─────────────────────────────────────────────────────────────────────────────
#  VarietyScore
# ─────────────────────────────────────────────────────────────────────────────

class TestVarietyScore:
    def test_valid_variety_score(self):
        vs = VarietyScore(variety_code="D197", variety_name="Golden Phoenix", confidence_score=0.75)
        assert vs.variety_code == "D197"

    def test_confidence_zero(self):
        vs = VarietyScore(variety_code="D13", variety_name="Kuk San", confidence_score=0.0)
        assert vs.confidence_score == 0.0

    def test_confidence_invalid_above_one(self):
        with pytest.raises(ValidationError):
            VarietyScore(variety_code="D13", variety_name="Kuk San", confidence_score=1.5)


# ─────────────────────────────────────────────────────────────────────────────
#  PredictionResponse
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionResponse:
    def _make(self, **kwargs) -> PredictionResponse:
        result = PredictionResult(
            variety_code="D200", variety_name="Musang King",
            local_name="D200", origin="Malaysia",
            description="Desc", confidence_score=0.90,
        )
        base = dict(
            success=True,
            prediction=result,
            all_varieties=[VarietyScore(variety_code="D200", variety_name="Musang King", confidence_score=0.90)],
            confidence_scores={"Musang King": 0.90},
            image_enhanced=True,
            inference_time_ms=18.5,
        )
        base.update(kwargs)
        return PredictionResponse(**base)

    def test_valid_response_created(self):
        resp = self._make()
        assert resp.success is True

    def test_inference_time_negative_raises(self):
        with pytest.raises(ValidationError):
            self._make(inference_time_ms=-1.0)

    def test_optional_fields_default_none(self):
        resp = self._make()
        assert resp.request_id    is None or isinstance(resp.request_id, str)
        assert resp.model_version is None or isinstance(resp.model_version, str)

    def test_success_default_true(self):
        resp = self._make()
        assert resp.success is True

    def test_serializes_to_dict(self):
        resp = self._make()
        d    = resp.model_dump()
        assert "prediction"        in d
        assert "confidence_scores" in d
        assert "all_varieties"     in d


# ─────────────────────────────────────────────────────────────────────────────
#  HealthResponse
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthResponse:
    def _make(self, **kwargs):
        base = dict(
            status="healthy",
            model_loaded=True,
            app_name="Durian Classification API",
            version="1.0.0",
        )
        base.update(kwargs)
        return HealthResponse(**base)

    def test_valid_health_response(self):
        resp = self._make()
        assert resp.status == "healthy"

    def test_degraded_status_valid(self):
        resp = self._make(status="degraded")
        assert resp.status == "degraded"

    def test_model_loaded_false(self):
        resp = self._make(model_loaded=False)
        assert resp.model_loaded is False

    def test_optional_fields_can_be_none(self):
        resp = self._make()
        # Semua optional boleh None
        assert resp.uptime_seconds is None or isinstance(resp.uptime_seconds, int)

    def test_with_all_optional_fields(self):
        resp = self._make(
            uptime_seconds=3600,
            memory_usage_mb=512.5,
            cpu_percent=2.1,
            rate_limiter_stats={"tracked_identifiers": 10},
            config_summary={"num_classes": 8},
        )
        assert resp.uptime_seconds == 3600
        assert resp.memory_usage_mb == 512.5


# ─────────────────────────────────────────────────────────────────────────────
#  ErrorResponse
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorResponse:
    def test_valid_error_response(self):
        err = ErrorResponse(error="InvalidImageException", detail="File kosong.")
        assert err.success is False

    def test_success_always_false(self):
        err = ErrorResponse(error="SomeError", detail="Something went wrong")
        assert err.success is False

    def test_optional_request_id(self):
        err = ErrorResponse(error="E", detail="D", request_id="req-123")
        assert err.request_id == "req-123"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            ErrorResponse(detail="Only detail, no error field")
