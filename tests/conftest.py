"""
Shared fixtures dan helpers untuk seluruh test suite.
"""
import base64
import io
import os
from typing import Dict, Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
#  Konstanta
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["D101", "D13", "D197", "D198", "D2", "D200", "D24", "D88"]
NUM_CLASSES = 8

VALID_API_KEY_1      = "dk_live_TestKey1ForFrontendAppXXXXXXXXX"
VALID_API_KEY_2      = "dk_live_TestKey2ForAdminServiceXXXXXXXX"
VALID_API_KEY_TEST   = "dk_test_TestKey3ForCICDPipelineXXXXXXXX"
INVALID_API_KEY      = "dk_live_ThisKeyDoesNotExistInConfig1234"
MALFORMED_API_KEY    = "not-a-valid-key-format"


# ─────────────────────────────────────────────────────────────────────────────
#  Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_image_bytes(
    size: tuple = (300, 300),
    color: tuple = (34, 139, 34),
    fmt: str = "JPEG",
    mode: str = "RGB",
) -> bytes:
    img = Image.new(mode, size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def make_png_bytes(size=(300, 300), color=(34, 139, 34)) -> bytes:
    return make_image_bytes(size=size, color=color, fmt="PNG")


def make_jpeg_bytes(size=(300, 300), color=(34, 139, 34)) -> bytes:
    return make_image_bytes(size=size, color=color, fmt="JPEG")


def make_webp_bytes(size=(300, 300), color=(34, 139, 34)) -> bytes:
    return make_image_bytes(size=size, color=color, fmt="WEBP")


def make_rgba_bytes(size=(300, 300)) -> bytes:
    img = Image.new("RGBA", size, color=(255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_grayscale_bytes(size=(300, 300)) -> bytes:
    img = Image.new("L", size, color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def to_base64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("utf-8")


def to_base64_uri(raw: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{to_base64(raw)}"


# ─────────────────────────────────────────────────────────────────────────────
#  Mock factories
# ─────────────────────────────────────────────────────────────────────────────

def make_uniform_probs(num_classes: int = NUM_CLASSES) -> np.ndarray:
    return np.ones(num_classes, dtype=np.float32) / num_classes


def make_peaked_probs(top_idx: int = 0, confidence: float = 0.92) -> np.ndarray:
    probs = np.full(NUM_CLASSES, (1.0 - confidence) / (NUM_CLASSES - 1), dtype=np.float32)
    probs[top_idx] = confidence
    return probs


def make_mock_onnx_session(output_probs: np.ndarray) -> MagicMock:
    session = MagicMock()
    session.run.return_value = [output_probs.reshape(1, -1)]
    return session


def make_mock_loader(output_probs: np.ndarray) -> MagicMock:
    loader = MagicMock()
    loader.session     = make_mock_onnx_session(output_probs)
    loader.input_name  = "image_input"
    loader.output_name = "predictions"
    loader.is_loaded   = True
    return loader


def make_valid_tensor(h=224, w=224, c=3) -> np.ndarray:
    return np.random.uniform(0, 255, (1, h, w, c)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Pytest fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def jpeg_bytes() -> bytes:
    return make_jpeg_bytes()


@pytest.fixture
def png_bytes() -> bytes:
    return make_png_bytes()


@pytest.fixture
def webp_bytes() -> bytes:
    return make_webp_bytes()


@pytest.fixture
def rgba_bytes() -> bytes:
    return make_rgba_bytes()


@pytest.fixture
def grayscale_bytes() -> bytes:
    return make_grayscale_bytes()


@pytest.fixture
def b64_jpeg(jpeg_bytes) -> str:
    return to_base64(jpeg_bytes)


@pytest.fixture
def b64_png(png_bytes) -> str:
    return to_base64(png_bytes)


@pytest.fixture
def b64_jpeg_uri(jpeg_bytes) -> str:
    return to_base64_uri(jpeg_bytes, "image/jpeg")


@pytest.fixture
def valid_tensor() -> np.ndarray:
    return make_valid_tensor()


@pytest.fixture
def uniform_probs() -> np.ndarray:
    return make_uniform_probs()


@pytest.fixture
def peaked_probs() -> np.ndarray:
    return make_peaked_probs(top_idx=5, confidence=0.92)  # D200 = index 5


@pytest.fixture
def mock_loader(peaked_probs) -> MagicMock:
    return make_mock_loader(peaked_probs)


@pytest.fixture(scope="session")
def env_with_keys(tmp_path_factory):
    """Set environment variables untuk API keys selama test session."""
    env_vars = {
        "API_KEY_1":         VALID_API_KEY_1,
        "API_KEY_1_NAME":    "Frontend Test",
        "API_KEY_1_SCOPES":  "predict,health",
        "API_KEY_1_TIER":    "standard",
        "API_KEY_2":         VALID_API_KEY_2,
        "API_KEY_2_NAME":    "Admin Test",
        "API_KEY_2_SCOPES":  "predict,health,admin",
        "API_KEY_2_TIER":    "premium",
        "API_KEY_3":         VALID_API_KEY_TEST,
        "API_KEY_3_NAME":    "CI-CD Test",
        "API_KEY_3_SCOPES":  "predict,health",
        "API_KEY_3_TIER":    "free",
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def mock_clip_pass():
    """CLIP selalu meloloskan gambar (dianggap durian)."""
    with patch("services.clip_service.CLIPService.is_durian", return_value=True):
        yield


@pytest.fixture
def mock_clip_fail():
    """CLIP selalu menolak gambar (dianggap bukan durian)."""
    with patch("services.clip_service.CLIPService.is_durian", return_value=False):
        yield


@pytest.fixture
def mock_image_processor(valid_tensor):
    """ImageProcessor.process() mengembalikan tensor valid."""
    with patch(
        "services.image_processor.ImageProcessor.process",
        return_value=(valid_tensor, True, 12.0),
    ):
        yield


@pytest.fixture
def mock_inference_service():
    """InferenceService.predict() mengembalikan PredictionResponse mock."""
    from schemas.response import PredictionResponse, PredictionResult, VarietyScore

    variety_scores = [
        VarietyScore(variety_code=c, variety_name=c, confidence_score=round(p, 6))
        for c, p in zip(CLASS_NAMES, make_peaked_probs(5, 0.92).tolist())
    ]
    variety_scores.sort(key=lambda v: v.confidence_score, reverse=True)

    mock_resp = PredictionResponse(
        success=True,
        prediction=PredictionResult(
            variety_code="D200",
            variety_name="Musang King",
            local_name="D200 / Musang King",
            origin="Malaysia",
            description="Deskripsi test",
            confidence_score=0.92,
        ),
        all_varieties=variety_scores,
        confidence_scores={c: round(p, 6) for c, p in zip(CLASS_NAMES, make_peaked_probs(5, 0.92).tolist())},
        image_enhanced=True,
        inference_time_ms=18.5,
        preprocessing_time_ms=12.0,
        model_version="1.0.0",
        request_id="test-req-001",
    )

    with patch(
        "services.inference_service.InferenceService.predict",
        return_value=mock_resp,
    ):
        yield mock_resp
