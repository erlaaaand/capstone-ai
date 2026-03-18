"""
Automated unit tests for the FastAPI application endpoints.

Tests /health and /predict boundaries. Mocks the InferenceService to
avoid loading the actual ONNX model into memory during API validation.
"""

import io
from typing import Dict, Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from core.exceptions import DurianServiceException


# Create a TestClient instance. 
# We don't trigger lifespan events by default with TestClient unless context managed,
# ensuring the real model isn't unexpectedly loaded.
client = TestClient(app)

# Dummy API Key matching what we will set in the environment mock
VALID_API_KEY = "test_super_secret_key"


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Fixture to set required environment variables for tests."""
    monkeypatch.setenv("API_KEY", VALID_API_KEY)


def test_health_check():
    """Test the GET /api/v1/health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_missing_api_key():
    """Test the POST /api/v1/predict endpoint without an API Key."""
    # Create dummy image
    file_content = b"fake_image_data"
    files = {"file": ("test.jpg", io.BytesIO(file_content), "image/jpeg")}
    
    response = client.post("/api/v1/predict", files=files)
    
    assert response.status_code == 403
    assert response.json()["detail"] == "Missing 'X-API-Key' header."


def test_predict_invalid_api_key():
    """Test the POST /api/v1/predict endpoint with an invalid API Key."""
    file_content = b"fake_image_data"
    files = {"file": ("test.jpg", io.BytesIO(file_content), "image/jpeg")}
    headers = {"X-API-Key": "wrong_key"}
    
    response = client.post("/api/v1/predict", files=files, headers=headers)
    
    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid API Key."


def test_predict_no_data_provided():
    """Test the POST /api/v1/predict endpoint with valid auth but no data."""
    headers = {"X-API-Key": VALID_API_KEY}
    response = client.post("/api/v1/predict", headers=headers)
    
    assert response.status_code == 400


def test_predict_success_mocked(mocker):
    """Test a successful POST /api/v1/predict using a mocked inference service.
    
    We bypass the real image processor and ONNX session to test the HTTP routing.
    """
    # 1. Mock the ImageProcessor to return a dummy np array
    mocker.patch(
        "app.api.routes.ImageProcessor.process", 
        return_value="dummy_tensor"
    )
    
    # 2. Mock the InferenceService to return a valid Pydantic response
    from schemas.response import PredictionResponse, PredictionResult
    
    mock_response = PredictionResponse(
        success=True,
        prediction=PredictionResult(class_name="Musang_King", confidence_score=0.98),
        confidence_scores={
            "Musang_King": 0.98,
            "Duri_Hitam": 0.01,
            "Sultan": 0.005,
            "Golden_Phoenix": 0.005
        },
        inference_time_ms=12.5
    )
    
    mocker.patch(
        "app.api.routes.InferenceService.predict", 
        return_value=mock_response
    )
    
    # 3. Submit valid file request
    # Create valid synthetic image bytes (we don't need real image since ImageProcessor is mocked)
    # BUT we do need the raw file reading logic in _validate_upload_file to pass (size/ext checks)
    file_content = b"synthetic_valid_image_bytes"
    files = {"file": ("test.jpg", io.BytesIO(file_content), "image/jpeg")}
    headers = {"X-API-Key": VALID_API_KEY}
    
    response = client.post("/api/v1/predict", files=files, headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["prediction"]["class_name"] == "Musang_King"
    assert data["prediction"]["confidence_score"] == 0.98
    assert "inference_time_ms" in data


def test_predict_invalid_extension():
    """Test the POST /api/v1/predict endpoint with an invalid file extension."""
    file_content = b"text_data"
    files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
    headers = {"X-API-Key": VALID_API_KEY}
    
    response = client.post("/api/v1/predict", files=files, headers=headers)
    
    assert response.status_code == 415  # Custom UnsupportedFileTypeException maps to 415
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "UnsupportedFileTypeException"
