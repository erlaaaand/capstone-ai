# Enterprise Durian Classification API

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-005C8A?style=for-the-badge&logo=onnx&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

---

🌐 **Bahasa / Language:** [![🇮🇩 Bahasa Indonesia](https://img.shields.io/badge/🇮🇩_Bahasa_Indonesia-tersedia-lightgrey?style=flat-square)](./README.md) [![🇬🇧 English](https://img.shields.io/badge/🇬🇧_English-active-blue?style=flat-square)](./README.en.md)

---

An AI backend system built with **FastAPI** for classifying durian varieties from images using deep learning. The **EfficientNetB0** model is exported to **ONNX** format for high-performance inference, equipped with enterprise security layers, an automatic image enhancement pipeline, and CLIP-based zero-shot image validation.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Supported Durian Varieties](#supported-durian-varieties)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Configuration](#installation--configuration)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Security & Authentication](#security--authentication)
- [Image Pipeline](#image-pipeline)
- [Deep Learning Model](#deep-learning-model)
- [Testing](#testing)
- [Advanced Configuration](#advanced-configuration)

---

## System Architecture

```
Request (Image)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                   FastAPI App                        │
│  ┌─────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │  Middleware  │→ │  Auth &    │→ │   Endpoint   │  │
│  │  (CORS, Log, │  │  Rate Limit│  │  /predict    │  │
│  │   Payload)  │  │            │  │  /health     │  │
│  └─────────────┘  └────────────┘  └──────┬───────┘  │
└──────────────────────────────────────────┼──────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                       │
                    ▼                      ▼                       ▼
          ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────┐
          │  CLIP Service   │   │  ImageProcessor  │   │InferenceService │
          │ (Zero-shot val) │   │ (Enhancement +   │   │ (ONNX Runtime)  │
          │                 │   │  Preprocessing)  │   │                 │
          └─────────────────┘   └──────────────────┘   └─────────────────┘
                                                                 │
                                                                 ▼
                                                     ┌──────────────────────┐
                                                     │  EfficientNetB0.onnx │
                                                     │  (6 Durian Classes)  │
                                                     └──────────────────────┘
```

Each image request passes through three main stages:
1. **CLIP Validation** — ensures the image is actually a durian fruit before further processing. CLIP and Image Processing run **in parallel** (`asyncio.gather`) to speed up response time.
2. **Image Processing** — automatic enhancement (white balance, CLAHE, sharpening) followed by letterbox resize to 224×224.
3. **ONNX Inference** — the EfficientNetB0 model produces probabilities for 6 varieties.

---

## Key Features

### 🚀 High-Performance Inference
ONNX Runtime is used as the inference backend, replacing native TensorFlow in production. The result: lower latency and smaller memory footprint. Automatic warmup on startup ensures the first request doesn't suffer from cold-start. The session is configured with graph optimization level `ORT_ENABLE_ALL` and automatic thread pooling.

### 🖼️ Advanced Image Processing Pipeline
Image enhancement runs automatically before inference, configurable per-feature via `.env`:
- **Auto White Balance** — corrects color cast caused by varying lighting conditions.
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) — improves local contrast without over-exposure (implemented via histogram equalization on the Y channel of YCbCr).
- **Unsharp Masking** — sharpens edges and durian skin texture.
- **Letterbox Resize** — preserves image aspect ratio without distortion when resizing to 224×224, with padding color `(114, 114, 114)`.

### 🛡️ Enterprise Security
- **API Key Authentication** using custom header `X-API-Key` or `Authorization: Bearer`.
- Supports **multiple API keys** (up to 19 keys, `API_KEY_1` through `API_KEY_19`) with different names, scopes, and tiers.
- **Zero-downtime key rotation**: mark old key as `deprecated`, add new key, remove old one after clients update.
- **Hot-reload keys** via `POST /api/v1/admin/reload-keys` without server restart.
- **Scopes**: `predict`, `health`, `admin`, `readonly`.
- **Rate Limiting** based on Sliding Window per API key, with burst protection (20 req/second).
- **Key Hashing**: API keys are stored as PBKDF2-HMAC-SHA256 hashes (100,000 iterations, random 16-byte salt). Validation uses `hmac.compare_digest()` to prevent timing attacks.
- Key format: `dk_live_<32char>` for production, `dk_test_<32char>` for testing.

### 🔄 Resilient Middleware Stack
Middleware order (outer to inner):
1. `PayloadSizeLimitMiddleware` — reject oversized payloads before reading (checks `Content-Length` header).
2. `GZipMiddleware` — compress responses ≥ 1KB.
3. `TrustedHostMiddleware` — host-header injection protection (active if `ALLOWED_HOSTS_STR` ≠ `*`).
4. `CORSMiddleware` — origin whitelist, methods `POST/GET/OPTIONS`, expose rate limit & request ID headers.
5. `RequestLoggingMiddleware` — logs every request/response with a unique UUID `request_id`, elapsed time, IP, and user agent.
6. `SecurityHeadersMiddleware` — injects 12 security headers (HSTS, CSP, X-Frame-Options, COOP, CORP, etc.).

### 📋 Structured Logging
All logs use JSON format (`JSONFormatter`) and include: UTC timestamp, level, module name, function, line number, and additional data. A separate `AuditLogger` (`audit` logger) records every auth success/failure, rate limit exceeded, deprecated key usage, and suspicious files.

### 🤖 CLIP-based Durian Validation
Before entering the classification model, images are validated using **CLIP** (`openai/clip-vit-base-patch32`) via zero-shot. Images are classified against 5 labels:
1. `a photo of a durian fruit`
2. `a photo of a person`
3. `a photo of an animal`
4. `a photo of a vehicle`
5. `a photo of random objects or scenery`

If a non-durian label dominates with confidence > 40%, the request is rejected. CLIP is **fail-open**: if the model fails to load or errors during inference, all images are still allowed through.

---

## Supported Durian Varieties

| Code | Popular Name | Local Name                                       | Origin                      |
| ---- | ------------ | ------------------------------------------------ | --------------------------- |
| D2   | Dato Nina    | D2 / Dato Nina                                   | Malaysia (Melaka)           |
| D13  | Golden Bun   | D13 / Golden Bun                                 | Malaysia (Johor)            |
| D24  | Sultan       | D24 / Sultan / Bukit Merah                       | Malaysia (Perak / Selangor) |
| D101 | Muar Gold    | D101 / Muar Gold / Johor Mas                     | Malaysia (Johor)            |
| D197 | Musang King  | D197 / Musang King / Raja Kunyit / Mao Shan Wang | Malaysia (Kelantan)         |
| D200 | Black Thorn  | D200 / Ochee / Duri Hitam / Black Thorn          | Malaysia (Penang)           |

> **Note:** The class order in the model follows the index order in `data/class_indices.json`: `D101(0), D13(1), D197(2), D2(3), D200(4), D24(5)`. Ensure `CLASS_NAMES` in `.env` matches the folder order during training (alphabetical).

---

## Project Structure

```
backend_ai/
│
├── app/
│   ├── api/
│   │   ├── __init__.py          # Router aggregator (prefix /api/v1)
│   │   ├── health.py            # GET /ping, GET /health, POST /admin/reload-keys
│   │   └── routes.py            # POST /predict — main inference endpoint
│   ├── core_dependencies.py     # verify_api_key, require_scope (FastAPI dependencies)
│   └── main.py                  # App factory, lifespan, middleware, exception handlers
│
├── core/
│   ├── __init__.py              # Re-export settings, logger, all exception classes
│   ├── config.py                # Settings (pydantic-settings), VARIETY_MAP, get_variety_info()
│   ├── exceptions.py            # DurianServiceException and 6 subclasses
│   ├── logger.py                # JSONFormatter, setup_logging(), get_logger()
│   ├── middleware.py            # SecurityHeaders, RequestLogging, PayloadSizeLimit, AuditLogger
│   ├── rate_limiter.py          # SlidingWindowRateLimiter (async, in-memory, background cleanup)
│   └── security.py             # APIKeyManager, hash/verify key, AuthResult, KeyScope, RateLimitTier
│
├── models/
│   ├── model_loader.py          # ONNXModelLoader (singleton, thread-safe, warmup inference)
│   └── weights/                 # ← place .onnx file here (not committed)
│
├── schemas/
│   ├── __init__.py              # Re-export all schemas
│   ├── request.py               # PredictionRequestBase64 (Pydantic, auto-strip data URI prefix)
│   └── response.py              # PredictionResponse, PredictionResult, VarietyScore, HealthResponse, ErrorResponse
│
├── services/
│   ├── __init__.py              # Re-export ImageProcessor, InferenceService
│   ├── clip_service.py          # CLIPService.is_durian() — zero-shot validation (lazy loading, thread-safe)
│   ├── image_processor.py       # ImageProcessor.process() — decode + enhance + letterbox resize
│   └── inference_service.py     # InferenceService.predict() — run ONNX + format response
│
├── tests/
│   └── test_clip_service.py     # Unit tests for CLIPService (lazy loading, thread safety, graceful degradation)
│
├── data/
│   └── class_indices.json       # Index → class code mapping (6 classes)
│
├── .env.example                 # Configuration template (read before setup)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Prerequisites

- **Python 3.9+**
- **Virtual environment** (highly recommended)
- **Trained ONNX model file** (`.onnx` format)
- GPU optional — ONNX Runtime automatically uses CUDA if available (`CUDAExecutionProvider`)

---

## Installation & Configuration

### 1. Clone & Create Virtual Environment

```bash
git clone https://github.com/erlaaaand/capstone-ai.git
cd backend_ai

python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Open `.env` and adjust the following values:

```env
# Path to the exported ONNX model file
MODEL_PATH=models/weights/efficientnet_b0.onnx

# Class order MUST match training folder order (alphabetical)
CLASS_NAMES=D101,D13,D197,D2,D200,D24

# Generate a new API key:
# python -c "import secrets; print('dk_live_' + secrets.token_urlsafe(24))"
API_KEY_1=dk_live_<generated_value>
API_KEY_1_NAME=Frontend App
API_KEY_1_SCOPES=predict,health
API_KEY_1_TIER=standard

# Add a second key for internal backend (admin scope)
API_KEY_2=dk_live_<generated_value>
API_KEY_2_NAME=Internal Backend
API_KEY_2_SCOPES=predict,health,admin
API_KEY_2_TIER=premium

# Allowed CORS origins (never use * in production)
CORS_ORIGINS_STR=https://app.example.com,https://admin.example.com
```

### 4. Prepare Model File

Place the trained `.onnx` file in the `models/weights/` directory:

```bash
# Example if the model already exists
cp /path/to/efficientnet_b0.onnx models/weights/
```

---

## Running the Application

### Development (with hot-reload)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Access Swagger UI (only if `DEBUG=True` in `.env`):

```
http://localhost:8000/docs
```

Check API status without auth:

```
GET http://localhost:8000/api/v1/ping
```

### Root Endpoint Info

```
GET http://localhost:8000/
```
Response contains an endpoint overview and service status (no auth required).

---

## API Reference

All endpoints under `/api/v1/` require the `X-API-Key` header (except `/ping`).

### `GET /api/v1/ping`

Public liveness check without authentication. Used for load balancers.

**Response:**
```json
{
  "status": "ok",
  "service": "Durian Classification API",
  "version": "1.0.0"
}
```

---

### `GET /api/v1/health`

Detailed service status. Requires a valid API key (any scope).

**Headers:**
```
X-API-Key: dk_live_xxx
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "app_name": "Durian Classification API",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "memory_usage_mb": 512.3,
  "cpu_percent": 2.1,
  "rate_limiter_stats": {
    "tracked_identifiers": 12,
    "cleanup_task_active": 1
  },
  "config_summary": {
    "num_classes": 6,
    "image_size": 224,
    "enhancement": true,
    "max_file_size_mb": 10
  }
}
```

> **Note:** Status `"degraded"` is returned if the ONNX model has not been loaded.

---

### `POST /api/v1/predict`

Classify durian variety from an image. Requires `predict` or `admin` scope.

**Input Option 1 — File Upload (`multipart/form-data`):**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "X-API-Key: dk_live_xxx" \
  -F "file=@durian.jpg"
```

**Input Option 2 — Base64 JSON:**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "X-API-Key: dk_live_xxx" \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "/9j/4AAQ...", "filename": "durian.jpg"}'
```

> Both cannot be sent simultaneously. The data URI prefix (`data:image/jpeg;base64,`) is automatically stripped by the validator.

**Success Response (200):**
```json
{
  "success": true,
  "prediction": {
    "variety_code": "D197",
    "variety_name": "Musang King",
    "local_name": "D197 / Musang King / Raja Kunyit / Mao Shan Wang",
    "origin": "Malaysia (Kelantan)",
    "description": "The king of Malaysian durians with thick golden-yellow flesh...",
    "confidence_score": 0.9231
  },
  "all_varieties": [
    { "variety_code": "D197", "variety_name": "Musang King", "confidence_score": 0.9231 },
    { "variety_code": "D200", "variety_name": "Black Thorn", "confidence_score": 0.0412 },
    { "variety_code": "D24", "variety_name": "Sultan", "confidence_score": 0.0201 },
    { "variety_code": "D101", "variety_name": "Muar Gold", "confidence_score": 0.0098 },
    { "variety_code": "D13", "variety_name": "Golden Bun", "confidence_score": 0.0041 },
    { "variety_code": "D2", "variety_name": "Dato Nina", "confidence_score": 0.0017 }
  ],
  "confidence_scores": {
    "Musang King": 0.9231,
    "Black Thorn": 0.0412,
    "Sultan": 0.0201,
    "Muar Gold": 0.0098,
    "Golden Bun": 0.0041,
    "Dato Nina": 0.0017
  },
  "image_enhanced": true,
  "inference_time_ms": 18.5,
  "preprocessing_time_ms": 12.3,
  "model_version": "1.0.0",
  "request_id": "a3b1c2d4-..."
}
```

**Error Responses:**

| Status | Error Code | Cause |
|--------|-----------|-------|
| 400 | `InvalidImageException` | Empty file, not a valid image, or not a durian image (CLIP) |
| 400 | - | Sending both file and JSON simultaneously / empty image data |
| 401 | - | No API key provided |
| 403 | - | Invalid API key or insufficient scope |
| 413 | `FileTooLargeException` | File exceeds limit (default 10MB) |
| 415 | `UnsupportedFileTypeException` | Unsupported extension (not jpg/jpeg/png/webp) |
| 422 | `ImageProcessingException` | Failed to preprocess image |
| 429 | - | Rate limit exceeded |
| 500 | `InferenceException` | Failed to run model inference |
| 503 | `ModelNotLoadedException` | Model not loaded at startup |

**Additional Response Headers:**
- `X-Request-ID` — unique UUID per request for tracing
- `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, `X-RateLimit-Policy` — rate limit info
- `Warning: 299` — appears if the API key is in deprecated status
- `X-API-Version` — current API version

---

### `POST /api/v1/admin/reload-keys`

Hot-reload API keys from environment variables without server restart. Requires `admin` scope.

**Headers:**
```
X-API-Key: dk_live_xxx  (key with admin scope)
```

**Success Response:**
```json
{
  "success": true,
  "message": "API keys and settings reloaded successfully.",
  "key_count": 3,
  "reloaded_by": "dk_live_xxxx...",
  "app_version": "1.0.0",
  "settings_refreshed": true
}
```

> This endpoint runs `load_dotenv(override=True)` so changes in the `.env` file are applied immediately. The `Settings` cache is also refreshed.

---

## Security & Authentication

### API Key Structure

```
dk_live_<random_32_characters>   ← production
dk_test_<random_32_characters>   ← testing / CI-CD
```

### Generate a New API Key

```python
import secrets
live = 'dk_live_' + secrets.token_urlsafe(24)
test = 'dk_test_' + secrets.token_urlsafe(24)
print('LIVE:', live)
print('TEST:', test)
```

Or via bash:
```bash
python -c "import secrets; print('dk_live_' + secrets.token_urlsafe(24))"
```

### Key Rotation (Zero-Downtime)

```bash
# 1. Generate a new key and add it to .env as API_KEY_4
# 2. Call POST /api/v1/admin/reload-keys (admin scope)
# 3. Mark old key as deprecated:
API_KEY_1_DEPRECATED=true
# 4. Call reload-keys again
# 5. After all clients update to the new key, remove API_KEY_1 from .env
```

While `deprecated=true`, the old key still works but the response will include a `Warning: 299` header.

### Rate Limit per Tier

| Tier | Limit | Burst |
|------|-------|-------|
| `free` | 60 req/min | 20 req/sec |
| `standard` | 300 req/min | 20 req/sec |
| `premium` | 1000 req/min | 20 req/sec |
| `unlimited` | ~Unlimited | 20 req/sec |

Rate limit identifier: `key:<prefix>` for successful authentication, `ip:<client_ip>` as fallback (limit 30/min).

Background cleanup task runs every 5 minutes to remove stale entries (inactive > 10 minutes).

### Key Storage Security

API keys are **not stored in plaintext** in memory. The system uses **PBKDF2-HMAC-SHA256** with 100,000 iterations and a random 16-byte salt. Hash comparison uses `hmac.compare_digest()` to prevent timing attacks. Lookup uses the key prefix as a dictionary index for O(1) performance.

### Key Expiration (Optional)

Each key can be configured with an expiration time using a Unix timestamp:

```env
API_KEY_1_EXPIRES_AT=1767225600
```

---

## Image Pipeline

Each image passes through the following pipeline in `ImageProcessor.process()`:

```
Input (bytes / base64 string)
        │
        ▼
   Decode & Verify      ← PIL verify() + re-open
        │
        ▼
  Convert to RGB        ← RGBA, grayscale, etc → RGB
        │
        ▼
  Letterbox Resize      ← resize to 224×224 with padding (114,114,114), aspect ratio preserved
        │
        ▼
 Enhancement (optional) ← controlled by ENABLE_ENHANCEMENT in .env
    ├── Auto White Balance    (gray world assumption)
    ├── CLAHE                 (histogram EQ on Y channel of YCbCr, alpha blending)
    └── Unsharp Masking       (GaussianBlur radius=2, amount=0.45)
        │
        ▼
  Output: float32       ← shape (1, 224, 224, 3), range [0, 255]
  numpy tensor          ← EfficientNetB0 handles normalization internally
```

> **Important:** The output tensor is in the range **[0, 255]**, NOT [0, 1]. EfficientNetB0 has an internal preprocessing layer that handles normalization during inference.

### Magic Bytes Validation

Before processing, uploaded files are validated using magic bytes:
- **JPEG**: `\xff\xd8\xff`
- **PNG**: `\x89PNG\r\n\x1a\n`
- **WebP**: `RIFF....WEBP`

Files whose extension doesn't match their magic bytes will be rejected and logged as `SUSPICIOUS_FILE`.

### Enhancement Configuration

```env
ENABLE_ENHANCEMENT=True
ENABLE_WHITE_BALANCE=True
ENABLE_CLAHE=True
ENABLE_SHARPENING=True
CLAHE_CLIP_LIMIT=2.0   # CLAHE strength: 1.0–4.0
```

---

## Deep Learning Model

### Architecture

The model uses **EfficientNetB0** as the backbone with a custom classification head:

```
Input (224×224×3)
    → EfficientNetB0 Backbone (ImageNet pretrained)
    → GlobalAveragePooling2D
    → Dense Head → Dense(6, Softmax)
```

Output: probabilities for 6 durian classes (D101, D13, D197, D2, D200, D24).

### ONNX Model

The model is stored in **ONNX** (Open Neural Network Exchange) format for:
- Cross-platform inference without TensorFlow dependency
- Automatic graph optimization via ONNX Runtime
- Automatic GPU (CUDA) and CPU support

Model specifications:
- Input: `float32[1, 224, 224, 3]`
- Output: `float32[1, 6]` (softmax probabilities)
- File size: ~18–20 MB (float32)

### Auto-Softmax

`InferenceService` automatically detects and handles model output:
- If the sum of probabilities ≈ 1.0 → use directly (output is already softmax)
- If the sum of probabilities is far from 1.0 → apply softmax automatically (output is still logits)

---

## Testing

```bash
# Run all tests
pytest tests/

# With verbose output
pytest tests/ -v

# Per-file test
pytest tests/test_clip_service.py -v
```

### CLIP Service Test Coverage

| Area | What's Tested |
|------|-----------| 
| **Lazy Loading** | Model not loaded on import, loaded on first `warmup()` / `is_durian()` call, `_load_attempted` flag prevents repeated reloads |
| **Graceful Degradation** | CLIP unavailable → allow all images (fail-open), loading failure → `_model` stays `None`, inference error → return `True` |
| **Classification** | High durian probability → `True`, non-durian dominant (>0.40) → `False`, non-durian low (≤0.40) → `True` (fail-open) |
| **Input Handling** | Accepts `bytes` and `base64` string, corrupt image → fail-safe (return `True`) |
| **Thread Safety** | 10 concurrent threads calling `warmup()` → model loaded only once (double-checked locking) |
| **Warmup Integration** | `warmup()` always returns `bool` (never raises exception), idempotent (safe to call repeatedly) |

---

## Advanced Configuration

All configuration is managed via `.env`. See `.env.example` for the full list.

### Key Variables

| Variable | Default | Description |
|----------|---------|-----------|
| `DEBUG` | `False` | Enable `/docs`, `/redoc`, `/openapi.json` |
| `LOG_LEVEL` | `INFO` | DEBUG/INFO/WARNING/ERROR/CRITICAL |
| `MODEL_PATH` | `models/weights/efficientnet_b0.onnx` | Path to model file |
| `CLASS_NAMES` | `D101,D13,D197,D2,D200,D24` | Class order (MUST match training folder order) |
| `IMAGE_SIZE` | `224` | Model input size (32–1024) |
| `MAX_FILE_SIZE_MB` | `10` | Upload file size limit |
| `ALLOWED_EXTENSIONS` | `jpg,jpeg,png,webp` | Allowed file extensions |
| `ENABLE_ENHANCEMENT` | `True` | Master switch for enhancement pipeline |
| `ENABLE_WHITE_BALANCE` | `True` | Auto white balance |
| `ENABLE_CLAHE` | `True` | Contrast adaptive histogram equalization |
| `ENABLE_SHARPENING` | `True` | Unsharp masking |
| `CLAHE_CLIP_LIMIT` | `2.0` | CLAHE strength (1.0–4.0) |
| `CORS_ORIGINS_STR` | `http://localhost:3000,...` | Comma-separated allowed origins |
| `ALLOWED_HOSTS_STR` | `*` | Allowed hosts (anti host-header injection) |
| `API_KEY_REQUIRED` | `True` | Require API key |

### Environment per Deployment

```env
# Development
DEBUG=True
LOG_LEVEL=DEBUG
CORS_ORIGINS_STR=http://localhost:3000,http://localhost:8080
ALLOWED_HOSTS_STR=*

# Production
DEBUG=False
LOG_LEVEL=INFO
CORS_ORIGINS_STR=https://app.example.com
ALLOWED_HOSTS_STR=api.example.com,*.example.com
```

### Exception Hierarchy

All custom exceptions inherit from `DurianServiceException`:

| Exception | HTTP Status | Case |
|-----------|------------|-------|
| `ModelNotLoadedException` | 503 | Model not ready |
| `ModelLoadException` | 500 | Failed to load model |
| `InvalidImageException` | 400 | File is not a valid image |
| `UnsupportedFileTypeException` | 415 | Unsupported extension |
| `FileTooLargeException` | 413 | File size exceeds limit |
| `ImageProcessingException` | 422 | Failed preprocessing |
| `InferenceException` | 500 | Failed to run model |

---

Developed by **Erland Agsya**.
