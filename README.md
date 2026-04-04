# Enterprise Durian Classification API

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-005C8A?style=for-the-badge&logo=onnx&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

Sistem backend AI berbasis **FastAPI** untuk mengklasifikasikan varietas durian dari gambar menggunakan deep learning. Model **EfficientNetB0** diekspor ke format **ONNX** untuk inferensi berperforma tinggi. Dilengkapi lapisan keamanan enterprise, pipeline enhancement gambar otomatis, dan validasi gambar berbasis CLIP (zero-shot).

---

## Daftar Isi

- [Arsitektur Sistem](#arsitektur-sistem)
- [Fitur Utama](#fitur-utama)
- [Varietas Durian yang Didukung](#varietas-durian-yang-didukung)
- [Struktur Proyek](#struktur-proyek)
- [Prasyarat](#prasyarat)
- [Instalasi & Konfigurasi](#instalasi--konfigurasi)
- [Menjalankan Aplikasi](#menjalankan-aplikasi)
- [Referensi API](#referensi-api)
- [Keamanan & Autentikasi](#keamanan--autentikasi)
- [Pipeline Gambar](#pipeline-gambar)
- [Model Deep Learning](#model-deep-learning)
- [Testing](#testing)
- [Konfigurasi Lanjutan](#konfigurasi-lanjutan)

---

## Arsitektur Sistem

```
Request (Gambar)
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
                                                     │  (6 Kelas Durian)    │
                                                     └──────────────────────┘
```

Setiap request gambar melewati tiga tahap utama:
1. **Validasi CLIP** — memastikan gambar memang berupa buah durian sebelum diproses lebih lanjut. CLIP dan Image Processing dijalankan secara **paralel** (`asyncio.gather`) untuk mempercepat response.
2. **Image Processing** — enhancement otomatis (white balance, CLAHE, sharpening) lalu letterbox resize ke 224×224.
3. **ONNX Inference** — model EfficientNetB0 menghasilkan probabilitas untuk 6 varietas.

---

## Fitur Utama

### 🚀 High-Performance Inference
ONNX Runtime digunakan sebagai backend inferensi, menggantikan TensorFlow native di production. Hasilnya: latensi lebih rendah dan footprint memori lebih kecil. Warmup otomatis saat startup memastikan request pertama tidak mengalami cold-start. Session dikonfigurasi dengan graph optimization level `ORT_ENABLE_ALL` dan thread pooling otomatis.

### 🖼️ Advanced Image Processing Pipeline
Enhancement gambar berjalan otomatis sebelum inferensi, dapat dikonfigurasi per-fitur melalui `.env`:
- **Auto White Balance** — koreksi cast warna akibat perbedaan kondisi pencahayaan.
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) — meningkatkan kontras lokal tanpa over-expose (diimplementasi via histogram equalization pada channel Y dari YCbCr).
- **Unsharp Masking** — mempertegas tepi dan tekstur kulit durian.
- **Letterbox Resize** — menjaga aspek rasio gambar tanpa distorsi saat resize ke 224×224, dengan padding warna `(114, 114, 114)`.

### 🛡️ Enterprise Security
- **API Key Authentication** menggunakan custom header `X-API-Key` atau `Authorization: Bearer`.
- Mendukung **multiple API keys** (hingga 19 key, `API_KEY_1` s.d. `API_KEY_19`) dengan nama, scope, dan tier berbeda.
- **Zero-downtime key rotation**: tandai key lama sebagai `deprecated`, tambah key baru, hapus lama setelah client update.
- **Hot-reload keys** via `POST /api/v1/admin/reload-keys` tanpa restart server.
- **Scopes**: `predict`, `health`, `admin`, `readonly`.
- **Rate Limiting** berbasis Sliding Window per API key, dengan burst protection (20 req/detik).
- **Key Hashing**: API key disimpan sebagai hash PBKDF2-HMAC-SHA256 (100.000 iterasi, salt acak 16 byte). Validasi menggunakan `hmac.compare_digest()` untuk mencegah timing attack.
- Format key: `dk_live_<32char>` untuk production, `dk_test_<32char>` untuk testing.

### 🔄 Resilient Middleware Stack
Urutan middleware (dari luar ke dalam):
1. `PayloadSizeLimitMiddleware` — tolak payload melebihi batas sebelum dibaca (cek `Content-Length` header).
2. `GZipMiddleware` — kompres respons ≥ 1KB.
3. `TrustedHostMiddleware` — proteksi host-header injection (aktif jika `ALLOWED_HOSTS_STR` ≠ `*`).
4. `CORSMiddleware` — whitelist origin, methods `POST/GET/OPTIONS`, expose rate limit & request ID headers.
5. `RequestLoggingMiddleware` — log setiap request/response dengan `request_id` UUID unik, elapsed time, IP, dan user agent.
6. `SecurityHeadersMiddleware` — inject 12 security headers (HSTS, CSP, X-Frame-Options, COOP, CORP, dll.).

### 📋 Structured Logging
Semua log menggunakan format JSON (`JSONFormatter`) dan mencakup: timestamp UTC, level, nama modul, function, line number, dan data tambahan. `AuditLogger` terpisah (`audit` logger) mencatat setiap auth success/failure, rate limit exceeded, penggunaan deprecated key, dan file mencurigakan.

### 🤖 CLIP-based Durian Validation
Sebelum masuk ke model klasifikasi, gambar divalidasi menggunakan **CLIP** (`openai/clip-vit-base-patch32`) secara zero-shot. Gambar diklasifikasikan terhadap 5 label:
1. `a photo of a durian fruit`
2. `a photo of a person`
3. `a photo of an animal`
4. `a photo of a vehicle`
5. `a photo of random objects or scenery`

Jika label non-durian dominan dengan confidence > 40%, request ditolak. CLIP bersifat **fail-open**: jika model gagal dimuat atau error saat inferensi, semua gambar tetap diizinkan.

---

## Varietas Durian yang Didukung

| Kode | Nama Populer | Nama Lokal                                       | Asal                        |
| ---- | ------------ | ------------------------------------------------ | --------------------------- |
| D2   | Dato Nina    | D2 / Dato Nina                                   | Malaysia (Melaka)           |
| D13  | Golden Bun   | D13 / Golden Bun                                 | Malaysia (Johor)            |
| D24  | Sultan       | D24 / Sultan / Bukit Merah                       | Malaysia (Perak / Selangor) |
| D101 | Muar Gold    | D101 / Muar Gold / Johor Mas                     | Malaysia (Johor)            |
| D197 | Musang King  | D197 / Musang King / Raja Kunyit / Mao Shan Wang | Malaysia (Kelantan)         |
| D200 | Black Thorn  | D200 / Ochee / Duri Hitam / Black Thorn          | Malaysia (Penang)           |

> **Catatan:** Urutan kelas dalam model mengikuti urutan indeks di `data/class_indices.json`: `D101(0), D13(1), D197(2), D2(3), D200(4), D24(5)`. Pastikan `CLASS_NAMES` di `.env` sesuai dengan urutan folder saat training (alfabetikal).

---

## Struktur Proyek

```
backend_ai/
│
├── app/
│   ├── api/
│   │   ├── __init__.py          # Router aggregator (prefix /api/v1)
│   │   ├── health.py            # GET /ping, GET /health, POST /admin/reload-keys
│   │   └── routes.py            # POST /predict — endpoint inferensi utama
│   ├── core_dependencies.py     # verify_api_key, require_scope (FastAPI dependencies)
│   └── main.py                  # App factory, lifespan, middleware, exception handlers
│
├── core/
│   ├── __init__.py              # Re-export settings, logger, semua exception classes
│   ├── config.py                # Settings (pydantic-settings), VARIETY_MAP, get_variety_info()
│   ├── exceptions.py            # DurianServiceException dan 6 turunannya
│   ├── logger.py                # JSONFormatter, setup_logging(), get_logger()
│   ├── middleware.py            # SecurityHeaders, RequestLogging, PayloadSizeLimit, AuditLogger
│   ├── rate_limiter.py          # SlidingWindowRateLimiter (async, in-memory, background cleanup)
│   └── security.py             # APIKeyManager, hash/verify key, AuthResult, KeyScope, RateLimitTier
│
├── models/
│   ├── model_loader.py          # ONNXModelLoader (singleton, thread-safe, warmup inference)
│   └── weights/                 # ← letakkan file .onnx di sini (tidak di-commit)
│
├── schemas/
│   ├── __init__.py              # Re-export semua schema
│   ├── request.py               # PredictionRequestBase64 (Pydantic, auto-strip data URI prefix)
│   └── response.py              # PredictionResponse, PredictionResult, VarietyScore, HealthResponse, ErrorResponse
│
├── services/
│   ├── __init__.py              # Re-export ImageProcessor, InferenceService
│   ├── clip_service.py          # CLIPService.is_durian() — validasi zero-shot (lazy loading, thread-safe)
│   ├── image_processor.py       # ImageProcessor.process() — decode + enhance + letterbox resize
│   └── inference_service.py     # InferenceService.predict() — jalankan ONNX + format response
│
├── tests/
│   └── test_clip_service.py     # Unit test CLIPService (lazy loading, thread safety, graceful degradation)
│
├── data/
│   └── class_indices.json       # Pemetaan index → kode kelas (6 kelas)
│
├── .env.example                 # Template konfigurasi (wajib dibaca sebelum setup)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Prasyarat

- **Python 3.9+**
- **Virtual environment** (sangat direkomendasikan)
- **File model ONNX** yang sudah terlatih (`.onnx` format)
- GPU opsional — ONNX Runtime otomatis menggunakan CUDA jika tersedia (`CUDAExecutionProvider`)

---

## Instalasi & Konfigurasi

### 1. Clone & Buat Virtual Environment

```bash
git clone https://github.com/erlaaaand/capstone-ai.git
cd backend_ai

python -m venv venv

# Aktivasi (Linux/Mac)
source venv/bin/activate

# Aktivasi (Windows)
.\venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Konfigurasi Environment

```bash
cp .env.example .env
```

Buka `.env` dan sesuaikan nilai-nilai berikut:

```env
# Path ke file model ONNX yang sudah diekspor
MODEL_PATH=models/weights/efficientnet_b0.onnx

# Urutan kelas WAJIB sesuai urutan folder training (alfabetikal)
CLASS_NAMES=D101,D13,D197,D2,D200,D24

# Generate API key baru:
# python -c "import secrets; print('dk_live_' + secrets.token_urlsafe(24))"
API_KEY_1=dk_live_<hasil_generate>
API_KEY_1_NAME=Frontend App
API_KEY_1_SCOPES=predict,health
API_KEY_1_TIER=standard

# Tambah key kedua untuk backend internal (scope admin)
API_KEY_2=dk_live_<hasil_generate>
API_KEY_2_NAME=Internal Backend
API_KEY_2_SCOPES=predict,health,admin
API_KEY_2_TIER=premium

# CORS origins yang diizinkan (jangan gunakan * di production)
CORS_ORIGINS_STR=https://app.example.com,https://admin.example.com
```

### 4. Siapkan File Model

Letakkan file `.onnx` hasil training ke direktori `models/weights/`:

```bash
# Contoh jika model sudah ada
cp /path/to/efficientnet_b0.onnx models/weights/
```

---

## Menjalankan Aplikasi

### Development (dengan hot-reload)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Akses Swagger UI (hanya jika `DEBUG=True` di `.env`):

```
http://localhost:8000/docs
```

Cek status API tanpa auth:

```
GET http://localhost:8000/api/v1/ping
```

### Info Root Endpoint

```
GET http://localhost:8000/
```
Response berisi overview endpoint dan status service (tanpa auth).

---

## Referensi API

Semua endpoint di bawah `/api/v1/` memerlukan header `X-API-Key` (kecuali `/ping`).

### `GET /api/v1/ping`

Liveness check publik tanpa autentikasi. Digunakan untuk load balancer.

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

Status detail service. Memerlukan API key valid (scope apapun).

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

> **Catatan:** Status `"degraded"` dikembalikan jika model ONNX belum ter-load.

---

### `POST /api/v1/predict`

Klasifikasi varietas durian dari gambar. Memerlukan scope `predict` atau `admin`.

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

> Tidak boleh mengirim keduanya sekaligus. Data URI prefix (`data:image/jpeg;base64,`) otomatis dicopot oleh validator.

**Response Sukses (200):**
```json
{
  "success": true,
  "prediction": {
    "variety_code": "D197",
    "variety_name": "Musang King",
    "local_name": "D197 / Musang King / Raja Kunyit / Mao Shan Wang",
    "origin": "Malaysia (Kelantan)",
    "description": "Raja durian Malaysia dengan daging kuning-emas tebal...",
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

**Response Error:**

| Status | Kode Error | Penyebab |
|--------|-----------|---------|
| 400 | `InvalidImageException` | File kosong, bukan gambar valid, atau bukan gambar durian (CLIP) |
| 400 | - | Kirim file dan JSON sekaligus / data gambar kosong |
| 401 | - | Tidak ada API key |
| 403 | - | API key invalid atau scope tidak cukup |
| 413 | `FileTooLargeException` | File melebihi batas (default 10MB) |
| 415 | `UnsupportedFileTypeException` | Ekstensi tidak didukung (bukan jpg/jpeg/png/webp) |
| 422 | `ImageProcessingException` | Gagal preprocessing gambar |
| 429 | - | Rate limit terlampaui |
| 500 | `InferenceException` | Gagal menjalankan inferensi model |
| 503 | `ModelNotLoadedException` | Model belum ter-load saat startup |

**Response Headers Tambahan:**
- `X-Request-ID` — UUID unik per request untuk tracing
- `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, `X-RateLimit-Policy` — info rate limit
- `Warning: 299` — muncul jika API key sedang dalam status deprecated
- `X-API-Version` — versi API saat ini

---

### `POST /api/v1/admin/reload-keys`

Hot-reload API keys dari environment variables tanpa restart server. Memerlukan scope `admin`.

**Headers:**
```
X-API-Key: dk_live_xxx  (key dengan scope admin)
```

**Response Sukses:**
```json
{
  "success": true,
  "message": "API keys dan settings berhasil di-reload.",
  "key_count": 3,
  "reloaded_by": "dk_live_xxxx...",
  "app_version": "1.0.0",
  "settings_refreshed": true
}
```

> Endpoint ini menjalankan `load_dotenv(override=True)` sehingga perubahan di file `.env` langsung diterapkan. `Settings` cache juga di-refresh.

---

## Keamanan & Autentikasi

### Struktur API Key

```
dk_live_<random_32_karakter>   ← production
dk_test_<random_32_karakter>   ← testing / CI-CD
```

### Generate API Key Baru

```python
import secrets
live = 'dk_live_' + secrets.token_urlsafe(24)
test = 'dk_test_' + secrets.token_urlsafe(24)
print('LIVE:', live)
print('TEST:', test)
```

Atau via bash:
```bash
python -c "import secrets; print('dk_live_' + secrets.token_urlsafe(24))"
```

### Rotasi Key (Zero-Downtime)

```bash
# 1. Generate key baru dan tambahkan ke .env sebagai API_KEY_4
# 2. Panggil POST /api/v1/admin/reload-keys (scope admin)
# 3. Tandai key lama sebagai deprecated:
API_KEY_1_DEPRECATED=true
# 4. Panggil reload-keys lagi
# 5. Setelah semua client update ke key baru, hapus API_KEY_1 dari .env
```

Selama `deprecated=true`, key lama masih bisa digunakan tapi response akan menyertakan `Warning: 299` header.

### Rate Limit per Tier

| Tier | Limit | Burst |
|------|-------|-------|
| `free` | 60 req/menit | 20 req/detik |
| `standard` | 300 req/menit | 20 req/detik |
| `premium` | 1000 req/menit | 20 req/detik |
| `unlimited` | ~Tidak terbatas | 20 req/detik |

Identifier rate limit: `key:<prefix>` untuk autentikasi sukses, `ip:<client_ip>` untuk fallback (limit 30/menit).

Background cleanup task berjalan setiap 5 menit untuk menghapus stale entries (tidak aktif > 10 menit).

### Keamanan Penyimpanan Key

API key **tidak disimpan plaintext** di memori. Sistem menggunakan **PBKDF2-HMAC-SHA256** dengan 100.000 iterasi dan salt acak 16 byte. Perbandingan hash menggunakan `hmac.compare_digest()` untuk mencegah timing attack. Lookup menggunakan key prefix sebagai index dictionary untuk performa O(1).

### Key Expiration (Opsional)

Setiap key dapat dikonfigurasi dengan waktu kadaluarsa menggunakan Unix timestamp:

```env
API_KEY_1_EXPIRES_AT=1767225600
```

---

## Pipeline Gambar

Setiap gambar melewati pipeline berikut di `ImageProcessor.process()`:

```
Input (bytes / base64 string)
        │
        ▼
   Decode & Verify      ← PIL verify() + re-open
        │
        ▼
  Konversi ke RGB       ← RGBA, grayscale, dll → RGB
        │
        ▼
  Letterbox Resize      ← resize ke 224×224 dengan padding (114,114,114), aspek rasio terjaga
        │
        ▼
 Enhancement (opsional) ← dikendalikan ENABLE_ENHANCEMENT di .env
    ├── Auto White Balance    (gray world assumption)
    ├── CLAHE                 (histogram EQ pada Y channel YCbCr, alpha blending)
    └── Unsharp Masking       (GaussianBlur radius=2, amount=0.45)
        │
        ▼
  Output: float32       ← shape (1, 224, 224, 3), range [0, 255]
  numpy tensor          ← EfficientNetB0 menangani normalisasi internal
```

> **Penting:** Tensor output berada dalam range **[0, 255]**, BUKAN [0, 1]. EfficientNetB0 yang memiliki preprocessing layer internal menangani normalisasi saat inferensi.

### Magic Bytes Validation

Sebelum processing, file upload divalidasi menggunakan magic bytes:
- **JPEG**: `\xff\xd8\xff`
- **PNG**: `\x89PNG\r\n\x1a\n`
- **WebP**: `RIFF....WEBP`

File yang ekstensinya tidak cocok dengan magic bytes akan ditolak dan di-log sebagai `SUSPICIOUS_FILE`.

### Konfigurasi Enhancement

```env
ENABLE_ENHANCEMENT=True
ENABLE_WHITE_BALANCE=True
ENABLE_CLAHE=True
ENABLE_SHARPENING=True
CLAHE_CLIP_LIMIT=2.0   # Kekuatan CLAHE: 1.0–4.0
```

---

## Model Deep Learning

### Arsitektur

Model menggunakan **EfficientNetB0** sebagai backbone dengan custom classification head:

```
Input (224×224×3)
    → EfficientNetB0 Backbone (ImageNet pretrained)
    → GlobalAveragePooling2D
    → Dense Head → Dense(6, Softmax)
```

Output: probabilitas untuk 6 kelas durian (D101, D13, D197, D2, D200, D24).

### ONNX Model

Model disimpan dalam format **ONNX** (Open Neural Network Exchange) untuk:
- Inferensi lintas platform tanpa dependensi TensorFlow
- Optimasi graph otomatis via ONNX Runtime
- Dukungan GPU (CUDA) dan CPU otomatis

Spesifikasi model:
- Input: `float32[1, 224, 224, 3]`
- Output: `float32[1, 6]` (probabilitas softmax)
- Ukuran file: ~18–20 MB (float32)

### Auto-Softmax

`InferenceService` secara otomatis mendeteksi dan menangani output model:
- Jika jumlah probabilitas ≈ 1.0 → gunakan langsung (output sudah softmax)
- Jika jumlah probabilitas jauh dari 1.0 → terapkan softmax otomatis (output masih logit)

---

## Testing

```bash
# Jalankan semua test
pytest tests/

# Dengan output verbose
pytest tests/ -v

# Test per file
pytest tests/test_clip_service.py -v
```

### Cakupan Test CLIP Service

| Area | Yang Diuji |
|------|-----------| 
| **Lazy Loading** | Model tidak dimuat saat import, dimuat saat `warmup()` / `is_durian()` pertama kali, `_load_attempted` flag mencegah reload berulang |
| **Graceful Degradation** | CLIP tidak tersedia → izinkan semua gambar (fail-open), kegagalan loading → `_model` tetap `None`, error saat inference → return `True` |
| **Klasifikasi** | Durian probability tinggi → `True`, non-durian dominan (>0.40) → `False`, non-durian rendah (≤0.40) → `True` (fail-open) |
| **Input Handling** | Menerima `bytes` dan `base64` string, gambar corrupt → fail-safe (return `True`) |
| **Thread Safety** | 10 thread simultan memanggil `warmup()` → model hanya dimuat 1x (double-checked locking) |
| **Warmup Integration** | `warmup()` selalu return `bool` (tidak pernah raise exception), idempotent (aman dipanggil berulang) |

---

## Konfigurasi Lanjutan

Semua konfigurasi dikelola via `.env`. Lihat `.env.example` untuk daftar lengkap.

### Variabel Penting

| Variabel | Default | Keterangan |
|----------|---------|-----------|
| `DEBUG` | `False` | Aktifkan `/docs`, `/redoc`, `/openapi.json` |
| `LOG_LEVEL` | `INFO` | DEBUG/INFO/WARNING/ERROR/CRITICAL |
| `MODEL_PATH` | `models/weights/efficientnet_b0.onnx` | Path ke file model |
| `CLASS_NAMES` | `D101,D13,D197,D2,D200,D24` | Urutan kelas (WAJIB sesuai folder training) |
| `IMAGE_SIZE` | `224` | Ukuran input model (32–1024) |
| `MAX_FILE_SIZE_MB` | `10` | Batas ukuran file upload |
| `ALLOWED_EXTENSIONS` | `jpg,jpeg,png,webp` | Ekstensi file yang diizinkan |
| `ENABLE_ENHANCEMENT` | `True` | Master switch enhancement pipeline |
| `ENABLE_WHITE_BALANCE` | `True` | Auto white balance |
| `ENABLE_CLAHE` | `True` | Contrast adaptive histogram equalization |
| `ENABLE_SHARPENING` | `True` | Unsharp masking |
| `CLAHE_CLIP_LIMIT` | `2.0` | Kekuatan CLAHE (1.0–4.0) |
| `CORS_ORIGINS_STR` | `http://localhost:3000,...` | Comma-separated allowed origins |
| `ALLOWED_HOSTS_STR` | `*` | Allowed hosts (anti host-header injection) |
| `API_KEY_REQUIRED` | `True` | Wajibkan API key |

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

Semua custom exception mewarisi `DurianServiceException`:

| Exception | HTTP Status | Kasus |
|-----------|------------|-------|
| `ModelNotLoadedException` | 503 | Model belum siap |
| `ModelLoadException` | 500 | Gagal memuat model |
| `InvalidImageException` | 400 | File bukan gambar valid |
| `UnsupportedFileTypeException` | 415 | Ekstensi tidak didukung |
| `FileTooLargeException` | 413 | Ukuran file melebihi batas |
| `ImageProcessingException` | 422 | Gagal preprocessing |
| `InferenceException` | 500 | Gagal menjalankan model |

---

Developed by **Erland Agsya**.
