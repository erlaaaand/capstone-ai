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
- [Pelatihan Model](#pelatihan-model)
- [Ekspor ke ONNX](#ekspor-ke-onnx)
- [Evaluasi Model](#evaluasi-model)
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
                                                     │  (8 Kelas Durian)    │
                                                     └──────────────────────┘
```

Setiap request gambar melewati tiga tahap utama:
1. **Validasi CLIP** — memastikan gambar memang berupa buah durian sebelum diproses lebih lanjut.
2. **Image Processing** — enhancement otomatis (white balance, CLAHE, sharpening) lalu resize ke 224×224.
3. **ONNX Inference** — model EfficientNetB0 menghasilkan probabilitas untuk 8 varietas.

---

## Fitur Utama

### 🚀 High-Performance Inference
ONNX Runtime digunakan sebagai backend inferensi, menggantikan TensorFlow native di production. Hasilnya: latensi lebih rendah dan footprint memori lebih kecil. Warmup otomatis saat startup memastikan request pertama tidak mengalami cold-start.

### 🖼️ Advanced Image Processing Pipeline
Enhancement gambar berjalan otomatis sebelum inferensi, dapat dikonfigurasi per-fitur melalui `.env`:
- **Auto White Balance** — koreksi cast warna akibat perbedaan kondisi pencahayaan.
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) — meningkatkan kontras lokal tanpa over-expose.
- **Unsharp Masking** — mempertegas tepi dan tekstur kulit durian.
- **Letterbox Resize** — menjaga aspek rasio gambar tanpa distorsi saat resize ke 224×224.

### 🛡️ Enterprise Security
- **API Key Authentication** menggunakan custom header `X-API-Key` atau `Authorization: Bearer`.
- Mendukung **multiple API keys** (hingga 20) dengan nama, scope, dan tier berbeda.
- **Zero-downtime key rotation**: tandai key lama sebagai `deprecated`, tambah key baru, hapus lama setelah client update.
- **Scopes**: `predict`, `health`, `admin`, `readonly`.
- **Rate Limiting** berbasis Sliding Window per API key, dengan burst protection (20 req/detik).
- Format key: `dk_live_<32char>` untuk production, `dk_test_<32char>` untuk testing.

### 🔄 Resilient Middleware Stack
Urutan middleware (dari luar ke dalam):
1. `PayloadSizeLimitMiddleware` — tolak payload melebihi batas sebelum dibaca.
2. `GZipMiddleware` — kompres respons ≥ 1KB.
3. `TrustedHostMiddleware` — proteksi host-header injection.
4. `CORSMiddleware` — whitelist origin.
5. `RequestLoggingMiddleware` — log setiap request/response dengan `request_id` unik.
6. `SecurityHeadersMiddleware` — inject security headers (HSTS, CSP, X-Frame-Options, dll.).

### 📋 Structured Logging
Semua log menggunakan format JSON (`JSONFormatter`) dan mencakup: timestamp UTC, level, nama modul, request_id, dan data tambahan. Audit log terpisah (`audit` logger) mencatat setiap auth success/failure, rate limit exceeded, penggunaan deprecated key, dan file mencurigakan.

### 🤖 CLIP-based Durian Validation
Sebelum masuk ke model klasifikasi, gambar divalidasi menggunakan **CLIP** (`openai/clip-vit-base-patch32`). Jika gambar bukan durian (misal: foto orang, kendaraan, pemandangan) dengan confidence > 40%, request langsung ditolak dengan pesan jelas.

---

## Varietas Durian yang Didukung

| Kode | Nama Populer | Nama Lokal | Asal |
|------|-------------|------------|------|
| D197 | Golden Phoenix | D197 / Jin Feng / Golden Phoenix | Malaysia |
| D24 | Sultan | D24 / Sultan | Malaysia |
| D198 | Red Prawn | D198 / Udang Merah / Red Prawn | Penang, Malaysia |
| D200 | Musang King | D200 / Musang King / Raja Kunyit / Mao Shan Wang | Kelantan / Gua Musang |
| D101 | Nyuk Kun | D101 / Nyuk Kun | Penang, Malaysia |
| D13 | Kuk San | D13 / Kuk San | Malaysia Barat |
| D2 | Chanee | D2 / Chanee | Thailand / Malaysia Utara |
| D88 | Tekka | D88 / Tekka | Johor, Malaysia |

> **Catatan:** Urutan kelas dalam model mengikuti urutan alfabetikal kode: `D101, D13, D197, D198, D2, D200, D24, D88`. Pastikan `CLASS_NAMES` di `.env` sesuai dengan urutan folder saat training.

---

## Struktur Proyek

```
backend_ai/
│
├── app/
│   ├── api/
│   │   ├── __init__.py          # Router aggregator (prefix /api/v1)
│   │   ├── health.py            # GET /ping (publik) & GET /health (protected)
│   │   └── routes.py            # POST /predict — endpoint inferensi utama
│   ├── core_dependencies.py     # verify_api_key, require_scope (FastAPI dependencies)
│   └── main.py                  # App factory, lifespan, middleware, exception handlers
│
├── core/
│   ├── config.py                # Settings (pydantic-settings), VARIETY_MAP, get_variety_info()
│   ├── exceptions.py            # DurianServiceException dan turunannya (7 tipe)
│   ├── logger.py                # JSONFormatter, setup_logging(), get_logger()
│   ├── middleware.py            # 4 middleware + AuditLogger
│   ├── rate_limiter.py          # SlidingWindowRateLimiter (async, in-memory)
│   └── security.py              # APIKeyManager, hash/verify key, AuthResult, KeyScope
│
├── models/
│   ├── model_loader.py          # ONNXModelLoader (singleton, thread-safe)
│   └── weights/                 # ← letakkan file .onnx di sini (tidak di-commit)
│
├── pipelines/
│   ├── ai_training_upgraded_2.ipynb  # Notebook training lengkap (Colab-ready)
│   ├── train.py                 # Script training CLI
│   ├── export_to_onnx.py        # Konversi .keras → .onnx
│   └── evaluate.py              # Evaluasi model: confusion matrix, per-class accuracy
│
├── schemas/
│   ├── request.py               # PredictionRequestBase64 (Pydantic)
│   └── response.py              # PredictionResponse, HealthResponse, ErrorResponse
│
├── services/
│   ├── clip_service.py          # CLIPService.is_durian() — validasi zero-shot
│   ├── image_processor.py       # ImageProcessor.process() — decode + enhance + resize
│   └── inference_service.py     # InferenceService.predict() — jalankan ONNX + format response
│
├── tests/
│   ├── test_api.py              # Unit & integration test endpoint FastAPI
│   └── test_inference.py        # Unit test ImageProcessor & InferenceService
│
├── data/
│   └── class_indices.json       # Pemetaan index → kode kelas
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
- **File model ONNX** yang sudah terlatih (lihat bagian [Pelatihan Model](#pelatihan-model) atau [Ekspor ke ONNX](#ekspor-ke-onnx))
- GPU opsional — ONNX Runtime otomatis menggunakan CUDA jika tersedia

---

## Instalasi & Konfigurasi

### 1. Clone & Buat Virtual Environment

```bash
git clone <repo-url>
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
CLASS_NAMES=D101,D13,D197,D198,D2,D200,D24,D88

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

# Atau jalankan pipeline export jika hanya punya .keras
python pipelines/export_to_onnx.py \
    --input models/weights/best_model.keras \
    --output models/weights/efficientnet_b0.onnx
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

Status detail service. Memerlukan API key dengan scope `health` atau `admin`.

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
  "rate_limiter_stats": { "tracked_identifiers": 12 },
  "config_summary": {
    "num_classes": 8,
    "image_size": 224,
    "enhancement": true,
    "max_file_size_mb": 10
  }
}
```

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

> Tidak boleh mengirim keduanya sekaligus.

**Response Sukses (200):**
```json
{
  "success": true,
  "prediction": {
    "variety_code": "D200",
    "variety_name": "Musang King",
    "local_name": "D200 / Musang King / Raja Kunyit / Mao Shan Wang",
    "origin": "Malaysia (Kelantan / Gua Musang)",
    "description": "Raja durian Malaysia dengan daging kuning-emas yang tebal...",
    "confidence_score": 0.9231
  },
  "all_varieties": [
    { "variety_code": "D200", "variety_name": "Musang King", "confidence_score": 0.9231 },
    { "variety_code": "D197", "variety_name": "Golden Phoenix", "confidence_score": 0.0412 }
  ],
  "confidence_scores": {
    "Musang King": 0.9231,
    "Golden Phoenix": 0.0412
  },
  "image_enhanced": true,
  "inference_time_ms": 18.5,
  "preprocessing_time_ms": 12.3,
  "model_version": "1.0.0",
  "request_id": "a3b1c2d4"
}
```

**Response Error:**

| Status | Kode Error | Penyebab |
|--------|-----------|---------|
| 400 | `InvalidImageException` | File kosong atau bukan gambar valid |
| 400 | - | Kirim file dan JSON sekaligus |
| 401 | - | Tidak ada API key |
| 403 | - | API key invalid atau scope tidak cukup |
| 413 | `FileTooLargeException` | File melebihi batas (default 10MB) |
| 415 | `UnsupportedFileTypeException` | Ekstensi tidak didukung (bukan jpg/png/webp) |
| 422 | `ImageProcessingException` | Gagal preprocessing gambar |
| 429 | - | Rate limit terlampaui |
| 503 | `ModelNotLoadedException` | Model belum ter-load saat startup |

**Response Headers Tambahan:**
- `X-Request-ID` — ID unik per request untuk tracing
- `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` — info rate limit
- `Warning: 299` — muncul jika API key sedang dalam status deprecated

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
# 2. Tandai key lama sebagai deprecated
API_KEY_1_DEPRECATED=true

# 3. Setelah semua client update ke key baru, hapus API_KEY_1 dari .env
```

Selama `deprecated=true`, key lama masih bisa digunakan tapi response akan menyertakan `Warning: 299` header.

### Rate Limit per Tier

| Tier | Limit | Burst |
|------|-------|-------|
| `free` | 60 req/menit | 20 req/detik |
| `standard` | 300 req/menit | 20 req/detik |
| `premium` | 1000 req/menit | 20 req/detik |
| `unlimited` | Tidak terbatas | 20 req/detik |

Identifier rate limit: `key:<prefix>` untuk autentikasi sukses, `ip:<client_ip>` untuk fallback (limit 30/menit).

### Keamanan Penyimpanan Key

API key **tidak disimpan plaintext** di memori. Sistem menggunakan **PBKDF2-HMAC-SHA256** dengan 100.000 iterasi dan salt acak 16 byte. Perbandingan hash menggunakan `hmac.compare_digest()` untuk mencegah timing attack.

---

## Pipeline Gambar

Setiap gambar melewati pipeline berikut di `ImageProcessor.process()`:

```
Input (bytes / base64)
        │
        ▼
   Decode & Verify      ← validasi header magic bytes
        │
        ▼
  Konversi ke RGB       ← RGBA, grayscale, dll → RGB
        │
        ▼
  Letterbox Resize      ← resize ke 224×224 dengan padding, aspek rasio terjaga
        │
        ▼
 Enhancement (opsional) ← dikendalikan ENABLE_ENHANCEMENT di .env
    ├── Auto White Balance
    ├── CLAHE
    └── Unsharp Masking
        │
        ▼
  Output: float32       ← shape (1, 224, 224, 3), range [0, 255]
  numpy tensor          ← EfficientNetB0 menangani normalisasi internal
```

> **Penting:** Tensor output berada dalam range **[0, 255]**, BUKAN [0, 1]. EfficientNetB0 yang digunakan `include_preprocessing=True` (default) menangani normalisasi internal saat training maupun inferensi.

Enhancement dapat dikonfigurasi per-komponen:
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

```
Input (224×224×3)
    → GPU Augmentation (RandomFlip, Rotation, Zoom, Translate, Contrast, Brightness, GaussianNoise)
    → EfficientNetB0 Backbone (ImageNet pretrained, frozen saat Phase 1)
    → GlobalAveragePooling2D
    → BN → Dropout(0.50)
    → Dense(512, GELU, L2=2e-4) → BN → Dropout(0.40)
    → Dense(256, GELU, L2=2e-4) → BN → Dropout(0.275)
    → Dense(128, GELU, L2=1e-4) → BN → Dropout(0.15)
    → Dense(8, Softmax, float32)
```

Total parameter: ~4.88 juta | Trainable (Phase 1): ~825 ribu

### Teknik Anti-Overfitting

- **Deduplication** via perceptual hash (pHash, threshold ≤ 8) sebelum training.
- **MixUp + CutMix** (50/50 per batch) untuk augmentasi data.
- **Label Smoothing** (ε = 0.12).
- **Cosine Annealing LR** di Phase 2.
- **Class Weighting** otomatis untuk dataset imbalanced.
- **OverfitMonitorCallback** — deteksi real-time gap train/val > 0.12.

### Hasil Training (v2)

| Fase | Best Val Accuracy |
|------|-------------------|
| Phase 1 (Feature Extraction, 20 epoch) | 55.44% |
| Phase 2 (Fine-Tuning, 45 epoch) | 72.12% |
| **TTA (n=5 augmentasi)** | **77.32%** |

---

## Pelatihan Model

### Menggunakan Notebook (Direkomendasikan)

Buka `pipelines/ai_training_upgraded_2.ipynb` di Google Colab (disarankan dengan GPU T4):

1. Upload dataset ke `/content/raw/` dengan struktur:
   ```
   raw/
   ├── train/
   │   ├── D101/  ← nama folder = kode kelas
   │   ├── D13/
   │   └── ...
   ├── valid/
   │   └── ...
   └── test/       ← opsional
       └── ...
   ```

2. Jalankan semua cell secara berurutan. Notebook akan:
   - Mendeteksi dan menghapus gambar duplikat otomatis.
   - Melatih model dalam dua fase.
   - Menghasilkan confusion matrix, Grad-CAM, training history.
   - Mengekspor model ke `.keras` dan TF SavedModel.
   - Mengemas semua output ke `.zip` untuk diunduh.

### Menggunakan Script CLI

```bash
python pipelines/train.py \
    --data_dir data/raw \
    --epochs_p1 20 \
    --epochs_p2 50 \
    --batch_size 32 \
    --mixed_prec     # tambahkan jika menggunakan GPU
```

---

## Ekspor ke ONNX

Setelah training selesai, konversi model ke ONNX:

```bash
python pipelines/export_to_onnx.py \
    --input  models/weights/best_model.keras \
    --output models/weights/efficientnet_b0.onnx \
    --opset  17
```

Script ini secara otomatis:
1. Memuat model `.keras` dan mengkonversi ke pure `float32` (menghapus `mixed_float16`).
2. Membangun **inference-only model** — augmentasi layer dicopot, hanya backbone + head tersisa.
3. Mengonversi ke ONNX via `tf2onnx`.
4. Memverifikasi output model menggunakan ONNX Runtime (warmup inference + cek output shape).

> Model ONNX yang benar berukuran ~18–20 MB (float32). Jika ~10 MB, ada masalah casting float16.

---

## Evaluasi Model

```bash
# Evaluasi standar
python pipelines/evaluate.py \
    --model    models/weights/best_model.keras \
    --test_dir data/raw/test

# Dengan Test-Time Augmentation (akurasi lebih tinggi)
python pipelines/evaluate.py \
    --model    models/weights/best_model.keras \
    --test_dir data/raw/test \
    --tta \
    --n_tta 5
```

Output yang dihasilkan (disimpan di `models/evaluation/`):
- `confusion_matrix_normalized.png` — heatmap confusion matrix ternormalisasi
- `confusion_matrix_counts.png` — heatmap dengan raw count
- `per_class_accuracy.png` — bar chart akurasi per kelas

---

## Testing

```bash
# Jalankan semua test
pytest tests/

# Dengan output verbose
pytest tests/ -v

# Test per file
pytest tests/test_api.py -v
pytest tests/test_inference.py -v
```

Cakupan test meliputi:

| Area | Yang Diuji |
|------|-----------|
| **API Endpoint** | Health check, autentikasi (missing/invalid key), validasi input (ekstensi, ukuran, format), response schema, semua 8 kelas di confidence_scores |
| **ImageProcessor** | Output shape (1,224,224,3), dtype float32, range [0,255], konversi mode (RGBA/grayscale→RGB), decode base64 dengan/tanpa prefix & padding, error handling |
| **InferenceService** | Top class benar, tidak ada double softmax, auto-softmax untuk logit output, class mismatch exception, invalid input shape |

---

## Konfigurasi Lanjutan

Semua konfigurasi dikelola via `.env`. Lihat `.env.example` untuk daftar lengkap.

### Variabel Penting

| Variabel | Default | Keterangan |
|----------|---------|-----------|
| `DEBUG` | `False` | Aktifkan `/docs`, `/redoc`, `/openapi.json` |
| `LOG_LEVEL` | `INFO` | DEBUG/INFO/WARNING/ERROR/CRITICAL |
| `MODEL_PATH` | `models/weights/efficientnet_b0.onnx` | Path ke file model |
| `CLASS_NAMES` | `D101,...` | Urutan kelas (WAJIB alfabetikal) |
| `IMAGE_SIZE` | `224` | Ukuran input model (32–1024) |
| `MAX_FILE_SIZE_MB` | `10` | Batas ukuran file upload |
| `ENABLE_ENHANCEMENT` | `True` | Master switch enhancement pipeline |
| `CORS_ORIGINS_STR` | — | Comma-separated allowed origins |
| `ALLOWED_HOSTS_STR` | `*` | Allowed hosts (anti host-header injection) |

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

---

## Lisensi

Developed for the **Erland Agsya**.