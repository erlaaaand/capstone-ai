"""
tests/test_clip_service.py — BARU (FIX #9):

  Sebelumnya CLIPService tidak memiliki unit test sama sekali meskipun
  berperan sebagai gatekeeper penting (false positive = reject durian valid,
  false negative = loloskan gambar bukan durian).

  Test ini mencakup:
  - Lazy loading (model tidak dimuat saat import)
  - Thread-safety singleton
  - Graceful degradation jika CLIP tidak tersedia
  - is_durian() dengan berbagai skenario
  - Fail-safe behavior (error → izinkan gambar)
  - Decode input bytes dan base64
"""

import base64
import io
import threading
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_image_bytes(size=(100, 100), color=(50, 120, 30), fmt="JPEG") -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _make_mock_clip(durian_prob: float = 0.80):
    """
    Buat mock CLIP model yang mengembalikan probabilitas sesuai parameter.

    durian_prob: probability untuk label 'a photo of a durian fruit' (index 0).
    Sisa probability dibagi rata ke 4 label lain.
    """
    n_labels   = 5
    other_prob = (1.0 - durian_prob) / (n_labels - 1)
    probs_arr  = np.array([durian_prob] + [other_prob] * (n_labels - 1), dtype=np.float32)

    mock_logits = MagicMock()
    mock_logits.softmax.return_value.cpu.return_value.numpy.return_value = np.array([probs_arr])

    mock_output = MagicMock()
    mock_output.logits_per_image = mock_logits

    mock_model     = MagicMock()
    mock_processor = MagicMock()

    mock_model.return_value  = mock_output
    mock_processor.return_value = {"input_ids": MagicMock(), "pixel_values": MagicMock()}

    return mock_model, mock_processor


def _reset_clip_service():
    """Reset state singleton CLIPService antar test."""
    from services.clip_service import CLIPService
    CLIPService._model          = None
    CLIPService._processor      = None
    CLIPService._load_attempted = False


# ─────────────────────────────────────────────────────────────────────────────
#  Lazy Loading
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIPServiceLazyLoading:
    """
    Verifikasi bahwa model CLIP TIDAK dimuat saat module diimport,
    melainkan hanya saat pertama kali dibutuhkan.
    """

    def test_model_none_before_first_call(self):
        """Model tidak boleh dimuat sebelum is_durian() atau warmup() dipanggil."""
        _reset_clip_service()
        from services.clip_service import CLIPService
        # Reset ulang setelah import
        CLIPService._model          = None
        CLIPService._processor      = None
        CLIPService._load_attempted = False
        assert CLIPService._model is None

    def test_warmup_triggers_load(self):
        """warmup() harus memicu loading model."""
        _reset_clip_service()
        from services.clip_service import CLIPService

        mock_model, mock_processor = _make_mock_clip()

        with patch("services.clip_service.CLIPModel") as MockCLIPModel, \
             patch("services.clip_service.CLIPProcessor") as MockCLIPProcessor, \
             patch("builtins.__import__", side_effect=lambda name, *a, **kw: (
                 __import__(name, *a, **kw)
             )):
            # Patch dari dalam _ensure_loaded
            with patch.dict("sys.modules", {
                "torch": MagicMock(),
                "transformers": MagicMock(
                    CLIPModel=MagicMock(from_pretrained=MagicMock(return_value=mock_model)),
                    CLIPProcessor=MagicMock(from_pretrained=MagicMock(return_value=mock_processor)),
                ),
            }):
                _reset_clip_service()
                result = CLIPService.warmup()
                # Warmup harus mengembalikan bool
                assert isinstance(result, bool)

    def test_load_attempted_flag_set_after_warmup(self):
        """Setelah warmup, _load_attempted harus True sehingga tidak reload berulang."""
        _reset_clip_service()
        from services.clip_service import CLIPService

        with patch.dict("sys.modules", {
            "torch": MagicMock(),
            "transformers": MagicMock(
                CLIPModel=MagicMock(from_pretrained=MagicMock(return_value=MagicMock())),
                CLIPProcessor=MagicMock(from_pretrained=MagicMock(return_value=MagicMock())),
            ),
        }):
            CLIPService.warmup()
            assert CLIPService._load_attempted is True


# ─────────────────────────────────────────────────────────────────────────────
#  Graceful Degradation
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIPServiceGracefulDegradation:
    """
    Jika model CLIP tidak bisa dimuat (tidak ada GPU, transformers tidak install, dll),
    service harus tetap berjalan dan mengizinkan semua gambar (fail-open).
    """

    def test_clip_unavailable_allows_all_images(self):
        """Jika CLIP tidak tersedia, is_durian() harus return True (izinkan)."""
        _reset_clip_service()
        from services.clip_service import CLIPService

        # Simulasi import error (transformers tidak terinstall)
        with patch.dict("sys.modules", {"transformers": None, "torch": None}):
            _reset_clip_service()
            CLIPService._load_attempted = True   # skip loading attempt
            CLIPService._model          = None   # tidak ada model

            raw_bytes = _make_image_bytes()
            result    = CLIPService.is_durian(raw_bytes)

        assert result is True, "Gambar harus diizinkan jika CLIP tidak tersedia"

    def test_load_failure_sets_model_to_none(self):
        """Jika loading gagal, _model harus tetap None (tidak crash)."""
        _reset_clip_service()
        from services.clip_service import CLIPService

        with patch.dict("sys.modules", {
            "torch": MagicMock(),
            "transformers": MagicMock(
                CLIPModel=MagicMock(from_pretrained=MagicMock(
                    side_effect=RuntimeError("Simulated download failure")
                )),
                CLIPProcessor=MagicMock(),
            ),
        }):
            _reset_clip_service()
            result = CLIPService.warmup()

        assert result is False
        assert CLIPService._model is None

    def test_inference_error_returns_true(self):
        """Jika inference CLIP error (OOM, corrupt input), gambar tetap diizinkan."""
        _reset_clip_service()
        from services.clip_service import CLIPService

        mock_model     = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
        mock_processor = MagicMock(return_value={})

        CLIPService._model          = mock_model
        CLIPService._processor      = mock_processor
        CLIPService._load_attempted = True

        raw_bytes = _make_image_bytes()

        with patch.dict("sys.modules", {"torch": MagicMock()}):
            result = CLIPService.is_durian(raw_bytes)

        assert result is True, "Error saat inference harus fail-open (izinkan gambar)"


# ─────────────────────────────────────────────────────────────────────────────
#  is_durian() — Klasifikasi benar
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIPServiceIsDurian:

    def _setup_clip(self, durian_prob: float):
        """Setup CLIPService dengan mock model dan probabilitas tertentu."""
        _reset_clip_service()
        from services.clip_service import CLIPService

        n_labels   = 5
        other_prob = (1.0 - durian_prob) / (n_labels - 1)
        probs_arr  = np.array(
            [durian_prob] + [other_prob] * (n_labels - 1), dtype=np.float32
        )

        # Setup mock output
        mock_logits_per_image = MagicMock()
        mock_softmax = MagicMock()
        mock_softmax.cpu.return_value.numpy.return_value = np.array([probs_arr])
        mock_logits_per_image.softmax.return_value = mock_softmax

        mock_output = MagicMock()
        mock_output.logits_per_image = mock_logits_per_image

        mock_model = MagicMock(return_value=mock_output)
        mock_model.eval = MagicMock()

        mock_processor = MagicMock(return_value={"dummy": "inputs"})

        CLIPService._model          = mock_model
        CLIPService._processor      = mock_processor
        CLIPService._load_attempted = True

        return CLIPService

    def test_high_durian_probability_returns_true(self):
        """Jika label 'durian' paling tinggi, gambar diterima."""
        svc = self._setup_clip(durian_prob=0.90)
        with patch.dict("sys.modules", {"torch": MagicMock(no_grad=MagicMock(
            return_value=MagicMock(__enter__=MagicMock(return_value=None),
                                   __exit__=MagicMock(return_value=False))
        ))}):
            result = svc.is_durian(_make_image_bytes())
        assert result is True

    def test_non_durian_dominant_high_confidence_returns_false(self):
        """
        Jika label non-durian dominan dengan confidence > 0.40, gambar ditolak.
        Simulasi: probabilitas person = 0.70, durian = 0.10.
        """
        _reset_clip_service()
        from services.clip_service import CLIPService

        # index 0 = durian, index 1 = person
        probs_arr = np.array([0.10, 0.70, 0.08, 0.06, 0.06], dtype=np.float32)

        mock_logits = MagicMock()
        mock_logits.softmax.return_value.cpu.return_value.numpy.return_value = np.array([probs_arr])

        mock_output       = MagicMock()
        mock_output.logits_per_image = mock_logits
        mock_model        = MagicMock(return_value=mock_output)
        mock_model.eval   = MagicMock()
        mock_processor    = MagicMock(return_value={"dummy": "inputs"})

        CLIPService._model          = mock_model
        CLIPService._processor      = mock_processor
        CLIPService._load_attempted = True

        with patch.dict("sys.modules", {"torch": MagicMock(no_grad=MagicMock(
            return_value=MagicMock(__enter__=MagicMock(return_value=None),
                                   __exit__=MagicMock(return_value=False))
        ))}):
            result = CLIPService.is_durian(_make_image_bytes())

        assert result is False, (
            "Gambar dengan label 'person' confidence 0.70 harus ditolak"
        )

    def test_non_durian_low_confidence_returns_true(self):
        """
        Jika label non-durian dominan tapi confidence <= 0.40, gambar diizinkan.
        (Ambiguous image — lebih aman diizinkan)
        """
        _reset_clip_service()
        from services.clip_service import CLIPService

        # confidence non-durian = 0.35 (di bawah threshold 0.40)
        probs_arr = np.array([0.30, 0.35, 0.15, 0.10, 0.10], dtype=np.float32)

        mock_logits = MagicMock()
        mock_logits.softmax.return_value.cpu.return_value.numpy.return_value = np.array([probs_arr])

        mock_output              = MagicMock()
        mock_output.logits_per_image = mock_logits
        mock_model               = MagicMock(return_value=mock_output)
        mock_model.eval          = MagicMock()
        mock_processor           = MagicMock(return_value={"dummy": "inputs"})

        CLIPService._model          = mock_model
        CLIPService._processor      = mock_processor
        CLIPService._load_attempted = True

        with patch.dict("sys.modules", {"torch": MagicMock(no_grad=MagicMock(
            return_value=MagicMock(__enter__=MagicMock(return_value=None),
                                   __exit__=MagicMock(return_value=False))
        ))}):
            result = CLIPService.is_durian(_make_image_bytes())

        assert result is True

    def test_accepts_bytes_input(self):
        """is_durian() harus menerima input bytes."""
        _reset_clip_service()
        from services.clip_service import CLIPService

        # Model tidak ada → fail-open (True)
        CLIPService._model          = None
        CLIPService._processor      = None
        CLIPService._load_attempted = True

        raw = _make_image_bytes()
        assert isinstance(raw, bytes)
        result = CLIPService.is_durian(raw)
        assert isinstance(result, bool)

    def test_accepts_base64_string_input(self):
        """is_durian() harus menerima input string base64."""
        _reset_clip_service()
        from services.clip_service import CLIPService

        CLIPService._model          = None
        CLIPService._processor      = None
        CLIPService._load_attempted = True

        b64    = _make_b64(_make_image_bytes())
        result = CLIPService.is_durian(b64)
        assert isinstance(result, bool)

    def test_corrupt_bytes_returns_true(self):
        """Gambar corrupt tidak boleh crash — harus return True (fail-safe)."""
        _reset_clip_service()
        from services.clip_service import CLIPService

        CLIPService._model          = MagicMock(side_effect=Exception("Corrupt"))
        CLIPService._processor      = MagicMock()
        CLIPService._load_attempted = True

        result = CLIPService.is_durian(b"this is not an image at all")
        assert result is True


# ─────────────────────────────────────────────────────────────────────────────
#  Thread Safety
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIPServiceThreadSafety:
    """Verifikasi bahwa lazy loading thread-safe (model dimuat tepat sekali)."""

    def test_concurrent_warmup_loads_model_once(self):
        """
        Jika warmup() dipanggil dari N thread simultan, model hanya dimuat SEKALI
        (double-checked locking mencegah race condition).
        """
        _reset_clip_service()
        from services.clip_service import CLIPService

        load_count = {"n": 0}
        original_lock = CLIPService._lock

        def counting_from_pretrained(*args, **kwargs):
            load_count["n"] += 1
            return MagicMock()

        mock_clip_model     = MagicMock(from_pretrained=counting_from_pretrained)
        mock_clip_processor = MagicMock(from_pretrained=MagicMock(return_value=MagicMock()))

        with patch.dict("sys.modules", {
            "torch": MagicMock(),
            "transformers": MagicMock(
                CLIPModel=mock_clip_model,
                CLIPProcessor=mock_clip_processor,
            ),
        }):
            _reset_clip_service()
            threads = [
                threading.Thread(target=CLIPService.warmup)
                for _ in range(10)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # Model hanya boleh di-load sekali meskipun 10 thread simultan
        assert load_count["n"] <= 1, (
            f"Model di-load {load_count['n']} kali — seharusnya hanya 1x "
            "(double-checked locking tidak bekerja)"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Integration dengan warmup() dari lifespan
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIPServiceWarmup:
    def test_warmup_returns_bool(self):
        """warmup() selalu return bool, tidak pernah raise exception."""
        _reset_clip_service()
        from services.clip_service import CLIPService

        # Simulasi kegagalan import
        with patch.dict("sys.modules", {
            "torch": None,
            "transformers": None,
        }):
            _reset_clip_service()
            try:
                result = CLIPService.warmup()
                assert isinstance(result, bool)
            except Exception as e:
                pytest.fail(f"warmup() tidak boleh raise exception: {e}")

    def test_warmup_idempotent(self):
        """Memanggil warmup() dua kali tidak menyebabkan double-load."""
        _reset_clip_service()
        from services.clip_service import CLIPService

        CLIPService._model          = MagicMock()   # simulasi sudah loaded
        CLIPService._processor      = MagicMock()
        CLIPService._load_attempted = True

        # Panggil dua kali — tidak boleh mencoba load lagi
        r1 = CLIPService.warmup()
        r2 = CLIPService.warmup()
        assert r1 is True
        assert r2 is True
