"""
Tests untuk models/model_loader.py — ONNXModelLoader singleton.
"""
import threading
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

import numpy as np
import pytest

from core.exceptions import ModelLoadException, ModelNotLoadedException


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fresh_loader():
    """Reset singleton dan kembalikan instance baru."""
    from models.model_loader import ONNXModelLoader
    ONNXModelLoader._instance = None
    return ONNXModelLoader()


def _mock_ort_session(num_classes: int = 8, img_size: int = 224):
    """Buat onnxruntime.InferenceSession palsu."""
    session = MagicMock()

    inp       = MagicMock()
    inp.name  = "image_input"
    inp.shape = [None, img_size, img_size, 3]

    out       = MagicMock()
    out.name  = "predictions"
    out.shape = [None, num_classes]

    session.get_inputs.return_value  = [inp]
    session.get_outputs.return_value = [out]

    probs = np.ones((1, num_classes), dtype=np.float32) / num_classes
    session.run.return_value = [probs]

    return session


# ─────────────────────────────────────────────────────────────────────────────
#  ONNXModelLoader — singleton
# ─────────────────────────────────────────────────────────────────────────────

class TestONNXModelLoaderSingleton:
    def test_same_instance_returned(self):
        from models.model_loader import get_model_loader
        l1 = get_model_loader()
        l2 = get_model_loader()
        assert l1 is l2

    def test_singleton_thread_safe(self):
        from models.model_loader import ONNXModelLoader
        ONNXModelLoader._instance = None
        instances = []

        def create():
            instances.append(ONNXModelLoader())

        threads = [threading.Thread(target=create) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Semua harus instance yang sama
        assert len(set(id(i) for i in instances)) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  ONNXModelLoader — load_model
# ─────────────────────────────────────────────────────────────────────────────

class TestONNXModelLoaderLoad:
    def test_load_nonexistent_file_raises_model_load_exception(self, tmp_path):
        loader = _fresh_loader()
        with pytest.raises(ModelLoadException):
            loader.load_model(str(tmp_path / "nonexistent.onnx"))

    def test_load_success_sets_is_loaded_true(self, tmp_path):
        fake_model = tmp_path / "model.onnx"
        fake_model.write_bytes(b"fake onnx")

        loader  = _fresh_loader()
        session = _mock_ort_session()

        with patch("onnxruntime.InferenceSession", return_value=session), \
             patch("onnxruntime.SessionOptions"), \
             patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            loader.load_model(str(fake_model))

        assert loader.is_loaded is True

    def test_load_success_sets_input_name(self, tmp_path):
        fake_model = tmp_path / "model.onnx"
        fake_model.write_bytes(b"fake onnx")
        loader  = _fresh_loader()
        session = _mock_ort_session()

        with patch("onnxruntime.InferenceSession", return_value=session), \
             patch("onnxruntime.SessionOptions"), \
             patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            loader.load_model(str(fake_model))

        assert loader.input_name == "image_input"

    def test_load_success_sets_output_name(self, tmp_path):
        fake_model = tmp_path / "model.onnx"
        fake_model.write_bytes(b"fake onnx")
        loader  = _fresh_loader()
        session = _mock_ort_session()

        with patch("onnxruntime.InferenceSession", return_value=session), \
             patch("onnxruntime.SessionOptions"), \
             patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            loader.load_model(str(fake_model))

        assert loader.output_name == "predictions"

    def test_class_mismatch_raises_model_load_exception(self, tmp_path):
        """Model dengan output berbeda dari CLASS_NAMES harus raise ModelLoadException."""
        fake_model = tmp_path / "model.onnx"
        fake_model.write_bytes(b"fake onnx")
        loader  = _fresh_loader()
        session = _mock_ort_session(num_classes=4)  # 4 kelas, config punya 8

        with patch("onnxruntime.InferenceSession", return_value=session), \
             patch("onnxruntime.SessionOptions"), \
             patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            with pytest.raises(ModelLoadException) as exc_info:
                loader.load_model(str(fake_model))
        assert "MISMATCH" in exc_info.value.detail.upper() or "mismatch" in exc_info.value.detail.lower()

    def test_ort_exception_raises_model_load_exception(self, tmp_path):
        fake_model = tmp_path / "model.onnx"
        fake_model.write_bytes(b"fake onnx")
        loader = _fresh_loader()

        with patch("onnxruntime.InferenceSession", side_effect=RuntimeError("ONNX parse error")), \
             patch("onnxruntime.SessionOptions"), \
             patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            with pytest.raises(ModelLoadException):
                loader.load_model(str(fake_model))

    def test_failed_load_leaves_is_loaded_false(self, tmp_path):
        loader = _fresh_loader()
        try:
            loader.load_model(str(tmp_path / "nonexistent.onnx"))
        except ModelLoadException:
            pass
        assert loader.is_loaded is False


# ─────────────────────────────────────────────────────────────────────────────
#  ONNXModelLoader — properties saat belum loaded
# ─────────────────────────────────────────────────────────────────────────────

class TestONNXModelLoaderNotLoaded:
    def test_session_raises_when_not_loaded(self):
        loader = _fresh_loader()
        with pytest.raises(ModelNotLoadedException):
            _ = loader.session

    def test_input_name_raises_when_not_loaded(self):
        loader = _fresh_loader()
        with pytest.raises(ModelNotLoadedException):
            _ = loader.input_name

    def test_output_name_raises_when_not_loaded(self):
        loader = _fresh_loader()
        with pytest.raises(ModelNotLoadedException):
            _ = loader.output_name

    def test_is_loaded_false_by_default(self):
        loader = _fresh_loader()
        assert loader.is_loaded is False


# ─────────────────────────────────────────────────────────────────────────────
#  ONNXModelLoader — unload_model
# ─────────────────────────────────────────────────────────────────────────────

class TestONNXModelLoaderUnload:
    def test_unload_sets_is_loaded_false(self, tmp_path):
        fake_model = tmp_path / "model.onnx"
        fake_model.write_bytes(b"fake onnx")
        loader  = _fresh_loader()
        session = _mock_ort_session()

        with patch("onnxruntime.InferenceSession", return_value=session), \
             patch("onnxruntime.SessionOptions"), \
             patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            loader.load_model(str(fake_model))

        assert loader.is_loaded is True
        loader.unload_model()
        assert loader.is_loaded is False

    def test_unload_when_not_loaded_no_crash(self):
        loader = _fresh_loader()
        # Tidak boleh raise exception
        loader.unload_model()
        assert loader.is_loaded is False

    def test_session_raises_after_unload(self, tmp_path):
        fake_model = tmp_path / "model.onnx"
        fake_model.write_bytes(b"fake onnx")
        loader  = _fresh_loader()
        session = _mock_ort_session()

        with patch("onnxruntime.InferenceSession", return_value=session), \
             patch("onnxruntime.SessionOptions"), \
             patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            loader.load_model(str(fake_model))

        loader.unload_model()

        with pytest.raises(ModelNotLoadedException):
            _ = loader.session
