"""
Tests untuk services/inference_service.py — InferenceService.predict().
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from core.exceptions import InferenceException, ModelNotLoadedException
from core.security import KeyScope


CLASS_NAMES = ["D101", "D13", "D197", "D198", "D2", "D200", "D24", "D88"]
NUM_CLASSES = 8


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tensor(h=224, w=224) -> np.ndarray:
    return np.random.uniform(0, 255, (1, h, w, 3)).astype(np.float32)


def _loader(probs: np.ndarray) -> MagicMock:
    session = MagicMock()
    session.run.return_value = [probs.reshape(1, -1)]
    loader  = MagicMock()
    loader.session     = session
    loader.input_name  = "image_input"
    loader.output_name = "predictions"
    loader.is_loaded   = True
    return loader


def _peaked(top_idx=5, conf=0.92) -> np.ndarray:
    probs = np.full(NUM_CLASSES, (1.0 - conf) / (NUM_CLASSES - 1), dtype=np.float32)
    probs[top_idx] = conf
    return probs


def _uniform() -> np.ndarray:
    return np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES


def _run(probs: np.ndarray, tensor=None, enhanced=True, preproc_ms=10.0):
    from services.inference_service import InferenceService
    loader = _loader(probs)
    tensor = tensor if tensor is not None else _tensor()
    with patch("services.inference_service.get_model_loader", return_value=loader), \
         patch("services.inference_service.settings") as ms:
        ms.class_names_list = CLASS_NAMES
        ms.APP_VERSION      = "1.0.0"
        return InferenceService.predict(tensor, enhanced, preproc_ms)


# ─────────────────────────────────────────────────────────────────────────────
#  Top prediction correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceTopPrediction:
    def test_top_class_d101_index_0(self):
        resp = _run(_peaked(top_idx=0, conf=0.90))
        assert resp.prediction.variety_code == "D101"

    def test_top_class_d200_index_5(self):
        resp = _run(_peaked(top_idx=5, conf=0.95))
        assert resp.prediction.variety_code == "D200"

    def test_top_class_d88_index_7(self):
        resp = _run(_peaked(top_idx=7, conf=0.88))
        assert resp.prediction.variety_code == "D88"

    def test_confidence_matches_probs(self):
        probs = _peaked(top_idx=2, conf=0.85)
        resp  = _run(probs)
        assert abs(resp.prediction.confidence_score - 0.85) < 0.01

    def test_confidence_between_0_and_1(self):
        resp = _run(_peaked())
        assert 0.0 <= resp.prediction.confidence_score <= 1.0

    def test_prediction_variety_name_not_empty(self):
        resp = _run(_peaked())
        assert resp.prediction.variety_name != ""

    def test_prediction_local_name_not_empty(self):
        resp = _run(_peaked())
        assert resp.prediction.local_name != ""

    def test_prediction_origin_not_empty(self):
        resp = _run(_peaked())
        assert resp.prediction.origin != ""

    def test_prediction_description_not_empty(self):
        resp = _run(_peaked())
        assert resp.prediction.description != ""

    def test_success_flag_true(self):
        resp = _run(_peaked())
        assert resp.success is True


# ─────────────────────────────────────────────────────────────────────────────
#  All varieties & confidence scores
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceAllVarieties:
    def test_all_varieties_contains_8_items(self):
        resp = _run(_uniform())
        assert len(resp.all_varieties) == NUM_CLASSES

    def test_all_varieties_sorted_descending(self):
        probs = np.array([0.1, 0.3, 0.05, 0.2, 0.05, 0.15, 0.1, 0.05], dtype=np.float32)
        resp  = _run(probs)
        scores = [v.confidence_score for v in resp.all_varieties]
        assert scores == sorted(scores, reverse=True)

    def test_confidence_scores_dict_has_8_keys(self):
        resp = _run(_uniform())
        assert len(resp.confidence_scores) == NUM_CLASSES

    def test_all_variety_codes_present_in_confidence_scores(self):
        resp   = _run(_uniform())
        scores = resp.confidence_scores
        # Kunci adalah display_name, nilai adalah float
        assert len(scores) == NUM_CLASSES
        for v in resp.all_varieties:
            assert v.variety_name in scores

    def test_confidence_scores_sum_near_one(self):
        resp  = _run(_peaked())
        total = sum(resp.confidence_scores.values())
        assert abs(total - 1.0) < 0.02

    def test_all_confidence_values_between_0_and_1(self):
        resp = _run(_uniform())
        for name, score in resp.confidence_scores.items():
            assert 0.0 <= score <= 1.0, f"{name}: {score} di luar [0,1]"

    def test_top_variety_matches_prediction(self):
        resp = _run(_peaked(top_idx=3, conf=0.93))
        assert resp.all_varieties[0].variety_code == resp.prediction.variety_code


# ─────────────────────────────────────────────────────────────────────────────
#  Metadata fields
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceMetadata:
    def test_inference_time_positive(self):
        resp = _run(_peaked())
        assert resp.inference_time_ms >= 0.0

    def test_preprocessing_time_passed_through(self):
        resp = _run(_peaked(), preproc_ms=25.5)
        assert resp.preprocessing_time_ms == 25.5

    def test_image_enhanced_flag_true(self):
        resp = _run(_peaked(), enhanced=True)
        assert resp.image_enhanced is True

    def test_image_enhanced_flag_false(self):
        resp = _run(_peaked(), enhanced=False)
        assert resp.image_enhanced is False

    def test_model_version_is_string(self):
        resp = _run(_peaked())
        assert isinstance(resp.model_version, str)


# ─────────────────────────────────────────────────────────────────────────────
#  Softmax & probability edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceProbabilityHandling:
    def test_valid_probs_no_double_softmax(self):
        """Jika output sudah berupa probabilitas, tidak boleh di-softmax lagi."""
        probs = _peaked(top_idx=0, conf=0.92)
        resp  = _run(probs)
        # Jika double softmax, confidence akan jauh lebih kecil dari 0.92
        assert resp.prediction.confidence_score > 0.5

    def test_logit_output_auto_softmax_applied(self):
        """Jika output berupa logits (sum != 1), softmax otomatis diterapkan."""
        logits = np.array([5.0, 0.1, 0.2, -0.5, 0.0, 0.1, -0.2, 0.3], dtype=np.float32)
        resp   = _run(logits)
        total  = sum(resp.confidence_scores.values())
        assert abs(total - 1.0) < 0.02

    def test_logit_top_class_is_highest_logit(self):
        logits = np.array([5.0, 0.1, 0.2, -0.5, 0.0, 0.1, -0.2, 0.3], dtype=np.float32)
        resp   = _run(logits)
        assert resp.prediction.variety_code == "D101"  # index 0 = D101

    def test_uniform_probs_all_equal(self):
        resp   = _run(_uniform())
        scores = list(resp.confidence_scores.values())
        assert max(scores) - min(scores) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
#  Error cases
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceErrors:
    def _run_with_class_mismatch(self, n_probs: int):
        from services.inference_service import InferenceService
        probs  = np.ones(n_probs, dtype=np.float32) / n_probs
        loader = _loader(probs)
        with patch("services.inference_service.get_model_loader", return_value=loader), \
             patch("services.inference_service.settings") as ms:
            ms.class_names_list = CLASS_NAMES  # 8 kelas
            ms.APP_VERSION      = "1.0.0"
            return InferenceService.predict(_tensor())

    def test_class_mismatch_raises_inference_exception(self):
        with pytest.raises(InferenceException) as exc_info:
            self._run_with_class_mismatch(n_probs=4)
        assert "sinkron" in exc_info.value.detail.lower() or "mismatch" in exc_info.value.detail.lower()

    def test_3d_tensor_raises_inference_exception(self):
        from services.inference_service import InferenceService
        loader = _loader(_peaked())
        bad_tensor = np.zeros((224, 224, 3), dtype=np.float32)  # 3D bukan 4D
        with patch("services.inference_service.get_model_loader", return_value=loader), \
             patch("services.inference_service.settings") as ms:
            ms.class_names_list = CLASS_NAMES
            ms.APP_VERSION      = "1.0.0"
            with pytest.raises(InferenceException):
                InferenceService.predict(bad_tensor)

    def test_wrong_spatial_dimensions_raises_inference_exception(self):
        from services.inference_service import InferenceService
        loader = _loader(_peaked())
        bad_tensor = np.zeros((1, 128, 128, 3), dtype=np.float32)  # bukan 224x224
        with patch("services.inference_service.get_model_loader", return_value=loader), \
             patch("services.inference_service.settings") as ms:
            ms.class_names_list = CLASS_NAMES
            ms.APP_VERSION      = "1.0.0"
            with pytest.raises(InferenceException):
                InferenceService.predict(bad_tensor)

    def test_model_not_loaded_raises_model_not_loaded(self):
        from services.inference_service import InferenceService
        loader = MagicMock()
        loader.session     = property(lambda self: (_ for _ in ()).throw(ModelNotLoadedException()))
        loader.is_loaded   = False

        # Simulasi ModelNotLoadedException saat akses session
        not_loaded_loader = MagicMock()
        not_loaded_loader.session.__get__ = MagicMock(side_effect=ModelNotLoadedException())
        type(not_loaded_loader).session = property(
            lambda self: (_ for _ in ()).throw(ModelNotLoadedException())
        )

        with patch("services.inference_service.get_model_loader", return_value=not_loaded_loader):
            with pytest.raises(ModelNotLoadedException):
                InferenceService.predict(_tensor())

    def test_onnx_runtime_error_raises_inference_exception(self):
        from services.inference_service import InferenceService
        loader          = _loader(_peaked())
        loader.session.run.side_effect = RuntimeError("ONNX Runtime error")
        with patch("services.inference_service.get_model_loader", return_value=loader), \
             patch("services.inference_service.settings") as ms:
            ms.class_names_list = CLASS_NAMES
            ms.APP_VERSION      = "1.0.0"
            with pytest.raises(InferenceException):
                InferenceService.predict(_tensor())

    def test_float16_tensor_converted_to_float32(self):
        """Tensor float16 harus di-convert otomatis."""
        from services.inference_service import InferenceService
        loader = _loader(_peaked())
        tensor_f16 = _tensor().astype(np.float16)
        with patch("services.inference_service.get_model_loader", return_value=loader), \
             patch("services.inference_service.settings") as ms:
            ms.class_names_list = CLASS_NAMES
            ms.APP_VERSION      = "1.0.0"
            # Harus tidak raise exception (auto-convert)
            resp = InferenceService.predict(tensor_f16)
            assert resp.success is True
