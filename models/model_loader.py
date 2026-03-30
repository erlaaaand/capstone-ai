import os
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort

from core.config import settings
from core.exceptions import ModelLoadException, ModelNotLoadedException
from core.logger import get_logger

logger = get_logger(__name__)


def _get_best_providers() -> List[str]:
    available = ort.get_available_providers()
    providers  = []

    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
        logger.info("GPU (CUDA) tersedia — menggunakan CUDAExecutionProvider.")
    else:
        logger.info("GPU tidak tersedia — menggunakan CPUExecutionProvider.")

    providers.append("CPUExecutionProvider")
    return providers


class ONNXModelLoader:

    _instance: Optional["ONNXModelLoader"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ONNXModelLoader":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._session     = None
                    inst._input_name  = None
                    inst._output_name = None
                    inst._is_loaded   = False
                    inst._providers   = []
                    cls._instance     = inst
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def session(self) -> ort.InferenceSession:
        if not self._is_loaded or self._session is None:
            raise ModelNotLoadedException()
        return self._session

    @property
    def input_name(self) -> str:
        if not self._is_loaded or self._input_name is None:
            raise ModelNotLoadedException()
        return self._input_name

    @property
    def output_name(self) -> str:
        if not self._is_loaded or self._output_name is None:
            raise ModelNotLoadedException()
        return self._output_name

    def load_model(self, model_path: Optional[str] = None) -> None:
        resolved: Path = Path(model_path) if model_path else settings.model_abs_path
        logger.info(f"Loading ONNX model dari: {resolved}")

        if not resolved.exists():
            msg = f"ONNX model tidak ditemukan: {resolved}"
            logger.error(msg)
            raise ModelLoadException(detail=msg)

        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            n_threads = min(os.cpu_count() or 4, 8)
            opts.intra_op_num_threads = n_threads
            opts.inter_op_num_threads = n_threads

            opts.enable_mem_pattern    = True
            opts.enable_cpu_mem_arena  = True

            self._providers = _get_best_providers()
            self._session   = ort.InferenceSession(
                str(resolved),
                sess_options=opts,
                providers=self._providers,
            )

            inp_meta  = self._session.get_inputs()[0]
            out_meta  = self._session.get_outputs()[0]
            self._input_name  = inp_meta.name
            self._output_name = out_meta.name

            input_shape:  List = inp_meta.shape
            output_shape: List = out_meta.shape

            logger.info(
                f"Model ter-load | provider={self._providers[0]} | "
                f"input='{self._input_name}' {input_shape} | "
                f"output='{self._output_name}' {output_shape}"
            )

            model_num_classes = output_shape[-1] if isinstance(output_shape[-1], int) else None
            config_num_classes = settings.num_classes

            if model_num_classes is not None and model_num_classes != config_num_classes:
                msg = (
                    f"MISMATCH kelas: model ONNX memiliki {model_num_classes} output, "
                    f"CLASS_NAMES di config mendefinisikan {config_num_classes} kelas. "
                    f"Periksa .env → CLASS_NAMES atau re-export model."
                )
                logger.error(msg)
                raise ModelLoadException(detail=msg)

            logger.info("Menjalankan warmup inference...")
            warmup_input = np.zeros(
                (1, settings.IMAGE_SIZE, settings.IMAGE_SIZE, 3), dtype=np.float32
            )
            _ = self._session.run(
                [self._output_name],
                {self._input_name: warmup_input},
            )
            logger.info("Warmup selesai. Model siap menerima request.")

            self._is_loaded = True

        except ModelLoadException:
            self._is_loaded = False
            self._session   = None
            raise
        except Exception as e:
            self._is_loaded = False
            self._session   = None
            msg = f"Gagal load ONNX model: {str(e)}"
            logger.error(msg)
            raise ModelLoadException(detail=msg) from e

    def unload_model(self) -> None:
        if self._session is not None:
            logger.info("Unloading ONNX model dari memory.")
            self._session     = None
            self._input_name  = None
            self._output_name = None
            self._is_loaded   = False
            self._providers   = []


def get_model_loader() -> ONNXModelLoader:
    return ONNXModelLoader()