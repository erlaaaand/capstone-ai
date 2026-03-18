"""
ONNX Model Loader for the Durian Classification API.

Implements a thread-safe Singleton pattern to load the ONNX model
into an onnxruntime.InferenceSession exactly once during application
startup. Subsequent calls return the cached session instance.
"""

import threading
from pathlib import Path
from typing import List, Optional

import onnxruntime as ort

from core.config import settings
from core.exceptions import ModelLoadException, ModelNotLoadedException
from core.logger import get_logger

logger = get_logger(__name__)


class ONNXModelLoader:
    """Thread-safe Singleton ONNX model loader.

    Ensures the ONNX InferenceSession is created only once, even when
    called from multiple threads (e.g., during concurrent startup events).

    Attributes:
        _instance: Singleton class instance.
        _lock: Threading lock for thread-safe initialization.
        _session: The loaded ONNX InferenceSession.
        _input_name: Name of the model's input tensor.
        _output_name: Name of the model's output tensor.
    """

    _instance: Optional["ONNXModelLoader"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ONNXModelLoader":
        """Create or return the singleton instance (double-checked locking)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._session = None
                    instance._input_name = None
                    instance._output_name = None
                    instance._is_loaded = False
                    cls._instance = instance
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        """Check if the model session is currently loaded."""
        return self._is_loaded

    @property
    def session(self) -> ort.InferenceSession:
        """Get the loaded ONNX InferenceSession.

        Returns:
            The active InferenceSession instance.

        Raises:
            ModelNotLoadedException: If the model has not been loaded yet.
        """
        if not self._is_loaded or self._session is None:
            raise ModelNotLoadedException()
        return self._session

    @property
    def input_name(self) -> str:
        """Get the model's input tensor name.

        Returns:
            The input tensor name string.

        Raises:
            ModelNotLoadedException: If the model has not been loaded yet.
        """
        if not self._is_loaded or self._input_name is None:
            raise ModelNotLoadedException()
        return self._input_name

    @property
    def output_name(self) -> str:
        """Get the model's output tensor name.

        Returns:
            The output tensor name string.

        Raises:
            ModelNotLoadedException: If the model has not been loaded yet.
        """
        if not self._is_loaded or self._output_name is None:
            raise ModelNotLoadedException()
        return self._output_name

    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load the ONNX model into an InferenceSession.

        Uses CPUExecutionProvider by default. Reads the model path from
        settings if not explicitly provided. Logs model metadata on success.

        Args:
            model_path: Optional override for the ONNX model file path.
                        Defaults to settings.MODEL_PATH.

        Raises:
            ModelLoadException: If the model file is missing or fails to load.
        """
        resolved_path: Path = Path(model_path) if model_path else settings.model_abs_path

        logger.info(f"Loading ONNX model from: {resolved_path}")

        if not resolved_path.exists():
            error_msg = f"Model file not found at: {resolved_path}"
            logger.error(error_msg)
            raise ModelLoadException(detail=error_msg)

        try:
            # Configure session options for production
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            session_options.intra_op_num_threads = 4
            session_options.inter_op_num_threads = 4

            # Create the inference session with CPU provider
            self._session = ort.InferenceSession(
                str(resolved_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )

            # Cache input/output tensor metadata
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self._is_loaded = True

            # Log model details
            input_shape: List[int] = self._session.get_inputs()[0].shape
            output_shape: List[int] = self._session.get_outputs()[0].shape

            logger.info(
                f"Model loaded successfully | "
                f"Input: {self._input_name} {input_shape} | "
                f"Output: {self._output_name} {output_shape}"
            )

        except Exception as e:
            self._is_loaded = False
            self._session = None
            error_msg = f"Failed to load ONNX model: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadException(detail=error_msg) from e

    def unload_model(self) -> None:
        """Release the loaded model session and free resources."""
        if self._session is not None:
            logger.info("Unloading ONNX model from memory.")
            self._session = None
            self._input_name = None
            self._output_name = None
            self._is_loaded = False


def get_model_loader() -> ONNXModelLoader:
    """Get the singleton ONNXModelLoader instance.

    Returns:
        The singleton ONNXModelLoader instance.
    """
    return ONNXModelLoader()
