"""
Inference service for the Durian Classification API.

Facilitates running an image tensor through the ONNX model session,
applying softmax to the logits, calculating confidences, and mapping
the results back to human-readable durian class names.
"""

import time
from typing import Dict, Tuple

import numpy as np

from core.config import settings
from core.exceptions import InferenceException
from core.logger import get_logger
from models.model_loader import get_model_loader
from schemas.response import PredictionResponse, PredictionResult

logger = get_logger(__name__)


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each set of scores in x.

    Robust implementation that subtracts the max for numerical stability.

    Args:
        x: Input numpy array (logits).

    Returns:
        Numpy array of the same shape with softmax probabilities.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class InferenceService:
    """Service class for running ONNX model predictions."""

    @staticmethod
    def predict(image_tensor: np.ndarray) -> PredictionResponse:
        """Run inference on a preprocessed image tensor.

        Args:
            image_tensor: A preprocessed numpy array of shape (1, 224, 224, 3)
                          and type float32.

        Returns:
            A PredictionResponse object containing the top prediction,
            confidence scores, and inference timing.

        Raises:
            InferenceException: If the ONNX session fails to execute.
        """
        logger.debug(f"Starting inference for tensor of shape {image_tensor.shape}")
        
        model_loader = get_model_loader()
        
        try:
            # Ensure model is loaded (raises ModelNotLoadedException if failed)
            session = model_loader.session
            input_name = model_loader.input_name
            output_name = model_loader.output_name
            
            # Start timing
            start_time = time.perf_counter()
            
            # Run inference
            # We supply the feed dict {input_name: input_data}
            outputs = session.run([output_name], {input_name: image_tensor})
            
            # End timing
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000.0
            
            logger.debug(f"Inference completed in {inference_time_ms:.2f} ms")
            
        except Exception as e:
            logger.error(f"ONNX inference session failed: {str(e)}")
            raise InferenceException(detail="Failed to execute model prediction.") from e
            
        try:
            # outputs[0] is typically shape (1, num_classes) containing logits or raw probabilities
            logits = outputs[0][0]  # Extract the single batch result
            
            # Apply softmax to get normalized probabilities (0.0 to 1.0)
            probabilities = softmax(logits)
            
            # Map probabilities to class names
            class_names = settings.class_names_list
            if len(probabilities) != len(class_names):
                error_msg = (
                    f"Model output shape mismatch. Model returned {len(probabilities)} "
                    f"classes, but config specifies {len(class_names)} classes."
                )
                logger.error(error_msg)
                raise InferenceException(detail="Internal config error: Class count mismatch.")
                
            # Create a dictionary of class_name: confidence
            confidence_map: Dict[str, float] = {
                name: float(prob) for name, prob in zip(class_names, probabilities)
            }
            
            # Find the index of the highest probability
            top_class_idx = int(np.argmax(probabilities))
            top_class_name = class_names[top_class_idx]
            top_confidence = float(probabilities[top_class_idx])
            
            logger.info(f"Prediction successful: {top_class_name} ({top_confidence:.4f})")
            
            # Construct response
            prediction_result = PredictionResult(
                class_name=top_class_name,
                confidence_score=top_confidence
            )
            
            return PredictionResponse(
                success=True,
                prediction=prediction_result,
                confidence_scores=confidence_map,
                inference_time_ms=round(inference_time_ms, 2)
            )
            
        except InferenceException:
            raise
        except Exception as e:
            logger.error(f"Failed to process inference outputs: {str(e)}")
            raise InferenceException(detail="Failed to post-process model outputs.") from e
