import io
import time
from typing import Any, Dict, Optional
import numpy as np
from PIL import Image

from app.ml.base_predictor import BaseImagePredictor
from app.ml.loaders import load_model_for_disease
from app.schemas.prediction import PredictionResult


class PneumoniaPreprocessingError(ValueError):
    """Raised when an uploaded X-ray image cannot be decoded or preprocessed."""
    pass


class PneumoniaInferenceError(RuntimeError):
    """Raised when DenseNet121 model execution fails."""
    pass


class PneumoniaPredictor(BaseImagePredictor):
    """
    Specialized computer vision predictor for Pneumonia & Lung Opacity screening from chest radiographs.
    
    Model Lineage:
    - Backbone: DenseNet121 (Keras / ImageNet weights)
    - Target: Binary chest X-ray classification (Lung Opacity vs No Lung Opacity)
    - Input: 224x224x3 RGB tensor (converted from grayscale frontal radiograph)
    - Preprocessing: Bilinear resize, grayscale-to-RGB expansion, ImageNet standardization
    - Decision Threshold: 0.30
    """

    def load_model(self) -> Any:
        """Loads the DenseNet121 .keras artifact from disk."""
        return load_model_for_disease(self.config)

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocesses raw chest radiograph bytes according to DenseNet specifications:
        1. Decode raw bytes into PIL Image
        2. Convert to Grayscale ('L') then expand to 3-channel RGB ('RGB')
        3. Resize to 224x224 with high-quality Bilinear resampling
        4. Normalize pixel values via ImageNet standard mean & std
        5. Expand dims into batch shape (1, 224, 224, 3)
        """
        if not image_bytes or len(image_bytes) == 0:
            raise PneumoniaPreprocessingError("Uploaded image payload is empty.")

        try:
            # Decode image from byte stream
            img = Image.open(io.BytesIO(image_bytes))

            # Convert to grayscale first (X-rays are single channel) then expand to 3 identical RGB channels
            img = img.convert("L").convert("RGB")

            # Extract target dimensions from config (default 224x224)
            dims = (
                self.config.image_spec.target_dimensions
                if self.config.image_spec and self.config.image_spec.target_dimensions
                else [224, 224]
            )
            target_h, target_w = dims[0], dims[1]

            # Resize (PIL expects (width, height))
            img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)

            # Convert to numpy float32 array
            arr = np.array(img, dtype=np.float32)

            # DenseNet ImageNet normalization: scale [0, 255] -> [0, 1] -> standardize
            arr = arr / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std

            # Expand batch dimension: (1, 224, 224, 3)
            batch = np.expand_dims(arr, axis=0)
            return batch

        except Exception as e:
            if isinstance(e, PneumoniaPreprocessingError):
                raise
            raise PneumoniaPreprocessingError(f"Failed to preprocess chest X-ray image: {str(e)}") from e

    def predict(self, input_data: bytes) -> PredictionResult:
        """
        Executes end-to-end inference on raw chest X-ray image bytes.
        """
        start_time = time.perf_counter()

        try:
            batch = self.preprocess_image(input_data)
            model = self.get_model()

            # Execute forward pass
            if hasattr(model, "predict"):
                raw_output = model.predict(batch, verbose=0)
            elif callable(model):
                raw_output = model(batch)
            else:
                raise PneumoniaInferenceError("DenseNet model is not callable and lacks predict() method.")

            # Convert torch/tensorflow/numpy tensor to scalar
            if hasattr(raw_output, "detach"):
                raw_output = raw_output.detach().cpu().numpy()
            elif hasattr(raw_output, "numpy"):
                raw_output = raw_output.numpy()

            raw_val = float(np.ravel(raw_output)[0])

            # Apply sigmoid if raw logits (< 0 or > 1)
            if raw_val < 0.0 or raw_val > 1.0:
                probability = float(1.0 / (1.0 + np.exp(-raw_val)))
            else:
                probability = float(np.clip(raw_val, 0.0, 1.0))

            is_positive, label = self.apply_threshold(probability)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return PredictionResult(
                disease_id=self.disease_id,
                disease_display_name=self.config.display_name,
                model_version=self.config.version,
                model_type=self.config.model_type,
                prediction_label=label,
                is_positive=is_positive,
                probability=round(probability, 4),
                decision_threshold=self.config.decision_threshold,
                metadata={
                    "latency_ms": round(elapsed_ms, 2),
                    "input_shape": list(batch.shape),
                    "model_backbone": "DenseNet121",
                },
            )

        except (PneumoniaPreprocessingError, ValueError):
            raise
        except Exception as e:
            raise PneumoniaInferenceError(
                f"Pneumonia image inference failed: {str(e)}"
            ) from e
