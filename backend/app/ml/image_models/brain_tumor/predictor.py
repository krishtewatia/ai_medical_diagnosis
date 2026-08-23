import io
import time
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from app.ml.base_predictor import BaseImagePredictor
from app.ml.loaders import load_model_for_disease
from app.schemas.prediction import PredictionResult


class BrainTumorPreprocessingError(ValueError):
    """Raised when an uploaded Brain MRI image cannot be decoded or preprocessed."""
    pass


class BrainTumorInferenceError(RuntimeError):
    """Raised when ResNet50 model execution fails."""
    pass


class BrainTumorPredictor(BaseImagePredictor):
    """
    Specialized computer vision predictor for Brain Tumor screening from axial brain MRI scans.
    
    Model Lineage:
    - Backbone: ResNet50 (Keras / ImageNet weights)
    - Target: Multiclass brain MRI classification (Glioma, Meningioma, No Tumor, Pituitary)
    - Input: 224x224x3 RGB tensor
    - Preprocessing: Bilinear resize, RGB conversion, ImageNet standardization
    - Decision Threshold: 0.50
    """

    CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

    def load_model(self) -> Any:
        """Loads the ResNet50 .keras artifact from disk."""
        return load_model_for_disease(self.config)

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocesses raw brain MRI scan bytes according to ResNet50 specifications:
        1. Decode raw bytes into PIL Image
        2. Ensure RGB 3-channel mode
        3. Resize to 224x224 with high-quality Bilinear resampling
        4. Normalize pixel values via ImageNet standard mean & std
        5. Expand dims into batch shape (1, 224, 224, 3)
        """
        if not image_bytes or len(image_bytes) == 0:
            raise BrainTumorPreprocessingError("Uploaded MRI image payload is empty.")

        try:
            img = Image.open(io.BytesIO(image_bytes))
            img = img.convert("RGB")

            dims = (
                self.config.image_spec.target_dimensions
                if self.config.image_spec and self.config.image_spec.target_dimensions
                else [224, 224]
            )
            target_h, target_w = dims[0], dims[1]

            img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)

            arr = np.array(img, dtype=np.float32)

            # ResNet50 ImageNet normalization
            arr = arr / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std

            batch = np.expand_dims(arr, axis=0)
            return batch

        except Exception as e:
            if isinstance(e, BrainTumorPreprocessingError):
                raise
            raise BrainTumorPreprocessingError(f"Failed to preprocess brain MRI image: {str(e)}") from e

    def predict(self, input_data: bytes) -> PredictionResult:
        """
        Executes end-to-end inference on raw brain MRI scan bytes.
        """
        start_time = time.perf_counter()

        try:
            batch = self.preprocess_image(input_data)
            model = self.get_model()

            if hasattr(model, "predict"):
                raw_output = model.predict(batch, verbose=0)
            elif callable(model):
                raw_output = model(batch)
            else:
                raise BrainTumorInferenceError("ResNet50 model is not callable and lacks predict() method.")

            if hasattr(raw_output, "detach"):
                raw_output = raw_output.detach().cpu().numpy()
            elif hasattr(raw_output, "numpy"):
                raw_output = raw_output.numpy()

            probs = np.ravel(raw_output)

            # Class index determination
            class_idx = int(np.argmax(probs))
            detected_subclass = self.CLASS_NAMES[class_idx] if class_idx < len(self.CLASS_NAMES) else f"Class {class_idx}"

            # Probability calculation:
            # If No Tumor is class index 2: tumor risk probability = 1.0 - P(No Tumor)
            no_tumor_prob = float(probs[2]) if len(probs) > 2 else 0.0
            tumor_risk_prob = float(np.clip(1.0 - no_tumor_prob, 0.0, 1.0))

            if class_idx == 2 or tumor_risk_prob < self.config.decision_threshold:
                is_positive = False
                label = self.config.negative_label
                final_prob = round(tumor_risk_prob, 4)
            else:
                is_positive = True
                label = f"Intracranial Neoplasm Detected ({detected_subclass})"
                final_prob = round(tumor_risk_prob, 4)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            class_breakdown = {
                name: round(float(probs[i]), 4)
                for i, name in enumerate(self.CLASS_NAMES)
                if i < len(probs)
            }

            return PredictionResult(
                disease_id=self.disease_id,
                disease_display_name=self.config.display_name,
                model_version=self.config.version,
                model_type=self.config.model_type,
                prediction_label=label,
                is_positive=is_positive,
                probability=final_prob,
                decision_threshold=self.config.decision_threshold,
                metadata={
                    "latency_ms": round(elapsed_ms, 2),
                    "input_shape": list(batch.shape),
                    "detected_subclass": detected_subclass,
                    "class_probabilities": class_breakdown,
                    "model_backbone": "ResNet50",
                },
            )

        except (BrainTumorPreprocessingError, ValueError):
            raise
        except Exception as e:
            raise BrainTumorInferenceError(
                f"Brain Tumor MRI inference failed: {str(e)}"
            ) from e
