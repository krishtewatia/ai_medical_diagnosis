from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.schemas.disease_config import DiseaseCategory, DiseaseConfig
from app.schemas.prediction import PredictionResult


class BasePredictor(ABC):
    """
    Abstract contract that all disease model predictors must implement.
    Decoupled from HTTP, routing, database persistence, and external presentation.
    """

    def __init__(self, config: DiseaseConfig):
        self.config = config
        self._model: Optional[Any] = None

    @property
    def disease_id(self) -> str:
        return self.config.id

    @property
    def model_path(self) -> Path:
        if not self.config.model_dir:
            raise ValueError(f"model_dir is not set on DiseaseConfig for '{self.config.id}'")
        return Path(self.config.model_dir) / self.config.artifact_filename

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model(self) -> Any:
        """Lazy-loads and caches the model instance."""
        if self._model is None:
            self._model = self.load_model()
        return self._model

    @abstractmethod
    def load_model(self) -> Any:
        """Loads and returns the underlying ML/DL model artifact from disk."""
        pass

    def apply_threshold(self, probability: float) -> Tuple[bool, str]:
        """
        Determines binary outcome and classification label based on the configured decision threshold.
        """
        is_positive = probability >= self.config.decision_threshold
        label = self.config.positive_label if is_positive else self.config.negative_label
        return is_positive, label

    @abstractmethod
    def predict(self, input_data: Any) -> PredictionResult:
        """
        Executes end-to-end inference on validated input data:
        Input -> Preprocessing -> Model Inference -> Thresholding -> PredictionResult
        """
        pass


class BaseTabularPredictor(BasePredictor, ABC):
    """
    Base contract for tabular disease models (e.g. Diabetes, Heart Disease, CKD).
    Expects input_data to be a dictionary of feature names and values.
    """

    def __init__(self, config: DiseaseConfig):
        if config.category != DiseaseCategory.TABULAR:
            raise ValueError(
                f"BaseTabularPredictor requires a TABULAR disease config, got '{config.category}'."
            )
        super().__init__(config)

    def validate_feature_names(self, features: Dict[str, Any]) -> None:
        """Ensures all required features defined in the config are present in input."""
        if not self.config.tabular_features:
            return

        expected = {f.name for f in self.config.tabular_features if f.required}
        provided = set(features.keys())
        missing = expected - provided
        if missing:
            raise ValueError(
                f"Missing required feature(s) for disease '{self.disease_id}': {sorted(list(missing))}"
            )

    @abstractmethod
    def preprocess_features(self, features: Dict[str, Any]) -> Any:
        """Transforms validated feature dictionary into a model-ready matrix/array/DataFrame."""
        pass

    @abstractmethod
    def predict(self, input_data: Dict[str, Any]) -> PredictionResult:
        pass


class BaseImagePredictor(BasePredictor, ABC):
    """
    Base contract for medical image classification models (e.g. Pneumonia, Brain Tumor).
    Expects input_data to be raw image bytes.
    """

    def __init__(self, config: DiseaseConfig):
        if config.category != DiseaseCategory.IMAGE:
            raise ValueError(
                f"BaseImagePredictor requires an IMAGE disease config, got '{config.category}'."
            )
        super().__init__(config)

    @abstractmethod
    def preprocess_image(self, image_bytes: bytes) -> Any:
        """Decodes, resizes, normalizes, and batches raw image bytes for CNN input."""
        pass

    @abstractmethod
    def predict(self, input_data: bytes) -> PredictionResult:
        pass
