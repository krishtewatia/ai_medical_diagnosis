from typing import Dict, List, Optional, Type

from app.ml.base_predictor import (
    BaseImagePredictor,
    BasePredictor,
    BaseTabularPredictor,
)
from app.ml.loaders import discover_and_load_disease_configs
from app.schemas.disease_config import (
    DiseaseCategory,
    DiseaseConfig,
    DiseasePublicInfo,
)


class DiseaseNotFoundError(KeyError):
    """Raised when a requested disease is not registered in the system."""
    pass


class DuplicateDiseaseError(ValueError):
    """Raised when attempting to register a disease with an existing ID without override."""
    pass


class PredictorNotRegisteredError(KeyError):
    """Raised when a predictor has not been registered or bound to a disease."""
    pass


class DiseaseRegistry:
    """
    Central, thread-safe single source of truth for disease configurations and their associated predictors.
    
    Responsibilities:
    - Register & store declarative disease configurations.
    - Prevent duplicate disease registrations.
    - Resolve and cache predictor instances dynamically.
    - Provide category filtering and public metadata for API presentation.
    - Completely decoupled from HTTP, database, and auth layers.
    """

    def __init__(self, auto_load: bool = True):
        self._configs: Dict[str, DiseaseConfig] = {}
        self._predictor_classes: Dict[str, Type[BasePredictor]] = {}
        self._predictor_instances: Dict[str, BasePredictor] = {}
        if auto_load:
            self.reload()

    def reload(self) -> None:
        """Discovers and reloads all disease configurations from disk."""
        discovered = discover_and_load_disease_configs()
        for config in discovered.values():
            self._configs[config.id] = config
        self._bind_builtin_predictors()

    def _bind_builtin_predictors(self) -> None:
        """Binds specialized disease predictor classes when available."""
        if "diabetes" in self._configs:
            try:
                from app.ml.tabular_models.diabetes.predictor import DiabetesPredictor
                self.register_predictor("diabetes", DiabetesPredictor)
            except Exception:
                pass

        if "heart_disease" in self._configs:
            try:
                from app.ml.tabular_models.heart_disease.predictor import HeartDiseasePredictor
                self.register_predictor("heart_disease", HeartDiseasePredictor)
            except Exception:
                pass

    def clear(self) -> None:
        """Clears all registered configurations and predictors (useful for unit testing)."""
        self._configs.clear()
        self._predictor_classes.clear()
        self._predictor_instances.clear()

    def register(
        self,
        config: DiseaseConfig,
        predictor_cls: Optional[Type[BasePredictor]] = None,
        allow_override: bool = False
    ) -> None:
        """
        Registers a disease configuration and optionally binds its predictor class.
        Raises DuplicateDiseaseError if already registered and allow_override is False.
        """
        if config.id in self._configs and not allow_override:
            raise DuplicateDiseaseError(
                f"Disease '{config.id}' is already registered. Set allow_override=True to overwrite."
            )

        self._configs[config.id] = config

        if predictor_cls is not None:
            self.register_predictor(config.id, predictor_cls)

    def register_predictor(
        self,
        disease_id: str,
        predictor_cls: Type[BasePredictor]
    ) -> None:
        """
        Binds a predictor class to an existing registered disease configuration.
        Validates category compatibility between the configuration and predictor.
        """
        config = self.get_or_raise(disease_id)

        # Validate category compatibility
        if config.category == DiseaseCategory.TABULAR and not issubclass(predictor_cls, BaseTabularPredictor):
            raise TypeError(
                f"Disease '{disease_id}' is TABULAR; predictor must inherit from BaseTabularPredictor."
            )
        if config.category == DiseaseCategory.IMAGE and not issubclass(predictor_cls, BaseImagePredictor):
            raise TypeError(
                f"Disease '{disease_id}' is IMAGE; predictor must inherit from BaseImagePredictor."
            )

        self._predictor_classes[disease_id] = predictor_cls
        # Invalidate any previously cached instance
        self._predictor_instances.pop(disease_id, None)

    def has_disease(self, disease_id: str) -> bool:
        """Returns True if the disease is registered."""
        return disease_id in self._configs

    def is_registered(self, disease_id: str) -> bool:
        """Alias for has_disease."""
        return self.has_disease(disease_id)

    def get(self, disease_id: str) -> Optional[DiseaseConfig]:
        """Retrieves configuration by disease ID, returning None if not found."""
        return self._configs.get(disease_id)

    def get_or_raise(self, disease_id: str) -> DiseaseConfig:
        """Retrieves configuration by disease ID or raises DiseaseNotFoundError."""
        config = self.get(disease_id)
        if not config:
            available = list(self._configs.keys())
            raise DiseaseNotFoundError(
                f"Disease '{disease_id}' is not registered. Available diseases: {available}"
            )
        return config

    def get_predictor(self, disease_id: str) -> BasePredictor:
        """
        Resolves, instantiates, and caches the predictor instance for a disease.
        Raises DiseaseNotFoundError if disease does not exist.
        Raises PredictorNotRegisteredError if no predictor is bound.
        """
        config = self.get_or_raise(disease_id)

        # Return cached instance if available
        if disease_id in self._predictor_instances:
            return self._predictor_instances[disease_id]

        # Check if predictor class is bound
        if disease_id not in self._predictor_classes:
            raise PredictorNotRegisteredError(
                f"No predictor has been registered for disease '{disease_id}'."
            )

        predictor_cls = self._predictor_classes[disease_id]
        instance = predictor_cls(config)
        self._predictor_instances[disease_id] = instance
        return instance

    def list_all(self) -> List[DiseaseConfig]:
        """Returns all registered configurations."""
        return list(self._configs.values())

    def list_active(self) -> List[DiseaseConfig]:
        """Returns only active disease configurations."""
        return [c for c in self._configs.values() if c.is_active]

    def list_by_category(self, category: DiseaseCategory) -> List[DiseaseConfig]:
        """Filters active configurations by category ('tabular' or 'image')."""
        return [c for c in self.list_active() if c.category == category]

    def get_public_info(self, disease_id: str) -> Optional[DiseasePublicInfo]:
        """Returns safe, frontend-ready public metadata for a disease."""
        config = self.get(disease_id)
        if not config or not config.is_active:
            return None
        return DiseasePublicInfo(
            id=config.id,
            version=config.version,
            display_name=config.display_name,
            category=config.category,
            input_type=config.input_type,
            short_description=config.short_description,
            tabular_features=config.tabular_features,
            image_spec=config.image_spec,
            positive_label=config.positive_label,
            negative_label=config.negative_label,
            supports_probability=config.supports_probability,
            metrics=config.metrics,
            clinical_purpose=config.clinical_purpose,
            disclaimer=config.disclaimer,
        )

    def list_public_info(self) -> List[DiseasePublicInfo]:
        """Returns list of public metadata for all active diseases."""
        return [
            info for d in self.list_active()
            if (info := self.get_public_info(d.id)) is not None
        ]


# Singleton instance pre-loaded for application use
disease_registry = DiseaseRegistry(auto_load=True)
