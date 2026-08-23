from typing import List, Optional

from app.ml.disease_registry import (
    DiseaseNotFoundError,
    DiseaseRegistry,
    disease_registry,
)
from app.schemas.disease import (
    DiseaseModelInfo,
    DiseaseResponse,
    DiseaseSafetyInfo,
)
from app.schemas.disease_config import DiseaseConfig


class DiseaseService:
    """
    Service layer for Disease Discovery.
    
    Responsibilities:
    - Acts as an intermediary between DiseaseRegistry and the API presentation layer.
    - Transforms internal declarative DiseaseConfig objects into public DiseaseResponse schemas.
    - Strips internal paths (e.g. model_dir, artifact_filename), preprocessing internals, and secrets.
    - Supports retrieving all active disease modules and querying specific diseases by slug.
    """

    def __init__(self, registry: Optional[DiseaseRegistry] = None):
        self.registry = registry if registry is not None else disease_registry

    def to_disease_response(self, config: DiseaseConfig) -> DiseaseResponse:
        """
        Converts an internal DiseaseConfig into a sanitized, frontend-ready DiseaseResponse.
        Hides filesystem directories, pickle references, and internal preprocessing details.
        """
        framework_val = config.framework.value if hasattr(config.framework, "value") else str(config.framework)

        model_info = DiseaseModelInfo(
            version=config.version,
            framework=framework_val,
            model_type=config.model_type,
            threshold=config.decision_threshold,
            supports_probability=config.supports_probability,
        )

        safety_info = DiseaseSafetyInfo(
            clinical_purpose=config.clinical_purpose,
            is_diagnostic_tool=config.is_diagnostic_tool,
            disclaimer=config.disclaimer,
        )

        return DiseaseResponse(
            id=config.id,
            display_name=config.display_name,
            category=config.category,
            input_type=config.input_type,
            description=config.short_description,
            is_active=config.is_active,
            required_fields=config.tabular_features or [],
            image_spec=config.image_spec,
            positive_label=config.positive_label,
            negative_label=config.negative_label,
            supports_probability=config.supports_probability,
            metrics=config.metrics or {},
            model_info=model_info,
            safety_info=safety_info,
        )

    def get_all_diseases(self, active_only: bool = True) -> List[DiseaseResponse]:
        """
        Returns all registered disease screening modules.
        Defaults to active modules only.
        """
        configs = self.registry.list_active() if active_only else self.registry.list_all()
        return [self.to_disease_response(config) for config in configs]

    def get_disease_by_id(self, disease_id: str) -> Optional[DiseaseResponse]:
        """
        Retrieves a single disease module by ID, returning None if not found or inactive.
        """
        if not disease_id or not isinstance(disease_id, str):
            return None

        cleaned_id = disease_id.strip().lower()
        config = self.registry.get(cleaned_id)

        if not config or not config.is_active:
            return None

        return self.to_disease_response(config)

    def get_disease_or_raise(self, disease_id: str) -> DiseaseResponse:
        """
        Retrieves a single disease module by ID or raises DiseaseNotFoundError.
        """
        resp = self.get_disease_by_id(disease_id)
        if not resp:
            raise DiseaseNotFoundError(f"Disease '{disease_id}' is not registered or not available.")
        return resp
