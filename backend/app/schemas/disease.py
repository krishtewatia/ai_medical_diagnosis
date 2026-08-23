from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.disease_config import (
    DiseaseCategory,
    FeatureDataType,
    ImageInputSpec,
    InputType,
    ModelFramework,
    TabularFeatureSpec,
)


class DiseaseModelInfo(BaseModel):
    """
    Public model telemetry, framework, and operational threshold information.
    Does not expose sensitive filesystem paths or model serialization secrets.
    """
    version: str = Field(..., description="Model version tag (e.g. 'v1').")
    framework: str = Field(..., description="Machine learning framework (e.g. 'scikit-learn', 'xgboost', 'pytorch').")
    model_type: str = Field(..., description="Underlying algorithm or architecture family (e.g. 'LogisticRegression', 'XGBClassifier', 'DenseNet121').")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Configured classification threshold, if applicable.")
    supports_probability: bool = Field(True, description="Whether the model provides calibrated probability estimates.")

    model_config = ConfigDict(populate_by_name=True)


class DiseaseSafetyInfo(BaseModel):
    """
    Clinical purpose, device limitations, and mandatory regulatory disclaimers.
    """
    clinical_purpose: str = Field(..., description="Intended screening purpose and scope of application.")
    is_diagnostic_tool: bool = Field(False, description="Flag explicitly indicating this is a screening tool, not a diagnostic medical device.")
    disclaimer: str = Field(..., description="Standard medical liability disclaimer.")

    model_config = ConfigDict(populate_by_name=True)


class DiseaseResponse(BaseModel):
    """
    Public schema returned by GET /diseases and GET /diseases/{disease}.
    Equips the frontend with dynamic configuration needed for:
    - Rendering disease discovery and selection cards
    - Dynamic form construction (tabular features, data types, units, validation constraints)
    - Medical image upload specs (formats, size limits, dimensions)
    - Displaying accuracy metrics, model info, and clinical safety disclaimers
    """
    id: str = Field(..., min_length=1, max_length=50, description="Unique disease identifier slug (e.g. 'diabetes', 'heart_disease').")
    display_name: str = Field(..., min_length=1, max_length=100, description="Human-readable title (e.g. 'Diabetes Risk Assessment').")
    category: DiseaseCategory = Field(..., description="Modal category ('tabular' or 'image').")
    input_type: InputType = Field(..., description="Input method ('form' for tabular, 'image_upload' for image).")
    description: str = Field(..., description="Overview description and clinical context.")
    is_active: bool = Field(True, description="Whether this disease screening module is actively deployed and available.")
    required_fields: List[TabularFeatureSpec] = Field(
        default_factory=list,
        description="List of required tabular feature specifications used for dynamic form construction."
    )
    image_spec: Optional[ImageInputSpec] = Field(
        None,
        description="Image specification rules if input_type is image_upload."
    )
    positive_label: str = Field(..., description="Label returned when positive risk or condition is detected.")
    negative_label: str = Field(..., description="Label returned when negative risk or condition is detected.")
    supports_probability: bool = Field(True, description="Whether the disease model computes calibrated confidence probabilities.")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Published validation metrics (e.g. accuracy, roc_auc, f1).")
    model_info: Optional[DiseaseModelInfo] = Field(None, description="Public model architecture and framework metadata.")
    safety_info: Optional[DiseaseSafetyInfo] = Field(None, description="Safety disclaimers and regulatory notices.")

    model_config = ConfigDict(populate_by_name=True)
