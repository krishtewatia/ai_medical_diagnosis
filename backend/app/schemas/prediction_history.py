from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionModelInfo(BaseModel):
    """
    Model telemetry and versioning metadata for the prediction record.
    """
    version: str = Field(..., min_length=1, max_length=50, description="Model version tag (e.g. 'v1').")
    model_type: str = Field(..., min_length=1, max_length=100, description="Model architecture or family (e.g. 'LogisticRegression', 'XGBoost', 'DenseNet121').")
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Configured classification threshold, or null if not applicable."
    )


class PredictionResultRecord(BaseModel):
    """
    Classification outcome, probability, and confidence scores.
    """
    prediction: str = Field(..., min_length=1, description="Standardized classification label (e.g. 'High Risk of Diabetes').")
    is_positive: bool = Field(..., description="Boolean flag indicating positive finding or elevated risk.")
    probability: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Calibrated probability score between 0.0 and 1.0, or null."
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Explicit confidence score between 0.0 and 1.0, or null."
    )


class PredictionHistoryCreate(BaseModel):
    """
    Internal schema used by the service layer to persist a verified prediction into MongoDB.
    Decoupled from client input; user_id is injected from JWT authentication.
    """
    user_id: str = Field(..., min_length=1, description="Hex string representation of the authenticated user's ObjectId.")
    disease: str = Field(..., min_length=1, max_length=50, description="Disease identifier matching the registry.")
    disease_display_name: str = Field(..., min_length=1, max_length=100, description="Human-readable disease title.")
    input_type: Literal["tabular", "image"] = Field(..., description="Input modality ('tabular' or 'image').")
    model: PredictionModelInfo
    input_data: Dict[str, Any] = Field(..., description="Sanitized snapshot of model inputs or image references.")
    result: PredictionResultRecord
    explanation: Optional[str] = Field(None, description="Decision-support narrative summary.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Technical execution telemetry (latency, feature count).")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp of creation.")

    @field_validator("disease")
    @classmethod
    def validate_disease_id(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not cleaned:
            raise ValueError("disease identifier cannot be empty.")
        return cleaned


class PredictionHistoryResponse(BaseModel):
    """
    Public history response schema returned by GET /history endpoints.
    Provides uniform structure for viewing historical screenings.
    """
    id: str = Field(..., description="Hex string representation of the MongoDB prediction _id.")
    user_id: str = Field(..., description="Owner user ID.")
    disease: str = Field(..., description="Disease identifier matching the registry.")
    disease_display_name: str = Field(..., description="Human-readable disease title.")
    input_type: Literal["tabular", "image"] = Field(..., description="Input modality ('tabular' or 'image').")
    model: PredictionModelInfo
    input_data: Dict[str, Any] = Field(..., description="Evaluated feature snapshot or image metadata.")
    result: PredictionResultRecord
    explanation: Optional[str] = Field(None, description="Decision-support narrative summary.")
    created_at: datetime = Field(..., description="UTC timestamp when prediction was recorded.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution telemetry.")

    model_config = ConfigDict(populate_by_name=True)


class PredictionHistoryListResponse(BaseModel):
    """
    Paginated history list response returned by GET /history.
    """
    items: List[PredictionHistoryResponse] = Field(default_factory=list, description="List of prediction history items.")
    total: int = Field(..., ge=0, description="Total count of historical predictions matching query.")
    limit: int = Field(..., ge=1, le=100, description="Maximum items per page.")
    skip: int = Field(..., ge=0, description="Items skipped.")
