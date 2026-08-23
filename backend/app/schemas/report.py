from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ReportModelInfo(BaseModel):
    model_type: str = Field(..., description="Model architecture family (e.g. 'LogisticRegression', 'DenseNet121').")
    version: str = Field(..., description="Model version string (e.g. 'v1').")
    threshold: Optional[float] = Field(None, description="Configured decision threshold.")


class MedicalReportResponse(BaseModel):
    """
    Standardized medical screening report response.
    Generated dynamically from historical MongoDB prediction records.
    """
    report_id: str = Field(..., description="Unique generated report identifier.")
    prediction_id: str = Field(..., description="Referenced MongoDB prediction identifier.")
    user_id: str = Field(..., description="Owner user identifier.")
    user_name: Optional[str] = Field(None, description="Patient / practitioner name.")
    user_email: Optional[str] = Field(None, description="Account email.")
    
    # Disease & Screening Lineage
    disease: str = Field(..., description="Disease identifier slug.")
    disease_display_name: str = Field(..., description="Human-readable disease title.")
    input_type: str = Field(..., description="Modal input type ('tabular' or 'image').")
    
    # Prediction Outputs
    prediction: str = Field(..., description="Outcome classification label.")
    is_positive: bool = Field(..., description="Flag indicating elevated screening risk.")
    probability: Optional[float] = Field(None, description="Calibrated risk probability.")
    model: ReportModelInfo = Field(..., description="Model architecture telemetry.")
    
    # Audit & Inputs
    prediction_date: str = Field(..., description="Original inference timestamp.")
    input_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of evaluated clinical features or image metadata.")
    explanation: Optional[str] = Field(None, description="AI clinical decision-support narrative.")
    
    # Regulatory Safety Notice
    disclaimer: str = Field(..., description="Screening limitation and medical disclaimer.")
    
    # Storage & Temporary Access
    storage_key: str = Field(..., description="Object storage path for generated PDF document.")
    download_url: Optional[str] = Field(None, description="Temporary signed access URL to stream/download the PDF.")
    created_at: str = Field(..., description="Report generation timestamp.")
