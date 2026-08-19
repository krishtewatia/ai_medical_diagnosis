from datetime import date, datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class GenderEnum(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class BloodTypeEnum(str, Enum):
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"
    UNKNOWN = "unknown"


class SmokingStatusEnum(str, Enum):
    NEVER = "never"
    FORMER = "former"
    CURRENT = "current"


class AlcoholConsumptionEnum(str, Enum):
    NONE = "none"
    OCCASIONAL = "occasional"
    MODERATE = "moderate"
    FREQUENT = "frequent"


class EmergencyContact(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    relationship: str = Field(..., min_length=2, max_length=50)
    phone: str = Field(..., min_length=5, max_length=25)


class MedicalProfileBase(BaseModel):
    date_of_birth: date
    gender: GenderEnum
    blood_type: Optional[BloodTypeEnum] = None
    height_cm: Optional[float] = Field(None, ge=30.0, le=300.0, description="Height in cm (30 to 300)")
    weight_kg: Optional[float] = Field(None, ge=1.0, le=500.0, description="Weight in kg (1 to 500)")
    allergies: List[str] = Field(default_factory=list)
    chronic_conditions: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    smoking_status: Optional[SmokingStatusEnum] = None
    alcohol_consumption: Optional[AlcoholConsumptionEnum] = None
    emergency_contact: Optional[EmergencyContact] = None

    @field_validator("date_of_birth")
    @classmethod
    def validate_past_dob(cls, v: date) -> date:
        if v >= date.today():
            raise ValueError("date_of_birth must be a date in the past.")
        return v


class MedicalProfileCreate(MedicalProfileBase):
    pass


class MedicalProfileUpdate(BaseModel):
    date_of_birth: Optional[date] = None
    gender: Optional[GenderEnum] = None
    blood_type: Optional[BloodTypeEnum] = None
    height_cm: Optional[float] = Field(None, ge=30.0, le=300.0)
    weight_kg: Optional[float] = Field(None, ge=1.0, le=500.0)
    allergies: Optional[List[str]] = None
    chronic_conditions: Optional[List[str]] = None
    current_medications: Optional[List[str]] = None
    smoking_status: Optional[SmokingStatusEnum] = None
    alcohol_consumption: Optional[AlcoholConsumptionEnum] = None
    emergency_contact: Optional[EmergencyContact] = None

    @field_validator("date_of_birth")
    @classmethod
    def validate_past_dob(cls, v: Optional[date]) -> Optional[date]:
        if v is not None and v >= date.today():
            raise ValueError("date_of_birth must be a date in the past.")
        return v


class MedicalProfileResponse(BaseModel):
    id: str
    user_id: str
    date_of_birth: date
    gender: GenderEnum
    blood_type: Optional[BloodTypeEnum] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    allergies: List[str] = Field(default_factory=list)
    chronic_conditions: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    smoking_status: Optional[SmokingStatusEnum] = None
    alcohol_consumption: Optional[AlcoholConsumptionEnum] = None
    emergency_contact: Optional[EmergencyContact] = None
    created_at: datetime
    updated_at: datetime
