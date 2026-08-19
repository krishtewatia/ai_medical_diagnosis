"""
Test suite for Medical Profile Schemas (Step 5 Verification).

Tests:
1. Valid profile -> accepted
2. Invalid gender -> rejected
3. Invalid blood type -> rejected
4. Invalid date format -> rejected
5. Invalid smoking status -> rejected
6. Partial update -> accepted
"""

import sys
from datetime import date, datetime, timezone
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError

from app.schemas.medical_profile import (
    AlcoholConsumptionEnum,
    BloodTypeEnum,
    EmergencyContact,
    GenderEnum,
    MedicalProfileCreate,
    MedicalProfileResponse,
    MedicalProfileUpdate,
    SmokingStatusEnum,
)


def test_valid_profile():
    data = {
        "date_of_birth": "1992-04-12",
        "gender": "female",
        "blood_type": "O+",
        "height_cm": 165.5,
        "weight_kg": 58.0,
        "allergies": ["Penicillin"],
        "chronic_conditions": ["Asthma"],
        "current_medications": ["Albuterol"],
        "smoking_status": "never",
        "alcohol_consumption": "occasional",
        "emergency_contact": {
            "name": "Alex Doe",
            "relationship": "Spouse",
            "phone": "+1234567890",
        },
    }
    profile = MedicalProfileCreate(**data)
    assert profile.gender == GenderEnum.FEMALE
    assert profile.blood_type == BloodTypeEnum.O_POSITIVE
    assert profile.date_of_birth == date(1992, 4, 12)
    print("  PASS: 1. Valid profile accepted")


def test_invalid_gender():
    data = {
        "date_of_birth": "1992-04-12",
        "gender": "invalid_gender_value",
    }
    try:
        MedicalProfileCreate(**data)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        print("  PASS: 2. Invalid gender rejected")


def test_invalid_blood_type():
    data = {
        "date_of_birth": "1992-04-12",
        "gender": "male",
        "blood_type": "C+",  # Nonexistent blood type
    }
    try:
        MedicalProfileCreate(**data)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        print("  PASS: 3. Invalid blood type rejected")


def test_invalid_date_format():
    data = {
        "date_of_birth": "12-04-1992",  # Not YYYY-MM-DD
        "gender": "male",
    }
    try:
        MedicalProfileCreate(**data)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        print("  PASS: 4. Invalid date format rejected")


def test_invalid_smoking_status():
    data = {
        "date_of_birth": "1992-04-12",
        "gender": "male",
        "smoking_status": "chain_smoker",  # Invalid enum
    }
    try:
        MedicalProfileCreate(**data)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        print("  PASS: 5. Invalid smoking status rejected")


def test_partial_update():
    # Updating only weight_kg
    update1 = MedicalProfileUpdate(weight_kg=75.0)
    assert update1.weight_kg == 75.0
    assert update1.date_of_birth is None
    assert update1.gender is None

    # Updating only smoking_status and current_medications
    update2 = MedicalProfileUpdate(
        smoking_status=SmokingStatusEnum.FORMER,
        current_medications=["Metformin 500mg"],
    )
    assert update2.smoking_status == SmokingStatusEnum.FORMER
    assert update2.current_medications == ["Metformin 500mg"]
    print("  PASS: 6. Partial update accepted")


def test_response_schema():
    now = datetime.now(timezone.utc)
    res_data = {
        "id": "64a1b2c3d4e5f6a7b8c9d0e2",
        "user_id": "64a1b2c3d4e5f6a7b8c9d0e1",
        "date_of_birth": "1990-01-01",
        "gender": "male",
        "blood_type": "A+",
        "height_cm": 180.0,
        "weight_kg": 75.0,
        "allergies": [],
        "chronic_conditions": [],
        "current_medications": [],
        "smoking_status": "never",
        "alcohol_consumption": "none",
        "emergency_contact": None,
        "created_at": now,
        "updated_at": now,
    }
    res = MedicalProfileResponse(**res_data)
    assert res.id == "64a1b2c3d4e5f6a7b8c9d0e2"
    assert res.user_id == "64a1b2c3d4e5f6a7b8c9d0e1"
    print("  PASS: 7. Response schema validated")


if __name__ == "__main__":
    print("\n=== Medical Profile Schemas Verification Tests ===\n")
    tests = [
        ("1. Valid profile", test_valid_profile),
        ("2. Invalid gender", test_invalid_gender),
        ("3. Invalid blood type", test_invalid_blood_type),
        ("4. Invalid date format", test_invalid_date_format),
        ("5. Invalid smoking status", test_invalid_smoking_status),
        ("6. Partial update", test_partial_update),
        ("7. Response schema", test_response_schema),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} - {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}\n")
    sys.exit(1 if failed else 0)
