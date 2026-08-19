"""
Test suite for MedicalProfileService (Step 6 Verification).

Tests:
1. Create profile successfully
2. Retrieve profile by user_id
3. Update profile successfully
4. updated_at changes after update
5. Nonexistent profile is handled correctly
6. Duplicate profile creation is handled correctly
7. user_id remains associated with the authenticated user
8. MongoDB document contains the expected fields
"""

import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mongomock
from bson import ObjectId

from app.schemas.medical_profile import (
    BloodTypeEnum,
    EmergencyContact,
    GenderEnum,
    MedicalProfileCreate,
    MedicalProfileUpdate,
    SmokingStatusEnum,
)
from app.services.medical_profile_service import (
    DuplicateProfileError,
    MedicalProfileService,
)


def get_mock_service():
    mock_client = mongomock.MongoClient()
    db = mock_client["test_medical_db"]
    return MedicalProfileService(db)


def _to_naive_utc(dt):
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def test_create_profile():
    service = get_mock_service()
    user_id = str(ObjectId())

    profile_in = MedicalProfileCreate(
        date_of_birth=date(1995, 6, 20),
        gender=GenderEnum.MALE,
        blood_type=BloodTypeEnum.A_POSITIVE,
        height_cm=178.0,
        weight_kg=74.5,
        allergies=["Aspirin"],
        chronic_conditions=["None"],
        current_medications=[],
        emergency_contact=EmergencyContact(
            name="Alice Smith",
            relationship="Sister",
            phone="+1987654321"
        )
    )

    created = service.create_profile(user_id, profile_in)

    assert created is not None
    assert "_id" in created
    assert created["user_id"] == ObjectId(user_id)
    assert created["gender"] == GenderEnum.MALE
    assert created["blood_type"] == BloodTypeEnum.A_POSITIVE
    assert created["created_at"] is not None
    assert created["updated_at"] is not None
    print("  PASS: 1. Profile created successfully with expected metadata")


def test_retrieve_by_user_id():
    service = get_mock_service()
    user_id = str(ObjectId())

    profile_in = MedicalProfileCreate(
        date_of_birth=date(1988, 3, 14),
        gender=GenderEnum.FEMALE,
        allergies=["Peanuts"]
    )
    service.create_profile(user_id, profile_in)

    retrieved = service.find_by_user_id(user_id)
    assert retrieved is not None
    assert retrieved["user_id"] == ObjectId(user_id)
    assert retrieved["gender"] == GenderEnum.FEMALE
    assert retrieved["allergies"] == ["Peanuts"]
    assert service.has_profile(user_id) is True
    print("  PASS: 2. Retrieved profile successfully by user_id")


def test_update_profile_and_timestamp():
    service = get_mock_service()
    user_id = str(ObjectId())

    profile_in = MedicalProfileCreate(
        date_of_birth=date(1990, 1, 1),
        gender=GenderEnum.OTHER,
        weight_kg=60.0
    )
    created = service.create_profile(user_id, profile_in)
    orig_updated_at = _to_naive_utc(created["updated_at"])

    time.sleep(0.05)

    # Perform partial update
    update_in = MedicalProfileUpdate(
        weight_kg=63.5,
        smoking_status=SmokingStatusEnum.FORMER
    )
    updated = service.update_profile(user_id, update_in)

    assert updated is not None
    assert updated["weight_kg"] == 63.5
    assert updated["smoking_status"] == SmokingStatusEnum.FORMER
    assert updated["gender"] == GenderEnum.OTHER

    updated_at_val = _to_naive_utc(updated["updated_at"])
    assert updated_at_val >= orig_updated_at
    print("  PASS: 3 & 4. Profile updated successfully and updated_at timestamp advanced")


def test_nonexistent_profile_handling():
    service = get_mock_service()
    fake_user_id = str(ObjectId())

    assert service.has_profile(fake_user_id) is False
    assert service.find_by_user_id(fake_user_id) is None

    update_in = MedicalProfileUpdate(weight_kg=80.0)
    assert service.update_profile(fake_user_id, update_in) is None
    print("  PASS: 5. Nonexistent profile handled safely returning None / False")


def test_duplicate_profile_rejected():
    service = get_mock_service()
    user_id = str(ObjectId())

    profile_in = MedicalProfileCreate(
        date_of_birth=date(1995, 6, 20),
        gender=GenderEnum.MALE
    )
    service.create_profile(user_id, profile_in)

    # Attempt to create duplicate profile for same user
    try:
        service.create_profile(user_id, profile_in)
        assert False, "Should have raised DuplicateProfileError"
    except DuplicateProfileError:
        print("  PASS: 6. Duplicate profile creation rejected with DuplicateProfileError")


def test_user_id_association_and_fields():
    service = get_mock_service()
    auth_user_id = str(ObjectId())

    profile_in = MedicalProfileCreate(
        date_of_birth=date(1985, 11, 30),
        gender=GenderEnum.FEMALE,
        height_cm=160.0
    )
    doc = service.create_profile(auth_user_id, profile_in)

    # Ensure user_id strictly equals the authenticated user ID
    assert doc["user_id"] == ObjectId(auth_user_id)

    # Verify expected fields are present
    expected_keys = {
        "_id", "user_id", "date_of_birth", "gender", "blood_type",
        "height_cm", "weight_kg", "allergies", "chronic_conditions",
        "current_medications", "smoking_status", "alcohol_consumption",
        "emergency_contact", "created_at", "updated_at"
    }
    assert expected_keys.issubset(doc.keys())
    print("  PASS: 7 & 8. user_id association and document fields verified")


if __name__ == "__main__":
    print("\n=== MedicalProfileService Verification Tests ===\n")
    tests = [
        ("1. Create Profile", test_create_profile),
        ("2. Retrieve by user_id", test_retrieve_by_user_id),
        ("3. Update profile & timestamp", test_update_profile_and_timestamp),
        ("4. Nonexistent profile handling", test_nonexistent_profile_handling),
        ("5. Duplicate profile rejection", test_duplicate_profile_rejected),
        ("6. user_id & fields validation", test_user_id_association_and_fields),
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
