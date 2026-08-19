"""
Step 8 Verification Script:
Verify live MongoDB collection, schemas, and unique index constraints.

Checks:
1. 'medical_profiles' collection exists and is accessible
2. Unique index on 'user_id' exists ({ 'user_id': 1 }, unique=True)
3. Profile creation inserts document with user_id stored as ObjectId
4. created_at and updated_at timestamps exist
5. Duplicate profile creation for same user is rejected by MongoDB unique constraint
6. Different user can create their own profile
7. Updating a profile updates in-place without creating a second document
8. 1-to-1 relationship strictly held between users and medical_profiles
"""

import sys
from datetime import date, datetime, timezone
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.database.connection import get_database
from app.schemas.medical_profile import (
    BloodTypeEnum,
    EmergencyContact,
    GenderEnum,
    MedicalProfileCreate,
    MedicalProfileUpdate,
)
from app.services.medical_profile_service import (
    DuplicateProfileError,
    MedicalProfileService,
)
from app.services.user_service import UserService


def run_atlas_verification():
    db = get_database()
    profile_service = MedicalProfileService(db)
    user_service = UserService(db)

    print("\n========================================================")
    print(" STEP 8 - VERIFY MONGODB 'medical_profiles' CONSTRAINTS")
    print("========================================================\n")

    # 1. Verify index
    print("1. Inspecting MongoDB Indexes on 'medical_profiles':")
    indexes = list(db["medical_profiles"].list_indexes())
    has_unique_user_id_index = False
    for idx in indexes:
        key_dict = idx.get("key", {})
        is_unique = idx.get("unique", False)
        print(f"   - Index: {idx.get('name')} | Key: {key_dict} | Unique: {is_unique}")
        if "user_id" in key_dict and is_unique:
            has_unique_user_id_index = True

    assert has_unique_user_id_index, "Unique index on user_id missing!"
    print("   -> [PASS] Unique index on user_id is active in MongoDB.\n")

    # Setup 2 test users
    temp_email_a = f"test_a_{int(datetime.now().timestamp())}@example.com"
    temp_email_b = f"test_b_{int(datetime.now().timestamp())}@example.com"

    user_a = user_service.create_user(name="Atlas Test User A", email=temp_email_a, password_hash="hash_a")
    user_b = user_service.create_user(name="Atlas Test User B", email=temp_email_b, password_hash="hash_b")

    user_a_id = str(user_a["_id"])
    user_b_id = str(user_b["_id"])

    try:
        # 2. Create profile for User A
        print("2. Creating profile for User A:")
        profile_data_a = MedicalProfileCreate(
            date_of_birth=date(1993, 7, 21),
            gender=GenderEnum.FEMALE,
            blood_type=BloodTypeEnum.B_POSITIVE,
            height_cm=170.0,
            weight_kg=65.0,
            allergies=["Pollen"],
            chronic_conditions=[],
            current_medications=[],
            emergency_contact=EmergencyContact(
                name="Emergency Contact A",
                relationship="Parent",
                phone="+123456789"
            )
        )
        created_a = profile_service.create_profile(user_a_id, profile_data_a)
        print(f"   - Document ID: {created_a['_id']}")
        print(f"   - user_id type: {type(created_a['user_id'])} ({created_a['user_id']})")
        print(f"   - created_at: {created_a.get('created_at')}")
        print(f"   - updated_at: {created_a.get('updated_at')}")

        assert isinstance(created_a["user_id"], ObjectId), "user_id must be stored as BSON ObjectId"
        assert "created_at" in created_a and "updated_at" in created_a
        print("   -> [PASS] Profile created with BSON ObjectId and UTC timestamps.\n")

        # 3. Duplicate profile creation rejection
        print("3. Testing duplicate profile rejection for User A:")
        try:
            profile_service.create_profile(user_a_id, profile_data_a)
            assert False, "Duplicate creation did not fail!"
        except DuplicateProfileError:
            print("   -> [PASS] Duplicate profile for User A was successfully blocked.\n")

        # 4. User B creates their own profile
        print("4. Creating profile for User B:")
        profile_data_b = MedicalProfileCreate(
            date_of_birth=date(1989, 2, 10),
            gender=GenderEnum.MALE,
            blood_type=BloodTypeEnum.O_POSITIVE,
            height_cm=180.0,
            weight_kg=78.0
        )
        created_b = profile_service.create_profile(user_b_id, profile_data_b)
        print(f"   - User B Profile ID: {created_b['_id']} (user_id: {created_b['user_id']})")
        assert created_b["_id"] != created_a["_id"]
        assert created_b["user_id"] == ObjectId(user_b_id)
        print("   -> [PASS] User B created independent profile.\n")

        # 5. In-place update check (no duplicate document creation)
        print("5. Updating User A profile (in-place modification):")
        count_before = db["medical_profiles"].count_documents({"user_id": ObjectId(user_a_id)})
        profile_service.update_profile(user_a_id, MedicalProfileUpdate(weight_kg=67.5))
        count_after = db["medical_profiles"].count_documents({"user_id": ObjectId(user_a_id)})
        assert count_before == 1 and count_after == 1, "Update must not create duplicate documents"
        updated_doc = profile_service.find_by_user_id(user_a_id)
        assert updated_doc["weight_kg"] == 67.5
        print(f"   - Document count for User A remained: {count_after}")
        print(f"   - Updated weight_kg: {updated_doc['weight_kg']}")
        print("   -> [PASS] Update executed in-place without duplicating documents.\n")

        print("========================================================")
        print(" SUMMARY: ALL MONGODB ATLAS VERIFICATION CHECKS PASSED")
        print("========================================================\n")

    finally:
        # Cleanup test records
        db["medical_profiles"].delete_many({"user_id": {"$in": [ObjectId(user_a_id), ObjectId(user_b_id)]}})
        db["users"].delete_many({"_id": {"$in": [ObjectId(user_a_id), ObjectId(user_b_id)]}})
        print("Cleaned up temporary verification test records from MongoDB.")


if __name__ == "__main__":
    run_atlas_verification()
