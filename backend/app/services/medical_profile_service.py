from datetime import date, datetime, timezone
from typing import Optional

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import ReturnDocument
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from app.schemas.medical_profile import MedicalProfileCreate, MedicalProfileUpdate


class DuplicateProfileError(Exception):
    """Raised when attempting to create a profile for a user that already has one."""
    pass


class MedicalProfileService:

    def __init__(self, database: Database):
        self.profiles = database["medical_profiles"]
        # Ensure unique index on user_id for 1-to-1 relationship
        try:
            self.profiles.create_index("user_id", unique=True)
        except Exception:
            pass

    def _parse_object_id(self, id_val: str) -> Optional[ObjectId]:
        try:
            return ObjectId(id_val)
        except (InvalidId, TypeError):
            return None

    def _serialize_dates(self, data: dict) -> dict:
        """Convert any date objects to ISO format string for MongoDB BSON compatibility."""
        serialized = {}
        for k, v in data.items():
            if isinstance(v, date) and not isinstance(v, datetime):
                serialized[k] = v.isoformat()
            elif isinstance(v, dict):
                serialized[k] = self._serialize_dates(v)
            else:
                serialized[k] = v
        return serialized

    def find_by_user_id(self, user_id: str) -> Optional[dict]:
        oid = self._parse_object_id(user_id)
        if not oid:
            return None
        return self.profiles.find_one({"user_id": oid})

    def has_profile(self, user_id: str) -> bool:
        return self.find_by_user_id(user_id) is not None

    def create_profile(
        self,
        user_id: str,
        profile_data: MedicalProfileCreate
    ) -> dict:
        oid = self._parse_object_id(user_id)
        if not oid:
            raise ValueError("Invalid user ID.")

        if self.has_profile(user_id):
            raise DuplicateProfileError("A medical profile already exists for this user.")

        now = datetime.now(timezone.utc)
        doc = self._serialize_dates(profile_data.model_dump())
        doc["user_id"] = oid
        doc["created_at"] = now
        doc["updated_at"] = now

        try:
            result = self.profiles.insert_one(doc)
            doc["_id"] = result.inserted_id
            return doc
        except DuplicateKeyError:
            raise DuplicateProfileError("A medical profile already exists for this user.")

    def update_profile(
        self,
        user_id: str,
        update_data: MedicalProfileUpdate
    ) -> Optional[dict]:
        oid = self._parse_object_id(user_id)
        if not oid:
            return None

        raw_updates = update_data.model_dump(exclude_unset=True)
        if not raw_updates:
            return self.find_by_user_id(user_id)

        updates = self._serialize_dates(raw_updates)
        updates["updated_at"] = datetime.now(timezone.utc)

        updated_doc = self.profiles.find_one_and_update(
            {"user_id": oid},
            {"$set": updates},
            return_document=ReturnDocument.AFTER
        )

        return updated_doc
