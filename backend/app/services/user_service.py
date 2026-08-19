from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from bson.errors import InvalidId
from pymongo.database import Database


class UserService:

    def __init__(self, database: Database):
        self.users = database["users"]

    def find_by_email(self, email: str) -> Optional[dict]:
        return self.users.find_one({
            "email": email.lower()
        })

    def find_by_id(self, user_id: str) -> Optional[dict]:
        try:
            return self.users.find_one({
                "_id": ObjectId(user_id)
            })
        except (InvalidId, TypeError):
            return None

    def create_user(
        self,
        name: str,
        email: str,
        password_hash: str
    ) -> dict:
        now = datetime.now(timezone.utc)

        user = {
            "name": name,
            "email": email.lower(),
            "password_hash": password_hash,
            "created_at": now,
            "updated_at": now
        }

        result = self.users.insert_one(user)

        user["_id"] = result.inserted_id

        return user