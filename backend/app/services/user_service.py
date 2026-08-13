from datetime import datetime, timezone

from pymongo.database import Database


class UserService:

    def __init__(self, database: Database):
        self.users = database["users"]

    def find_by_email(self, email: str):
        return self.users.find_one({
            "email": email.lower()
        })

    def create_user(
        self,
        name: str,
        email: str,
        password_hash: str
    ):
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