import certifi
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError

from app.core.config import settings


_client: Optional[MongoClient] = None
_database: Optional[Database] = None


def get_mongo_client() -> MongoClient:
    """Lazily initializes and returns the PyMongo MongoClient singleton."""
    global _client
    if _client is None:
        try:
            ca_file = certifi.where()
            _client = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=5000,
                tlsCAFile=ca_file
            )
        except Exception:
            _client = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=5000
            )
    return _client


def get_database() -> Database:
    """Returns the application MongoDB database instance."""
    global _database
    if _database is None:
        client = get_mongo_client()
        _database = client[settings.database_name]
    return _database


def check_database_connection() -> bool:
    """Verifies that the MongoDB connection is alive."""
    try:
        client = get_mongo_client()
        client.admin.command("ping")
        return True
    except (PyMongoError, Exception) as e:
        print(f"MongoDB connection error: {e}")
        return False