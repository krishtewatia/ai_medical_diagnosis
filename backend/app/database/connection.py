import certifi
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from app.core.config import settings


try:
    ca_file = certifi.where()
    client = MongoClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=10000,
        tlsCAFile=ca_file
    )
except Exception:
    client = MongoClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=10000
    )

database = client[settings.database_name]


def get_database():
    return database


def check_database_connection():
    try:
        client.admin.command("ping")
        return True

    except PyMongoError as e:
        print(f"MongoDB connection error: {e}")
        return False