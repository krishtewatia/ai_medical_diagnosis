"""
Database Index Initialization & Verification Script for Production MongoDB Atlas.

Ensures the following critical production indexes exist:
1. users: { email: 1 } [UNIQUE]
2. medical_profiles: { user_id: 1 } [UNIQUE]
3. predictions: { user_id: 1, created_at: -1 }
4. predictions: { user_id: 1, disease: 1, created_at: -1 }
"""

import sys
from pathlib import Path
from pymongo import ASCENDING, DESCENDING, MongoClient

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings


def init_database_indexes():
    print(f"Connecting to MongoDB database '{settings.database_name}'...")
    client = MongoClient(settings.mongodb_uri)
    db = client[settings.database_name]

    # 1. users indexes
    print("\n1. Provisioning 'users' collection indexes...")
    users_col = db["users"]
    users_col.create_index([("email", ASCENDING)], unique=True, name="idx_users_email_unique")
    print("   -> Created index: idx_users_email_unique on users.email [UNIQUE]")

    # 2. medical_profiles indexes
    print("\n2. Provisioning 'medical_profiles' collection indexes...")
    profiles_col = db["medical_profiles"]
    profiles_col.create_index([("user_id", ASCENDING)], unique=True, name="idx_profiles_user_id_unique")
    print("   -> Created index: idx_profiles_user_id_unique on medical_profiles.user_id [UNIQUE]")

    # 3. predictions indexes
    print("\n3. Provisioning 'predictions' collection indexes...")
    preds_col = db["predictions"]
    preds_col.create_index(
        [("user_id", ASCENDING), ("created_at", DESCENDING)],
        name="idx_predictions_user_created_at"
    )
    print("   -> Created index: idx_predictions_user_created_at on {user_id: 1, created_at: -1}")

    preds_col.create_index(
        [("user_id", ASCENDING), ("disease", ASCENDING), ("created_at", DESCENDING)],
        name="idx_predictions_user_disease_created_at"
    )
    print("   -> Created index: idx_predictions_user_disease_created_at on {user_id: 1, disease: 1, created_at: -1}")

    print("\n========================================================")
    print("  ALL PRODUCTION DATABASE INDEXES SUCCESSFULLY PROVISIONED!")
    print("========================================================\n")


if __name__ == "__main__":
    try:
        init_database_indexes()
    except Exception as e:
        print(f"\n[ERROR] Failed to provision indexes: {e}")
        sys.exit(1)
