from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from bson import ObjectId
from bson.errors import InvalidId
from pymongo import ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.errors import PyMongoError

from app.database.connection import get_database
from app.schemas.prediction_history import (
    PredictionHistoryCreate,
    PredictionHistoryResponse,
    PredictionModelInfo,
    PredictionResultRecord,
)


class PredictionHistoryError(Exception):
    """Base exception for prediction history operations."""
    pass


class InvalidPredictionIdError(ValueError, PredictionHistoryError):
    """Raised when an invalid ObjectId string is supplied for a prediction."""
    pass


class InvalidUserIdError(ValueError, PredictionHistoryError):
    """Raised when an invalid ObjectId string is supplied for a user."""
    pass


class PredictionNotFoundError(KeyError, PredictionHistoryError):
    """Raised when a specific prediction is not found or does not belong to the user."""
    pass


class PredictionHistoryService:
    """
    MongoDB service layer for managing the immutable 'predictions' collection.
    
    Guarantees:
    1. Strict User Isolation: Every query and insertion is bound to the authenticated user's ObjectId.
    2. Immutability: Historical records are append-only; no update/edit operations exist.
    3. Index-Backed Queries: Optimized with compound indexes for chronologically ordered and disease-filtered feeds.
    """

    MAX_LIMIT = 100
    DEFAULT_LIMIT = 20

    def __init__(self, db: Optional[Database] = None):
        self.db = db if db is not None else get_database()
        self.collection = self.db["predictions"]

    def ensure_indexes(self) -> None:
        """
        Creates compound indexes on the predictions collection:
        1. { user_id: 1, created_at: -1 } -> User feed ordered newest first
        2. { user_id: 1, disease: 1, created_at: -1 } -> Disease-filtered user feed
        """
        try:
            self.collection.create_index(
                [("user_id", ASCENDING), ("created_at", DESCENDING)],
                name="idx_user_history_created_at"
            )
            self.collection.create_index(
                [("user_id", ASCENDING), ("disease", ASCENDING), ("created_at", DESCENDING)],
                name="idx_user_disease_history"
            )
        except (PyMongoError, Exception) as e:
            # Avoid crashing if index creation is restricted in test environments
            print(f"Warning: Index creation on predictions collection failed: {e}")

    def _parse_user_id(self, user_id: Union[str, ObjectId]) -> ObjectId:
        """Converts user_id to ObjectId, raising InvalidUserIdError if malformed."""
        if isinstance(user_id, ObjectId):
            return user_id
        if not user_id or not isinstance(user_id, str) or not ObjectId.is_valid(user_id):
            raise InvalidUserIdError(f"Invalid user ID format: '{user_id}'")
        return ObjectId(user_id)

    def _parse_prediction_id(self, prediction_id: Union[str, ObjectId]) -> ObjectId:
        """Converts prediction_id to ObjectId, raising InvalidPredictionIdError if malformed."""
        if isinstance(prediction_id, ObjectId):
            return prediction_id
        if not prediction_id or not isinstance(prediction_id, str) or not ObjectId.is_valid(prediction_id):
            raise InvalidPredictionIdError(f"Invalid prediction ID format: '{prediction_id}'")
        return ObjectId(prediction_id)

    def _format_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Converts Mongo ObjectIds to clean string representations for API responses."""
        if not doc:
            return doc
        formatted = dict(doc)
        formatted["id"] = str(formatted.pop("_id"))
        formatted["user_id"] = str(formatted["user_id"])
        return formatted

    def create_prediction(
        self,
        user_id: Union[str, ObjectId],
        payload: Union[PredictionHistoryCreate, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Persists a verified prediction record into MongoDB.
        
        user_id is derived strictly from authentication.
        """
        u_id = self._parse_user_id(user_id)

        # Validate with Pydantic if raw dict provided
        if isinstance(payload, dict):
            # Inject authenticated user_id
            payload_data = dict(payload)
            payload_data["user_id"] = str(u_id)
            record = PredictionHistoryCreate(**payload_data)
        else:
            record = payload

        doc = record.model_dump()
        # Convert user_id to BSON ObjectId for storage
        doc["user_id"] = u_id
        doc["_id"] = ObjectId()
        if "created_at" not in doc or doc["created_at"] is None:
            doc["created_at"] = datetime.now(timezone.utc)

        try:
            self.collection.insert_one(doc)
            return self._format_doc(doc)
        except PyMongoError as e:
            raise PredictionHistoryError(f"Database error while saving prediction record: {str(e)}") from e

    def get_user_predictions(
        self,
        user_id: Union[str, ObjectId],
        disease: Optional[str] = None,
        skip: int = 0,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Retrieves paginated history for the authenticated user, newest first.
        Optionally filters by disease ID.
        """
        u_id = self._parse_user_id(user_id)
        clamped_skip = max(0, skip)
        clamped_limit = max(1, min(limit, self.MAX_LIMIT))

        query: Dict[str, Any] = {"user_id": u_id}
        if disease and disease.strip():
            query["disease"] = disease.strip().lower()

        try:
            cursor = (
                self.collection.find(query)
                .sort("created_at", DESCENDING)
                .skip(clamped_skip)
                .limit(clamped_limit)
            )
            return [self._format_doc(doc) for doc in cursor]
        except PyMongoError as e:
            raise PredictionHistoryError(f"Database error while querying user prediction history: {str(e)}") from e

    def count_user_predictions(
        self,
        user_id: Union[str, ObjectId],
        disease: Optional[str] = None
    ) -> int:
        """Counts total predictions for pagination calculation."""
        u_id = self._parse_user_id(user_id)
        query: Dict[str, Any] = {"user_id": u_id}
        if disease and disease.strip():
            query["disease"] = disease.strip().lower()

        try:
            return self.collection.count_documents(query)
        except PyMongoError as e:
            raise PredictionHistoryError(f"Database error while counting predictions: {str(e)}") from e

    def get_prediction_by_id(
        self,
        user_id: Union[str, ObjectId],
        prediction_id: Union[str, ObjectId]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single prediction by ID.
        
        CRITICAL SECURITY REQUIREMENT:
        Query matches both _id AND user_id to strictly prevent User A from reading User B's record.
        """
        u_id = self._parse_user_id(user_id)
        p_id = self._parse_prediction_id(prediction_id)

        try:
            doc = self.collection.find_one({"_id": p_id, "user_id": u_id})
            return self._format_doc(doc) if doc else None
        except PyMongoError as e:
            raise PredictionHistoryError(f"Database error while retrieving prediction: {str(e)}") from e
