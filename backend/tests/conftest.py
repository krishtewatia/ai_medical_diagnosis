import pytest
import mongomock
from app.main import app
from app.database.connection import get_database

_shared_mock_client = mongomock.MongoClient()
_shared_mock_db = _shared_mock_client["test_medical_api_db"]

@pytest.fixture(autouse=True)
def ensure_db_override():
    # Guarantee consistent mock database across test executions
    if get_database not in app.dependency_overrides:
        app.dependency_overrides[get_database] = lambda: _shared_mock_db
    yield
