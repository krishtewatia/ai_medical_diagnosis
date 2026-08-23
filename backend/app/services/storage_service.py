import hashlib
import hmac
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field

from app.core.config import settings


class StoredImageMetadata(BaseModel):
    """
    Standardized image metadata recorded in MongoDB history records.
    Strictly contains references and telemetry, NEVER raw binary image data.
    """
    file_name: str = Field(..., description="Original sanitized client filename.")
    content_type: str = Field(..., description="MIME content type (e.g. 'image/png').")
    file_size: int = Field(..., description="Total size in bytes.")
    storage_key: str = Field(..., description="Unique server-generated object storage path.")
    uploaded_at: str = Field(..., description="UTC timestamp in ISO 8601 format.")
    sha256: str = Field(..., description="SHA-256 cryptographic hash of image bytes.")


class StorageError(Exception):
    """Base exception for all storage service failures."""
    pass


class StorageUploadError(StorageError):
    """Raised when an object fails to upload to storage."""
    pass


class StorageNotFoundError(StorageError):
    """Raised when a requested object is missing in storage."""
    pass


class StorageSecurityError(StorageError):
    """Raised when a signed URL or path fails security verification."""
    pass


class StorageService:
    """
    Production-grade object storage service for clinical medical images.
    Supports Cloudflare R2 / AWS S3 S3-compatible backends and a robust local driver for offline environments.
    
    Guarantees:
    - Never stores raw binary data in MongoDB.
    - Prevents path traversal via server-generated random UUID keys.
    - Provides cryptographic SHA-256 integrity checksums.
    - Generates short-lived, signed temporary access URLs for frontend presentation.
    - Supports atomic cleanup on inference or database persistence failure.
    """

    def __init__(
        self,
        driver: Optional[str] = None,
        local_dir: Optional[str] = None,
        bucket_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
        signed_url_expire_seconds: Optional[int] = None
    ):
        self.driver = (driver or settings.storage_driver).lower()
        self.local_dir = Path(local_dir or settings.storage_local_dir)
        self.bucket_name = bucket_name or settings.storage_bucket_name
        self.endpoint_url = endpoint_url or settings.storage_endpoint_url
        self.access_key_id = access_key_id or settings.storage_access_key_id
        self.secret_access_key = secret_access_key or settings.storage_secret_access_key
        self.region_name = region_name or settings.storage_region_name
        self.signed_url_expire_seconds = signed_url_expire_seconds or settings.storage_signed_url_expire_seconds

        # Ensure local directory exists if using local driver
        if self.driver == "local":
            self.local_dir.mkdir(parents=True, exist_ok=True)

        self._s3_client: Optional[Any] = None

    def _get_s3_client(self) -> Any:
        """Initializes and returns boto3 S3 client for S3/R2 storage."""
        if self._s3_client is None:
            try:
                import boto3
                from botocore.config import Config
                self._s3_client = boto3.client(
                    "s3",
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.access_key_id,
                    aws_secret_access_key=self.secret_access_key,
                    region_name=self.region_name,
                    config=Config(signature_version="s3v4")
                )
            except ImportError:
                raise StorageError("boto3 package is required for S3/R2 object storage.")
        return self._s3_client

    def generate_storage_key(
        self,
        disease_id: str,
        user_id: str,
        filename: Optional[str] = None
    ) -> str:
        """
        Generates a secure, randomized server-side storage path.
        Pattern: medical_images/{disease_id}/{user_id}/{uuid}.{ext}
        """
        safe_disease = "".join(c for c in disease_id if c.isalnum() or c in "-_").lower()
        safe_user = "".join(c for c in str(user_id) if c.isalnum() or c in "-_")
        
        ext = "png"
        if filename:
            parsed_ext = Path(filename).suffix.lstrip(".").lower()
            if parsed_ext in ["png", "jpg", "jpeg", "dicom", "dcm"]:
                ext = parsed_ext

        unique_id = uuid.uuid4().hex
        return f"medical_images/{safe_disease}/{safe_user}/{unique_id}.{ext}"

    def upload_file(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str,
        disease_id: str,
        user_id: str
    ) -> StoredImageMetadata:
        """
        Uploads raw image bytes to object storage and returns metadata reference.
        """
        if not image_bytes or len(image_bytes) == 0:
            raise StorageUploadError("Cannot upload empty image bytes.")

        storage_key = self.generate_storage_key(
            disease_id=disease_id,
            user_id=user_id,
            filename=filename
        )

        sha256_hash = hashlib.sha256(image_bytes).hexdigest()
        uploaded_at = datetime.now(timezone.utc).isoformat()

        # 1. S3 / R2 Upload
        if self.driver in ["s3", "r2"]:
            try:
                s3 = self._get_s3_client()
                s3.put_object(
                    Bucket=self.bucket_name,
                    Key=storage_key,
                    Body=image_bytes,
                    ContentType=content_type,
                    Metadata={
                        "user_id": str(user_id),
                        "disease_id": disease_id,
                        "sha256": sha256_hash,
                        "uploaded_at": uploaded_at,
                    }
                )
            except Exception as e:
                raise StorageUploadError(f"Failed to upload object to S3/R2 storage: {str(e)}") from e

        # 2. Local Filesystem Driver Upload
        else:
            try:
                target_path = self.local_dir / storage_key
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, "wb") as f:
                    f.write(image_bytes)
            except Exception as e:
                raise StorageUploadError(f"Failed to write image to local storage driver: {str(e)}") from e

        return StoredImageMetadata(
            file_name=Path(filename).name,
            content_type=content_type,
            file_size=len(image_bytes),
            storage_key=storage_key,
            uploaded_at=uploaded_at,
            sha256=sha256_hash
        )

    def get_object(self, storage_key: str) -> bytes:
        """Retrieves raw object bytes from storage."""
        if not storage_key:
            raise StorageNotFoundError("Storage key cannot be empty.")

        # S3 / R2
        if self.driver in ["s3", "r2"]:
            try:
                s3 = self._get_s3_client()
                response = s3.get_object(Bucket=self.bucket_name, Key=storage_key)
                return response["Body"].read()
            except Exception as e:
                raise StorageNotFoundError(f"Object '{storage_key}' not found in S3/R2: {str(e)}") from e

        # Local
        else:
            target_path = self.local_dir / storage_key
            if not target_path.exists() or not target_path.is_file():
                raise StorageNotFoundError(f"Local storage object not found: {storage_key}")
            with open(target_path, "rb") as f:
                return f.read()

    def delete_object(self, storage_key: str) -> bool:
        """Deletes object from storage. Used for cleanup on failed predictions or explicit removal."""
        if not storage_key:
            return False

        # S3 / R2
        if self.driver in ["s3", "r2"]:
            try:
                s3 = self._get_s3_client()
                s3.delete_object(Bucket=self.bucket_name, Key=storage_key)
                return True
            except Exception as e:
                print(f"Warning: Failed to delete S3 object '{storage_key}': {e}")
                return False

        # Local
        else:
            try:
                target_path = self.local_dir / storage_key
                if target_path.exists() and target_path.is_file():
                    target_path.unlink()
                    return True
                return False
            except Exception as e:
                print(f"Warning: Failed to delete local storage object '{storage_key}': {e}")
                return False

    def has_object(self, storage_key: str) -> bool:
        """Returns True if object exists in storage."""
        if not storage_key:
            return False

        if self.driver in ["s3", "r2"]:
            try:
                s3 = self._get_s3_client()
                s3.head_object(Bucket=self.bucket_name, Key=storage_key)
                return True
            except Exception:
                return False
        else:
            target_path = self.local_dir / storage_key
            return target_path.exists() and target_path.is_file()

    def get_signed_url(
        self,
        storage_key: str,
        expires_in: Optional[int] = None
    ) -> str:
        """
        Generates a secure, short-lived signed access URL for frontend rendering.
        """
        exp = expires_in or self.signed_url_expire_seconds

        # S3 / R2 Presigned URL
        if self.driver in ["s3", "r2"]:
            try:
                s3 = self._get_s3_client()
                return s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket_name, "Key": storage_key},
                    ExpiresIn=exp
                )
            except Exception as e:
                raise StorageError(f"Failed to generate S3 presigned URL: {str(e)}") from e

        # Local HMAC Signed Access URL
        else:
            expiry_timestamp = int(time.time()) + exp
            message = f"{storage_key}:{expiry_timestamp}".encode("utf-8")
            signature = hmac.new(
                settings.jwt_secret_key.encode("utf-8"),
                message,
                hashlib.sha256
            ).hexdigest()
            return f"/api/storage/media/{storage_key}?expires={expiry_timestamp}&signature={signature}"

    def verify_signed_url(
        self,
        storage_key: str,
        expires: int,
        signature: str
    ) -> bool:
        """Verifies local HMAC signed URL validity and timestamp expiration."""
        if int(time.time()) > expires:
            raise StorageSecurityError("Signed media access URL has expired.")

        message = f"{storage_key}:{expires}".encode("utf-8")
        expected = hmac.new(
            settings.jwt_secret_key.encode("utf-8"),
            message,
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected):
            raise StorageSecurityError("Invalid signed media URL signature.")

        return True


# Default singleton instance
default_storage_service = StorageService()
