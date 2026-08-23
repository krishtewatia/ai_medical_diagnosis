import io
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_current_user
from app.services.storage_service import (
    StorageNotFoundError,
    StorageSecurityError,
    default_storage_service,
)

router = APIRouter(
    prefix="/storage",
    tags=["Storage & Media"]
)


@router.get(
    "/signed-url",
    status_code=status.HTTP_200_OK,
    summary="Generate Temporary Signed Access URL for Medical Image"
)
def get_signed_media_url(
    storage_key: str = Query(..., description="Object storage path for the medical image."),
    current_user: dict = Depends(get_current_user),
):
    """
    Generates a secure, short-lived signed access URL for frontend preview.
    Ensures user isolation: user can only request signed URLs for their own stored images.
    """
    user_id = str(current_user["_id"])
    key_parts = storage_key.strip().split("/")

    # Expected key pattern: medical_images/{disease}/{user_id}/{uuid}.{ext}
    if len(key_parts) >= 3 and key_parts[2] != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: cannot access another user's clinical image."
        )

    try:
        url = default_storage_service.get_signed_url(storage_key)
        return {"storage_key": storage_key, "signed_url": url}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate signed URL: {str(e)}"
        )


@router.get(
    "/media/{path:path}",
    summary="Stream Media with Signed HMAC Verification"
)
def stream_signed_media(
    path: str,
    expires: int = Query(..., description="Expiration timestamp."),
    signature: str = Query(..., description="HMAC-SHA256 signature."),
):
    """
    Serves stored medical image file under local driver with valid signature verification.
    """
    try:
        default_storage_service.verify_signed_url(storage_key=path, expires=expires, signature=signature)
    except StorageSecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )

    try:
        data = default_storage_service.get_object(path)
        ext = path.split(".")[-1].lower()
        media_type = f"image/{ext}" if ext in ["png", "jpg", "jpeg"] else "application/octet-stream"
        return Response(content=data, media_type=media_type)
    except StorageNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Requested medical image was not found."
        )
