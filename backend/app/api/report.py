from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.api.dependencies import get_current_user
from app.schemas.report import MedicalReportResponse
from app.services.report_service import (
    ReportAccessDeniedError,
    ReportNotFoundError,
    default_report_service,
)

router = APIRouter(
    prefix="/reports",
    tags=["Medical Reports"]
)


@router.get(
    "/{prediction_id}",
    response_model=MedicalReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve or Generate Medical Screening Report"
)
def get_or_generate_report(
    prediction_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Generates a structured medical screening report and signed PDF download reference
    from an immutable historical prediction record.
    Enforces strict user ownership: users can only generate reports for their own predictions.
    """
    user_id = str(current_user["_id"])
    user_name = current_user.get("name")
    user_email = current_user.get("email")

    try:
        return default_report_service.get_or_generate_report(
            prediction_id=prediction_id,
            user_id=user_id,
            user_name=user_name,
            user_email=user_email
        )
    except ReportNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record '{prediction_id}' not found."
        )
    except ReportAccessDeniedError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: cannot generate a report for another user's prediction."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate medical screening report: {str(e)}"
        )


@router.get(
    "/{prediction_id}/download",
    summary="Download Clinical Screening Report PDF"
)
def download_report_pdf(
    prediction_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Generates and streams the raw PDF binary directly with application/pdf Content-Type.
    Enforces strict user ownership.
    """
    user_id = str(current_user["_id"])
    user_name = current_user.get("name")
    user_email = current_user.get("email")

    try:
        pdf_bytes, filename = default_report_service.get_report_pdf_bytes(
            prediction_id=prediction_id,
            user_id=user_id,
            user_name=user_name,
            user_email=user_email
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except ReportNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record '{prediction_id}' not found."
        )
    except ReportAccessDeniedError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: cannot download a report for another user's prediction."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stream report PDF: {str(e)}"
        )
