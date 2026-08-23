import io
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from bson import ObjectId

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.database.connection import get_database
from app.schemas.prediction_history import PredictionHistoryResponse
from app.schemas.report import MedicalReportResponse, ReportModelInfo
from app.services.prediction_history_service import PredictionHistoryService
from app.services.storage_service import StorageService, default_storage_service


class ReportError(Exception):
    """Base exception for report generation failures."""
    pass


class ReportNotFoundError(ReportError):
    """Raised when the referenced prediction record is not found."""
    pass


class ReportAccessDeniedError(ReportError):
    """Raised when a user attempts to access another user's report."""
    pass


class ReportService:
    """
    Service responsible for generating, persisting, and securing Clinical PDF Screening Reports.
    
    Guarantees:
    - Generated from immutable historical prediction records (never reruns ML inference).
    - Strict user ownership validation.
    - PDF binaries stored in object storage; MongoDB stores references only.
    - Professional, formatted clinical layout with clear non-diagnostic screening disclaimers.
    """

    def __init__(
        self,
        history_service: Optional[PredictionHistoryService] = None,
        storage_service: Optional[StorageService] = None,
        database: Optional[Any] = None
    ):
        self._history_service = history_service
        self._storage_service = storage_service
        self.db = database

    @property
    def history_service(self) -> PredictionHistoryService:
        if self._history_service is None:
            self._history_service = PredictionHistoryService(db=self.db)
        return self._history_service

    @property
    def storage_service(self) -> StorageService:
        if self._storage_service is None:
            self._storage_service = default_storage_service
        return self._storage_service

    def build_pdf_document(self, data: MedicalReportResponse) -> bytes:
        """
        Renders a clinical screening report PDF using ReportLab.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()

        # Custom Clinical Styles
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
            alignment=0,
            spaceAfter=4
        )

        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#64748b"),
            spaceAfter=12
        )

        section_heading = ParagraphStyle(
            "SectionHeading",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=10,
            spaceAfter=6
        )

        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155")
        )

        banner_title_pos = ParagraphStyle(
            "BannerTitlePos",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#991b1b")
        )

        banner_title_neg = ParagraphStyle(
            "BannerTitleNeg",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#065f46")
        )

        disclaimer_style = ParagraphStyle(
            "DisclaimerStyle",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#475569")
        )

        story = []

        # 1. Header Banner
        story.append(Paragraph("AI MEDICAL DIAGNOSIS CLINICAL REPORT", title_style))
        story.append(
            Paragraph(
                f"Automated AI Decision-Support Assessment • Document Reference: {data.report_id}",
                subtitle_style
            )
        )
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=12))

        # 2. Patient & Report Overview Metadata Table
        meta_table_data = [
            [
                Paragraph("<b>Patient / Practitioner:</b>", body_style),
                Paragraph(f"{data.user_name or 'Anonymous'} ({data.user_email or 'User'})", body_style),
                Paragraph("<b>Screening Date:</b>", body_style),
                Paragraph(data.prediction_date[:10], body_style),
            ],
            [
                Paragraph("<b>Disease Module:</b>", body_style),
                Paragraph(data.disease_display_name, body_style),
                Paragraph("<b>Model Family / Version:</b>", body_style),
                Paragraph(f"{data.model.model_type} ({data.model.version})", body_style),
            ],
            [
                Paragraph("<b>Evaluation Modality:</b>", body_style),
                Paragraph(data.input_type.capitalize(), body_style),
                Paragraph("<b>Decision Threshold:</b>", body_style),
                Paragraph(f"{(data.model.threshold * 100):.0f}%" if data.model.threshold is not None else "N/A", body_style),
            ],
        ]

        meta_table = Table(meta_table_data, colWidths=[130, 150, 130, 120])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 14))

        # 3. Primary Screening Outcome Banner
        is_pos = data.is_positive
        prob_text = f"Calibrated Risk Probability: {(data.probability * 100):.1f}%" if data.probability is not None else "Risk Probability: Categorical"
        
        banner_bg = colors.HexColor("#fee2e2") if is_pos else colors.HexColor("#d1fae5")
        banner_border = colors.HexColor("#f87171") if is_pos else colors.HexColor("#34d399")
        banner_title = banner_title_pos if is_pos else banner_title_neg

        outcome_data = [
            [
                Paragraph(f"<b>SCREENING OUTCOME:</b> {data.prediction.upper()}", banner_title),
            ],
            [
                Paragraph(
                    f"{prob_text} • Intended for clinician risk stratification and triaging.",
                    body_style
                )
            ]
        ]
        outcome_table = Table(outcome_data, colWidths=[530])
        outcome_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), banner_bg),
            ("BOX", (0, 0), (-1, -1), 1, banner_border),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ]))
        story.append(outcome_table)
        story.append(Spacer(1, 14))

        # 4. Input Summary Section
        story.append(Paragraph("Evaluated Clinical Features & Input Parameters", section_heading))
        
        input_rows = []
        if data.input_summary:
            items = list(data.input_summary.items())
            # Format inputs in a 2-column or key-value table
            for i in range(0, len(items), 2):
                col1_key, col1_val = items[i]
                col1_text = f"<b>{col1_key}:</b> {col1_val}"
                
                if i + 1 < len(items):
                    col2_key, col2_val = items[i + 1]
                    col2_text = f"<b>{col2_key}:</b> {col2_val}"
                else:
                    col2_text = ""

                input_rows.append([
                    Paragraph(col1_text, body_style),
                    Paragraph(col2_text, body_style)
                ])
        else:
            input_rows.append([Paragraph("No input feature record stored.", body_style), Paragraph("", body_style)])

        input_table = Table(input_rows, colWidths=[265, 265])
        input_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(input_table)
        story.append(Spacer(1, 14))

        # 5. AI Decision Support Explanation
        if data.explanation:
            story.append(Paragraph("AI Decision-Support Narrative", section_heading))
            exp_table = Table([[Paragraph(data.explanation, body_style)]], colWidths=[530])
            exp_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(exp_table)
            story.append(Spacer(1, 14))

        # 6. Regulatory Disclaimer & Safety Notice
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=8))
        
        disclaimer_text = (
            "<b>IMPORTANT CLINICAL NOTICE:</b> "
            + (data.disclaimer or "This AI screening report is generated by a computer algorithm for preliminary decision-support and record-keeping purposes only. It is NOT a definitive medical diagnosis and should never substitute professional medical evaluation, laboratory testing, or formal radiological review by a licensed healthcare practitioner.")
        )
        story.append(Paragraph(disclaimer_text, disclaimer_style))

        # Build PDF into memory
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    def get_or_generate_report(
        self,
        prediction_id: str,
        user_id: Union[str, ObjectId],
        user_name: Optional[str] = None,
        user_email: Optional[str] = None
    ) -> MedicalReportResponse:
        """
        Retrieves an existing report or generates a new one from the stored prediction record.
        Strictly verifies ownership: users can only generate reports for their own predictions.
        """
        uid_str = str(user_id)

        # 1. Fetch prediction record with ownership check
        item_doc = self.history_service.get_prediction_by_id(
            user_id=uid_str,
            prediction_id=prediction_id
        )
        if not item_doc:
            raise ReportNotFoundError(f"Prediction record '{prediction_id}' not found.")

        item = PredictionHistoryResponse.model_validate(item_doc)

        # 2. Build structured report data
        report_id = f"RPT-{uuid.uuid4().hex[:12].upper()}"
        created_at = datetime.now(timezone.utc).isoformat()
        storage_key = f"medical_reports/{item.disease}/{uid_str}/{report_id}.pdf"

        report_response = MedicalReportResponse(
            report_id=report_id,
            prediction_id=item.id,
            user_id=uid_str,
            user_name=user_name,
            user_email=user_email,
            disease=item.disease,
            disease_display_name=item.disease_display_name,
            input_type=item.input_type,
            prediction=item.result.prediction,
            is_positive=item.result.is_positive,
            probability=item.result.probability,
            model=ReportModelInfo(
                model_type=item.model.model_type,
                version=item.model.version,
                threshold=item.model.threshold,
            ),
            prediction_date=item.created_at.isoformat() if hasattr(item.created_at, "isoformat") else str(item.created_at),
            input_summary=item.input_data or {},
            explanation=item.explanation,
            disclaimer="This AI screening report is an automated decision-support output and does not represent a confirmed medical diagnosis.",
            storage_key=storage_key,
            created_at=created_at,
        )

        # 3. Generate PDF bytes and store in object storage
        pdf_bytes = self.build_pdf_document(report_response)
        
        self.storage_service.upload_file(
            image_bytes=pdf_bytes,
            filename=f"{report_id}.pdf",
            content_type="application/pdf",
            disease_id=f"reports_{item.disease}",
            user_id=uid_str
        )

        # 4. Generate temporary signed URL
        signed_url = self.storage_service.get_signed_url(storage_key)
        report_response.download_url = signed_url

        return report_response

    def get_report_pdf_bytes(
        self,
        prediction_id: str,
        user_id: Union[str, ObjectId],
        user_name: Optional[str] = None,
        user_email: Optional[str] = None
    ) -> Tuple[bytes, str]:
        """
        Retrieves raw PDF bytes and filename for direct HTTP downloading.
        """
        report_data = self.get_or_generate_report(
            prediction_id=prediction_id,
            user_id=user_id,
            user_name=user_name,
            user_email=user_email
        )
        pdf_bytes = self.build_pdf_document(report_data)
        filename = f"{report_data.disease}_{report_data.report_id}.pdf"
        return pdf_bytes, filename


# Default singleton instance
default_report_service = ReportService()
