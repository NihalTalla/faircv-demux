"""Report endpoint — downloadable markdown generated from the audit's own output."""

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import AuditResult

router = APIRouter(prefix="/api/audits", tags=["reports"])


@router.get("/{audit_id}/report")
def get_report(audit_id: str, db: Session = Depends(get_db), download: bool = False):
    result = db.get(AuditResult, audit_id)
    if result is None or not result.report_markdown:
        raise HTTPException(
            status_code=409,
            detail="No report available — the audit has not completed.",
        )
    if download:
        return Response(
            content=result.report_markdown,
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="faircv-audit-{audit_id}.md"'
            },
        )
    return {"audit_id": audit_id, "format": "markdown", "content": result.report_markdown}
