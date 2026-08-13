"""Audit endpoints: create, run (background), status, results, delete."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Audit, AuditResult, Dataset
from ..schemas import AuditCreate, AuditOut, AuditRunOut
from ..services.audit_runner import run_audit_job

router = APIRouter(prefix="/api/audits", tags=["audits"])


def _to_out(a: Audit) -> AuditOut:
    return AuditOut(
        id=a.id, name=a.name, status=a.status, stage=a.stage, progress=a.progress,
        dataset_mode=a.dataset_mode, dataset_id=a.dataset_id,
        protected_attributes=a.protected_attributes, config=a.config, error=a.error,
        created_at=a.created_at.isoformat() if a.created_at else None,
        completed_at=a.completed_at.isoformat() if a.completed_at else None,
    )


def _get_audit_or_404(db: Session, audit_id: str) -> Audit:
    a = db.get(Audit, audit_id)
    if a is None:
        raise HTTPException(status_code=404, detail="Audit not found")
    return a


def _require_result(audit_id: str, db: Session) -> AuditResult:
    r = db.get(AuditResult, audit_id)
    if r is None:
        raise HTTPException(
            status_code=409,
            detail="Audit has no results yet — it has not been run or did not complete.",
        )
    return r


@router.post("", response_model=AuditOut, status_code=201)
def create_audit(body: AuditCreate, db: Session = Depends(get_db)):
    """Create an audit. Demo mode uses the frozen FairCV dataset with the
    fixed gender/ethnicity attributes; csv mode needs a dataset_id and a
    config with 'label_column'."""
    if body.dataset_mode == "demo":
        attrs = set(body.protected_attributes)
        allowed = {"gender", "ethnicity"}
        if not attrs <= allowed:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Demo audits support protected attributes {sorted(allowed)} only "
                    f"(got {sorted(attrs)})."
                ),
            )
        dataset_id = None
    else:
        if not body.dataset_id:
            raise HTTPException(
                status_code=422, detail="dataset_id is required for csv-mode audits."
            )
        dataset = db.get(Dataset, body.dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Referenced dataset not found.")
        cfg = body.config or {}
        if not cfg.get("label_column"):
            raise HTTPException(
                status_code=422,
                detail="config.label_column is required for csv-mode audits.",
            )
        # Validate protected attributes exist in the uploaded columns
        for attr in body.protected_attributes:
            if attr not in (dataset.columns or []):
                raise HTTPException(
                    status_code=422,
                    detail=f"Protected attribute '{attr}' is not a column of the uploaded dataset.",
                )
        dataset_id = body.dataset_id

    audit = Audit(
        name=body.name,
        status="pending",
        stage=None,
        progress=0.0,
        dataset_mode=body.dataset_mode,
        dataset_id=dataset_id,
        protected_attributes=body.protected_attributes,
        config=body.config,
    )
    db.add(audit)
    db.commit()
    db.refresh(audit)
    return _to_out(audit)


@router.get("", response_model=list[AuditOut])
def list_audits(db: Session = Depends(get_db)):
    return [_to_out(a) for a in db.query(Audit).order_by(Audit.created_at.desc()).all()]


@router.get("/{audit_id}", response_model=AuditOut)
def get_audit(audit_id: str, db: Session = Depends(get_db)):
    return _to_out(_get_audit_or_404(db, audit_id))


@router.post("/{audit_id}/run", response_model=AuditRunOut, status_code=202)
def run_audit(audit_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Start the audit in the background; poll GET /api/audits/{id} for status."""
    audit = _get_audit_or_404(db, audit_id)
    if audit.status in ("running", "validating"):
        raise HTTPException(status_code=409, detail="Audit is already running.")
    audit.status = "pending"
    audit.error = None
    db.commit()
    background_tasks.add_task(run_audit_job, audit_id)
    return AuditRunOut(
        id=audit.id, status="queued", stage=None, progress=0.0,
        message="Audit queued. Poll GET /api/audits/{id} for progress.",
    )


@router.get("/{audit_id}/metrics")
def audit_metrics(audit_id: str, db: Session = Depends(get_db)):
    result = _require_result(audit_id, db)
    return {
        "audit_id": audit_id,
        "metrics": result.metrics or [],
        "per_group": result.per_group or [],
        "performance": result.performance or {},
    }


@router.get("/{audit_id}/statistics")
def audit_statistics(audit_id: str, db: Session = Depends(get_db)):
    result = _require_result(audit_id, db)
    return {"audit_id": audit_id, "statistics": result.statistics or []}


@router.get("/{audit_id}/robustness")
def audit_robustness(audit_id: str, db: Session = Depends(get_db)):
    result = _require_result(audit_id, db)
    return {"audit_id": audit_id, "robustness": result.robustness or {}}


@router.delete("/{audit_id}", status_code=204)
def delete_audit(audit_id: str, db: Session = Depends(get_db)):
    audit = _get_audit_or_404(db, audit_id)
    result = db.get(AuditResult, audit_id)
    if result is not None:
        db.delete(result)
    db.delete(audit)
    db.commit()
