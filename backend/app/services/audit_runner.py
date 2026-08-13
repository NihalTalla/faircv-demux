"""Audit execution service.

Runs the verified audit engine (demo or generic mode) in a background job,
writes stage/progress to the Audit row, and persists the full result payload
(metrics, statistics, per-group, robustness, dataset summary, report) into
``audit_results``.

Logging discipline: only audit IDs and exception types are logged — never
dataset contents.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import settings
from ..database import SessionLocal
from ..models import Audit, AuditResult
from fairness_engine import demo_pipeline, generic_pipeline, robustness
from ..validation import DatasetValidationError, validate_csv

logger = logging.getLogger("faircv.audit_runner")


# ── JSON safety ──────────────────────────────────────────────────────────────
def _jsonable(value):
    """Recursively convert numpy types / NaN / DataFrame to JSON-safe values."""
    if isinstance(value, pd.DataFrame):
        return [_jsonable(r) for r in value.to_dict(orient="records")]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


def _float(v):
    f = float(v)
    return None if f != f else f


def _set_audit(db, audit, **fields):
    for k, v in fields.items():
        setattr(audit, k, v)
    db.commit()


# ── Job entry point (called from BackgroundTasks) ────────────────────────────
def run_audit_job(audit_id: str) -> None:
    db = SessionLocal()
    try:
        audit = db.get(Audit, audit_id)
        if audit is None:
            logger.warning("audit job for unknown id %s", audit_id)
            return
        _set_audit(db, audit, status="running", stage="validating",
                   progress=5, error=None, completed_at=None)

        payload = _execute(db, audit)

        _set_audit(db, audit, status="completed", stage=None, progress=100,
                   completed_at=datetime.now(timezone.utc))

        db.add(AuditResult(
            audit_id=audit.id,
            metrics=payload["metrics"],
            statistics=payload["statistics"],
            per_group=payload["per_group"],
            performance=payload["performance"],
            robustness=payload["robustness"],
            dataset_summary=payload["dataset_summary"],
            report_markdown=payload["report_markdown"],
        ))
        db.commit()
        logger.info("audit %s completed", audit_id)
    except Exception as exc:  # noqa: BLE001 — surface every failure to the user
        try:
            audit = db.get(Audit, audit_id)
            if audit is not None:
                _set_audit(db, audit, status="failed", stage=None,
                           error=f"{type(exc).__name__}: {exc}")
        except Exception:
            logger.exception("failed to record error for audit %s", audit_id)
        logger.warning("audit %s failed: %s", audit_id, type(exc).__name__)


# ── Execution ────────────────────────────────────────────────────────────────
def _execute(db, audit: Audit) -> dict:
    if audit.dataset_mode == "demo":
        return _execute_demo(db, audit)
    return _execute_csv(db, audit)


def _execute_demo(db, audit: Audit) -> dict:
    db_path = settings.FAIRCV_DB_PATH
    if not db_path.exists():
        raise RuntimeError(
            "The frozen demo dataset is not available on this server. "
            f"Set FAIRCV_DB_PATH to point at FairCVdb.npy (expected at {db_path})."
        )

    _set_audit(db, audit, stage="training", progress=20)
    result = demo_pipeline.run_faircv_audit(
        str(db_path), n_boot=settings.AUDIT_N_BOOT, seed=settings.AUDIT_SEED
    )

    # Robustness protocols (top-1000 + p75) over the freshly trained models
    _set_audit(db, audit, stage="robustness", progress=80)
    median_lookup = {
        (r["model"], r["attribute"]): r
        for r in result["metrics"].to_dict(orient="records")
    }
    rob = robustness.demo_robustness(
        result["runtime"], n_top=settings.TOP_N_DEMO,
        n_boot=settings.AUDIT_N_BOOT, seed=settings.AUDIT_SEED,
        median_lookup=median_lookup,
    )

    return _assemble_payload(db, audit, result, rob)


def _execute_csv(db, audit: Audit) -> dict:
    from ..models import Dataset

    dataset = db.get(Dataset, audit.dataset_id) if audit.dataset_id else None
    if dataset is None or not dataset.stored_path:
        raise DatasetValidationError(
            "The referenced dataset was not found. Re-upload the CSV and create a new audit."
        )
    from .storage import resolve_upload

    path = resolve_upload(dataset.stored_path)

    cfg = audit.config or {}
    label_column = cfg.get("label_column")
    if not label_column:
        raise DatasetValidationError(
            "No label column configured. Provide 'label_column' in the audit config."
        )
    feature_columns = cfg.get("feature_columns")
    test_ratio = float(cfg.get("test_ratio", settings.AUDIT_TEST_RATIO))

    # Re-validate with human-readable errors before running
    validate_csv(str(path), label_column, audit.protected_attributes, feature_columns)

    _set_audit(db, audit, stage="training", progress=25)
    df = pd.read_csv(path)

    result = generic_pipeline.run_generic_audit(
        df,
        label_column=label_column,
        protected_attributes=audit.protected_attributes,
        feature_columns=feature_columns,
        test_ratio=test_ratio,
        seed=settings.AUDIT_SEED,
        n_boot=settings.AUDIT_N_BOOT,
        model_name=audit.name or "Model-1",
        top_n_pct=settings.TOP_N_PCT,
    )

    _set_audit(db, audit, stage="robustness", progress=85)
    return _assemble_payload(db, audit, result, result["robustness"])


def _assemble_payload(db, audit, result, rob) -> dict:
    _set_audit(db, audit, stage="report", progress=95)

    metrics = _jsonable(result["metrics"])
    statistics = _jsonable(result["statistics"])
    per_group = _jsonable(result["per_group"])
    performance = _jsonable(result["performance"])
    dataset_summary = _jsonable(result["dataset_summary"])
    robustness_payload = _jsonable(rob)

    from .report_builder import build_report

    payload = {
        "metrics": metrics,
        "statistics": statistics,
        "per_group": per_group,
        "performance": performance,
        "robustness": robustness_payload,
        "dataset_summary": dataset_summary,
        "report_markdown": None,
    }
    payload["report_markdown"] = build_report(audit, payload)
    return payload
