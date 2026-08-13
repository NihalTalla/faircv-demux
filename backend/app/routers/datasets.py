"""Dataset endpoints: upload, list, inspect (including the frozen demo)."""

import io

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Dataset
from ..schemas import DatasetOut
from ..security import validate_upload
from ..services.storage import delete_upload, save_upload, stored_rel_path

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _to_out(d: Dataset) -> DatasetOut:
    return DatasetOut(
        id=d.id, name=d.name, source=d.source, filename=d.filename,
        row_count=d.row_count, column_count=d.column_count,
        columns=d.columns, groups=d.groups, label_column=d.label_column,
        created_at=d.created_at.isoformat() if d.created_at else None,
    )


@router.get("/demo", response_model=DatasetOut)
def demo_dataset():
    """Structural info about the frozen FairCV demo dataset (read-only)."""
    from fairness_engine.demo_info import demo_info

    info = demo_info()
    if info is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The frozen demo dataset (FairCVdb.npy) is not available on this "
                "server. Demo audits require it locally."
            ),
        )
    return DatasetOut(
        id="demo",
        name="FairCVdb (frozen release)",
        source="demo",
        filename="FairCVdb.npy",
        row_count=info["rows"],
        column_count=info["columns"],
        columns=info["columns_list"],
        groups=info["groups"],
        label_column=None,
        created_at=None,
    )


@router.post("", response_model=DatasetOut, status_code=201)
def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a CSV dataset. Structural checks only — protected attributes and
    label are validated when the audit is configured."""
    filename = file.filename or "upload.csv"
    data = file.file.read()
    size = len(data)
    validate_upload(filename, size)

    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse '{filename}' as CSV: {exc}") from exc
    if df.empty:
        raise HTTPException(status_code=422, detail="The uploaded file contains no data rows.")

    path = save_upload(data, filename)
    columns = [str(c) for c in df.columns]

    dataset = Dataset(
        name=filename,
        source="csv",
        filename=filename,
        stored_path=stored_rel_path(path),
        row_count=int(len(df)),
        column_count=len(columns),
        columns=columns,
        groups=None,          # groups are computed per protected attribute at audit time
        label_column=None,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return _to_out(dataset)


@router.get("", response_model=list[DatasetOut])
def list_datasets(db: Session = Depends(get_db)):
    return [_to_out(d) for d in db.query(Dataset).order_by(Dataset.created_at.desc()).all()]


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    d = db.get(Dataset, dataset_id)
    if d is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return _to_out(d)


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    d = db.get(Dataset, dataset_id)
    if d is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    from ..models import Audit

    in_use = db.query(Audit).filter(Audit.dataset_id == dataset_id).count()
    if in_use:
        raise HTTPException(
            status_code=409,
            detail=f"Dataset is used by {in_use} audit(s); delete those audits first.",
        )
    delete_upload(d.stored_path)
    db.delete(d)
    db.commit()
