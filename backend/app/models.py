"""ORM models — minimal, MVP-sized schema.

Only what a completed audit needs to be retrievable by ID:
  datasets      — uploaded CSVs (+ metadata); the frozen demo dataset is
                  referenced implicitly by audits with dataset_mode='demo'
  audits        — audit metadata, status, progress, error
  audit_results — the full result payload (metrics, statistics, per-group,
                  robustness, dataset summary, report) as JSON per audit

A ``users`` table is intentionally absent: the MVP has no authentication.
Reports are generated on demand from ``audit_results`` and are not persisted.
"""

import uuid
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(255), nullable=False)
    source = Column(String(16), nullable=False, default="csv")  # "demo" | "csv"
    filename = Column(String(255), nullable=True)               # sanitized original name
    stored_path = Column(String(512), nullable=True)            # relative to backend/
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    columns = Column(JSON, nullable=True)                       # [str, ...]
    groups = Column(JSON, nullable=True)                        # {attr: {group: count}}
    label_column = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Audit(Base):
    __tablename__ = "audits"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(255), nullable=False)
    status = Column(String(32), nullable=False, default="pending")
    # pending | validating | running | completed | failed
    stage = Column(String(64), nullable=True)
    progress = Column(Float, nullable=True)                     # 0..100
    dataset_mode = Column(String(16), nullable=False)           # "demo" | "csv"
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=True)
    protected_attributes = Column(JSON, nullable=False)         # [str, ...]
    config = Column(JSON, nullable=True)                        # label col, features, ...
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    dataset = relationship("Dataset")


class AuditResult(Base):
    __tablename__ = "audit_results"

    audit_id = Column(String(36), ForeignKey("audits.id"), primary_key=True)
    metrics = Column(JSON, nullable=True)
    statistics = Column(JSON, nullable=True)
    per_group = Column(JSON, nullable=True)
    performance = Column(JSON, nullable=True)
    robustness = Column(JSON, nullable=True)
    dataset_summary = Column(JSON, nullable=True)
    report_markdown = Column(Text, nullable=True)
    generated_at = Column(DateTime, default=datetime.utcnow)
