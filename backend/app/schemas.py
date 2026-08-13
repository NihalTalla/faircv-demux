"""Pydantic schemas for the API."""

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Datasets ─────────────────────────────────────────────────────────────────
class DatasetOut(BaseModel):
    id: str
    name: str
    source: str
    filename: Optional[str] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    columns: Optional[list[str]] = None
    groups: Optional[dict[str, dict[str, int]]] = None
    label_column: Optional[str] = None
    created_at: Optional[str] = None


# ── Audits ───────────────────────────────────────────────────────────────────
class AuditCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    dataset_mode: str = Field(..., pattern="^(demo|csv)$")
    dataset_id: Optional[str] = None
    protected_attributes: list[str] = Field(..., min_length=1)
    # For demo mode: leave empty (gender + ethnicity are fixed by the frozen spec).
    # For csv mode: {"label_column": "...", "feature_columns": [...], "test_ratio": 0.2}
    config: Optional[dict[str, Any]] = None


class AuditOut(BaseModel):
    id: str
    name: str
    status: str
    stage: Optional[str] = None
    progress: Optional[float] = None
    dataset_mode: str
    dataset_id: Optional[str] = None
    protected_attributes: list[str]
    config: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class AuditRunOut(BaseModel):
    id: str
    status: str
    stage: Optional[str] = None
    progress: Optional[float] = None
    message: str


# ── Report ───────────────────────────────────────────────────────────────────
class ReportOut(BaseModel):
    audit_id: str
    format: str
    content: str


# ── Assistant ─────────────────────────────────────────────────────────────────
class AssistantMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str = Field(..., max_length=8000)


class AssistantChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    audit_id: Optional[str] = None
    context_page: Optional[str] = None
    history: list["AssistantMessage"] = Field(default_factory=list)


class AssistantChatResponse(BaseModel):
    reply: str
    has_audit_context: bool
    audit_id: Optional[str] = None
    model_used: str
