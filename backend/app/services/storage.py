"""File storage for uploaded datasets.

Uploads live under ``backend/data/uploads/`` (git-ignored), stored with a
UUID prefix so filenames can never collide or traverse directories. Deletion
only ever touches files inside the upload directory.
"""

import uuid
from pathlib import Path

from ..config import BASE_DIR, settings
from ..security import sanitize_filename

UPLOAD_DIR = settings.UPLOAD_DIR


def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_upload(file_bytes: bytes, original_name: str) -> Path:
    """Persist an upload under a unique, sanitized name. Returns the Path."""
    ensure_dirs()
    safe = sanitize_filename(original_name)
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}_{safe}"
    dest.write_bytes(file_bytes)
    return dest


def stored_rel_path(path: Path) -> str:
    """Store paths relative to backend/ so they stay valid across machines."""
    return str(path.resolve().relative_to(BASE_DIR.resolve()))


def resolve_upload(rel_path: str) -> Path:
    """Resolve a stored path, refusing anything outside the upload directory."""
    p = (BASE_DIR / rel_path).resolve()
    root = UPLOAD_DIR.resolve()
    if not (p == root or root in p.parents):
        raise ValueError("Refusing to access a path outside the upload directory")
    return p


def delete_upload(rel_path: str | None) -> None:
    if not rel_path:
        return
    try:
        resolve_upload(rel_path).unlink(missing_ok=True)
    except ValueError:
        pass  # never delete outside the upload dir; nothing to do
