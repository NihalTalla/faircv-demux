"""Upload security: size limits, file-type allowlist, filename sanitization.

Uploaded files are treated as sensitive data. The product never executes
anything from an upload — files are parsed as CSV only (pandas), stored under
a UUID-prefixed name in a git-ignored directory, and never logged by content.
"""

import re
from pathlib import Path

from fastapi import HTTPException

from .config import settings

MAX_BYTES = settings.MAX_UPLOAD_MB * 1024 * 1024


def sanitize_filename(name: str) -> str:
    """Strip directories and dangerous characters; keep a safe basename.

    Handles both ``/`` and ``\\`` separators (Windows uploads) and replaces
    anything outside a safe charset with ``_``.
    """
    base = Path(name).name  # strips any path components
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return base[:200] or "upload.csv"


def validate_upload(filename: str, size: int) -> None:
    """Reject unsupported types and oversized files with clear errors."""
    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext or '(none)'}'. "
                f"Allowed: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}."
            ),
        )
    if size > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {size / 1e6:.1f} MB exceeds the "
                f"{settings.MAX_UPLOAD_MB} MB limit."
            ),
        )
