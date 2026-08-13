"""Assistant router — proxies chat requests to the Groq-backed assistant service.

The Groq API key lives ONLY in settings (backend .env).
This router is the sole entry point from the frontend; it never exposes the key.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas import AssistantChatRequest, AssistantChatResponse
from ..services import assistant_service

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


@router.post("/chat", response_model=AssistantChatResponse)
def chat(body: AssistantChatRequest, db: Session = Depends(get_db)):
    try:
        return assistant_service.chat(db, body)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        log.exception("assistant chat failed")
        raise HTTPException(
            status_code=500,
            detail="The assistant service encountered an error. Check the backend logs.",
        )
