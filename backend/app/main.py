"""FairCV Bias Audit — API entry point.

Run locally:  uvicorn app.main:app --reload
Docs:         http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import init_db
from .routers import assistant, audits, datasets, reports

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="FairCV Bias Audit API",
    description=(
        "Deterministic fairness audit engine: dataset validation, model scoring, "
        "DPD/DIR/EOD/EO/KL metrics, KS/Kruskal-Wallis/chi-square + Holm, Cohen's "
        "effect sizes, 2,000-sample bootstrap 95% CIs, and median/p75/top-N "
        "hiring-rule robustness. Powered by the frozen, verified FairCV audit math."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(audits.router)
app.include_router(reports.router)
app.include_router(assistant.router)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "engine": "faircv-audit-v2 (frozen, verified)",
        "demo_dataset_available": settings.FAIRCV_DB_PATH.exists(),
    }
