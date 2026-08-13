# FairCV Bias Audit — Backend API

FastAPI service exposing the **frozen, verified FairCV audit engine** as a
REST API. The scientific math is extracted verbatim into
[`fairness_engine/`](fairness_engine/); the equivalence test
(`tests/test_equivalence.py`) proves the extracted engine reproduces the
frozen `results/*.csv` byte-for-byte.

## Stack

- **API:** FastAPI + uvicorn
- **Engine:** pandas / numpy / scipy / scikit-learn (the frozen audit stack)
- **Storage:** SQLAlchemy — SQLite by default (zero infra); Postgres via
  `DATABASE_URL` (Supabase/Render compatible, see `docker-compose.yml`)

## Run

```bash
cd backend
pip install -r requirements.txt

# default: SQLite at backend/data/faircv.db
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive API docs.

### Postgres (optional)

```bash
docker compose up -d db          # from the repo root
export DATABASE_URL="postgresql+psycopg2://faircv:faircv@localhost:5432/faircv"
```

## Demo mode requires the frozen dataset

`dataset_mode="demo"` runs the frozen M1–M6 audit and needs `FairCVdb.npy`
(local-only, git-ignored). Point at it with:

```bash
export FAIRCV_DB_PATH=/absolute/path/to/FairCVdb.npy
```

## Tests

```bash
cd backend
python -m pytest                          # full suite (incl. slow)
python -m pytest -m "not slow"            # fast subset (no FairCVdb.npy needed)
```

The **equivalence test** is the load-bearing guarantee: it re-runs the
extracted engine on `FairCVdb.npy` and asserts the regenerated
`metrics.csv` / `statistical_tests.csv` / `per_group_metrics.csv` are
**byte-identical** to the frozen artifacts, and that the demo robustness
protocols (top-1000, p75) reproduce the frozen point estimates and verdicts.

## API surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | engine + dataset availability |
| GET | `/api/datasets/demo` | frozen demo dataset info |
| POST | `/api/datasets` | upload a CSV (validated, size-limited) |
| GET | `/api/datasets` / `/{id}` / DELETE `/{id}` | manage uploads |
| POST | `/api/audits` | create an audit (demo or csv mode) |
| POST | `/api/audits/{id}/run` | run in background (returns 202) |
| GET | `/api/audits/{id}` | status / progress / error |
| GET | `/api/audits/{id}/metrics` | DPD/DIR/EOD/EO/KL + per-group + performance |
| GET | `/api/audits/{id}/statistics` | KS / KW / χ² / Holm / effect sizes |
| GET | `/api/audits/{id}/robustness` | median / p75 / top-N protocols |
| GET | `/api/audits/{id}/report` | markdown report (`?download=true`) |

## Design notes

- **Layer separation:** the frozen research baseline (`faircv_audit_v2.py`,
  `results/*.csv`, figures) is never imported or modified. The product
  consumes extracted copies of the pure math; the equivalence test enforces
  that the copies are exact.
- **Wording discipline:** the report describes *observed disparities* with
  confidence intervals and never asserts causation. A DIR below 0.80 with a
  CI crossing 0.80 is reported as *borderline*, not as proof of
  discrimination.
- **Security:** uploads are size-limited, extension-allowlisted,
  UUID-named inside `backend/data/uploads/` (git-ignored), never executed,
  and never logged by content.
- **Errors are human:** dataset validation failures return readable messages
  ("Protected attribute 'gender' contains only one group…") instead of
  tracebacks.
