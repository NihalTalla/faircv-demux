"""Shared fixtures. Environment is configured BEFORE any app import so the
settings object picks up an isolated test database."""

import os
import tempfile
from pathlib import Path

_tmp_dir = tempfile.mkdtemp(prefix="faircv_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_tmp_dir) / 'test.db'}"
os.environ["AUDIT_N_BOOT"] = "100"   # keep API tests fast; the equivalence test
                                     # explicitly uses the frozen n_boot=2000

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def sample_csv(tmp_path):
    """A small, valid audit input: numeric label, 2 protected attrs, features."""
    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame({
        "score": rng.normal(0.5, 0.15, n).clip(0, 1),
        "feature1": rng.normal(size=n),
        "feature2": rng.normal(size=n),
        "gender": rng.choice(["M", "F"], n),
        "ethnicity": rng.choice(["A", "B", "C"], n),
    })
    p = tmp_path / "sample.csv"
    df.to_csv(p, index=False)
    return str(p)
