"""End-to-end API tests: health, upload, create/run audits, results, report.

CSV-mode flow is fast (small synthetic file, AUDIT_N_BOOT=100 from conftest).
The demo flow (slow) runs the frozen FairCV pipeline and asserts the frozen
M4/ethnicity DIR point estimate appears in the API response.
"""

import json

import pytest


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "engine" in body


# ── Datasets ─────────────────────────────────────────────────────────────────
def test_upload_csv(client, sample_csv):
    with open(sample_csv, "rb") as f:
        r = client.post("/api/datasets", files={"file": ("sample.csv", f, "text/csv")})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["row_count"] == 300
    assert body["column_count"] == 5
    assert "gender" in body["columns"]


def test_upload_wrong_extension(client, sample_csv):
    with open(sample_csv, "rb") as f:
        r = client.post("/api/datasets", files={"file": ("sample.exe", f, "application/octet-stream")})
    assert r.status_code == 415


def test_upload_empty_file(client):
    r = client.post("/api/datasets", files={"file": ("empty.csv", b"", "text/csv")})
    assert r.status_code == 422  # clear parse error, not a traceback
    r = client.post("/api/datasets", files={"file": ("ok.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 201


def test_demo_dataset_info(client):
    r = client.get("/api/datasets/demo")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "demo"
    assert body["groups"]["ethnicity"]  # present when FairCVdb.npy exists locally


# ── Audit lifecycle: csv mode (fast) ─────────────────────────────────────────
def test_csv_audit_flow(client, sample_csv):
    with open(sample_csv, "rb") as f:
        up = client.post("/api/datasets", files={"file": ("sample.csv", f, "text/csv")})
    dataset_id = up.json()["id"]

    r = client.post("/api/audits", json={
        "name": "csv-test",
        "dataset_mode": "csv",
        "dataset_id": dataset_id,
        "protected_attributes": ["gender", "ethnicity"],
        "config": {"label_column": "score"},
    })
    assert r.status_code == 201, r.text
    audit_id = r.json()["id"]

    r = client.post(f"/api/audits/{audit_id}/run")
    assert r.status_code == 202, r.text

    # TestClient runs background tasks before returning, so it should be done
    r = client.get(f"/api/audits/{audit_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "completed", r.json().get("error")

    # Metrics
    m = client.get(f"/api/audits/{audit_id}/metrics").json()
    assert len(m["metrics"]) == 2  # gender + ethnicity
    for row in m["metrics"]:
        for key in ("DPD", "DIR", "EOD", "EO", "KL_extreme",
                    "DIR_ci_lo", "DIR_ci_hi", "EEOC_pass"):
            assert key in row

    # Statistics
    s = client.get(f"/api/audits/{audit_id}/statistics").json()
    assert len(s["statistics"]) >= 4  # KS pairs + chi2 (+ KW for 3-group ethnicity)

    # Robustness: p75 + topN rows for each attribute
    rob = client.get(f"/api/audits/{audit_id}/robustness").json()["robustness"]
    protocols = {(r["attribute"], r["protocol"]) for r in rob["rows"]}
    assert ("gender", "p75") in protocols
    assert ("gender", "topN") in protocols
    assert ("ethnicity", "p75") in protocols

    # Report
    rep = client.get(f"/api/audits/{audit_id}/report")
    assert rep.status_code == 200
    assert "FairCV Bias Audit — Report" in rep.json()["content"]


def test_csv_audit_bad_attr(client, sample_csv):
    with open(sample_csv, "rb") as f:
        up = client.post("/api/datasets", files={"file": ("sample.csv", f, "text/csv")})
    dataset_id = up.json()["id"]
    r = client.post("/api/audits", json={
        "name": "bad-attr",
        "dataset_mode": "csv",
        "dataset_id": dataset_id,
        "protected_attributes": ["nonexistent"],
        "config": {"label_column": "score"},
    })
    assert r.status_code == 422
    assert "not a column" in r.json()["detail"]


def test_csv_audit_missing_label(client, sample_csv):
    with open(sample_csv, "rb") as f:
        up = client.post("/api/datasets", files={"file": ("sample.csv", f, "text/csv")})
    dataset_id = up.json()["id"]
    r = client.post("/api/audits", json={
        "name": "no-label",
        "dataset_mode": "csv",
        "dataset_id": dataset_id,
        "protected_attributes": ["gender"],
        "config": {},
    })
    assert r.status_code == 422
    assert "label_column" in r.json()["detail"]


# ── Audit lifecycle: demo mode (slow — frozen pipeline) ──────────────────────
@pytest.mark.slow
def test_demo_audit_frozen_headline(client, monkeypatch):
    """Run the frozen demo pipeline through the API and confirm the frozen
    M4/ethnicity DIR point estimate (0.7711) appears in the response.
    Skipped when FairCVdb.npy is not present locally."""
    import os
    from pathlib import Path

    from app.config import settings

    # The frozen CI bounds require the full 2,000 resamples (conftest uses 100
    # for speed); restore the frozen bootstrap size for this test.
    monkeypatch.setattr(settings, "AUDIT_N_BOOT", 2000)

    if not Path(settings.FAIRCV_DB_PATH).exists():
        pytest.skip("FairCVdb.npy not present locally")

    r = client.post("/api/audits", json={
        "name": "demo-frozen-check",
        "dataset_mode": "demo",
        "protected_attributes": ["gender", "ethnicity"],
    })
    assert r.status_code == 201, r.text
    audit_id = r.json()["id"]

    r = client.post(f"/api/audits/{audit_id}/run")
    assert r.status_code == 202, r.text

    status = client.get(f"/api/audits/{audit_id}").json()
    assert status["status"] == "completed", status.get("error")

    m = client.get(f"/api/audits/{audit_id}/metrics").json()
    m4 = [row for row in m["metrics"]
          if row["model"] == "M4-Ethnicity-Biased (CV7)" and row["attribute"] == "ethnicity"]
    assert len(m4) == 1
    assert abs(m4[0]["DIR"] - 0.7711) < 1e-3          # frozen point estimate
    assert m4[0]["EEOC_pass"] is False                 # below 0.80 -> not passing
    assert m4[0]["DIR_ci_hi"] >= 0.80                  # CI crosses 0.80 (borderline)

    # Robustness protocols available for demo
    rob = client.get(f"/api/audits/{audit_id}/robustness").json()["robustness"]
    demo_protocols = {r["protocol"] for r in rob["rows"]}
    assert {"top1000", "p75"} <= demo_protocols


def test_create_demo_audit_rejects_unknown_attr(client):
    r = client.post("/api/audits", json={
        "name": "bad-demo",
        "dataset_mode": "demo",
        "protected_attributes": ["age"],
    })
    assert r.status_code == 422
    assert "Demo audits support" in r.json()["detail"]
