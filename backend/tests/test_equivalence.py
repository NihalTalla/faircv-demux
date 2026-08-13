"""Equivalence test — the extracted engine must reproduce the frozen result
CSVs byte-for-byte when run with the frozen FairCV configuration.

This is the machine-checkable guarantee that the product consumes the
verified audit logic rather than a re-implementation of it.

Requires ``FairCVdb.npy`` (local, git-ignored). Marked ``slow`` because the
full pipeline (6 models × 2 attributes × 2000 bootstrap resamples) takes a
couple of minutes. Run with:  pytest -m slow tests/test_equivalence.py
"""

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from fairness_engine import audit_core
from fairness_engine.demo_pipeline import run_faircv_audit

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "FairCVdb.npy"
RESULTS_DIR = REPO_ROOT / "results"

pytestmark = pytest.mark.slow


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def assert_identical(df: pd.DataFrame, frozen_path: Path, label: str):
    tmp = Path(__file__).parent / f"_eq_{label}.csv"
    try:
        df.to_csv(tmp, index=False)
        assert sha256(tmp) == sha256(frozen_path), (
            f"{label} is NOT byte-identical to {frozen_path.name}. "
            f"Engine={sha256(tmp)} Frozen={sha256(frozen_path)}"
        )
    finally:
        tmp.unlink(missing_ok=True)


@pytest.mark.skipif(not DB_PATH.exists(), reason="FairCVdb.npy not present locally")
def test_engine_reproduces_frozen_csvs():
    result = run_faircv_audit(str(DB_PATH))  # n_boot=2000, seed=42 (defaults)

    assert_identical(result["metrics"], RESULTS_DIR / "metrics.csv", "metrics")
    assert_identical(result["statistics"], RESULTS_DIR / "statistical_tests.csv", "statistics")
    assert_identical(result["per_group"], RESULTS_DIR / "per_group_metrics.csv", "per_group")

    # Structural invariants from the freeze (row counts / shape)
    assert len(result["metrics"]) == 12
    assert len(result["statistics"]) == 42
    assert len(result["per_group"]) == 30
    assert result["dataset_summary"]["train_rows"] == 19200
    assert result["dataset_summary"]["test_rows"] == 4800
    assert result["dataset_summary"]["columns"] == 51

    # Spot-check a headline frozen value (M4/ethnicity DIR point estimate)
    row = result["metrics"]
    m4 = row[(row["model"] == "M4-Ethnicity-Biased (CV7)") & (row["attribute"] == "ethnicity")]
    assert abs(float(m4["DIR"].iloc[0]) - 0.7711) < 5e-4
    assert abs(float(m4["DIR_ci_lo"].iloc[0]) - 0.7180) < 5e-4
    assert abs(float(m4["DIR_ci_hi"].iloc[0]) - 0.8282) < 5e-4

@pytest.mark.skipif(not DB_PATH.exists(), reason="FairCVdb.npy not present locally")
def test_robustness_protocols_match_frozen():
    """Demo robustness (top-1000, p75) must reproduce the frozen point
    estimates and verdicts exactly; CI bounds match the frozen method within
    bootstrap sampling tolerance (the frozen script's CIs come from its own
    RNG draw order, so the percentile bounds differ only by sampling noise)."""
    from fairness_engine import robustness

    result = run_faircv_audit(str(DB_PATH))  # n_boot=2000, seed=42
    median_lookup = {(r["model"], r["attribute"]): r
                     for r in result["metrics"].to_dict(orient="records")}
    rob = robustness.demo_robustness(
        result["runtime"], n_top=1000, n_boot=audit_core.N_BOOT, seed=audit_core.RNG_SEED,
        median_lookup=median_lookup,
    )
    rows = pd.DataFrame(rob["rows"])

    frozen_top = pd.read_csv(RESULTS_DIR / "robustness" / "top1000_metrics.csv")
    frozen_p75 = pd.read_csv(RESULTS_DIR / "robustness" / "p75_metrics.csv")

    for protocol, frozen in (("top1000", frozen_top), ("p75", frozen_p75)):
        ours = rows[rows.protocol == protocol].reset_index(drop=True)
        merged = ours.merge(
            frozen, on=["model", "attribute"], suffixes=("_ours", "_frozen")
        )
        assert len(merged) == 12
        # Point estimates must match to machine precision (deterministic)
        assert (merged["DPD_ours"] - merged["DPD_frozen"]).abs().max() < 1e-9
        assert (merged["DIR_ours"] - merged["DIR_frozen"]).abs().max() < 1e-9
        # Verdicts must match exactly
        assert (merged["verdict_change_ours"] != merged["verdict_change_frozen"]).sum() == 0
        # CI bounds: same method/seed, bootstrap sampling tolerance
        assert (merged["DIR_ci_lo_ours"] - merged["DIR_ci_lo_frozen"]).abs().max() < 0.02
        assert (merged["DIR_ci_hi_ours"] - merged["DIR_ci_hi_frozen"]).abs().max() < 0.02
