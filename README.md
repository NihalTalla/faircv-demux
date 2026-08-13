# FairCV Bias Audit

This repository contains the reproducible audit of the FairCV/FairCVtest automated screening experiment.

## Main deliverables

- `FINAL_REPORT.md` / `FINAL_REPORT.pdf` — final technical audit report (frozen evidence only; all claims machine-verified)
- `EXECUTIVE_BRIEF.md` / `EXECUTIVE_BRIEF.pdf` — one-page executive brief
- `PROJECT_FREEZE.md` — freeze declaration and artifact inventory
- `faircv_audit.py` — original audit implementation (frozen)
- `faircv_audit_v2.py` — corrected/frozen audit implementation (R-1..R-15)
- `dataset_ground_truth.md` — verified dataset structure and semantics
- `audit_code_review.md` — review of the original audit
- `faircv_audit_v2_validation.md` — validation of the corrected audit
- `topn_robustness.py` — top-1000 / 75th-percentile robustness analysis (results in `results/robustness/`)
- `topn_screening.py` — Top-N screening analysis (2%–50%)
- `bio_arm.py`, `bio_leak_strong.py`, `bio_leak_bert.py`, `bio_ft_distilbert.py` — bio/text companion arm
- `verify_final_report.py` — verification pass 1 (metrics-CSV checks, 189 checks)
- `verify_final_report_pass2.py` — verification pass 2 (prose-artifact cross-checks, 225 checks)
- `export_pdf.py` — markdown → PDF export helper

## Dataset

The original `FairCVdb.npy` dataset is intentionally not committed to Git because of its size.

## Status

The audit methodology and results are **frozen** (see `PROJECT_FREEZE.md`). Final numerical claims were independently verified against the generated result artifacts (two verification passes, all checks passing).
