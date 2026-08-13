# Project Freeze Declaration — FairCV Bias Audit

**Date:** 2026-08-13
**Status: PROJECT FROZEN.** The audit methodology, all experiment code, all result artifacts, and the final report are declared immutable as of this date.

---

## 1. What "frozen" means

- **No further methodology changes** to the numeric audit (`faircv_audit_v2.py`), the companion analyses, or any result artifact.
- **No frozen artifact may be overwritten.** All outputs of any future work must be new files, clearly marked as post-freeze additions.
- **Future work is allowed only as additive companion experiments** (new scripts, new `results/…` subfolders, new appendices) that leave the frozen set untouched, mirroring how `topn_screening`, `topn_robustness`, and the bio arm were added.
- **Ground truth for any downstream claim is the frozen artifact set below.**

## 2. Frozen artifact inventory

### 2.1 Core numeric audit (methodology + results)

| Artifact | Role |
|---|---|
| `faircv_audit_v2.py` | corrected/frozen audit implementation (R-1..R-15; seed 42, n_boot 2000) |
| `faircv_audit.py` | original audit (unchanged; numerics sound, narrative flawed — superseded by v2) |
| `results/metrics.csv` | 12 rows: DPD/DIR/EOD/EO/KL point estimates + bootstrap CIs + effect sizes + EEOC |
| `results/statistical_tests.csv` | 42 rows: KS(+Holm)/Kruskal–Wallis/χ² |
| `results/per_group_metrics.csv` | 30 rows: N/SR/TPR/FPR/PPV/mean-score per group |
| `results/audit_report.txt` | full v2 console report |
| `results/fig1..fig5*.png` | selection rates, heatmap, distributions, coefficients, CI plots |

### 2.2 Verification & ground truth

| Artifact | Role |
|---|---|
| `dataset_ground_truth.md` | verified dataset schema, distributions, bias construction, cols 31–50 finding |
| `audit_code_review.md` | v1 review (BUG-1..11, N-1..N-10, R-1..R-15) |
| `faircv_audit_v2_validation.md` | independent validation of v2 numerics + dataset defect (§5) |
| `ground_truth_verify.py`, `gt_verify_run.txt` | independent ground-truth verification harness + run log |

### 2.3 Companion analyses (frozen, additive)

| Artifact | Role |
|---|---|
| `topn_screening.py`, `results/topn_metrics.csv`, `results/topn_per_group.csv`, `results/fig6_topn_selection.png`, `topn_screening.md` | top-2%..50% threshold sweep |
| `topn_robustness.py`, `results/robustness/top1000_metrics.csv`, `top1000_per_group.csv`, `p75_metrics.csv`, `p75_per_group.csv`, `fig9_robustness_dir.png`, `results/robustness/robustness_report.md` | repo-protocol robustness (top-1000 + 75th percentile) |
| `bio_arm.py`, `bio_leak_strong.py`, `bio_leak_bert.py`, `bio_ft_distilbert.py` | bio/text companion arm scripts |
| `results/bio_leakage.csv`, `results/bio_metrics.csv`, `results/bio_leak_strong.csv`, `results/bio_leak_finetune.csv`, `results/bio_report.txt`, `fig7`/`fig8` | bio/text outputs (incl. partial 1-epoch fine-tune, test AUC 0.7592) |
| `results/models/distilbert_ft_ep1.pt` | fine-tune checkpoint (partial run) |
| `bio_arm.md` | full bio/text analysis write-up |

### 2.4 Final deliverables (frozen)

| Artifact | Role |
|---|---|
| `FINAL_REPORT.md`, `FINAL_REPORT.pdf` | final technical audit report (12-page PDF) |
| `EXECUTIVE_BRIEF.md`, `EXECUTIVE_BRIEF.pdf` | one-page executive brief |
| `README.md` | repository entry point (updated for frozen state) |
| `verify_final_report.py` | verification pass 1 — 189 checks vs metric CSVs |
| `verify_final_report_pass2.py` | verification pass 2 — 225 checks vs prose artifacts (audit_report.txt, docs, topn/bio CSVs) |
| `export_pdf.py` | markdown → PDF export helper |

## 3. Reproducibility

Deterministic (seed 42 for model + bootstrap). On Windows, run with `PYTHONIOENCODING=utf-8` when output is redirected (cp1252 console).

| Step | Command |
|---|---|
| Numeric audit (rewrites `results/*.csv` + `audit_report.txt` — run only to reproduce) | `python faircv_audit_v2.py` |
| Top-N screening sweep | `python topn_screening.py` |
| Top-1000 / p75 robustness | `python topn_robustness.py` |
| Bio/text arm | `python bio_arm.py` (then `python bio_leak_strong.py`, `python bio_leak_bert.py [max_len]`, `python bio_ft_distilbert.py`) |
| Verification pass 1 | `python verify_final_report.py` |
| Verification pass 2 | `python verify_final_report_pass2.py` |
| PDF export | `python export_pdf.py FINAL_REPORT.md FINAL_REPORT.pdf` (add `--compact` for the brief) |

## 4. Verification status (at freeze)

- **Pass 1** (`verify_final_report.py`): 189/189 checks pass — every numeric claim in `FINAL_REPORT.md` (tables, CIs, tests, effect sizes, top-1000/p75, top-2%) verified against `results/metrics.csv`, `results/statistical_tests.csv`, `results/per_group_metrics.csv`, `results/topn_metrics.csv`, `results/robustness/*`.
- **Pass 2** (`verify_final_report_pass2.py`): 225/225 checks pass — `audit_report.txt` ↔ `metrics.csv` consistency (all points + CIs), dataset ground-truth claims, code-review claims, validation-doc claims, `topn_screening.md` ↔ CSVs, and all bio companion numbers (incl. the fine-tune AUC 0.7592 and the "exceeds word-level baseline" finding).
- PDFs verified: `FINAL_REPORT.pdf` (12 pages, all sections incl. Appendix A), `EXECUTIVE_BRIEF.pdf` (1 page).

## 5. Freeze statement

The FairCV bias audit is complete and frozen. Headline conclusions (see `FINAL_REPORT.md` §1 and `EXECUTIVE_BRIEF.md`): one demonstrated, magnitude-borderline disparity (M4/ethnicity, EEOC DIR 0.77 with CI straddling 0.80); one statistically supported error-rate disparity hidden by a single selection screen (M3/gender EOD 0.198); gender parity robust everywhere; no demonstrated face-proxy bias increase (association only); all verdicts unchanged under the repository's own top-1000 / 75th-percentile hiring protocol. Any modification to this frozen state invalidates the verification status recorded here.
