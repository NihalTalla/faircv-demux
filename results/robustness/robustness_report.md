# Robustness Validation — Repository Selection Protocol (Top-1000 + 75th-Percentile)

**Script:** `topn_robustness.py` (companion; the frozen `faircv_audit_v2.py` was **not** modified)
**Outputs:** `results/robustness/top1000_metrics.csv`, `top1000_per_group.csv`, `p75_metrics.csv`, `p75_per_group.csv`, `fig9_robustness_dir.png`
**Date:** 2026-08-13 · seed 42 · bootstrap n = 2,000 · comparison target: frozen median-threshold results (`results/metrics.csv`, `results/per_group_metrics.csv`)

---

## 1. What was tested

The frozen audit defines *hired* as `predicted score ≥ 0.5` (median-binarised training labels + default LR decision threshold) — an arbitrary rule. This pass replaces **only the selection rule** with the hiring protocol actually documented in the official FairCVtest repository (`FairCV.py`, BiDAlab/FairCVtest):

| Protocol | Repo function | Rule |
|---|---|---|
| **A — Top-1000** | `computeTopScore` / `testDemographicParity` | hire the **top 1,000** of 4,800 test scores (top ≈20.8%) |
| **B — 75th percentile** | `testEqualityOfOpportunity(..., p=75)` | threshold = **75th percentile of the training labels**; hired = score ≥ threshold; EEO test: per-group TPR = P(hired \| truly qualified) where truly qualified = label ≥ threshold |

Models M1–M6 are trained exactly as in the frozen audit (same features, same median-binarised labels, same seed), so predicted scores are identical; only the hiring rule changes. For Protocol B the threshold is taken from each model's own training label set (blind for M1/M2/M5/M6, gender-biased for M3, ethnicity-biased for M4), mirroring the frozen audit's per-label-set binarisation.

---

## 2. Reproducibility validation (models are identical to the frozen ones)

Before trusting the new rules, the script recomputes the median-split metrics from the freshly trained models and compares them against the frozen CSVs:

- **max |drift| vs `metrics.csv` (DPD/DIR): 1.1e-16 → models reproduce the frozen audit exactly.**
- **Internal consistency check:** the blind-label p75 thresholds come out at 0.4999 (75th percentile of blind train labels) — essentially the median-split 0.5 decision boundary. Under Protocol B, M1/M2/M5/M6 therefore reproduce the frozen selection rates to 6 decimals, an independent confirmation that both pipelines are the same models.

---

## 3. Protocol A — top-1000 selection (point estimates, bootstrap 95% CIs)

DIR = worst-group selection rate ÷ best-group selection rate; EEOC pass if DIR ≥ 0.80.

| Model | attr | top-1000 DIR [95% CI] | median DIR [95% CI] | EEOC (top-1000 / median) |
|---|---|---|---|---|
| M1-Fair (CV7) | gender | 0.906 [0.811, 0.989] | 0.975 [0.919, 0.999] | PASS / PASS |
| M1-Fair (CV7) | ethnicity | 0.909 [0.790, 0.978] | 0.963 [0.887, 0.989] | PASS / PASS |
| M2-Multimodal (CV7+Face) | gender | 0.891 [0.795, 0.987] | 0.970 [0.914, 0.999] | PASS / PASS |
| M2-Multimodal (CV7+Face) | ethnicity | 0.912 [0.793, 0.978] | 0.961 [0.886, 0.988] | PASS / PASS |
| M3-Gender-Biased (CV7) | gender | 0.895 [0.805, 0.989] | 0.974 [0.912, 0.999] | PASS / PASS |
| M3-Gender-Biased (CV7) | ethnicity | 0.934 [0.800, 0.981] | 0.965 [0.890, 0.990] | PASS / PASS |
| M4-Ethnicity-Biased (CV7) | gender | 0.989 [0.879, 0.998] | 0.980 [0.924, 0.999] | PASS / PASS |
| **M4-Ethnicity-Biased (CV7)** | **ethnicity** | **0.710 [0.619, 0.810]** | **0.771 [0.718, 0.828]** | **FAIL / FAIL** |
| M5-Robust (CV9) | gender | 0.950 [0.864, 0.997] | 0.992 [0.934, 0.999] | PASS / PASS |
| M5-Robust (CV9) | ethnicity | 0.886 [0.767, 0.965] | 0.954 [0.881, 0.987] | PASS / PASS |
| M6-Robust (CV9+Face) | gender | 0.950 [0.853, 0.997] | 0.992 [0.935, 0.999] | PASS / PASS |
| M6-Robust (CV9+Face) | ethnicity | 0.897 [0.773, 0.968] | 0.958 [0.887, 0.987] | PASS / PASS |

**Per-group selection rates, top-1000 vs median split** (worst-off group in **bold**):

| Model | attr | Group | top-1000 SR | median SR |
|---|---|---|---|---|
| M1 | gender | Male / Female | 0.198 / 0.219 | 0.471 / 0.483 |
| M1 | ethnicity | G1 / G2 / G3 | 0.218 / **0.198** / 0.210 | 0.488 / **0.470** / 0.472 |
| M2 | gender | Male / Female | 0.197 / 0.220 | 0.470 / 0.484 |
| M2 | ethnicity | G1 / G2 / G3 | 0.218 / **0.198** / 0.209 | 0.490 / **0.470** / 0.471 |
| M3 | gender | Male / Female | 0.197 / 0.220 | 0.468 / 0.481 |
| M3 | ethnicity | G1 / G2 / G3 | 0.213 / **0.199** / 0.213 | 0.484 / **0.467** / 0.472 |
| **M4** | **ethnicity** | G1 / G2 / G3 | 0.246 / 0.205 / **0.174** | 0.543 / 0.455 / **0.419** |
| M5 | ethnicity | G1 / G2 / G3 | 0.223 / **0.198** / 0.204 | 0.501 / **0.478** / 0.479 |
| M6 | ethnicity | G1 / G2 / G3 | 0.222 / **0.199** / 0.204 | 0.502 / 0.482 / **0.481** |

(Full 30-row table in `top1000_per_group.csv`.)

**Readings:**
- **Only M4/ethnicity fails EEOC under top-1000**, DIR 0.710 vs 0.771 under the median split — the failure is *slightly worse* under the repo's own protocol, and χ² (group × hired) is significant (p = 4×10⁻⁶). Its CI [0.619, 0.810] still just straddles 0.80, so the failure remains "real but borderline" — the same caveat as the frozen result.
- **G3 stays the worst-off ethnicity group under top-1000** (0.174 vs 0.246 for G1) — the ×0.75 label penalty shows through in selection rates regardless of how hiring is defined.
- **Gender parity holds everywhere** (all gender DIR ≥ 0.891, max DPD 0.024). The only gender cell even approaching significance is **M2 gender χ² p = 0.045** (marginal; multiple-comparison caveat applies).
- Worst-off-group "flips" (M5/M6 gender Male↔Female; M6 ethnicity G2↔G3) occur only where DIR ≥ 0.89 — near-zero disparity, noise level.

---

## 4. Protocol B — 75th-percentile threshold

Thresholds (each model's own train-label 75th percentile) and selected counts: blind-label models 0.4999 (n ≈ 2,288–2,342, ≈48%), M3 0.4597 (2,455), M4 0.5191 (2,194).

| Model | attr | DIR [95% CI] | EEOC | EOD [95% CI] | median EOD [95% CI] |
|---|---|---|---|---|---|
| M1 | gender | 0.975 [0.918, 0.999] | PASS | 0.000 [0.000, 0.017] | 0.012 [0.001, 0.046] |
| M1 | ethnicity | 0.963 [0.889, 0.990] | PASS | 0.002 [0.002, 0.026] | 0.039 [0.011, 0.082] |
| M2 | gender | 0.970 [0.917, 0.998] | PASS | 0.002 [0.000, 0.019] | 0.016 [0.001, 0.053] |
| M2 | ethnicity | 0.961 [0.882, 0.988] | PASS | 0.002 [0.002, 0.027] | 0.041 [0.012, 0.082] |
| M3 | gender | 0.991 [0.934, 0.999] | PASS | **0.087 [0.069, 0.106]** | **0.198 [0.164, 0.231]** |
| M3 | ethnicity | 0.961 [0.893, 0.989] | PASS | 0.017 [0.005, 0.054] | 0.067 [0.029, 0.111] |
| M4 | gender | 0.975 [0.916, 0.999] | PASS | 0.030 [0.002, 0.068] | 0.011 [0.001, 0.047] |
| **M4** | **ethnicity** | **0.751 [0.694, 0.807]** | **FAIL** | **0.170 [0.145, 0.198]** | **0.255 [0.219, 0.289]** |
| M5 | gender | 0.992 [0.931, 0.999] | PASS | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.018] |
| M5 | ethnicity | 0.954 [0.884, 0.987] | PASS | 0.000 [0.000, 0.000] | 0.004 [0.002, 0.028] |
| M6 | gender | 0.992 [0.933, 0.999] | PASS | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.019] |
| M6 | ethnicity | 0.958 [0.886, 0.990] | PASS | 0.000 [0.000, 0.000] | 0.003 [0.002, 0.026] |

**Readings:**
- **Same EEOC verdicts as the frozen audit.** M4/ethnicity fails (DIR 0.751, CI [0.694, 0.807], χ² p = 2.6×10⁻¹³). M3/gender passes selection parity and is *closer* to parity than under the median split (DIR 0.991 vs 0.974).
- **M3's gender error-rate disparity survives the rule change**: gender EOD 0.087 (CI excluding zero) — still by far the largest error-rate disparity of any model×attribute, and the direction is preserved (Female TPR 1.00 vs Male 0.913). The EOD magnitude differs from the median-split 0.198 because the ground-truth threshold differs (75th percentile vs median); the *conclusion* — M3 passes selection parity but shows substantial error-rate disparity — is unchanged.
- **M4/ethnicity EOD 0.170 [0.145, 0.198]** (vs 0.255 under the median split) — G3's TPR is 1.00, G1's 0.83: the ethnicity bias shows through in error rates too.
- **M5/M6 EOD = 0.000 exactly** (TPR 1.0 for every group — near-perfect separation, AUC 0.996). Note the p75 EOD CI [0, 0] collapses because the model is essentially deterministic at this threshold.

---

## 5. Do the frozen audit's conclusions change?

| Finding (frozen audit, median threshold) | Under repo protocol (top-1000, p75) | Verdict |
|---|---|---|
| **F1** Gender parity: all models pass, DPD tiny | Passes under both rules (all gender DIR ≥ 0.891, DPD ≤ 0.024); only M2's top-1000 χ² is marginally significant (p = 0.045) | **Robust — unchanged** |
| **F2** M4 ethnicity FAILs EEOC (DIR 0.771) | Fails under top-1000 (0.710 [0.619, 0.810]) and p75 (0.751 [0.694, 0.807]); χ² significant under both; G3 worst-off group under both | **Robust — unchanged** (slightly *worse* point estimate at top-1000; the CI-straddling-0.80 caveat remains) |
| **F3** M3 passes selection parity but gender EOD = 0.198 | Selection parity still passes; p75 gender EOD 0.087 (CI excludes 0), direction preserved | **Robust — unchanged** |
| **F4** M1/M2/M5/M6 ethnicity pass EEOC | Pass under top-1000 (0.886–0.912) and p75 (0.954–0.963) | **Robust — unchanged** under the repo's own protocol (contrast: they fail only at the much tighter top-2%/100 protocol tested in `topn_screening.md`) |
| **F5** M2 does not clearly increase disparity (DPD CI overlap; ethnicity KL decreases) | Top-1000: M2 gender DPD 0.0239 vs M1 0.0206 (CIs overlap), ethnicity DPD 0.0192 vs 0.0198; only weak top-1000 gender χ² (p = 0.045) | **Broadly unchanged** — no consistent face-proxy signal |

**Bottom line:** the arbitrary median threshold was **not** the driver of any headline finding. Under the repository's own documented selection protocol (top-1000 hiring, 75th-percentile EEO threshold), every frozen conclusion holds: gender parity is robust, M4's ethnicity EEOC failure persists (and is slightly stronger at top-1000), M3's error-rate disparity persists, and the blind models still pass the ethnicity screen at the repo's actual strictness (≈21% hired) — the earlier top-2% failures were specific to that much tighter, low-power screen.

---

## 6. Caveats & limitations

- Bootstrap CIs are percentile, n = 2,000, seed 42, with extreme groups re-derived per resample — the same convention as the frozen audit. Top-1000 CIs are wider than median-split CIs (fewer selections, ~1,000 of 4,800).
- The p75 EOD and median EOD use **different ground-truth thresholds** (75th percentile vs median of labels), so magnitudes are not directly comparable — only ordering/qualitative conclusions are.
- M3/M4 TPR/EOD values are against artificially biased label sets (frozen-audit R-9 caveat applies); selection-rate metrics are neutral across models.
- Models are regularised logistic regressions on the numeric profile block; blind-face block (cols 31–50) remains excluded; conclusions are specific to this setup.
- M2 gender χ² (p = 0.045) at top-1000 is marginal and uncorrected for multiple comparisons — read as weak evidence, not a finding.
- Protocol B follows the frozen audit's per-label-set threshold convention; the upstream repo applies the p75 EEO test to the *blind* labels only (same thresholds for every model). Both conventions agree on the headline results here.

## 7. Freeze status

`faircv_audit_v2.py` was **not** modified and no existing results were overwritten. All robustness outputs are new files under `results/robustness/`; `topn_robustness.py` joins the companion-script set alongside `topn_screening.py`.
