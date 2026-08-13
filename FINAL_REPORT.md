# FairCV Bias Audit — Final Report

**Date:** 2026-08-13
**Status:** Final. Compiled exclusively from the frozen evidence artifacts listed in §11. No methodology was changed, no code was modified, and **no new experiments were run** to produce this report.
**Primary sources of truth:** `results/metrics.csv`, `results/statistical_tests.csv`, `results/per_group_metrics.csv`, `results/audit_report.txt` (frozen audit run), `dataset_ground_truth.md`, `audit_code_review.md`, `faircv_audit_v2_validation.md`, `results/topn_metrics.csv` / `topn_screening.md`, and `results/robustness/*` (top-1000 / 75th-percentile robustness).

**Conventions used throughout.** *Point estimate* = value computed on the 4,800-profile test set. *Bootstrap 95% CI* = percentile interval from 2,000 resamples (seed 42), extreme groups re-derived per resample. *Statistically supported* = the relevant CI excludes the null and/or the test is significant at α = 0.05 (Holm-corrected where multiple comparisons occur). *Borderline / uncertain* = the point estimate crosses a decision threshold but the CI straddles it. *Association* is never reported as causation.

---

## 1. Executive Summary

1. **Gender parity is robust across every model, every metric, and every hiring rule tested.** All gender DPD point estimates are ≤ 0.0143 (median split) / ≤ 0.024 (top-1000), all gender DIR values are ≥ 0.89, all gender χ² tests are non-significant, and the lowest gender DPD (M5, 0.0038) has a CI [0.0005, 0.0331] consistent with near-parity. **Statistically supported.**

2. **Only M4 (ethnicity-biased labels) fails the EEOC four-fifths rule on ethnicity** — DIR 0.771 (point estimate), CI [0.718, 0.828]. The *underlying disparity is statistically unambiguous* (DPD CI [0.0905, 0.1573] excludes 0; χ² p = 4.1×10⁻¹²; Kruskal–Wallis p = 7.9×10⁻¹⁷; all pairwise KS significant after Holm). However, the *specific 0.80 verdict is borderline*: the DIR CI straddles 0.80, so "fails EEOC" is a point-estimate failure with a real but magnitude-uncertain disparity. G3 (the group penalised ×0.75 in the label construction) is the lowest-selection group under every hiring rule.

3. **M3 (gender-biased labels) passes selection-rate parity but shows the largest error-rate disparity in the study**: gender EOD 0.198 with CI [0.164, 0.231] — entirely above 0, so **statistically supported**, not borderline. Female TPR 0.873 vs Male 0.675 (against the biased labels). Passing a single 0.80 selection screen is insufficient for a fairness audit.

4. **Adding the face-feature block (M1 vs M2) was not associated with a clear increase in disparity.** Gender DPD +0.0021 and ethnicity DPD +0.0013 (both CIs overlap), and ethnicity KL *decreased* (0.0295 → 0.0255). This is an **association under one evaluation setup**, not evidence of causation, and equally there is no evidence the face block is "unbiased". The dataset's intended SensitiveNets control (cols 31–50) is unusable — it is a constant vector in the official file.

5. **The arbitrary median-split hiring rule was not the driver of any headline conclusion.** Under the repository's own documented selection protocol (top-1000 hiring; 75th-percentile EEO threshold), every frozen verdict repeats: M4/ethnicity still fails (top-1000 DIR 0.710 [0.619, 0.810]; p75 DIR 0.751 [0.694, 0.807]), gender parity still holds, and M3's error-rate disparity still appears. The only threshold-sensitivity is at the far tighter top-2% screen (96 of 4,800), where even blind-label models show point-estimate ethnicity DIR 0.53–0.69 — low-power, wide-CI evidence of a threshold effect, not a demonstrated disparity.

6. **The original audit (v1) was numerically sound but narratively flawed.** Its computed metrics reproduce exactly, but its report made false/overstated claims (M3 "violates EEOC", a "significant KS p-value", and an M2 ethnicity-KL increase). All were corrected in the frozen v2 audit (R-1–R-15, §4), and the conclusions below are those of the corrected audit.

7. **Companion bio/text arm (Appendix A, outside the numeric audit):** gender leaks from free text — even redacted bios (AUC 0.71; ~0.76 under a partial one-epoch fine-tune) and names (0.97) — while ethnicity does not leak through any text channel. Yet text-only hiring models show the *lowest* gender disparity in the project (DPD 0.0039). **Leakage alone does not produce disparate hiring in this testbed; label bias does.**

---

## 2. Scope, Data, and Methods

### 2.1 Dataset (`FairCVdb.npy`, verified in `dataset_ground_truth.md`)

- Pickled dict with 14 keys; numeric profiles 19,200 train / 4,800 test (80/20), both splits balanced on gender (49.8/50.2% train) and ethnicity (~33/33/33%).
- Columns: 0–1 protected attributes (ethnicity G1/G2/G3, gender Male/Female — **excluded from all models**); 2–3 occupation (10 categories) and suitability (4 levels) — used only by the CV9 robustness arms; 4–10 seven CV competencies; 11–30 face embedding (20-d, L2 norm 1, genuine per-identity variance); 31–50 "blind face" embedding — **degenerate: a single constant vector plus ~1e-6 noise in the official file** (see §3.3).
- Labels: continuous hiring probabilities in [0,1]. Median (train): blind 0.4135, gender-biased 0.3659, ethnicity-biased 0.4148.
- **Verified bias construction** (ratio biased/blind label, blind > 0.05): gender-biased penalises **Female ×0.75**; ethnicity-biased penalises **G3 ×0.75** and boosts **G1 ×1.25**; all other groups unchanged. This matches the paper's "penalty factor" description.
- Train/test integrity: 0 row overlap (no leakage); names drawn from a finite pool (1,162 names repeat across splits — not row-level leakage).

### 2.2 Models M1–M6 (trained identically in v2 and both robustness passes)

| Model | Features | Training labels | Purpose |
|---|---|---|---|
| M1-Fair (CV7) | CV7 (cols 4–10) | Blind | Fair-label, no-face control |
| M2-Multimodal (CV7+Face) | CV7 + face (11–30) | Blind | Face-proxy test (same labels as M1; only features differ) |
| M3-Gender-Biased (CV7) | CV7 | Gender-biased | Constructed gender-bias arm |
| M4-Ethnicity-Biased (CV7) | CV7 | Ethnicity-biased | Constructed ethnicity-bias arm |
| M5-Robust (CV9) | CV9 (2–10) | Blind | Robustness arm (full CV set) |
| M6-Robust (CV9+Face) | CV9 + face | Blind | Robustness arm |

All are regularised logistic regressions on standardised features (C = 1.0, seed 42) — deterministic. Binarisation: `label ≥ train median → hired` is an **audit decision, not the papers' protocol** (R-4/N-4); the papers screen by top-N score, which the robustness analysis (§8) addresses.

### 2.3 Metrics (as implemented in `faircv_audit_v2.py`)

- **DPD** = max SR − min SR (demographic-parity difference; 0 = parity).
- **DIR** = worst-group SR ÷ best-group SR; **EEOC pass** if DIR ≥ 0.80 (four-fifths rule).
- **EOD** = max TPR − min TPR (magnitude; group with highest TPR reported); **EO** = max(ΔTPR, ΔFPR).
- **KL** = KL(best-SR-group ‖ worst-SR-group score histogram, 50 bins); pairwise KL mean also reported (R-11).
- **χ²** on group × hired; **KS** pairwise on score distributions with Holm correction within each model×attribute block; **Kruskal–Wallis** for 3-group ethnicity.
- **Effect sizes**: Cohen's d (extreme-SR-pair scores), Cohen's h (extreme-SR-pair SRs).
- **Bootstrap 95% CIs** (n = 2,000, seed 42, percentile method) for DPD/DIR/EOD/EO/KL.

### 2.4 Key caveats that shape interpretation

- **R-9:** M3/M4 are evaluated against their own artificially biased test labels; their TPR/FPR/PPV/EOD are relative to constructed ground truth. Only selection-rate (SR-based) metrics are neutral across models. M1-vs-M3/M4 is the *label-bias* contrast; M1-vs-M2 is the *proxy-feature* contrast.
- **R-12:** M1–M4 use the CV7 feature set (occupation and suitability — the strongest merit predictor, corr ≈ 0.48 with the blind label — excluded). M5/M6 show that adding these features materially changes performance and disparity; M1–M4 conclusions are specific to CV7.
- KL bootstrap CIs are upward-biased relative to the plug-in point estimate (standard for plug-in histogram KL); read them as conservative upper ranges.

---

## 3. Dataset Ground Truth (verified, not assumed)

### 3.1 Feature and label verification highlights

| Item | Verified value | Source |
|---|---|---|
| Gender coding | 0 = Male, 1 = Female; perfect pronoun separation in bios | `dataset_ground_truth.md` §3 |
| Ethnicity coding | {0,1,2} = G1/G2/G3 (README placeholders; no source maps codes to Black/Asian/Caucasian) | §3 |
| Occupation→suitability | suitability is deterministic per labour sector; corr with blind label ≈ 0.48 | §4 |
| Bias construction | Gender: Female ×0.75; Ethnicity: G1 ×1.25, G3 ×0.75 (train means 0.7502 / 1.2481 / 0.7478) | §5, `gt_verify_run.txt` |
| Missing values | none (NaN = 0, Inf = 0 everywhere) | §7 |
| Train/test | 19,200/4,800, balanced, 0 row overlap | §8 |

### 3.2 Label statistics (train median / test hired rate)

| Label set | train median | test hired rate (median split) |
|---|---|---|
| Blind | 0.4135 | 0.4929 |
| Gender-biased | 0.3659 | 0.4946 |
| Ethnicity-biased | 0.4148 | 0.4885 |

### 3.3 Critical dataset finding: the "blind face" block (cols 31–50) is degenerate

- All 19,200 train and 4,800 test rows lie within 1e-4 of the same constant 20-dim vector (per-column std ≈ 2–3e-6 vs 0.14–0.21 for the real face block).
- This is a property of the **official release**: our local file matches the repo's Git-LFS object (identical size and sha256).
- Consequences: (a) the intended SensitiveNets control arm is **unavailable**; any reproduction of the paper's "agnostic" scenarios from this file would train on near-constant features; (b) M2/M6 use cols 11–30, which are genuine embeddings (per-identity variance; gender AUC 0.93), so the M1-vs-M2 comparison stands.

---

## 4. Audit Lineage: Original Issues and the R-1–R-15 Corrections

### 4.1 What the original audit (`faircv_audit.py`) got right and wrong

**Right:** the computed numerics. Every v1 metric reproduced within 5e-4 by an independent harness and exactly by v2.

**Wrong (narrative/methodology):** the printed report made claims its own numbers contradicted — confirmed as BUG-1..11 and N-1..N-10 in `audit_code_review.md`:

| # | Issue (v1) | Correction (v2) |
|---|---|---|
| Obs-3 / BUG-1 | Claimed "M3 and M4 violate the EEOC 80% rule" — false: M3 passes both attributes; only M4/ethnicity fails (DIR 0.771) | **R-1** |
| Obs-3 / BUG-2 | Claimed a "significant KS p-value" — no KS p < 0.05 exists in v1's own output (M1 gender KS p = 0.318) | **R-2** |
| Obs-2 / BUG-3 | Claimed M2 raised KL vs M1 — false for ethnicity (0.0255 < 0.0295); only gender DPD/KL increase | **R-3** |
| Obs-1 / BUG-4 | "Lowest bias across all metrics" — overstated (M4 gender EOD 0.011 < M1 0.012) | **R-4** |
| BUG-5 / N-5 | No multi-group test for ethnicity; no CIs, effect sizes, or multiple-comparison correction anywhere | **R-5, R-13** |
| BUG-6 | DIR labelled "minority ÷ majority" — code computes worst/best SR | **R-6** |
| BUG-7 | EOD unsigned, direction lost | **R-7** |
| BUG-8 | Invented ethnicity names "Grp-A/B/C" | **R-8** |
| BUG-9 / N-6 | M3/M4 evaluated against their own biased labels; cross-model performance compared as if comparable | **R-9** |
| BUG-10 | Heatmap DIR colour scale inverted | **R-10** |
| BUG-11 | KL reported only for the extreme pair, middle group ignored | **R-11** |
| N-1 | Feature set undocumented (occupation/suitability dropped); no CV9 robustness arm | **R-12** |
| N-7 | Bias construction (×0.75/×1.25) never stated | **R-14** |
| N-8 | Metrics printed but not exported | **R-15** |
| N-2, N-3 | Degenerate cols 31–50 and bios/names/images scope undocumented | **R-14** (documented; control unavailable) |
| N-10 | M2 design ≠ paper's Scenario 4 (blind labels vs gender-biased labels + face) | documented; no parity claim with the paper |

**Net:** the corrected audit (v2) keeps the model design and metric math, fixes the narrative, adds multi-group tests, bootstrap uncertainty, effect sizes, machine-readable exports, and explicit scope/caveat statements. All results in §5–§8 are v2 outputs.

---

## 5. Complete M1–M6 Results (frozen audit, median-split hiring)

### 5.1 Model performance

| Model | acc | F1 | AUC |
|---|---|---|---|
| M1-Fair (CV7) | 0.793 | 0.786 | 0.888 |
| M2-Multimodal (CV7+Face) | 0.795 | 0.789 | 0.888 |
| M3-Gender-Biased (CV7) | 0.766 | 0.759 | 0.852 |
| M4-Ethnicity-Biased (CV7) | 0.760 | 0.750 | 0.843 |
| M5-Robust (CV9) | **0.966** | **0.965** | **0.996** |
| M6-Robust (CV9+Face) | 0.965 | 0.965 | 0.996 |

> R-9: M1–M4 performance values are **not** directly comparable (M3/M4 train on different, artificially biased label sets and are evaluated against them). Only M1/M2/M5/M6 (blind labels) share a common target.

### 5.2 Complete fairness table (point estimates, test n = 4,800)

g* = gender, e* = ethnicity. EEOC pass = DIR ≥ 0.80.

| Model | gDPD | gDIR | gEOD | gEO | gKL | eDPD | eDIR | eEOD | eEO | eKL | EEOC g / e |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M1-Fair (CV7) | +0.0122 | 0.9747 | 0.0124 | 0.0185 | 0.0183 | +0.0180 | 0.9631 | 0.0391 | 0.0391 | 0.0295 | PASS / PASS |
| M2-Multimodal (CV7+Face) | +0.0143 | 0.9705 | 0.0159 | 0.0193 | 0.0187 | +0.0193 | 0.9607 | 0.0415 | 0.0415 | 0.0255 | PASS / PASS |
| M3-Gender-Biased (CV7) | +0.0125 | 0.9739 | **0.1977** | **0.1977** | 0.0192 | +0.0168 | 0.9654 | 0.0667 | 0.0667 | 0.0301 | PASS / PASS |
| M4-Ethnicity-Biased (CV7) | +0.0096 | 0.9800 | 0.0113 | 0.0146 | 0.0257 | **+0.1243** | **0.7711** | **0.2548** | **0.2548** | 0.0775 | PASS / **FAIL** |
| M5-Robust (CV9) | +0.0038 | 0.9923 | 0.0001 | 0.0030 | 0.0235 | +0.0230 | 0.9541 | 0.0036 | 0.0036 | 0.1870 | PASS / PASS |
| M6-Robust (CV9+Face) | +0.0041 | 0.9916 | 0.0008 | 0.0014 | 0.0180 | +0.0210 | 0.9580 | 0.0030 | 0.0030 | 0.1033 | PASS / PASS |

### 5.3 Per-group selection rates and error rates

| Model | attr | Group | N | SR | TPR | FPR | PPV |
|---|---|---|---|---|---|---|---|
| M1 | gender | Male / Female | 2437 / 2363 | 0.471 / 0.483 | 0.767 / 0.780 | 0.179 / 0.198 | 0.808 / 0.791 |
| M1 | ethnicity | G1 / G2 / G3 | 1595 / 1588 / 1617 | 0.488 / 0.470 / 0.472 | 0.792 / 0.753 / 0.774 | 0.174 / 0.203 / 0.187 | 0.824 / 0.777 / 0.797 |
| M2 | gender | Male / Female | 2437 / 2363 | 0.470 / 0.484 | 0.768 / 0.784 | 0.177 / 0.196 | 0.810 / 0.794 |
| M2 | ethnicity | G1 / G2 / G3 | 1595 / 1588 / 1617 | 0.490 / 0.470 / 0.471 | 0.796 / 0.755 / 0.776 | 0.174 / 0.203 / 0.181 | 0.825 / 0.778 / 0.803 |
| M3 | gender | Male / Female | 2437 / 2363 | 0.468 / 0.481 | **0.675 / 0.873** | 0.107 / 0.271 | 0.917 / 0.633 |
| M3 | ethnicity | G1 / G2 / G3 | 1595 / 1588 / 1617 | 0.484 / 0.467 / 0.472 | 0.779 / 0.713 / 0.739 | 0.205 / 0.229 / 0.200 | 0.782 / 0.752 / 0.790 |
| M4 | gender | Male / Female | 2437 / 2363 | 0.467 / 0.477 | 0.743 / 0.732 | 0.211 / 0.226 | 0.766 / 0.761 |
| **M4** | **ethnicity** | G1 / G2 / **G3** | 1595 / 1588 / 1617 | **0.543 / 0.455 / 0.419** | 0.686 / 0.728 / 0.941 | 0.070 / 0.201 / 0.273 | 0.970 / 0.772 / 0.490 |
| M5 | gender | Male / Female | 2437 / 2363 | 0.488 / 0.484 | 0.959 / 0.959 | 0.025 / 0.028 | 0.974 / 0.970 |
| M5 | ethnicity | G1 / G2 / G3 | 1595 / 1588 / 1617 | 0.501 / 0.478 / 0.479 | 0.960 / 0.958 / 0.957 | 0.028 / 0.026 / 0.027 | 0.972 / 0.972 / 0.972 |
| M6 | gender | Male / Female | 2437 / 2363 | 0.490 / 0.486 | 0.959 / 0.960 | 0.028 / 0.030 | 0.971 / 0.969 |
| M6 | ethnicity | G1 / G2 / G3 | 1595 / 1588 / 1617 | 0.502 / 0.482 / 0.481 | 0.960 / 0.961 / 0.958 | 0.029 / 0.031 / 0.028 | 0.971 / 0.967 / 0.970 |

Reading the per-group table:
- **M4/ethnicity:** G3's selection rate collapses to 0.419 vs G1's 0.543 (DIR 0.771) while G3's TPR against the biased labels is *highest* (0.941) — the constructed ×0.75 penalty on G3 appears in hiring rates, and the label-bias inversion shows up in error rates (G1 PPV 0.970 vs G3 0.490).
- **M3/gender:** Male/Female selection rates are near-equal (0.468/0.481) yet Male TPR 0.675 vs Female 0.873 — the selection-rate screen hides a large error-rate gap.
- **M5/M6:** near-perfect classifiers (AUC 0.996) with tiny TPR/FPR gaps — but ethnicity DPD (0.0230/0.0210) is *higher* than M1 (0.0180), and ethnicity KL is large (0.187/0.103) with wide CIs. "More predictive" ≠ unambiguously "fairer".

### 5.4 Bootstrap 95% CIs (n = 2,000, seed 42, percentile)

| Model × attr | DPD | DIR | EOD |
|---|---|---|---|
| M1 gender | [0.0007, 0.0405] | [0.9189, 0.9986] | [0.0007, 0.0462] |
| M1 ethnicity | [0.0052, 0.0575] | [0.8869, 0.9891] | [0.0111, 0.0824] |
| M2 gender | [0.0006, 0.0430] | [0.9144, 0.9988] | [0.0010, 0.0526] |
| M2 ethnicity | [0.0056, 0.0579] | [0.8862, 0.9882] | [0.0120, 0.0818] |
| M3 gender | [0.0006, 0.0436] | [0.9121, 0.9987] | **[0.1645, 0.2314]** |
| M3 ethnicity | [0.0049, 0.0550] | [0.8903, 0.9899] | [0.0287, 0.1110] |
| M4 gender | [0.0006, 0.0373] | [0.9243, 0.9988] | [0.0007, 0.0467] |
| **M4 ethnicity** | **[0.0905, 0.1573]** | **[0.7180, 0.8282]** | **[0.2187, 0.2894]** |
| M5 gender | [0.0005, 0.0331] | [0.9345, 0.9991] | [0.0002, 0.0183] |
| M5 ethnicity | [0.0062, 0.0614] | [0.8810, 0.9874] | [0.0022, 0.0275] |
| M6 gender | [0.0004, 0.0329] | [0.9346, 0.9993] | [0.0002, 0.0187] |
| M6 ethnicity | [0.0065, 0.0585] | [0.8872, 0.9868] | [0.0020, 0.0262] |

(Full KL and EO CIs in `results/audit_report.txt` §5; KL CIs are conservative upper ranges.)

---

## 6. Statistical Validation

### 6.1 Significance tests (`results/statistical_tests.csv`, 42 rows)

| Model | gender KS (min p_adj) | gender χ² p | ethnicity KW p | ethnicity χ² p | ethnicity KS (min p_adj) |
|---|---|---|---|---|---|
| M1 | 0.318 | 0.414 | 0.603 | 0.547 | 1.000 |
| M2 | 0.176 | 0.336 | 0.538 | 0.458 | 1.000 |
| M3 | 0.254 | 0.400 | 0.623 | 0.619 | 1.000 |
| **M4** | 0.243 | 0.526 | **7.9e-17** | **4.1e-12** | **5.7e-14** |
| M5 | 0.761 | 0.817 | 0.379 | 0.345 | 1.000 |
| M6 | 0.871 | 0.797 | 0.492 | 0.409 | 1.000 |

- **Only M4/ethnicity shows any significant group differences**: all three pairwise KS survive Holm (G1–G2 p_adj 6.5e-7; G1–G3 5.7e-14; G2–G3 1.7e-3), KW p = 7.9e-17, χ² p = 4.1e-12.
- Every other model × attribute is non-significant — including **all** gender comparisons and all blind-label models. This directly refutes v1's "significant KS" narrative and supports the weakened M2 claim.

### 6.2 Effect sizes (extreme-SR pair; Cohen's d / h)

| Model | gender | ethnicity |
|---|---|---|
| M1 | 0.026 / 0.024 | 0.034 / 0.036 |
| M2 | 0.031 / 0.029 | 0.035 / 0.039 |
| M3 | 0.025 / 0.025 | 0.032 / 0.034 |
| M4 | 0.025 / 0.019 | **0.298 / 0.249** |
| M5 | 0.008 / 0.008 | 0.046 / 0.046 |
| M6 | 0.006 / 0.008 | 0.045 / 0.042 |

Only M4/ethnicity shows a non-trivial effect size.

### 6.3 Point estimates vs statistically supported results

| Claim | Point estimate | Statistical status | Verdict label |
|---|---|---|---|
| M4/ethnicity fails EEOC (median) | DIR 0.771 < 0.80 | DIR CI [0.718, 0.828] **straddles 0.80**; disparity itself significant (DPD CI excludes 0; χ²/KW/KS all significant at α = 0.05, p from 1.7×10⁻³ to 7.9×10⁻¹⁷) | **Point-estimate FAIL, borderline verdict, statistically real disparity** |
| M3 gender error-rate disparity | EOD 0.198 | CI [0.1645, 0.2314] entirely > 0 | **Statistically supported** |
| M1/M2/M5/M6 gender parity | all DIR ≥ 0.9705 | all gender DIR CI lower bounds ≥ 0.91; all χ² ns | **Statistically supported** (pass) |
| M1/M2/M5/M6 ethnicity parity | all DIR 0.954–0.963 | all ethnicity DIR CIs entirely > 0.80 at the median split; all tests ns | **Statistically supported** (pass) |
| M5/M6 "fairer" claim | lower gender DPD, but higher ethnicity DPD/KL | gender DPD CIs overlap 0; ethnicity KL CIs extremely wide ([0.179, 0.620]) | **Not established** — mixed, do not claim |
| M2 adds disparity (face proxy) | gender DPD +0.0021 | DPD CIs overlap (M1 [0.0007, 0.0405] vs M2 [0.0006, 0.0430]); ethnicity KL decreases | **Not supported** (see §7) |

---

## 7. Face-Feature Findings (M1 vs M2)

Same blind labels; only the features differ (CV7 vs CV7+face). All deltas are **associations under this evaluation setup**, not demonstrated causal mechanisms.

| Metric | M1 | M2 | Δ (M2 − M1) | CI overlap? |
|---|---|---|---|---|
| gender DPD | 0.0122 | 0.0143 | +0.0021 | yes ([0.0007, 0.0405] vs [0.0006, 0.0430]) |
| gender KL (extreme) | 0.0183 | 0.0187 | +0.0004 | yes |
| ethnicity DPD | 0.0180 | 0.0193 | +0.0013 | yes |
| ethnicity KL (extreme) | 0.0295 | 0.0255 | **−0.0040** | — |

- Gender DPD/KL increase slightly; ethnicity KL *decreases*. Both gender DPD CIs overlap the other's interval.
- Correct statement: *under the evaluated FairCV configuration, adding the face-feature block did not provide strong evidence of increased demographic disparity relative to the CV-only baseline.* Equally, there is no evidence the face embeddings are "unbiased".
- Validity note: M2 uses cols 11–30, genuine per-image embeddings (gender AUC 0.93), so the comparison is meaningful; the intended agnostic-embedding control (cols 31–50) is a constant vector in the official file and cannot serve as a SensitiveNets-style control (§3.3).

---

## 8. Robustness: Hiring-Rule Sensitivity (Top-1000 and 75th-Percentile Protocols)

### 8.1 What was tested (`topn_robustness.py`, outputs in `results/robustness/`)

The median split is an audit decision. The robustness pass re-runs the audit under the hiring rule **documented in the official FairCVtest repository** (`FairCV.py`), training M1–M6 exactly as v2 (verified identical: max drift vs frozen `metrics.csv` = 1.1e-16):

| Protocol | Repo function | Rule |
|---|---|---|
| **A — Top-1000** | `computeTopScore` / `testDemographicParity` | hire the top 1,000 of 4,800 test scores (top ≈20.8%) |
| **B — 75th percentile** | `testEqualityOfOpportunity(..., p=75)` | threshold = 75th percentile of the training labels; hired = score ≥ threshold; EEO test: per-group TPR = P(hired \| label ≥ threshold) |

Bootstrap 95% CIs (n = 2,000, seed 42) computed for DPD/DIR/EOD under each rule. Protocol B uses each model's own training label set (mirroring v2's binarisation); the upstream repo applies the p75 EEO test to blind labels only — both conventions agree on the headline results here. Internal consistency: the blind-label p75 thresholds come out at 0.4999 ≈ the median-split 0.5 boundary, so Protocol B reproduces the frozen selection rates to 6 decimals for M1/M2/M5/M6.

### 8.2 DIR under the three hiring rules (point estimate [95% CI])

| Model × attr | median | top-1000 | p75 |
|---|---|---|---|
| M1 gender | 0.975 [0.919, 0.999] | 0.906 [0.811, 0.989] | 0.975 [0.918, 0.999] |
| M1 ethnicity | 0.963 [0.887, 0.989] | 0.909 [0.790, 0.978] | 0.963 [0.889, 0.990] |
| M2 gender | 0.970 [0.914, 0.999] | 0.891 [0.795, 0.987] | 0.970 [0.917, 0.998] |
| M2 ethnicity | 0.961 [0.886, 0.988] | 0.912 [0.793, 0.978] | 0.961 [0.882, 0.988] |
| M3 gender | 0.974 [0.912, 0.999] | 0.895 [0.805, 0.989] | 0.991 [0.934, 0.999] |
| M3 ethnicity | 0.965 [0.890, 0.990] | 0.934 [0.800, 0.981] | 0.961 [0.893, 0.989] |
| M4 gender | 0.980 [0.924, 0.999] | 0.989 [0.879, 0.998] | 0.975 [0.916, 0.999] |
| **M4 ethnicity** | **0.771 [0.718, 0.828]** | **0.710 [0.619, 0.810]** | **0.751 [0.694, 0.807]** |
| M5 gender | 0.992 [0.934, 0.999] | 0.950 [0.864, 0.997] | 0.992 [0.931, 0.999] |
| M5 ethnicity | 0.954 [0.881, 0.987] | 0.886 [0.767, 0.965] | 0.954 [0.884, 0.987] |
| M6 gender | 0.992 [0.935, 0.999] | 0.950 [0.853, 0.997] | 0.992 [0.933, 0.999] |
| M6 ethnicity | 0.958 [0.887, 0.987] | 0.897 [0.773, 0.968] | 0.958 [0.886, 0.990] |

**No EEOC verdict changes under either protocol.** M4/ethnicity fails under all three rules; all others pass under all three.

### 8.3 Key robustness readings

- **M4/ethnicity:** fails at top-1000 (0.710) and p75 (0.751); χ² significant under both (top-1000 p = 3.9×10⁻⁶; p75 p = 2.6×10⁻¹³); G3 remains the worst-off group under every rule. The DIR CI still straddles 0.80 under both protocols, so the borderline-magnitude caveat persists across all three hiring rules.
- **M4/ethnicity per-group top-1000 selection rates:** G1 0.246, G2 0.205, G3 0.174 — the ×0.75 label penalty shows through in selection rates regardless of how hiring is defined.
- **Gender:** all pass under every rule (min DIR 0.891 at top-1000; max DPD 0.024). The only gender cell approaching significance is M2 gender χ² at top-1000 (p = 0.045) — marginal and uncorrected for multiple comparisons; read as weak, not a finding.
- **M3 gender EOD under p75:** 0.087 [0.069, 0.106] — still by far the largest error-rate disparity in the study, direction preserved (Female TPR 1.00 vs Male 0.913). Magnitude differs from the median-split 0.198 because the ground-truth threshold differs (75th percentile vs median); the *conclusion* is unchanged.
- **Blind-label models (M1/M2/M5/M6) ethnicity:** pass under the repo's own top-1000 protocol (DIR 0.886–0.912) and under p75 (0.954–0.963). Note the top-1000 CIs dip below 0.80 at their lower ends (lo ≈ 0.77–0.79), so these are point-estimate passes with some uncertainty — weaker than the median-split or p75 passes, whose CIs sit entirely above 0.80.
- **Worst-off-group "flips"** (M5/M6 gender; M6 ethnicity) occur only where DIR ≥ 0.89 — near-zero disparity, noise level.

### 8.4 Interaction with the earlier top-N screening companion (`topn_screening.md`, `results/topn_metrics.csv`)

The earlier companion swept thresholds 2%–50%: at the **top-2%** screen (96 of 4,800 — the papers' demo scale), even the blind-label models show point-estimate ethnicity DIR 0.53–0.69 (M2 0.527 [0.325, 0.855], M1 0.655 [0.364, 0.900], M5 0.690, M6 0.672; the gender-biased M3 shows 0.646), all with very wide CIs and non-significant χ² except M2 (p = 0.046). M4/ethnicity fails at every threshold (top-2% DIR 0.481, CI [0.236, 0.706] — entirely below 0.80 — χ² p = 0.014). The honest reading, consistent with `topn_screening.md`: **the top-2% failures are low-power evidence of a threshold effect (96 selections), not demonstrated disparities** — and they do not appear at the repository's own top-1000 protocol.

### 8.5 Verdict on conclusion robustness

| Frozen finding | Under top-1000 / p75 | Status |
|---|---|---|
| F1 Gender parity everywhere | Holds at every rule | **Robust** |
| F2 M4/ethnicity EEOC fail | Fails at every rule (point estimates 0.71–0.77; CIs straddle 0.80) | **Robust** (borderline magnitude in all rules) |
| F3 M3 passes selection parity, large gender EOD | Holds under p75 (EOD 0.087, CI excludes 0) | **Robust** |
| F4 Blind models pass ethnicity EEOC | Pass at top-1000 and p75 (fail only at the tight top-2% screen) | **Robust** at repo protocol; qualified at top-2% |
| F5 No clear face-proxy disparity signal | No consistent M2-vs-M1 signal at top-1000 either | **Robust** |

---

## 9. Conclusions

1. **The corrected audit's four findings are robust to the hiring-rule definition.** The median threshold was not the driver of any headline result: gender parity holds everywhere, M4/ethnicity is the only EEOC failure under every rule, M3's selection-parity-but-error-rate-disparity profile persists, and the face-proxy signal remains weak.

2. **M4/ethnicity is the only demonstrated disparity, and it is real but magnitude-borderline.** The disparity is statistically unambiguous (all tests significant at α = 0.05 — p from 1.7×10⁻³ to 7.9×10⁻¹⁷; DPD CI excludes 0; effect size Cohen's d = 0.298), yet the DIR CI straddles the 0.80 threshold under the median split, top-1000, and p75. Report it as: *point-estimate EEOC failure (DIR 0.77) with a borderline-magnitude disparity that is statistically distinguishable from parity* — not as a "beyond doubt" failure and not as "not a failure".

3. **A single selection screen is insufficient.** M3 passes every selection-rate criterion yet has the study's largest error-rate disparity (gender EOD 0.198, CI [0.164, 0.231]); M4's error rates are inverted relative to its selection rates (G3 highest TPR, lowest PPV). Multi-criterion auditing (selection parity + error parity + score distributions) is necessary.

4. **The face-embedding question remains open, in both directions.** Under this setup, adding face features was *associated* with small, overlapping-CI changes (gender DPD +0.0021; ethnicity KL −0.0040). This does not support the v1 claim that face embeddings "inject bias", and it does not demonstrate that they are unbiased. The intended control arm (agnostic embeddings, cols 31–50) is unusable in the official dataset, so a definitive answer cannot be produced from this file.

5. **The dataset itself has a notable defect.** The "blind face embedding" block is a constant vector in the official release, disabling the SensitiveNets control the benchmark is designed to provide. Any reproduction of the paper's "agnostic" scenarios from this file would silently train on near-constant features.

6. **All conclusions are scoped to this synthetic testbed** (numeric profiles; regularised logistic regressions; CV7/CV9 feature sets; constructed bias). They describe associations under a simulated recruitment system and do not generalise to real hiring systems.

---

## 10. Limitations

- **Scope:** numeric profile block only (cols 0–10, 11–30). Bios (text), names, and images are out of scope; raw images are not present in this folder. Text-based experiments exist as separate companion artifacts (e.g., `bio_arm.md`) and are not part of this numeric audit.
- **Model class:** regularised logistic regressions only; conclusions are specific to this model class and feature sets.
- **Artificial outcome:** the median binarisation (and, in the robustness pass, the top-1000/p75 rules) is an evaluation choice, not a real hiring process. The papers' demo uses top-100 score screening, which the robustness analysis addresses only as far as the documented repo protocol goes.
- **Constructed ground truth:** M3/M4 metrics against their own biased labels (R-9); cross-model performance comparisons are only valid among blind-label models.
- **Borderline statistics:** several CIs straddle decision thresholds (M4 DIR under all rules; blind-model ethnicity DIR lower bounds at top-1000); point estimates should always be read together with their CIs.
- **Uncertainty conventions:** bootstrap CIs re-derive extreme groups per resample (standard for max-min statistics); KL CIs are conservative upper ranges.
- **No causal claims:** all differences are associations under a synthetic, controlled testbed.

---

## Appendix A — Companion Bio/Text Arm (outside the frozen numeric audit)

**Status marker: this appendix summarises a SEPARATE companion experiment — NOT part of the
frozen numeric audit.** The numeric audit above covers profile columns 0–30. The bio/text arm
(`bio_arm.py`, `bio_leak_strong.py`, `bio_leak_bert.py`, `bio_ft_distilbert.py`; outputs
`results/bio_leakage.csv`, `results/bio_metrics.csv`, `results/bio_leak_strong.csv`,
`results/bio_leak_finetune.csv`, `results/bio_report.txt`, `fig7`/`fig8`) investigates the
free-text `Bios` field (original and redacted variants) and the `Names` field. It reuses the
frozen data pipeline and v2 fairness conventions (median-binarised blind label, same metrics and
bootstrap CIs), so its fairness numbers are directly comparable to M1–M6. Nothing in this
appendix alters the numeric audit's conclusions. Full analysis: `bio_arm.md`. All figures below
were verified against the bio CSVs.

### A.1 Demographic leakage through language (`results/bio_leakage.csv`)

| Channel | Target | Test acc | Test AUC | Reading |
|---|---|---|---|---|
| original bios | gender | 0.9975 | 1.0000 | Perfect — pronouns ("He/His" vs "She/Her") |
| blind (redacted) bios | gender | 0.6452 | 0.7091 | Strong residual leak despite redaction |
| names (control) | gender | 0.9165 | 0.9711 | Names are a near-perfect gender channel |
| original bios | ethnicity | 0.3229 | 0.4945 | Chance |
| blind bios | ethnicity | 0.3246 | 0.4969 | Chance |
| names (control) | ethnicity | 0.3235 | 0.4954 | Chance |

- **Gender leaks massively, even from the "blind" version.** The residual-cue scan of the redacted
  bios identifies the surviving cues (ratio = stronger side / weaker side, train): `husband` ratio
  19.1, `wife` 9.3, `mother` 3.6, `father` 2.4, `sister` 2.4. A naive pronoun redaction does not
  make text gender-blind.
- **Ethnicity does not leak through any text channel** (AUC ≈ 0.49–0.50, acc ≈ 0.32 ≈ 1/3). This
  is a dataset property: the synthetic G1/G2/G3 labels are not name/bio-consistent, so ethnicity
  is only reachable through the numeric profile block and the face embedding in this testbed. Do
  not generalise to real data, where names are strongly ethnicity-correlated.

### A.2 Text-only hiring models vs M1 (`results/bio_metrics.csv`)

| Model | acc | AUC | attr | DPD | DIR | EOD | KL | χ² p | KW p | EEOC |
|---|---|---|---|---|---|---|---|---|---|---|
| BIO-BioBlind (redacted bios) | 0.669 | 0.714 | gender | 0.0039 | 0.9923 | 0.0075 | 0.0222 | 0.81 | — | PASS |
| BIO-BioBlind | 0.669 | 0.714 | ethnicity | 0.0344 | 0.9347 | 0.0102 | 0.0778 | 0.13 | 0.18 | PASS |
| BIO-BioOriginal (original bios) | 0.672 | 0.713 | gender | 0.0121 | 0.9761 | 0.0168 | 0.0401 | 0.42 | — | PASS |
| BIO-BioOriginal | 0.672 | 0.713 | ethnicity | 0.0436 | 0.9170 | 0.0315 | 0.0538 | 0.040 | 0.14 | PASS |
| M1-Fair (CV7, numeric; re-fitted in-script, acc 0.7929) | 0.793 | — | gender | 0.0122 | 0.9747 | 0.0124 | 0.0183 | — | — | PASS |
| M1-Fair (CV7) | 0.793 | — | ethnicity | 0.0180 | 0.9631 | 0.0391 | 0.0295 | — | — | PASS |

1. **Text models are weaker performers** (acc 0.67 / AUC 0.71 vs M1's 0.79) — the scraped bios are
   only loosely coupled to the synthetic merit score.
2. **Leakage did not translate into outcome disparity.** Despite recovering gender at AUC 0.71, the
   redacted-bio model has the *lowest* gender DPD in the project (0.0039, a third of M1's 0.0122),
   and all four bio cells pass EEOC. Reason: the blind label is gender/ethnicity-neutral given the
   profile, so gender-linked text features carry little predictive weight. **In this synthetic
   testbed, demographic leakage is real but does not mechanically produce disparate hiring** —
   disparity materialises when the *labels* are biased (M3/M4), not when features merely contain
   demographic information.
3. **Naive redaction partially works.** Seeing the original text roughly triples gender DPD (0.0121
   vs 0.0039) and produces the arm's only nominally-significant test (BioOriginal/ethnicity χ²
   p = 0.040 — borderline, not significant under any multiple-comparison view).

### A.3 Stronger-model bound — and a qualification from a partial fine-tune

The leak was stress-tested with four model tiers (frozen `bio_leak_strong.csv`): T0 word TF-IDF
**0.7091**, T1 char TF-IDF 0.7037, T2 char CNN 0.6589, T3 frozen all-MiniLM-L6-v2 + LR 0.6799.
None of these exceeds the word-level baseline, which `bio_arm.md` therefore calls a "tight bound".

**Qualification from the frozen evidence:** a separate fine-tuning run (`bio_ft_distilbert.py`,
designed to survive crashes and append partial results) logged **one epoch** in
`results/bio_leak_finetune.csv`: test AUC **0.7592** (train AUC 0.8217, test acc 0.6846, ~5 h CPU,
maxlen 160). This **exceeds** the word-level baseline (0.7592 > 0.7091). Because the run is
partial (1 of the default 2 epochs) and single-configuration, 0.7592 is a lower-bound indication
that a *tuned* transformer recovers somewhat more gender from the redacted bios than the four
frozen feature-extraction tiers — i.e. `bio_arm.md`'s "tight bound" statement holds for the frozen
tiers only, and the recoverable signal is at least ~0.76 under a fine-tuned model. This does not
change the hiring-disparity conclusion in A.2 (leakage ≠ disparate outcomes), nor any numeric-audit
conclusion.

### A.4 What the bio arm adds

- **The modalities now complete the picture:** face embeddings encode demographics strongly but
  show no material disparity increase (M2); numeric profiles are neutral features carrying the M4
  label-bias story; text leaks *gender* (even redacted) but not *ethnicity*, and shows no disparity
  amplification in hiring.
- **A clean separation of two claims the original audit conflated:** "the model can read a
  protected attribute from the input" is TRUE for gender (text AUC 0.71–1.00, names 0.97, face
  0.93–0.99) and FALSE for ethnicity via text; but "reading it causes disparate outcomes" is NOT
  demonstrated anywhere except where the training labels themselves are biased (M3/M4). The proxy
  hypothesis (features → bias) remains unsupported; the label-bias mechanism is the demonstrated
  one.
- **Practical lesson:** pronoun/name redaction is insufficient anonymisation of free text;
  kinship/family and domain vocabulary re-encode gender. A real "blind CV" pipeline must audit the
  text representation itself (e.g. with the leakage-AUC test used here).
- **Caveats (from `bio_arm.md`):** the ethnicity-chance result is specific to this synthetic
  dataset; hiring fairness uses the median-binarised blind label (not top-N screening); the
  dataset's own redaction differs from a purpose-built pipeline; leakage AUCs have no bootstrap
  CIs; the fine-tuned transformer result is a partial, single-epoch run (see A.3).

---

## 11. Evidence Inventory (frozen artifacts used)

| Artifact | Content | Used for |
|---|---|---|
| `results/metrics.csv` | 12 rows: DPD/DIR/EOD/EO/KL + CIs + effect sizes + EEOC | §5, §6, §8.2 |
| `results/statistical_tests.csv` | 42 rows: KS(+Holm)/KW/χ² | §6.1 |
| `results/per_group_metrics.csv` | 30 rows: N/SR/TPR/FPR/PPV/mean-score | §5.3 |
| `results/audit_report.txt` | full v2 console report | §5.2–5.4, §6 |
| `dataset_ground_truth.md` | dataset schema, distributions, bias construction, cols 31–50 | §3 |
| `audit_code_review.md` | v1 review, BUG-1..11, N-1..N-10, R-1..R-15 | §4 |
| `faircv_audit_v2_validation.md` | independent validation of v2 numerics | §5–§7 |
| `results/topn_metrics.csv`, `topn_screening.md` | top-2%..50% threshold sweep | §8.4 |
| `results/robustness/*` | top-1000 / p75 protocols + CIs + figure | §8 |
| `gt_verify_run.txt`, `ground_truth_verify.py` | independent ground-truth verification run | §3 |
| `faircv_audit.py` / `faircv_audit_v2.py` | original and corrected audits (frozen, unmodified) | §4 |
| `bio_arm.md`, `results/bio_leakage.csv`, `results/bio_metrics.csv`, `results/bio_report.txt` | bio/text companion arm (leakage + hiring) — outside the numeric audit | Appendix A |
| `bio_leak_strong.csv`, `bio_leak_finetune.csv`, `fig7`/`fig8` | bio stronger-model ladder + partial fine-tune | Appendix A.3 |
| `bio_arm.py`, `bio_leak_strong.py`, `bio_leak_bert.py`, `bio_ft_distilbert.py` | companion scripts (frozen, unmodified) | Appendix A |

**Verification note:** every numerical claim above was transcribed from the artifacts in this table (or computed by the frozen companion scripts that produced them); no new experiments were run and no artifact was modified to produce this report. The claims were machine-checked against the CSVs with `verify_final_report.py` (189 checks, all passing).
