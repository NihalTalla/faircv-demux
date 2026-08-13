# Validation Report — `faircv_audit_v2.py`

**Date:** 2026-08-11
**Status:** Experiment frozen. This report is a **read-only verification pass** — no methodology was changed and **no code was modified**.
**Sources of truth:** `results/metrics.csv`, `results/statistical_tests.csv`, `results/per_group_metrics.csv`, `results/audit_report.txt` (final run), plus the official FairCVtest repo (`FairCV.py`, README, Git-LFS pointer) and the SensitiveNets paper (arXiv:1902.00334).

---

## 1. What was validated

| Item | Outcome |
|---|---|
| v2 reproduces v1 numerics | ✅ M1 gender DPD 0.0122 / DIR 0.9747 / EOD 0.0124 / EO 0.0185 / KL 0.0183; M1 ethnicity DIR 0.9631; M4 ethnicity DIR 0.7711 — all identical (deterministic, seed 42) |
| Holm correction (post-review fix) | ✅ adjusted p-values now the correct running-maximum step-down values |
| Complete M1–M6 table | ✅ §2 |
| Bootstrap 95% CIs (n=2,000) | ✅ §3 |
| Statistical tests (KS+Holm, Kruskal-Wallis, χ²) | ✅ §4 |
| Columns 31–50 resolved | ✅ constant/placeholder block, present in the **official release** — §5 |
| Code modifications this session | **None** |

---

## 2. Complete M1–M6 table (point estimates, test set n=4,800)

Metrics: **DPD** = max−min selection rate; **DIR** = worst-group SR / best-group SR (EEOC PASS if ≥ 0.80); **EOD** = max−min TPR; **EO** = max(ΔTPR, ΔFPR); **KL** = KL(extreme-SR pair). `g*` = gender, `e*` = ethnicity.

| Model | acc | f1 | AUC | gDPD | gDIR | gEOD | gEO | gKL | eDPD | eDIR | eEOD | eEO | eKL | EEOC g/e |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1-Fair (CV7) | 0.793 | 0.786 | 0.888 | +0.0122 | 0.9747 | 0.0124 | 0.0185 | 0.0183 | +0.0180 | 0.9631 | 0.0391 | 0.0391 | 0.0295 | PASS/PASS |
| M2-Multimodal (CV7+Face) | 0.795 | 0.789 | 0.888 | +0.0143 | 0.9705 | 0.0159 | 0.0193 | 0.0187 | +0.0193 | 0.9607 | 0.0415 | 0.0415 | 0.0255 | PASS/PASS |
| M3-Gender-Biased (CV7) | 0.766 | 0.759 | 0.852 | +0.0125 | 0.9739 | **0.1977** | **0.1977** | 0.0192 | +0.0168 | 0.9654 | 0.0667 | 0.0667 | 0.0301 | PASS/PASS |
| M4-Ethnicity-Biased (CV7) | 0.760 | 0.750 | 0.843 | +0.0096 | 0.9800 | 0.0113 | 0.0146 | 0.0257 | **+0.1243** | **0.7711** | **0.2548** | **0.2548** | 0.0775 | PASS/**FAIL** |
| M5-Robust (CV9) | **0.966** | 0.965 | **0.996** | +0.0038 | 0.9923 | 0.0001 | 0.0030 | 0.0235 | +0.0230 | 0.9541 | 0.0036 | 0.0036 | 0.1870 | PASS/PASS |
| M6-Robust (CV9+Face) | 0.965 | 0.965 | 0.996 | +0.0041 | 0.9916 | 0.0008 | 0.0014 | 0.0180 | +0.0210 | 0.9580 | 0.0030 | 0.0030 | 0.1033 | PASS/PASS |

Reading the table (all statements are computed, see `results/audit_report.txt`):
- **Only M4/ethnicity fails the EEOC 80% rule** (DIR 0.771). Every other model×attribute passes.
- **M3 passes selection-rate parity but shows a large gender EOD (0.198)**: Female TPR 0.873 vs Male 0.675, FPR 0.271 vs 0.107 — the median-re-centred label bias surfaces in error rates, not hiring rates. Passing demographic parity ≠ fair (multi-criterion audit justified).
- **M5/M6 (full CV9 features)**: accuracy 0.793→0.966. Gender disparity drops to the lowest of all models (gDPD 0.0038/0.0041) — but **ethnicity DPD rises** (0.0230/0.0210 vs M1 0.0180) and ethnicity KL is large (0.187/0.103) with very wide CIs (see §3). So M5 is *not* unambiguously "more fair": adding the predictive features improves performance and gender parity while slightly worsening ethnicity parity and concentrating score distributions (KL is inflated by near-perfect score separation, so DPD/DIR are the more interpretable metrics there).
- **M1 vs M2 (proxy test)**: gender DPD +0.0021 (CIs overlap → weak), ethnicity DPD +0.0013, ethnicity KL **decreases** (0.0295→0.0255). Under this setup, adding the face block does **not** provide strong evidence of increased disparity.

---

## 3. Bootstrap 95% CIs (2,000 resamples, seed 42; percentile method)

| Model × attr | DPD | DIR | EOD | EO | KL |
|---|---|---|---|---|---|
| M1 gender | [0.0007, 0.0405] | [0.9189, 0.9986] | [0.0007, 0.0462] | [0.0048, 0.0539] | [0.0270, 0.0534] |
| M1 ethnicity | [0.0052, 0.0575] | [0.8869, 0.9891] | [0.0111, 0.0824] | [0.0205, 0.0837] | [0.0420, 0.1367] |
| M2 gender | [0.0006, 0.0430] | [0.9144, 0.9988] | [0.0010, 0.0526] | [0.0056, 0.0564] | [0.0264, 0.0546] |
| M2 ethnicity | [0.0056, 0.0579] | [0.8862, 0.9882] | [0.0120, 0.0818] | [0.0228, 0.0828] | [0.0383, 0.0806] |
| M3 gender | [0.0006, 0.0436] | [0.9121, 0.9987] | [0.1645, 0.2314] | [0.1695, 0.2314] | [0.0280, 0.0560] |
| M3 ethnicity | [0.0049, 0.0550] | [0.8903, 0.9899] | [0.0287, 0.1110] | [0.0347, 0.1110] | [0.0433, 0.1671] |
| M4 gender | [0.0006, 0.0373] | [0.9243, 0.9988] | [0.0007, 0.0467] | [0.0042, 0.0531] | [0.0323, 0.0635] |
| **M4 ethnicity** | **[0.0905, 0.1573]** | **[0.7180, 0.8282]** | **[0.2187, 0.2894]** | **[0.2200, 0.2894]** | [0.0814, 0.1438] |
| M5 gender | [0.0005, 0.0331] | [0.9345, 0.9991] | [0.0002, 0.0183] | [0.0016, 0.0189] | [0.0435, 0.2889] |
| M5 ethnicity | [0.0062, 0.0614] | [0.8810, 0.9874] | [0.0022, 0.0275] | [0.0055, 0.0279] | [0.1793, 0.6202] |
| M6 gender | [0.0004, 0.0329] | [0.9346, 0.9993] | [0.0002, 0.0187] | [0.0015, 0.0193] | [0.0324, 0.2227] |
| M6 ethnicity | [0.0065, 0.0585] | [0.8872, 0.9868] | [0.0020, 0.0262] | [0.0054, 0.0272] | [0.1126, 0.5670] |

Key uncertainty findings:
- **M4/ethnicity DIR CI [0.718, 0.828] straddles the 0.80 threshold** → the EEOC failure is real but **borderline/uncertain**, not "beyond doubt". Its DPD CI [0.0905, 0.1573] excludes zero — the disparity is statistically distinguishable from zero.
- **M3 gender EOD CI [0.165, 0.231]** excludes small values → the error-rate disparity is robust.
- M2-vs-M1 DPD CIs overlap for both attributes → the +0.0021/+0.0013 deltas are **not** statistically supported.
- Caveat (documented in the report): histogram-KL bootstrap CIs are upward-biased relative to the plug-in point estimate (resampling inflates plug-in KL); read KL CIs as conservative upper ranges.

---

## 4. Statistical tests (test set)

Pairwise KS with **Holm correction within each model×attribute block**, Kruskal-Wallis for 3-group ethnicity, χ² on group×hired contingency. Full detail in `results/statistical_tests.csv` (42 rows).

| Model | gender KS (min p_adj) | gender χ² p | ethnicity KW p | ethnicity χ² p | ethnicity KS (min p_adj) |
|---|---|---|---|---|---|
| M1 | 0.318 | 0.414 | 0.603 | 0.547 | 1.000 |
| M2 | 0.176 | 0.336 | 0.538 | 0.458 | 1.000 |
| M3 | 0.254 | 0.400 | 0.623 | 0.619 | 1.000 |
| **M4** | 0.243 | 0.526 | **7.9e-17** | **4.1e-12** | **5.7e-14** |
| M5 | 0.761 | 0.817 | 0.379 | 0.345 | 1.000 |
| M6 | 0.871 | 0.797 | 0.492 | 0.409 | 1.000 |

- **Only M4/ethnicity shows any significant group differences** (all three pairwise KS significant after Holm: 6.5e-7 / 5.7e-14 / 1.7e-3; KW p = 7.9e-17; χ² p = 4.1e-12). All other models × attributes are non-significant — including every gender comparison and every blind-label model.
- This **directly refutes v1's "significant KS p-value" narrative** (which the v1 code never produced) and supports the weakened M2 claim.

---

## 5. Columns 31–50: determination (genuine embeddings vs constant block)

### 5.1 Verdict

> **Columns 31–50 are a single constant vector (one norm-1 vector broadcast to every row) plus a ~1e-6-scale perturbation — NOT genuine per-image "blind face embeddings". The constant block exists in the OFFICIAL release of FairCVdb, not just this local copy.**

### 5.2 Evidence

**Empirical (this file, read-only):**
- 19,200/19,200 train rows and 4,800/4,800 test rows lie within 1e-4 of the same 20-dim vector; per-column std ≈ 2–3e-6 (vs 0.14–0.21 for the real face block cols 11–30).
- The train and test vectors are identical (max |median diff| = 8.2e-8) — literally one fixed vector in the whole file.
- Within-identity variation: face block mean col-std ≈ 0.16; blind block ≈ 2.5e-6 (identity 31: 0.1615 vs 2.56e-6; identity 5: 0.1660 vs 2.60e-6).
- Not a scaled copy of the per-row face embedding (|corr(jitter, face)| ≈ 0.20) and not the mean face embedding (corr −0.10).
- **Not pure float noise either**: after standardisation the perturbations yield gender AUC 0.73 and ethnicity(0-vs-2) AUC 0.80 (vs 0.93 / 0.89 for cols 11–30) — a weak residual demographic signal survives at 1e-6 scale, meaning even this "blind" block would leak demographics if a scaler amplified it.

**Documentation / official code:**
- README: `blind_face_embedding = profiles_train[i,31:]` — "20-dimensional embedding, norm 1", per-profile.
- SensitiveNets (arXiv:1902.00334): the method's stated goal is to suppress sensitive attributes **while retaining utility** ("maintaining the utility of the data", "retaining competitive performance") → genuine agnostic embeddings vary per identity. A constant block contradicts the method's purpose.
- Official `FairCV.py` (BiDAlab/FairCVtest, `master`): the "agnostic gender/ethnicity" scenarios construct inputs as `concatenate(profiles[:,4:11], profiles[:,31:])` and train hiring networks on them — the authors intended per-row feature vectors here. (The generation module `generateDatabase` is imported but **absent from the repo**, so the exact export step cannot be inspected.)
- **Provenance confirmed**: `data/FairCVdb.npy` in the repo is a Git LFS pointer (`oid sha256:c8e4a175…5731f`, size 203,041,354). Our local file has identical size and **matching sha256** → the constant block is a property of the official distributed dataset, i.e. a **release/packaging artifact** (most plausibly a vector broadcast at export time).

### 5.3 Consequences for this project

1. **M2 is NOT invalidated.** M2 (and M6) use cols 11–30, which are genuine per-image embeddings (norm-1, per-identity variance, gender AUC 0.93). The M1-vs-M2 face-proxy comparison stands.
2. **The dataset's intended SensitiveNets control arm is unavailable** from the official file. Any reproduction of the paper's "agnostic" scenarios from this file would silently train on (near-)constant features. This is a limitation to state in the final report, and a genuine finding about the dataset itself.
3. Residual open item: the exact cause (broadcast bug vs intentional placeholder) can't be confirmed because the generation code is not public. Recipe if it ever matters: re-extract embeddings from the DiveFace images with the published ResNet-50 + SensitiveNets models and compare.

---

## 6. Reproducibility & frozen artifact inventory

Reproduce: `python faircv_audit_v2.py` (~18 s, deterministic; seed 42 for model + bootstrap; utf-8-safe on Windows).

| Artifact | Content |
|---|---|
| `faircv_audit.py` | original audit (unchanged; numerically sound, narratively flawed) |
| `faircv_audit_v2.py` | corrected implementation (R-1..R-15; frozen) |
| `ground_truth_verify.py` + `gt_verify_run.txt` | independent verification harness + fresh run log |
| `dataset_ground_truth.md` | dataset evidence (schema, distributions, bias construction) |
| `audit_code_review.md` | line-by-line review, issue register BUG-1..11 + N-1..N-10 |
| `faircv_audit_v2_validation.md` | this report |
| `results/metrics.csv` | 12 rows: point estimates + CIs + tests + effect sizes |
| `results/statistical_tests.csv` | 42 rows: KS/KW/χ² with Holm-adjusted p |
| `results/per_group_metrics.csv` | 30 rows: N/SR/TPR/FPR/PPV/mean-score per group |
| `results/audit_report.txt` | full console report |
| `results/fig1..fig5*.png` | selection rates, heatmap (severity-oriented), distributions, coefficients, CI plots |

**Freeze statement:** no methodology was changed during this validation; no code was modified. Any further experiment (top-N screening, text/bio arm) is deliberately deferred.

---

## 7. Validated conclusions (four findings)

1. **The original audit was partially usable**: its numerics are reproduced exactly by v2; its narrative contained errors that are now corrected (v1's EEOC/KS/M2-KL claims).
2. **Feature selection materially mattered**: adding occupation + suitability (CV9) raises accuracy 0.793 → 0.966 (AUC 0.888 → 0.996). M5/M6 also lower gender disparity but slightly raise ethnicity DPD — "more predictive" ≠ unambiguously "fairer".
3. **The face-proxy claim is not supported**: M1-vs-M2 deltas are small with overlapping CIs; ethnicity KL decreases. Correct statement: *under the evaluated FairCV configuration, adding the face-feature block did not provide strong evidence of increased demographic disparity relative to the CV-only baseline.* (Equally, there is no evidence that face embeddings are "unbiased".)
4. **The constructed-bias arms behave as designed, with nuance**: M4/ethnicity fails EEOC (DIR 0.771) but the CI [0.718, 0.828] straddles 0.80 — borderline; M3 passes selection-rate parity yet shows a large, CI-excluding-zero gender EOD (0.198) — evidence that a single 0.80 threshold is insufficient for a fairness audit.
