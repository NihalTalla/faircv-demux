# Code Review — `faircv_audit.py`

**Date:** 2026-08-11
**Method:** Line-by-line review of `faircv_audit.py` against (a) the dataset ground truth (`dataset_ground_truth.md`), (b) the official FairCVtest README/paper, and (c) a fresh independent recomputation (`ground_truth_verify.py`, run output in `gt_verify_run.txt`, 2026-08-11). No code in this review was modified.

**Verdict legend:** 🟢 Valid · 🟡 Needs clarification/documentation · 🔴 Incorrect · ⚠️ Unsupported claim

---

## 1. Executive summary

The audit script is **numerically sound but narratively flawed**:

- All computed metrics (DPD, DIR, EOD, EO, KL for M1 gender/ethnicity; DPD/DIR for M4-ethnicity) **reproduce within 5e-4** of the script's own printed output. The underlying math is correct.
- The **printed narrative makes 3 false/overstated claims** (Obs-1, Obs-2, Obs-3), including a "significant KS p-value" that the code itself never produces.
- The **methodology has gaps**: no multi-group significance test for ethnicity, no uncertainty quantification, M3/M4 evaluated against their own biased labels, and an undocumented feature-set choice that drops the strongest merit feature.
- **11 issues** were previously identified in `ground_truth_verify.py` (BUG-1..11); this review confirms them and adds findings N-1..N-10.

---

## 2. Component-by-component review

### 2.1 Loading ([1/6])
| Component | Verdict | Notes |
|---|---|---|
| `np.load(DATA_PATH, allow_pickle=True).item()` | 🟢 | Correct — the file is a pickled dict (see ground truth §1). All 8 keys used exist with the expected shapes. |
| Split sizes 19,200 / 4,800 | 🟢 | Verified (80/20, balanced). |

### 2.2 Feature extraction ([2/6])
| Component | Verdict | Notes |
|---|---|---|
| `gender = P[:,1]`, `ethnicity = P[:,0]` | 🟢 | Column indices correct per README. |
| `CV_COLS = range(4, 11)` (7 features) | 🟡🔴 | **Drops cols 2–3 (occupation, suitability).** Suitability is the single strongest blind-label predictor (corr ≈ 0.48; part of the paper's score formula). M1/M3/M4 therefore train on 7 of 9 available profile features. The docstring comment's *meanings* for cols 4–10 are correct, but calling them "CV merit features" while silently excluding occupation/suitability is undocumented. **Finding N-1.** |
| `FACE_COLS = range(11, 31)` (20 features) | 🟢 | Correct (face embedding, norm 1). |
| M2 input = CV + FACE (27 cols) | 🟢 | Matches stated design. |
| Ignoring cols 31–50 (blind face) | 🟡 | The dataset ships an agnostic-embedding control arm, but in this file that block is a **constant vector** (ground truth §6), so it is unusable anyway. The script never references it — should be documented as a limitation (can't run a SensitiveNets control). **Finding N-2.** |
| Ignoring `Bios`, `Names`, `Image List` | 🟡 | FairCVtest is explicitly multimodal (structured + image + text); the audit is numeric-profiles-only. Valid scope, must be disclosed; claims should be scoped accordingly. **Finding N-3.** |

### 2.3 Label binarisation ([2/6])
| Component | Verdict | Notes |
|---|---|---|
| `threshold = median(train labels)`, `label ≥ threshold → hired` | 🟡 | Defensible as a ~50% base-rate binarisation, **but it is an audit-invented outcome**. The paper's own evaluation screens the *top-N scores* (top 100 of 4,800), not "above the median". "Above median = hired" has no source in the dataset or paper. Must be documented as an artificial outcome. **Finding N-4.** |
| Independent median per label set (blind / gender / eth) | 🟡 | Normalises every model to ~50% base rate, erasing absolute score shifts caused by the bias penalty (e.g. female median drops from 0.414 → 0.366 under gender-biased labels). Fine for within-model selection-rate analysis; makes *absolute* score comparisons across models meaningless. |

### 2.4 Models ([3/6])
| Component | Verdict | Notes |
|---|---|---|
| M1 = CV-only + blind labels | 🟢 | Sound control (fair labels, no face). |
| M2 = CV+face + blind labels | 🟢 | Clean proxy test: **same labels as M1, features differ** — the M1-vs-M2 delta is attributable to adding face features. ⚠️ Note this is **not** the paper's Scenario 4 (paper uses *gender-biased* labels + face); the audit's design is stronger for the proxy claim but must not be described as reproducing the paper's experiments. **Finding N-10.** |
| M3/M4 = CV-only + biased labels | 🟡🔴 | **Evaluated against their own biased test labels** → TPR/FPR/PPV are relative to an artificially constructed ground truth; cross-model performance comparison (acc/AUC/F1 across M1..M4) is apples-to-oranges. Selection-rate disparity M1-vs-M3 *is* interpretable (same features, different labels), but this distinction is never made. **Findings N-6, N-9** (below). |
| LogisticRegression (C=1.0, max_iter=1000, seed 42), StandardScaler pipeline | 🟢 | Deterministic and reproducible; appropriate for an audit baseline. No CV/hyperparameter search — acceptable, disclose. |

### 2.5 Fairness metrics ([4/6], `audit_group`)
| Component | Verdict | Notes |
|---|---|---|
| DPD = max SR − min SR | 🟢 | Correct demographic-parity difference. |
| DIR = min SR / max SR | 🟡 | Computes **worst-group / best-group** ratio — the right thing for an audit. ⚠️ The printed narrative calls it "minority SR ÷ majority SR" — wrong: "minority" here means demographic minority, not lowest-selected group; these need not coincide. **(BUG-6).** |
| EEOC pass = DIR ≥ 0.80 | 🟢 | Correct four-fifths-rule threshold. |
| EOD = max TPR − min TPR | 🟡 | Magnitude only — **sign/direction lost** (which group is favoured). The script's own note "Positive = favoured group has higher recall" is meaningless since the value is always ≥ 0. **(BUG-7).** |
| EO = max(\|ΔTPR\|, \|ΔFPR\|) | 🟢 | Valid equalized-odds violation magnitude. |
| KL divergence (histogram, `rel_entr`) | 🟢 | Implementation correct (verified). 🟡 Computed only for the **highest- vs lowest-SR group pair** — for 3-group ethnicity the middle group is ignored; all pairwise KLs should be reported (the paper itself averages the 3 pairwise KLs for ethnicity). **(BUG-11).** |
| KS test `if len(groups) == 2 else NaN` | 🟡🟢 | Correctly *avoids* an invalid 2-sample KS on 3 groups, but leaves **no alternative significance test for ethnicity** (needs Kruskal–Wallis or pairwise KS + multiple-comparison correction). **(BUG-5).** |
| No confidence intervals / bootstrap / effect sizes | 🔴 | None of the disparity metrics carry uncertainty. With n=4,800 the point estimates are stable, but no statistical test distinguishes "observed disparity" from "disparity we can be confident in". Missing Phase-7 of the project plan. **Finding N-5.** |

### 2.6 Visualisations
| Component | Verdict | Notes |
|---|---|---|
| Fig 1 selection rates | 🟢 | Accurate. Note the dashed "50% base rate" line is only meaningful for blind-label models. |
| Fig 2 heatmap | 🔴 | **DIR colour direction is inverted**: `RdYlGn_r` with column normalisation makes the *best* (highest) DIR render as the *most severe*. Visually misrepresents the very metric used for the EEOC verdict. **(BUG-10).** |
| Fig 3 score distributions | 🟢 | Title prints `p_KS=nan` for ethnicity rows (KS not computed) — visible symptom of BUG-5. |
| Fig 4 coefficients | 🟢 | Fine. |

### 2.7 Printed narrative ([8/6] summary)
| Claim (verbatim intent) | Verdict | Evidence |
|---|---|---|
| Obs-1: "M1 … lowest bias across all metrics" | 🟡 | Overstated. M1 has the lowest DPD on most metrics, but M4 has *lower gender EOD* (0.011 vs 0.012). "Across all metrics" is false; "most metrics" is true. **(BUG-4).** |
| Obs-2: "M2 shows measurably higher DPD and KL than M1 … face embeddings act as a proxy … inject bias at inference time" | ⚠️ | True for **gender** DPD/KL. **False for ethnicity KL**: M2 KL (0.0255) < M1 KL (0.0295) — independently reproduced. The claim must be qualified per attribute. **(BUG-3).** Also: this is an *association*, not a demonstrated causal mechanism; wording should be "adding face embeddings was associated with increased disparity under this setup". |
| Obs-3: "M3 and M4 … EEOC 80% rule is violated … significant KS p-value" | 🔴 | Two errors: (a) **M3 PASSES EEOC for both gender and ethnicity** — only M4/ethnicity fails (DIR 0.771); (b) **no KS p-value < 0.05 exists for any model** (gender KS p ≈ 0.32 for M1; ethnicity KS is NaN by design). Both sub-claims are unsupported/false. **(BUG-1, BUG-2).** |

### 2.8 Other observations
- The script never states **who the biased labels penalise** (gender-biased = ×0.75 on females; ethnicity-biased = ×0.75 on code 2, ×1.25 on code 0). This context is essential for interpreting M3/M4 and should be printed. **Finding N-7.**
- Results are console-only + 4 PNGs; **no CSV/JSON of metrics** is exported, so the numbers are ephemeral. Export `metrics.csv`/`statistical_tests.csv` (deliverable 4). **Finding N-8.**
- `ground_truth_verify.py` (the sibling verification script) fails on Windows when output is redirected unless `PYTHONIOENCODING=utf-8` (cp1252 console, `→` char). **Finding N-9.**
- `GENDER_LABELS`/`ETHNICITY_LABELS` names ("Grp-A/B/C"): gender naming verified; ethnicity naming invented (README uses G1/G2/G3). **(BUG-8).**

---

## 3. Issue register

### Confirmed from `ground_truth_verify.py` (freshly reproduced 2026-08-11)
| ID | Sev. | Location | Issue |
|---|---|---|---|
| BUG-1 | 🔴 | Narrative, Obs-3 | Claims "M3 and M4" violate EEOC; M3 **passes** both attributes. Only M4/ethnicity fails (DIR 0.771). |
| BUG-2 | 🔴 | Narrative, Obs-3 | Claims "significant KS p-value"; no gender KS p < 0.05 (M1: 0.318), ethnicity KS is NaN. |
| BUG-3 | 🔴 | Narrative, Obs-2 | "M2 higher KL than M1" is false for ethnicity (0.0255 < 0.0295). |
| BUG-4 | 🟡 | Narrative, Obs-1 | "Lowest bias across all metrics" overstated (M4 gender EOD 0.011 < M1 0.012). |
| BUG-5 | 🔴 | `audit_group`, KS | 2-group-only KS leaves ethnicity untested; no Kruskal–Wallis/pairwise + correction. |
| BUG-6 | 🟡 | DIR printout | "minority ÷ majority" wording ≠ code (min-SR / max-SR). |
| BUG-7 | 🟡 | EOD | Unsigned; direction of disparity lost; printed sign note is meaningless. |
| BUG-8 | 🟡 | Header/labels | "Group A/B/C" ethnicity names unsupported (README: G1/G2/G3). |
| BUG-9 | 🟡 | M3/M4 eval | TPR/FPR/PPV computed against biased ground truth; uninterpretable as selection quality; cross-model metric comparison misleading. |
| BUG-10 | 🔴 | Fig 2 | DIR colour scale inverted in heatmap. |
| BUG-11 | 🟡 | KL | Only extreme-SR pair reported for 3-group ethnicity; middle group ignored. |

### New findings from this review
| ID | Sev. | Location | Issue |
|---|---|---|---|
| N-1 | 🟡 | `CV_COLS` | Cols 2–3 (occupation, suitability) dropped without documentation; suitability is the strongest merit predictor (corr 0.48) and part of the paper's score formula → M1/M3/M4 train on 7/9 features. |
| N-2 | 🟡 | Feature set | Blind face block (31–50) — the dataset's control for the face-proxy claim — is degenerate (constant vector) in this file and never used; no SensitiveNets-style control is possible from this file. |
| N-3 | 🟡 | Scope | Bios/Names/ImageList unused; audit is numeric-only within a multimodal benchmark — disclose scope in the report. |
| N-4 | 🟡 | Binarisation | Median-split "hired" label is an artificial outcome (paper uses top-N score screening); document rationale. |
| N-5 | 🔴 | Statistics | No CIs/bootstrap/tests/effect sizes/multiple-comparison correction for any disparity metric. |
| N-6 | 🔴 | Model eval | M1..M4 performance numbers are not comparable (labels differ by construction); report must separate "label-bias" effect (M1 vs M3) from "proxy" effect (M1 vs M2). |
| N-7 | 🟡 | Reporting | The bias direction (females ×0.75; ethnicity code 2 ×0.75 / code 0 ×1.25) is never stated — essential context for M3/M4. |
| N-8 | 🟡 | Outputs | Metrics only printed, not exported to CSV/JSON. |
| N-9 | 🟡 | Tooling | `ground_truth_verify.py` requires `PYTHONIOENCODING=utf-8` on Windows (cp1252). |
| N-10 | ⚠️ | M2 claim | M2 (blind labels + face) ≠ paper's Scenario 4 (gender-biased labels + face); don't claim parity with the paper's experiments. |

---

## 4. Required changes (for `faircv_audit_v2.py`)

**Narrative fixes (must-do, cheap):**
- R-1 Obs-3: "Only M4/ethnicity fails the EEOC 80% rule (DIR 0.771); M3 passes both attributes."
- R-2 Obs-3: delete "significant KS p-value"; report actual p-values and state none are significant at α=0.05.
- R-3 Obs-2: qualify "higher DPD/KL" to gender; state ethnicity KL is lower for M2 (0.0255 vs 0.0295).
- R-4 Obs-1: "lowest bias on most metrics", not "all".
- R-6: relabel DIR as "worst-group SR / best-group SR".
- R-7: report signed EOD (which group has higher TPR) per attribute.
- R-8: use G1/G2/G3 (README) or "group 0/1/2" instead of invented Grp-A/B/C.

**Methodology fixes (should-do):**
- R-5: add Kruskal–Wallis for 3-group ethnicity; add pairwise KS/KS-style tests with Holm/Bonferroni correction.
- R-11: report all pairwise KL values for ethnicity (3 pairs) and/or the mean (paper convention).
- R-9: add an explicit note that M3/M4 TPR/FPR/PPV are against biased ground truth; frame M1-vs-M3 as the label-bias contrast.
- R-13 (new): bootstrap 95% CIs for DPD/DIR/EOD (e.g., 2,000 resamples); report effect sizes; apply multiple-comparison correction across all tests.
- R-12 (new): document the feature set (9 available, 7 used; occupation/suitability excluded) and why; consider adding M2' with all 9 CV features + face for robustness.
- R-14 (new): state the bias construction in the report (×0.75 female; ×0.75/×1.25 ethnicity) and scope all claims to the numeric-profile experiment (Bios/Names/ImageList out of scope).
- R-15 (new): export `results/metrics.csv` and `results/statistical_tests.csv`.

**Visualisation fix:**
- R-10: invert/normalise DIR separately in the heatmap (higher DIR = better = green), or exclude DIR and plot it beside.

---

## 5. Final verdict

| Dimension | Verdict |
|---|---|
| Numerics (DPD/DIR/EOD/EO/KL/KS) | 🟢 Correct — all independently reproduced within 5e-4 |
| Narrative | 🔴 3 false/overstated claims (Obs-1, Obs-2-ethnicity, Obs-3 ×2) |
| Statistical rigour | 🟡 Missing multi-group tests, uncertainty, and correction for multiple comparisons |
| Design clarity | 🟡 M3/M4 vs biased labels; feature-set choice undocumented; bias direction unreported |
| Visualisation | 🔴 DIR heatmap inverted |
| Reproducibility | 🟢 Deterministic (seed 42) but metrics not exported to disk |

**The script should not be presented as-is.** It is a solid numerical skeleton whose conclusions are overstated and partially contradicted by its own numbers. `faircv_audit_v2.py` should keep the model design and metric math, fix the narrative, add the multi-group tests + bootstrap uncertainty, export machine-readable results, and document the feature-set and label-construction choices.
