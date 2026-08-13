# Top-N Screening Experiment — Do the Audit Conclusions Change?

**Script:** `topn_screening.py` (companion to the frozen `faircv_audit_v2.py`)
**Outputs:** `results/topn_metrics.csv` (72 rows), `results/topn_per_group.csv` (180 rows), `results/fig6_topn_selection.png`
**Date:** 2026-08-11 · seed 42 · bootstrap n = 1,000 · ~20 s runtime

---

## 1. Objective

The frozen audit (`faircv_audit_v2.py`) defines *hired* as `predicted score ≥ median` of a
binarised label — an arbitrary rule the papers do not use. The FairCV papers evaluate hiring as
**top-N score screening** (demo: top-100 of 4,800 ≈ top-2%; repo code: top-1,000 ≈ top-20%).

This experiment replaces **only the selection rule** — the models M1–M6 are trained exactly as in
v2 (same features, same median-binarised training labels, same seed, so predicted scores are
identical). Hiring is then re-defined as the top **2 / 5 / 10 / 20 / 30 / 50 %** of predicted
scores on the test set, and the fairness audit (DPD, DIR/EEOC, χ², per-group shares, label-based
EOD, bootstrap 95% CIs) is re-run at each threshold.

The question: **does the arbitrary median threshold change any of the audit's headline conclusions?**

---

## 2. Validation — the companion reproduces the frozen audit

Before trusting the thresholds, the top-50% rule was sanity-checked against the frozen
median-split results (`results/metrics.csv`), comparing both DIR magnitudes and the **identity of
the worst-off group**:

| Model | attr | top-50% DIR | median-split DIR | worst-off group (top50 / median) |
|---|---|---|---|---|
| M1 | gender | 0.9876 | 0.9747 | Male / Male ✓ |
| M1 | ethnicity | 0.9694 | 0.9631 | G3 / G2 (noise-level flip, DIR≈0.96) |
| M2 | gender | 0.9713 | 0.9705 | Male / Male ✓ |
| M2 | ethnicity | 0.9599 | 0.9607 | G3 / G2 (noise-level flip, DIR≈0.96) |
| M3 | gender | 0.9892 | 0.9739 | Male / Male ✓ |
| M3 | ethnicity | 0.9563 | 0.9654 | G2 / G2 ✓ |
| M4 | gender | 0.9729 | 0.9800 | Male / Male ✓ |
| M4 | ethnicity | 0.7673 | 0.7711 | G3 / G3 ✓ |
| M5 | gender | 0.9909 | 0.9923 | Female / Female ✓ |
| M5 | ethnicity | 0.9411 | 0.9541 | G2 / G2 ✓ |
| M6 | gender | 0.9810 | 0.9916 | Female / Female ✓ |
| M6 | ethnicity | 0.9650 | 0.9580 | G2 / G3 (noise-level flip, DIR≈0.96) |

All DIR magnitudes agree within ~0.015. The three "flips" (M1/M2/M6 ethnicity) occur **only where
DIR ≈ 0.96 — i.e. near-zero disparity** — so the worst-off designation is at noise level, not a
substantive reversal. (The old "direction" check compared DIR vs 1.0 and could not even see these;
the improved check compares worst-off group identity directly.) **Validation passes.**

---

## 3. Results — ethnicity DIR by model × threshold

DIR = worst-group selection rate / best-group selection rate. **P** = EEOC pass (≥ 0.80), **X** = fail.

| Model | 2% | 5% | 10% | 20% | 30% | 50% | median-split |
|---|---|---|---|---|---|---|---|
| M1-Fair (CV7) | **X** 0.655 | P 0.814 | P 0.988 | P 0.913 | P 0.969 | P 0.969 | P 0.963 |
| M2-Multimodal (CV7+Face) | **X** 0.527 | P 0.811 | P 0.965 | P 0.919 | P 0.937 | P 0.960 | P 0.961 |
| M3-Gender-Biased (CV7) | **X** 0.646 | P 0.814 | P 0.995 | P 0.910 | P 0.973 | P 0.956 | P 0.965 |
| **M4-Ethnicity-Biased (CV7)** | **X** 0.481 | **X** 0.691 | **X** 0.752 | **X** 0.702 | **X** 0.762 | **X** 0.767 | **X** 0.771 |
| M5-Robust (CV9) | **X** 0.690 | P 0.865 | P 0.886 | P 0.881 | P 0.940 | P 0.941 | P 0.954 |
| M6-Robust (CV9+Face) | **X** 0.672 | P 0.865 | P 0.914 | P 0.884 | P 0.943 | P 0.965 | P 0.958 |

**Gender DPD is tiny at every threshold** (max 0.020 across all 36 gender cells); all gender cells
pass EEOC (min DIR 0.82), and all gender χ² tests are non-significant. Gender conclusions do not
change anywhere.

---

## 4. Statistical significance (χ², group × hired) for the failures

| Model × ethnicity | 2% | 5% | 10% | 20% | 30% | 50% |
|---|---|---|---|---|---|---|
| M1 | 0.23 | 0.38 | 0.99 | 0.42 | 0.85 | 0.60 |
| M2 | **0.046** | 0.37 | 0.94 | 0.49 | 0.49 | 0.42 |
| M3 | 0.22 | 0.39 | 0.93 | 0.38 | 0.84 | 0.53 |
| **M4** | **0.014** | **0.043** | **0.029** | **2.2e-6** | **1.2e-6** | **2.0e-13** |
| M5 | 0.33 | 0.49 | 0.86 | 0.55 | 0.62 | 0.34 |
| M6 | 0.28 | 0.49 | 0.89 | 0.42 | 0.75 | 0.51 |

- **M4's ethnicity disparity is significant at every threshold** and becomes overwhelming as the
  threshold loosens (p = 0.014 → 2×10⁻¹³).
- Among blind-label models, **only M2's top-2% failure is significant** (p = 0.046) — and only
  marginally so.

### Bootstrap 95% CIs at top-2% (the papers' protocol; 96 of 4,800 selected)

| Model × ethnicity | DIR | 95% CI |
|---|---|---|
| M4 | 0.481 | **[0.236, 0.706]** — entirely below 0.80 |
| M2 | 0.527 | [0.325, 0.855] — straddles 0.80 |
| M1 | 0.655 | [0.364, 0.900] — straddles 0.80 |
| M5 | 0.690 | [0.387, 0.905] |
| M6 | 0.672 | [0.365, 0.898] |
| M3 | 0.646 | [0.359, 0.922] |

### M4 ethnicity per-group selection rates (G1 = boosted ×1.25, G3 = penalised ×0.75)

| threshold | G1 | G2 | G3 |
|---|---|---|---|
| 2% | 0.0245 | 0.0239 | **0.0118** |
| 20% | 0.2389 | 0.1940 | **0.1676** |
| 50% | 0.5730 | 0.4880 | **0.4397** |

G3 is the worst-off group at **every** threshold — the label-bias penalty (×0.75) shows through in
selection rates regardless of how hiring is defined.

### Label-based EOD (true = top-k% of the model's own test labels)

At k = 50%, M3's gender EOD = **0.178** — consistent with v2's 0.198 under the median split
(definitions differ slightly: quantile of continuous labels here vs binarised labels in v2, so this
is "consistent with", not an exact reproduction). **The M3 finding — passes selection-rate parity
but shows substantial error-rate disparity — survives the change of selection rule.**

---

## 5. Do the frozen audit's conclusions change?

| Finding (frozen audit) | Under top-N screening | Verdict |
|---|---|---|
| **F1** Gender parity: all models pass, DPD tiny | Passes at **every** threshold (2%–50%) | **Robust — unchanged** |
| **F2** M4 ethnicity FAILs EEOC (DIR 0.771) | Fails at **all six** thresholds; worst at 2% (0.481, CI entirely < 0.80); significant at every threshold | **Robust — strengthened** (the CI-straddling caveat from the median split disappears at 2%: the whole CI is below 0.80) |
| **F3** M3 passes selection parity but gender EOD = 0.198 | EOD 0.178 at 50% (and 0.32 at 2%) | **Robust — unchanged** |
| **F4** M1/M2/M5/M6 ethnicity pass EEOC | **Fail at top-2%** (the papers' own protocol); pass at 5%+. Only M2's failure is marginally significant | **Threshold-dependent — conclusion must be qualified** |
| **F5** M2 does not clearly increase disparity (DPD CI overlap; ethnicity KL decreases) | Still holds at 5%–50%. At 2% only, M2 has the **worst** DIR (0.527) of the blind-label models and the only significant χ² — but its CI straddles 0.80 | **Broadly unchanged; weak, low-power counter-signal at 2% only** |

### What this means for the final report

1. **The median threshold was not the driver of the headline findings.** The two strongest
   results — M4's ethnicity failure and M3's gender EOD — hold across all six selection rules.
   Gender parity is robust everywhere.

2. **One conclusion needs re-qualification.** "The fair/blind models pass the EEOC ethnicity
   screen" is true for loose screening (≥ 5%) and for the median split, but **not** under the
   papers' tight top-2% protocol (top-100/4,800), where every model fails — albeit with wide CIs
   and (except M2) non-significant χ². The honest statement is:
   > *At the tightest screening (top-2%), even the blind-label models show point-estimate
   > ethnicity DIR values of 0.53–0.69, but with bootstrap CIs spanning 0.36–0.92 and χ²
   > non-significant except M2 (p = 0.046). This is low-power evidence of a threshold effect, not
   > a demonstrated disparity.*

3. **A new nuance for the face-proxy question (F5):** the only place M2 shows *worse* disparity
   than M1 is the tightest screen (DIR 0.527 vs 0.655). This is a weak, marginally-significant,
   wide-CI signal — consistent with the overall finding that the original "face embeddings cause
   bias" claim remains **unsupported**, while noting the interaction with screening strictness as
   an open question rather than a confirmed effect.

---

## 6. Caveats & limitations

- **Top-2% is low-power:** 96 selections of 4,800. Per-group counts are still adequate (χ²
  expected counts ≈ 25–32), but DIR CIs are very wide ([0.24, 0.92] across models). Top-2%
  verdicts should be read as screening evidence, not precise estimates.
- **EOD_labels differs from v2's EOD** (quantile of continuous labels vs binarised labels) — used
  for the M3 check only; the two are "consistent", not identical.
- **M3/M4 label-based metrics are against artificially biased ground truth** (v2 R-9 caveat
  applies); only selection-rate metrics are neutral across models.
- Models are regularised logistic regressions on the numeric profile block; the constant
  blind-face block (cols 31–50) remains excluded; the conclusions are specific to this setup.
- Bootstrap CIs re-derive extreme groups per resample (standard for max-min statistics), the same
  convention as the frozen audit.

---

## 7. Freeze status

`faircv_audit_v2.py` was **not** modified. `topn_screening.py` is a companion script and joins the
frozen artifact set alongside `results/topn_metrics.csv`, `results/topn_per_group.csv`,
`results/fig6_topn_selection.png`, `faircv_audit_v2_validation.md`, `dataset_ground_truth.md`,
`audit_code_review.md`, and `gt_verify_run.txt`.
