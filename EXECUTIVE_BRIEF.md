# FairCV Bias Audit — Executive Brief

**Date:** 2026-08-13 · Evidence: `FINAL_REPORT.md` (data: `results/metrics.csv`, `statistical_tests.csv`, `per_group_metrics.csv`, `audit_report.txt`). Numbers are point estimates with bootstrap 95% CIs (n = 2,000, seed 42).

**Bottom line.** In the synthetic FairCVdb testbed (24,000 CVs, 19,200 train / 4,800 test), the corrected audit finds **one demonstrated demographic disparity — M4/ethnicity, real but magnitude-borderline** — plus an **error-rate disparity hidden by a single selection screen** (M3/gender). Everything else is near-parity, and **no verdict changes** when hiring is redefined by the repository's own top-1000 / 75th-percentile protocol.

---

## The one EEOC failure — point estimate vs statistics

| Model × attr (hiring rule) | DIR [95% CI] | Verdict |
|---|---|---|
| M4-Ethnicity-Biased / ethnicity — **median split** | 0.771 [0.718, 0.828] | **FAIL — borderline** |
| M4 / ethnicity — **top-1000** (repo protocol) | 0.710 [0.619, 0.810] | **FAIL — borderline** |
| M4 / ethnicity — **75th percentile** | 0.751 [0.694, 0.807] | **FAIL — borderline** |

- The point estimate fails the EEOC 0.80 rule under every rule, but the CI **straddles 0.80**, so the *magnitude* of the failure is uncertain.
- The *underlying disparity is statistically unambiguous*: χ² p = 4×10⁻¹², Kruskal–Wallis p = 8×10⁻¹⁷, DPD CI [0.0905, 0.1573] excludes 0, Cohen's d = 0.298. G3 (the group penalised ×0.75 in the label construction) is the lowest-selected group everywhere.
- **Honest phrasing:** "point-estimate EEOC failure with a statistically real but magnitude-borderline disparity" — not categorical, and not "no failure".

## The hidden error-rate gap (M3)

M3 (gender-biased labels, female ×0.75) **passes selection parity** (gender DIR 0.974) yet shows the study's largest error-rate disparity: **gender EOD 0.198, CI [0.164, 0.231] — entirely above 0, statistically supported** (Female TPR 0.873 vs Male 0.675). A single 0.80 selection screen is insufficient for a fairness audit.

## Everything else passes

- **Gender:** all models pass all rules (DPD ≤ 0.024; all χ² non-significant). **Statistically supported.**
- **Blind-label ethnicity (M1/M2/M5/M6):** DIR 0.954–0.963 at the median split, 0.886–0.912 at top-1000, 0.954–0.963 at p75 — all pass (only the much tighter top-2% screen, 96 of 4,800, shows wide-CI, low-power dips).
- **M5/M6 (full CV9):** accuracy 0.793 → 0.966, AUC 0.996, gender DPD drops to 0.0038 — but ethnicity DPD *rises* (0.023) and ethnicity KL is large with wide CIs. "More predictive" ≠ unambiguously "fairer".

## Face features — association, not causation

M1 (CV7) vs M2 (CV7+face), identical labels: gender DPD +0.0021 (CIs overlap), ethnicity DPD +0.0013, ethnicity KL **decreases** (0.0295 → 0.0255). No demonstrated bias increase from face embeddings — and no demonstrated "unbiased" either. The intended control (SensitiveNets block, cols 31–50) is a **constant vector in the official dataset**, so the question stays open.

## Robustness of the conclusions

The audit's median-split hiring rule was **not the driver** of any finding: M4/ethnicity fails, gender parity holds, and M3's error-rate gap persists under top-1000 (the repo's own protocol) and the 75th-percentile EEO threshold. Only the top-2% screen adds a low-power, wide-CI threshold nuance.

## Bio/text companion arm (separate experiment — Appendix A of `FINAL_REPORT.md`)

Gender leaks from free text — even redacted bios (AUC 0.71; ~0.76 under a partial one-epoch fine-tune) and names (0.97) — while ethnicity does not leak through any text channel. Yet the text-only hiring model shows the project's **lowest gender DPD (0.0039)**. **Leakage alone does not produce disparate hiring; label bias does.**

## Caveats

Synthetic testbed · numeric profiles only · logistic regressions · constructed label bias (M3/M4) · associations, not causation · the constant blind-face block (cols 31–50) removes the intended control arm · no generalisation to real hiring systems.
