# Dataset Ground Truth — `FairCVdb.npy`

**Project:** Verification-first FairCV bias audit
**Date:** 2026-08-11
**Status:** Immutable snapshot — the dataset file was **only read**, never modified (a checksum-able snapshot approach is recommended before any downstream work).
**Evidence sources:**
1. Direct interrogation of `FairCVdb.npy` (numpy, read-only)
2. Official FairCVtest GitHub README — https://github.com/BiDAlab/FairCVtest (authoritative column schema)
3. Peña et al., *Bias in Multimodal AI: Testbed for Fair Automatic Recruitment*, CVPR-W 2020 (arXiv:2004.07173) — dataset construction
4. Peña et al., *FairCVtest Demo*, ICMI 2020 (arXiv:2009.07025) — recruitment scenarios

---

## 1. File structure — this is NOT a raw `(24000, 51)` array

`FairCVdb.npy` is a **pickled Python dict** (numpy object array). It must be loaded with
`np.load(path, allow_pickle=True).item()`. It contains **14 keys**:

| Key | dtype | Shape | Contents |
|---|---|---|---|
| `Profiles Train` | float64 | (19200, 51) | Numeric profile features |
| `Profiles Test` | float64 | (4800, 51) | Numeric profile features |
| `Bios Train` | `<U968` | (19200, 2) | Bio text: `[original, gender-blinded]` |
| `Bios Test` | `<U968` | (4800, 2) | Bio text: `[original, gender-blinded]` |
| `Names Train` | `<U20` | (19200,) | First names |
| `Names Test` | `<U20` | (4800,) | First names |
| `Blind Labels Train` | float64 | (19200,) | Unbiased hiring score |
| `Blind Labels Test` | float64 | (4800,) | Unbiased hiring score |
| `Biased Labels Train (Gender)` | float64 | (19200,) | Gender-biased hiring score |
| `Biased Labels Test (Gender)` | float64 | (4800,) | Gender-biased hiring score |
| `Biased Labels Train (Ethnicity)` | float64 | (19200,) | Ethnicity-biased hiring score |
| `Biased Labels Test (Ethnicity)` | float64 | (4800,) | Ethnicity-biased hiring score |
| `Image List Train` | `<U51` | (19200,) | Relative paths to face photographs |
| `Image List Test` | `<U51` | (4800,) | Relative paths to face photographs |

> ⚠️ **The labels are stored separately from the profiles.** There is no label column in `Profiles`; the hiring target lives in the 6 label arrays. The audit script maps these keys correctly.

> ⚠️ **The image files themselves are not present in this project folder** — only `Image List` paths. Any experiment must use the precomputed face embeddings (cols 11–30), not raw pixels.

---

## 2. Column schema of `Profiles` (51 columns)

Authoritative mapping from the official README (verified against observed value ranges):

| Cols | Meaning | Values (observed) | Used by audit? |
|---|---|---|---|
| 0 | **Ethnicity** | {0, 1, 2} = G1, G2, G3 (README) | Protected (excluded) |
| 1 | **Gender** | {0 = Male, 1 = Female} | Protected (excluded) |
| 2 | **Occupation** | 0–9 (10 categories, see §4) | ❌ dropped |
| 3 | **Suitability** | {0.25, 0.5, 0.75, 1.0} | ❌ dropped |
| 4 | Education attainment | {0.2, 0.4, 0.6, 0.8, 1.0} | ✅ CV feature |
| 5 | Previous experience | {0, 0.2, …, 1.0} | ✅ CV feature |
| 6 | Recommendation letter | {0, 1} (10.4% = 1) | ✅ CV feature |
| 7 | Availability | {0.2, …, 1.0} (skewed high) | ✅ CV feature |
| 8–10 | Language proficiency ×3 | {0, 0.2, …, 1.0} | ✅ CV features |
| 11–30 | **Face embedding** (ResNet-50 bottleneck) | 20-dim, **exact L2 norm = 1.0** | ✅ M2 only |
| 31–50 | **Blind face embedding** (SensitiveNets-agnostic) | 20-dim, norm 1, **but degenerate — see §6** | ❌ never used |

**Observed value counts (train), cols 0–10:**

```
col0 ethnicity : 0→6405  1→6412  2→6383
col1 gender    : 0→9563  1→9637
col2 occupation: 0→1598 1→1599 2→1593 3→1597 4→1586 5→1607 6→2398 7→2391 8→2390 9→2441
col3 suitabil. : 0.25→4790  0.5→4831  0.75→4790  1.0→4789
col4 education : 0.2→2508  0.4→6000  0.6→6920  0.8→3078  1.0→694
col5 exper.    : 0→359  0.2→2624  0.4→6093  0.6→6439  0.8→3148  1.0→537
col6 rec-letter: 0→17206  1→1994
col7 availab.  : 0.2→318  0.4→528  0.6→1609  0.8→4219  1.0→12526
col8-10 lang   : 6 levels each, roughly uniform (~3000/level)
```

---

## 3. Protected attributes

### Gender (col 1) — **verified 0 = Male, 1 = Female**
- README: `gender = profiles_train[i,1]  # 0 = Male, 1 = Female`
- **Independent internal evidence (bio pronouns):** across all 19,200 train bios, every row with "he/his" is gender 0 and every row with "she/her" is gender 1:

  | | his/he rows | her/she rows |
  |---|---|---|
  | gender = 0 | 9,057 | 0 |
  | gender = 1 | 0 | 8,740 |

  (1,403 rows contain no gendered pronoun.) Perfect separation → col 1 is unambiguously a gendered attribute with the README mapping.

### Ethnicity (col 0) — **verified values {0, 1, 2}; names are G1/G2/G3, NOT "Group A/B/C"**
- README: `ethnicity = profiles_train[i,0]  # 0 = G1, 1 = G2, 3 = G3` (the `3 = G3` is a README typo; data is `2 = G3`).
- The paper (§3.1) states the three groups come from **DiveFace** and correspond to **Black, Asian, and Caucasian** — but **no source maps code → ethnic group**. Codes 0/1/2 cannot be named from the dataset or its documentation alone.
- The audit script's "Grp-A / Grp-B / Grp-C" names are **invented** (they are not even the G1/G2/G3 placeholder names from the README).
- Image-path prefixes independently decode the coding: `H/M` = gender 0/1 and `A/B/N` = ethnicity 0/1/2 (see §5).

---

## 4. Occupation & suitability

README occupation mapping: `nurse(0) surgeon(1) physician(2) journalist(3) photographer(4) filmmaker(5) teacher(6) professor(7) attorney(8) accountant(9)`.

**Suitability is a deterministic function of occupation sector** (occupations grouped into 4 labour sectors):

| Occupation (col 2) | Sector | Suitability (col 3) | Mean blind label |
|---|---|---|---|
| nurse, surgeon, physician (0–2) | healthcare | 0.75 | 0.450–0.466 |
| journalist, photographer, filmmaker (3–5) | media | 0.25 | 0.324–0.336 |
| teacher, professor (6–7) | education | 1.00 | 0.502–0.505 |
| attorney, accountant (8–9) | legal/finance | 0.50 | 0.381–0.389 |

**Suitability is the single strongest predictor of the blind label** (label means 0.329 → 0.504 across its 4 levels; Pearson corr ≈ 0.48). The paper's score formula is a linear combination of candidate competencies (12 features in the paper; 9 released) + Gaussian noise — suitability is explicitly one of the components. **The audit drops both cols 2 and 3** (see `audit_code_review.md`, finding N-1).

---

## 5. Labels — distribution and bias construction

All labels are **continuous hiring probabilities in [0, 1]** ("target score"; 0 = worst candidate, 1 = best).

| Array | min | median | mean | max | #zeros |
|---|---|---|---|---|---|
| Blind Train | 0.0000 | 0.4135 | 0.4185 | 0.9232 | 1 |
| Blind Test | 0.0380 | 0.4116 | 0.4153 | 1.0000 | 0 |
| Gender-biased Train | 0.0000 | 0.3659 | 0.3713 | 0.8996 | 20 |
| Gender-biased Test | 0.0000 | 0.3640 | 0.3688 | 0.9430 | 6 |
| Ethnicity-biased Train | 0.0000 | 0.4148 | 0.4186 | 1.0000 | 15 |
| Ethnicity-biased Test | 0.0000 | 0.4106 | 0.4150 | 1.0000 | 5 |

**Bias construction — verified empirically as a multiplicative group penalty** (ratio of biased/blind label, train, blind > 0.05):

| Label set | Group | Ratio biased/blind |
|---|---|---|
| Gender-biased | gender 0 (Male) | 1.0000 (no change) |
| Gender-biased | gender 1 (Female) | **0.7502 (×0.75 penalty)** |
| Ethnicity-biased | ethnicity 0 | **1.2481 (×1.25 boost)** |
| Ethnicity-biased | ethnicity 1 | 1.0000 (no change) |
| Ethnicity-biased | ethnicity 2 | **0.7478 (×0.75 penalty)** |

This matches the paper: "biased scores are generated by applying a penalty factor to certain individuals belonging to a particular demographic group". **The gender-biased labels penalise females (×0.75); the ethnicity-biased labels penalise code 2 (×0.75) and boost code 0 (×1.25).**

Label agreement (train, Pearson): blind vs gender-biased **0.933**, blind vs ethnicity-biased **0.855**, gender- vs ethnicity-biased **0.798**.

Blind labels are **nearly independent of protected attributes** (binarised blind hiring rate: gender 0.498/0.502; ethnicity 0.511/0.497/0.492), confirming the "agnostic target" design — this is what makes M1 a fair-label control.

---

## 6. Face embedding blocks (cols 11–30 and 31–50)

| Block | Shape | L2 norm per row | Value range | Max \|corr\| w/ gender | Max \|corr\| w/ ethnicity |
|---|---|---|---|---|---|
| 11–30 (face) | (19200, 20) | **exactly 1.0** (std 0) | [−0.721, 0.767] | 0.456 | 0.306 |
| 31–50 (blind face) | (19200, 20) | exactly 1.0 | [−0.446, 0.304] | 0.185 | 0.259 |

**Critical finding — the "blind face embedding" block is degenerate.**
- Every one of the 19,200 train rows lies within 1e-4 of the same constant 20-dim vector (mean max-abs deviation ≈ 6e-6; per-column std ≈ 2–3e-6).
- I.e., **cols 31–50 are a single constant vector (norm 1) plus float noise**. As model features they carry zero variance — any standard scaler amplifies pure noise, and no model can extract identity or demographic signal from them.
- The README describes this block as the SensitiveNets gender/ethnicity-agnostic embedding — a natural *control arm* for the face-proxy hypothesis. **In this file, that control is unavailable** (whether this is a packaging bug in the released file or intentional cannot be determined here).
- The face block (11–30) does carry demographic signal (max |corr| 0.46 with gender, 0.31 with ethnicity), which is the basis for the M1-vs-M2 proxy test.

---

## 7. Missing values & duplicates

- **No missing values anywhere:** NaN = 0, Inf = 0 for Profiles Train/Test and all label arrays. All 51 columns complete.
- Within-train competency profiles (cols 2–10): 16,971 unique combinations of 19,200 (88%); max frequency of a single combination = 5. Face embeddings: 8 exact (4-dp) duplicate rows out of 19,200. No pathological duplication.

---

## 8. Train/test integrity

| Check | Result |
|---|---|
| Sizes | Train 19,200 / Test 4,800 = **80/20** (paper: same split) |
| Total | 24,000 ✓ |
| Gender balance | Train 49.8/50.2%, Test 50.8/49.2% — balanced ✓ |
| Ethnicity balance | Train 33.4/33.4/33.2%, Test 33.2/33.1/33.7% — balanced ✓ |
| Row overlap | 0 of 4,800 test profiles match any train profile (cols 0–30, rounded 6 dp) — **no obvious leakage** ✓ |
| Names | 1,162 names appear in both splits — names come from a **finite name pool** re-sampled across profiles; not row-level leakage, but names alone cannot separate splits |
| Image prefix consistency | Prefix→(gender, ethnicity) mapping identical in train and test ✓ |

Bio structure: `Bios[:, 0]` = original text with gendered pronouns/names; `Bios[:, 1]` = same text with those markers replaced by `_` (gender-blinded variant). Names in `Bios[0]` also appear in `Names`.

---

## 9. Phase-1 claim ledger

| Check | Required | Verdict |
|---|---|---|
| Shape | 19,200 / 4,800 × 51 | ✅ Verified (as two dict keys; not one 24,000-row array) |
| Columns | Meaning of all 51 | ✅ 51 mapped via README; ⚠️ 2 open items: ethnicity code→group names, language identities |
| Gender | Unique values + counts | ✅ {0, 1}, 0 = Male (README + bio-pronoun proof) |
| Ethnicity | Unique values + counts | ✅ {0, 1, 2} balanced; names = G1/G2/G3 placeholders (not A/B/C) |
| Labels | Range/distribution | ✅ Continuous [0, 1] probabilities; blind mean ≈ 0.42 |
| Missing values | Count by column | ✅ Zero across all arrays |
| Train/test | No obvious leakage | ✅ 80/20, balanced, 0 row overlap |
| Protected attrs | Genuinely protected | ✅ Gender & ethnicity are cols 0–1, excluded from the label formula (blind labels ~independent); ⚠️ names, original bios, and face images are *sensitive proxies* per the paper but are not "protected columns" |
| Face features | Dimensions | ✅ 20-dim, norm 1 (cols 11–30) |
| CV features | Dimensions | ✅ 9 features (cols 2–10: occupation, suitability + 7 competencies) — **audit uses only 7 (cols 4–10)** |

---

## 10. Open questions & caveats

1. **Ethnicity code → ethnic group** (Black/Asian/Caucasian per paper): no source maps code 0/1/2 to groups. The image-prefix letters are A/B/N, not informative of the group names.
2. **Which 3 languages** are encoded in cols 8–10: not documented in the README (paper mentions 8 languages for the unpublished variant).
3. **Blind face block degeneracy:** cols 31–50 are a constant vector in this file — flag for any experiment that assumes usable agnostic face features. Worth verifying against the original Google-Drive copy of FairCVdb before relying on it.
4. **Names/Bios cleanliness:** name strings contain artifacts (leading quotes, `#170`, `(Diana`, `*Dr.`) — a text pipeline would need cleaning; the numeric audit does not use them.
5. **Reproducibility artifact:** `gt_verify_run.txt` in this folder is the fresh output of `ground_truth_verify.py` (run 2026-08-11, `PYTHONIOENCODING=utf-8`); on Windows the script fails without that env var when output is redirected (cp1252 console).
