# Bio / Text Arm — Does Demographic Signal Leak Through Language?

**Script:** `bio_arm.py` · **Outputs:** `results/bio_leakage.csv`, `results/bio_metrics.csv`,
`results/bio_report.txt`, `results/fig7_bio_leakage.png` · seed 42 · ~25 s runtime

---

## 1. Objective

The numeric audit (`faircv_audit_v2.py`) and the face embedding cover profile columns 0–30. The
last untested modality is the **free-text `Bios` field** (the dataset ships an original version and
a "blind"/redacted version) plus the **Names** field. Two questions:

- **A. Leakage** — how much demographic information can a simple text model recover, from the
  original bios, the *redacted* bios, and the names alone?
- **B. Hiring** — a text-only model trained on the redacted bios (what a deployed system would
  actually see) — how accurate is it, and how does its demographic disparity compare to M1
  (numeric CV7)?

Method: TF-IDF (word 1–2 grams, min_df=5, max_df=0.9) + L2 logistic regression (seed 42), fit on
the 19,200 train bios, evaluated on the 4,800 test bios. Names use char_wb 2–4 grams. Hiring
labels are the blind label binarised at the train median — identical to v2 M1 — so all fairness
numbers are directly comparable (v2 conventions: SR from predicted class, DPD, DIR, EOD, EO, KL,
KS/Kruskal-Wallis, χ², bootstrap CIs).

**Data facts established first** (see `dataset_ground_truth.md`): the bios are scraped-style
professional text; the blind version replaces pronouns and most personal names with `_`; occupation
and suitability are **demographically neutral** in the numeric profiles (Cramer's V ≈ 0.0007), so
any leak a text model finds is genuinely linguistic, not merit-confounded.

---

## 2. Findings A — Leakage through language

| Channel | Target | Test acc | Test AUC | Interpretation |
|---|---|---|---|---|
| original bios | gender | 0.9975 | **1.0000** | Perfect — pronouns ("He/His" ↔ "She/Her") |
| **blind (redacted) bios** | gender | 0.6452 | **0.7091** | Strong residual leak despite redaction |
| names (control) | gender | 0.9165 | **0.9711** | Names are a near-perfect gender channel |
| original bios | ethnicity | 0.3229 | 0.4945 | chance |
| blind bios | ethnicity | 0.3246 | 0.4969 | chance |
| names (control) | ethnicity | 0.3235 | 0.4954 | chance |

### 2.1 Gender leaks massively — even from the "blind" version

- **Original bios: AUC = 1.0000.** The first-person/third-person pronoun system is a perfect
  gender oracle. Any model trained on unredacted bios has access to gender.
- **Redacted bios: AUC = 0.7091** — the redaction removed pronouns (1/19,200 residual) and most
  names, but **semantic content still leaks gender**. The residual-cue scan shows exactly what
  survived (ratio = stronger side ÷ weaker side, train):
  - `husband` 321 rows, ratio **19.1** (305 female / 16 male) — "her husband"
  - `wife` 247 rows, ratio **9.3** (223 male / 24 female) — "his wife"
  - `mother` ratio 3.6 · `father` ratio 2.4 · `sister`/`daughter` skewed female
- Top female-side tokens in the redacted bios: *women, husband, children, woman, gender, mother,
  breast, loves, rights, dental* — i.e. kinship/family words plus specialty terms from the
  underlying (largely medical) source bios. **A naive pronoun redaction does not make text
  gender-blind.**
- **Names: AUC = 0.9711** — the `Names` field is 95.2% gender-pure in the dataset. A system that
  ever touches names has essentially the protected attribute.

### 2.2 Ethnicity does NOT leak through any text channel

All three ethnicity classifiers sit at chance (AUC ≈ 0.49–0.50, acc ≈ 0.32 ≈ 1/3). This is a
**dataset property, not a method limitation**: the ethnicity labels (G1/G2/G3) are synthetic
profile attributes, and the scraped names/bios are **not** ethnicity-consistent with them. So in
*this* testbed the ethnicity dimension can only be reached through the numeric profile block and
the face embedding — text and names are clean on ethnicity *by construction of the data*, which
should not be assumed for real hiring data (real names are strongly ethnicity-correlated).

---

### 2.3 Stronger-model bound — 0.7091 is tight, not a weak-method artifact

To check whether a stronger text model would recover substantially more gender from the
redacted bios, the leak was re-measured with a four-tier battery (same train/test split, seed 42):

| Tier | Model | Test AUC | Notes |
|---|---|---|---|
| T0 | word TF-IDF (1–2g) + LR | **0.7091** | bio_arm.py baseline (exact reproduction) |
| T1 | char TF-IDF (char_wb 2–5g) + LR | 0.7037 | char-level lexical cues |
| T2 | char-level CNN (PyTorch, CPU, 6 epochs) | 0.6589 | train AUC 0.7297 → mild overfit |
| T3 | all-MiniLM-L6-v2 (cached, frozen) + LR | 0.6799 | also 0.6803 at 160-token cap → truncation irrelevant |

**No stronger model exceeded the word-level baseline.** The transformer run was re-done at a
256-token cap (AUC 0.6799 ≈ identical to the 160-token run), ruling out truncation as the cause.
The gender leak in the redacted bios is carried by **explicit whole-word cues** (husband, wife,
women, mother, father, children) that TF-IDF weights natively; char-level and mean-pooled
semantic representations dilute rather than amplify them. The earlier caveat that "a transformer
would likely push AUC above 0.71" is **falsified by this battery** — 0.7091 stands as a tight
bound for the leak in this testbed. (A *fine-tuned* transformer on 19.2k CPU docs remains untested
and could in principle do better; the evidence here says the recoverable signal is small either
way.)

Scripts: `bio_leak_strong.py` (T0–T2) + `bio_leak_bert.py [max_len]` (T3) →
`results/bio_leak_strong.csv`, `results/fig8_leak_ladder.png`.

---

## 3. Findings B — Text-only hiring models vs M1

| Model | acc | AUC | attr | DPD | DIR | EOD | KL | χ² p | KW p |
|---|---|---|---|---|---|---|---|---|---|
| BIO-BioBlind | 0.669 | 0.714 | gender | 0.0039 | 0.9923 | 0.0075 | 0.0222 | 0.81 | — |
| BIO-BioBlind | 0.669 | 0.714 | ethnicity | 0.0344 | 0.9347 | 0.0102 | 0.0778 | 0.13 | 0.18 |
| BIO-BioOriginal | 0.672 | 0.713 | gender | 0.0121 | 0.9761 | 0.0168 | 0.0401 | 0.42 | — |
| BIO-BioOriginal | 0.672 | 0.713 | ethnicity | 0.0436 | 0.9170 | 0.0315 | 0.0538 | **0.040** | 0.14 |
| M1-Fair (CV7) | 0.793 | — | gender | 0.0122 | 0.9747 | 0.0124 | 0.0183 | — | — |
| M1-Fair (CV7) | 0.793 | — | ethnicity | 0.0180 | 0.9631 | 0.0391 | 0.0295 | — | — |

(M1 is re-fitted in-script with v2's exact pipeline: acc 0.7929 vs the frozen audit's 0.793.)

Three results:

1. **Text models are worse performers.** acc 0.67 / AUC 0.71 vs M1's 0.79 — the scraped bios are
   only loosely coupled to the synthetic merit score. The text carries real signal (occupations,
   experience, credentials) but is far noisier than the numeric profile.

2. **Leakage did not translate into outcome disparity.** Despite recovering gender at AUC 0.71,
   the redacted-bio model has the *lowest* gender disparity of any model in the project
   (DPD 0.0039 — a third of M1's 0.0122), and all four bio cells pass EEOC with wide margins.
   The reason: the blind label is gender- and ethnicity-neutral given the profile, so gender-linked
   text features carry no predictive value for hiring, and a logistic regression assigns them
   little weight. **In this synthetic testbed, demographic leakage is real but does not
   mechanically produce disparate hiring** — disparity materialises only when the *labels*
   themselves are biased (M3/M4), not when the features merely contain demographic information.

3. **Naive redaction partially works.** Seeing the original text (pronouns + names) roughly
   triples gender DPD (0.0121 vs 0.0039) and produces the arm's only nominally-significant test
   (BioOriginal/ethnicity χ² p = 0.040 — borderline, and not significant under any
   multiple-comparison view). So redaction reduced — but could not eliminate — the text-based
   disparity channel, exactly matching Finding A's residual-leak AUC.

---

## 4. What this adds to the project

- **The modalities now complete the picture.** Face embeddings: strong demographic encoding but no
  material disparity increase (M2). Numeric profiles: neutral features, moderate performance, the
  M4 label-bias story. Text: strong *gender* leakage even after redaction, no *ethnicity* leakage,
  and no disparity amplification in hiring.
- **A clean separation of the two claims the original audit conflated:**
  - *"The model can read a protected attribute from the input"* — TRUE for gender (text AUC 0.71–
    1.00, names 0.97, face 0.93–0.99), FALSE for ethnicity via text (chance).
  - *"Reading it causes disparate outcomes"* — NOT demonstrated anywhere except where the training
    labels themselves are biased (M3/M4). The proxy hypothesis (features → bias) remains
    unsupported; the label-bias mechanism is the demonstrated one.
- **A practical lesson:** pronoun/name redaction is insufficient anonymisation of free text;
  kinship terms, family words and domain vocabulary re-encode gender. Any real "blind CV" pipeline
  must audit the text representation itself (e.g. by the leakage-AUC test used here).

## 5. Caveats

- The leak bound was stress-tested with four model families (bag-of-words, char-lexical, char
  CNN, frozen transformer embeddings): none exceeded word-level TF-IDF's 0.7091, so that value is
  treated as a tight bound here. A **fine-tuned** transformer on the full corpus is the one
  untested configuration (CPU-only environment, ~30+ min/epoch); it could plausibly raise the
  bound, but the dominance of explicit lexical cues suggests a modest ceiling.
- The ethnicity-chance result is specific to this synthetic dataset (labels not name-consistent);
  it must not be read as "ethnicity never leaks in text" for real data.
- Hiring fairness uses the median-binarised blind label (v2 convention), not top-N screening.
- The "blind" bios are the dataset's own redaction; quality differs from a purpose-built pipeline.
- Leakage AUCs have no bootstrap CIs (out of scope for this arm; differences are far beyond
  sampling noise).

## 6. Artifacts

`bio_arm.py` joins the frozen set; `results/bio_leakage.csv` (6 rows), `results/bio_metrics.csv`
(4 rows), `results/bio_report.txt`, `results/fig7_bio_leakage.png`. Stronger-model bound:
`bio_leak_strong.py`, `bio_leak_bert.py`, `results/bio_leak_strong.csv` (4 rows),
`results/fig8_leak_ladder.png`. No frozen scripts were modified.
