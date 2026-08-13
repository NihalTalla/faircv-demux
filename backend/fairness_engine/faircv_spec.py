"""Frozen FairCV demo specification — the exact configuration the frozen
audit uses (``faircv_audit_v2.py`` L90-109, L395-425, and the label loading
at L330-345).

This file is CONFIGURATION, not math. The math lives in ``audit_core.py``.
Nothing here may be changed without invalidating the byte-identical
equivalence test against ``results/metrics.csv``,
``results/statistical_tests.csv`` and ``results/per_group_metrics.csv``.
"""

# ── Feature columns (verified in dataset_ground_truth.md §2) ────────────────
CV7_COLS = list(range(4, 11))        # 7 competencies (education … 3 languages)
CV9_COLS = list(range(2, 11))        # + occupation, suitability
FACE_COLS = list(range(11, 31))      # face embedding (20-d, norm 1)
BLIND_FACE_COLS = list(range(31, 51))  # degenerate in this file — never used

FEATURE_SLICES = {
    "cv7": CV7_COLS,
    "cv9": CV9_COLS,
    "cv7+face": CV7_COLS + FACE_COLS,
    "cv9+face": CV9_COLS + FACE_COLS,
}

# ── Protected attributes (column indices in Profiles) ───────────────────────
GENDER_COL = 1
ETHNICITY_COL = 0
GENDER_LABELS = {0: "Male", 1: "Female"}
ETHNICITY_LABELS = {0: "G1", 1: "G2", 2: "G3"}   # R-8: README placeholder names

ATTRIBUTES = [
    ("gender", GENDER_COL, GENDER_LABELS),
    ("ethnicity", ETHNICITY_COL, ETHNICITY_LABELS),
]

# ── Label sets (dataset dict keys) ──────────────────────────────────────────
LABEL_KEYS = {
    "blind":       ("Blind Labels Train", "Blind Labels Test"),
    "gender-bias": ("Biased Labels Train (Gender)", "Biased Labels Test (Gender)"),
    "eth-bias":    ("Biased Labels Train (Ethnicity)", "Biased Labels Test (Ethnicity)"),
}

# ── Model configurations (faircv_audit_v2.py L414-421) ──────────────────────
# (name, short_name, feature-slice key, label-set key)
MODELS = [
    ("M1-Fair (CV7)",            "M1-Fair",      "cv7",      "blind"),
    ("M2-Multimodal (CV7+Face)", "M2-Multimodal", "cv7+face", "blind"),
    ("M3-Gender-Biased (CV7)",   "M3-Gender-Bias", "cv7",     "gender-bias"),
    ("M4-Ethnicity-Biased (CV7)", "M4-Eth-Bias",  "cv7",      "eth-bias"),
    ("M5-Robust (CV9)",          "M5-CV9",       "cv9",      "blind"),
    ("M6-Robust (CV9+Face)",     "M6-CV9+Face",  "cv9+face", "blind"),
]

DATASET_KEYS = [
    "Profiles Train", "Profiles Test",
    "Blind Labels Train", "Blind Labels Test",
    "Biased Labels Train (Gender)", "Biased Labels Test (Gender)",
    "Biased Labels Train (Ethnicity)", "Biased Labels Test (Ethnicity)",
]
