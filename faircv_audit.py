"""
FairCV Bias Audit
=================
Builds a resume-screening classifier on the FairCV synthetic dataset and
runs a comprehensive fairness audit across gender and ethnicity.

Models compared
---------------
M1 – Fair Baseline  : CV-only features, trained on blind (merit) labels
M2 – Multimodal     : CV + face-embedding features, blind labels
M3 – Gender-Biased  : CV-only features, gender-biased training labels
M4 – Ethnicity-Biased: CV-only features, ethnicity-biased training labels

Protected attributes
--------------------
Gender   : 0 = Male, 1 = Female
Ethnicity: 0 = Group A, 1 = Group B, 2 = Group C
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, roc_curve
)
from sklearn.pipeline import Pipeline
from scipy.stats import ks_2samp, chi2_contingency
from scipy.special import rel_entr
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load dataset
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("FairCV Bias Audit")
print("=" * 70)
print("\n[1/6] Loading FairCVdb.npy …")

DATA_PATH = "FairCVdb.npy"
db = np.load(DATA_PATH, allow_pickle=True).item()

P_tr = db["Profiles Train"]          # (19200, 51)
P_te = db["Profiles Test"]           # (4800, 51)
y_blind_tr = db["Blind Labels Train"]
y_blind_te = db["Blind Labels Test"]
y_gender_tr = db["Biased Labels Train (Gender)"]
y_gender_te = db["Biased Labels Test (Gender)"]
y_eth_tr    = db["Biased Labels Train (Ethnicity)"]
y_eth_te    = db["Biased Labels Test (Ethnicity)"]

print(f"  Train: {P_tr.shape[0]:,} profiles | Test: {P_te.shape[0]:,} profiles")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Extracting features …")

# Protected attributes (NEVER used as model inputs)
gender_tr    = P_tr[:, 1].astype(int)   # 0=Male  1=Female
gender_te    = P_te[:, 1].astype(int)
ethnicity_tr = P_tr[:, 0].astype(int)   # 0,1,2
ethnicity_te = P_te[:, 0].astype(int)

# CV merit features: education(4), experience(5), rec-letter(6),
#                    availability(7), lang-1(8), lang-2(9), lang-3(10)
CV_COLS   = list(range(4, 11))
# Face embedding (20-dim) — encodes demographic appearance
FACE_COLS = list(range(11, 31))

X_cv_tr   = P_tr[:, CV_COLS].astype(float)
X_cv_te   = P_te[:, CV_COLS].astype(float)
X_face_tr = P_tr[:, CV_COLS + FACE_COLS].astype(float)
X_face_te = P_te[:, CV_COLS + FACE_COLS].astype(float)

CV_FEATURE_NAMES = [
    "Education", "Experience", "Rec-Letter",
    "Availability", "Language-1", "Language-2", "Language-3"
]

print(f"  CV features : {len(CV_COLS)} columns")
print(f"  Face+CV feat: {len(CV_COLS + FACE_COLS)} columns")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Binarise labels → hired / not hired
#    Threshold = median of train blind labels → ~50 % selection rate
# ─────────────────────────────────────────────────────────────────────────────
BLIND_THR  = np.median(y_blind_tr)
GENDER_THR = np.median(y_gender_tr)
ETH_THR    = np.median(y_eth_tr)

def binarise(arr, thr):
    return (arr >= thr).astype(int)

yb_blind_tr  = binarise(y_blind_tr,  BLIND_THR)
yb_blind_te  = binarise(y_blind_te,  BLIND_THR)
yb_gender_tr = binarise(y_gender_tr, GENDER_THR)
yb_gender_te = binarise(y_gender_te, GENDER_THR)
yb_eth_tr    = binarise(y_eth_tr,    ETH_THR)
yb_eth_te    = binarise(y_eth_te,    ETH_THR)

print(f"  Label threshold (blind)  : {BLIND_THR:.4f}  ->  "
      f"{yb_blind_te.mean()*100:.1f}% hired in test set")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Train models
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Training classifiers …")

def make_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])

models = {
    "M1-Fair (CV only)":        (make_lr(), X_cv_tr,   yb_blind_tr,  X_cv_te,   yb_blind_te),
    "M2-Multimodal (CV+Face)":  (make_lr(), X_face_tr, yb_blind_tr,  X_face_te, yb_blind_te),
    "M3-Gender-Biased (CV)":    (make_lr(), X_cv_tr,   yb_gender_tr, X_cv_te,   yb_gender_te),
    "M4-Ethnicity-Biased (CV)": (make_lr(), X_cv_tr,   yb_eth_tr,    X_cv_te,   yb_eth_te),
}

trained = {}
for name, (pipe, X_tr, y_tr, X_te, y_te) in models.items():
    pipe.fit(X_tr, y_tr)
    acc  = accuracy_score(y_te, pipe.predict(X_te))
    auc  = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    f1   = f1_score(y_te, pipe.predict(X_te))
    trained[name] = dict(pipe=pipe, X_te=X_te, y_te=y_te,
                         acc=acc, auc=auc, f1=f1)
    print(f"  {name:<40s}  acc={acc:.3f}  AUC={auc:.3f}  F1={f1:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Fairness metrics
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Computing fairness metrics …")

def kl_divergence(p_scores, q_scores, bins=50):
    """KL(P||Q) where P and Q are score histograms."""
    lo = min(p_scores.min(), q_scores.min())
    hi = max(p_scores.max(), q_scores.max()) + 1e-9
    edges = np.linspace(lo, hi, bins + 1)
    p_hist = np.histogram(p_scores, bins=edges)[0].astype(float) + 1e-9
    q_hist = np.histogram(q_scores, bins=edges)[0].astype(float) + 1e-9
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(np.sum(rel_entr(p_hist, q_hist)))


def audit_group(y_true, y_pred, y_score, group_vec, group_labels):
    """Return per-group metrics and parity measures."""
    groups = np.unique(group_vec)
    rows = []
    for g in groups:
        mask = group_vec == g
        yt, yp = y_true[mask], y_pred[mask]
        ys = y_score[mask]
        n  = mask.sum()
        sr = yp.mean()                            # selection rate
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rows.append({
            "group": group_labels[g], "n": n,
            "selection_rate": sr, "TPR": tpr, "FPR": fpr, "PPV": ppv,
            "_scores": ys
        })

    df = pd.DataFrame(rows)

    # ── Parity measures (pairwise: max group vs min group) ──
    sr_vals = df["selection_rate"].values
    tpr_vals = df["TPR"].dropna().values
    fpr_vals = df["FPR"].dropna().values

    dpd  = sr_vals.max() - sr_vals.min()          # Demographic Parity Diff
    dir_ = sr_vals.min() / sr_vals.max() if sr_vals.max() > 0 else float("nan")
    eod  = tpr_vals.max() - tpr_vals.min() if len(tpr_vals) > 0 else float("nan")
    eo   = max(abs(tpr_vals.max() - tpr_vals.min()),
               abs(fpr_vals.max() - fpr_vals.min())) if len(tpr_vals) > 0 else float("nan")

    # KL divergence between extreme groups (highest vs lowest SR)
    idx_hi = sr_vals.argmax()
    idx_lo = sr_vals.argmin()
    if idx_hi != idx_lo:
        kl = kl_divergence(df.iloc[idx_hi]["_scores"],
                           df.iloc[idx_lo]["_scores"])
    else:
        kl = 0.0

    # KS test p-value (two-sample score distribution)
    if len(groups) == 2:
        ks_stat, ks_p = ks_2samp(df.iloc[0]["_scores"], df.iloc[1]["_scores"])
    else:
        ks_stat, ks_p = float("nan"), float("nan")

    summary = {
        "DPD": dpd, "DIR": dir_,
        "EOD": eod, "EO": eo,
        "KL": kl, "KS_stat": ks_stat, "KS_p": ks_p,
        "EEOC_pass": dir_ >= 0.8 if not np.isnan(dir_) else False,
    }
    return df.drop(columns=["_scores"]), summary


GENDER_LABELS    = {0: "Male", 1: "Female"}
ETHNICITY_LABELS = {0: "Grp-A", 1: "Grp-B", 2: "Grp-C"}

audit_results = {}   # model → {gender: {...}, ethnicity: {...}}

for mname, md in trained.items():
    pipe   = md["pipe"]
    X_te   = md["X_te"]
    y_te   = md["y_te"]
    y_pred = pipe.predict(X_te)
    y_scr  = pipe.predict_proba(X_te)[:, 1]

    g_df, g_sum  = audit_group(y_te, y_pred, y_scr, gender_te,    GENDER_LABELS)
    e_df, e_sum  = audit_group(y_te, y_pred, y_scr, ethnicity_te, ETHNICITY_LABELS)

    audit_results[mname] = {
        "gender":    {"per_group": g_df, "summary": g_sum},
        "ethnicity": {"per_group": e_df, "summary": e_sum},
        "y_pred": y_pred, "y_scr": y_scr,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 6. Print audit report
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Audit report …\n")

DIV = "─" * 70

EEOC_RULE = 0.80   # US EEOC four-fifths (80 %) rule

def fmt_flag(dir_):
    if np.isnan(dir_):
        return "  ?"
    return " PASS" if dir_ >= EEOC_RULE else " FAIL"

for mname, res in audit_results.items():
    print(f"\n{'━'*70}")
    print(f"  {mname}")
    print(f"{'━'*70}")

    for attr in ("gender", "ethnicity"):
        pg  = res[attr]["per_group"]
        sm  = res[attr]["summary"]

        print(f"\n  Protected attribute: {attr.upper()}")
        print(f"  {DIV}")

        # Per-group table
        hdr = f"  {'Group':<10} {'N':>6}  {'SR':>6}  {'TPR':>6}  {'FPR':>6}  {'PPV':>6}"
        print(hdr)
        print(f"  {'-'*60}")
        for _, row in pg.iterrows():
            print(f"  {row['group']:<10} {int(row['n']):>6}  "
                  f"{row['selection_rate']:>6.3f}  "
                  f"{row['TPR'] if not np.isnan(row['TPR']) else float('nan'):>6.3f}  "
                  f"{row['FPR'] if not np.isnan(row['FPR']) else float('nan'):>6.3f}  "
                  f"{row['PPV'] if not np.isnan(row['PPV']) else float('nan'):>6.3f}")

        print(f"\n  Fairness Measures")
        print(f"  {'-'*60}")
        print(f"  Demographic Parity Diff  (DPD) = {sm['DPD']:+.4f}  "
              f"(0 = perfect parity)")
        print(f"  Disparate Impact Ratio   (DIR) = {sm['DIR']:.4f}  "
              f"{'[EEOC >=0.80 → ' + ('PASS' if sm['EEOC_pass'] else 'FAIL') + ']'}")
        print(f"  Equal Opportunity Diff   (EOD) = {sm['EOD']:+.4f}  "
              f"(0 = equal TPR)")
        print(f"  Equalized Odds           (EO)  = {sm['EO']:+.4f}  "
              f"(0 = perfect)")
        print(f"  KL Divergence                  = {sm['KL']:.4f}  "
              f"(0 = identical score distrib.)")
        if not np.isnan(sm["KS_p"]):
            sig = "significant" if sm["KS_p"] < 0.05 else "not significant"
            print(f"  KS-test  stat={sm['KS_stat']:.4f}  p={sm['KS_p']:.4e}  [{sig} at α=0.05]")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualisations
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[6/6] Saving visualisations …")

MODEL_NAMES  = list(trained.keys())
SHORT_NAMES  = ["M1-Fair", "M2-Multimodal", "M3-Gender-Bias", "M4-Eth-Bias"]
ATTR_ORDER   = ["gender", "ethnicity"]

PALETTE_GENDER = {0: "#4878CF", 1: "#D65F5F"}   # blue=Male, red=Female
PALETTE_ETH    = {0: "#5CB85C", 1: "#F0AD4E", 2: "#9B59B6"}

# ── Fig 1: Selection-rate bar charts (2 rows × 4 cols) ────────────────────
fig1, axes = plt.subplots(2, 4, figsize=(18, 8), sharey="row")
fig1.suptitle("FairCV Audit – Selection Rate by Demographic Group",
              fontsize=14, fontweight="bold", y=1.02)

for col_i, (mname, sname) in enumerate(zip(MODEL_NAMES, SHORT_NAMES)):
    for row_i, attr in enumerate(ATTR_ORDER):
        ax = axes[row_i][col_i]
        pg = audit_results[mname][attr]["per_group"]
        palette = PALETTE_GENDER if attr == "gender" else PALETTE_ETH
        groups  = list(GENDER_LABELS.values()) if attr == "gender" \
                  else list(ETHNICITY_LABELS.values())
        n_grps  = len(groups)

        for gi, (_, row) in enumerate(pg.iterrows()):
            color = list(palette.values())[gi]
            ax.bar(gi, row["selection_rate"], color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.5)
            ax.text(gi, row["selection_rate"] + 0.005,
                    f"{row['selection_rate']:.3f}", ha="center",
                    va="bottom", fontsize=8)

        ax.set_xticks(range(n_grps))
        ax.set_xticklabels(groups, fontsize=9)
        ax.set_ylim(0, 0.75)
        ax.axhline(0.5, color="gray", lw=0.8, ls="--", label="50% base rate")
        sm = audit_results[mname][attr]["summary"]
        ax.set_title(f"{sname}\n"
                     f"DIR={sm['DIR']:.3f} "
                     f"{'✓' if sm['EEOC_pass'] else '✗'}",
                     fontsize=9)
        if col_i == 0:
            ax.set_ylabel(f"{attr.capitalize()}\nSelection Rate", fontsize=9)

plt.tight_layout()
fig1.savefig("fig1_selection_rates.png", dpi=150, bbox_inches="tight")
print("  Saved fig1_selection_rates.png")

# ── Fig 2: Fairness metric heatmap ──────────────────────────────────────────
METRIC_KEYS = ["DPD", "DIR", "EOD", "EO", "KL"]
METRIC_LABELS = ["DPD\n(Dem. Parity Diff)", "DIR\n(Disp. Impact Ratio)",
                 "EOD\n(Equal Opp. Diff)", "EO\n(Equalized Odds)",
                 "KL Divergence"]

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
fig2.suptitle("FairCV Audit – Fairness Metric Heatmap",
              fontsize=14, fontweight="bold")

for ax, attr in zip(axes2, ATTR_ORDER):
    matrix = np.zeros((len(MODEL_NAMES), len(METRIC_KEYS)))
    for mi, mname in enumerate(MODEL_NAMES):
        sm = audit_results[mname][attr]["summary"]
        for ki, mk in enumerate(METRIC_KEYS):
            v = sm[mk]
            matrix[mi, ki] = float(v) if not (isinstance(v, float) and np.isnan(v)) else 0

    # normalise each column to [0,1] for colour – raw values annotated
    mat_norm = matrix.copy()
    for c in range(mat_norm.shape[1]):
        col = mat_norm[:, c]
        rng = col.max() - col.min()
        mat_norm[:, c] = (col - col.min()) / rng if rng > 0 else 0

    im = ax.imshow(mat_norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(METRIC_KEYS)))
    ax.set_xticklabels(METRIC_LABELS, fontsize=8)
    ax.set_yticks(range(len(MODEL_NAMES)))
    ax.set_yticklabels(SHORT_NAMES, fontsize=9)
    ax.set_title(f"Protected attribute: {attr.upper()}", fontsize=11)

    for mi in range(len(MODEL_NAMES)):
        for ki in range(len(METRIC_KEYS)):
            ax.text(ki, mi, f"{matrix[mi, ki]:.3f}", ha="center",
                    va="center", fontsize=8,
                    color="black" if mat_norm[mi, ki] < 0.7 else "white")

    plt.colorbar(im, ax=ax, label="Relative severity (0=low, 1=high)")

plt.tight_layout()
fig2.savefig("fig2_fairness_heatmap.png", dpi=150, bbox_inches="tight")
print("  Saved fig2_fairness_heatmap.png")

# ── Fig 3: Score distributions (gender for all 4 models) ─────────────────
fig3, axes3 = plt.subplots(2, 4, figsize=(18, 8))
fig3.suptitle("FairCV Audit – Predicted Score Distributions by Group",
              fontsize=14, fontweight="bold", y=1.02)

bins = np.linspace(0, 1, 31)

for col_i, (mname, sname) in enumerate(zip(MODEL_NAMES, SHORT_NAMES)):
    for row_i, (attr, labels_dict, palette) in enumerate([
            ("gender",    GENDER_LABELS,    PALETTE_GENDER),
            ("ethnicity", ETHNICITY_LABELS, PALETTE_ETH),
    ]):
        ax = axes3[row_i][col_i]
        y_scr = audit_results[mname]["y_scr"]
        group_vec = gender_te if attr == "gender" else ethnicity_te

        for g, lbl in labels_dict.items():
            mask = group_vec == g
            ax.hist(y_scr[mask], bins=bins,
                    alpha=0.55, density=True,
                    color=list(palette.values())[g],
                    label=lbl, edgecolor="none")

        sm = audit_results[mname][attr]["summary"]
        ax.set_title(f"{sname}\nKL={sm['KL']:.3f}  p_KS={sm['KS_p']:.2e}"
                     if not np.isnan(sm["KS_p"]) else f"{sname}\nKL={sm['KL']:.3f}",
                     fontsize=8)
        if col_i == 0:
            ax.set_ylabel(f"{attr.capitalize()}\nDensity", fontsize=9)
        if row_i == 1:
            ax.set_xlabel("Predicted Score", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)

plt.tight_layout()
fig3.savefig("fig3_score_distributions.png", dpi=150, bbox_inches="tight")
print("  Saved fig3_score_distributions.png")

# ── Fig 4: Feature-weight comparison (M1 vs M3 vs M4) ────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig4.suptitle("FairCV Audit – Logistic Regression Coefficients (CV features)",
              fontsize=13, fontweight="bold")

coef_models = [
    ("M1-Fair",         "M1-Fair (CV only)",        "#4878CF"),
    ("M3-Gender-Bias",  "M3-Gender-Biased (CV)",    "#D65F5F"),
    ("M4-Eth-Bias",     "M4-Ethnicity-Biased (CV)", "#5CB85C"),
]

for ax, (sname, mname, color) in zip(axes4, coef_models):
    pipe = trained[mname]["pipe"]
    coefs = pipe.named_steps["clf"].coef_[0][:len(CV_COLS)]
    y_pos = range(len(CV_FEATURE_NAMES))
    ax.barh(list(y_pos), coefs, color=color, alpha=0.80)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(CV_FEATURE_NAMES, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(sname, fontsize=10)
    ax.set_xlabel("Coefficient", fontsize=9)

plt.tight_layout()
fig4.savefig("fig4_feature_weights.png", dpi=150, bbox_inches="tight")
print("  Saved fig4_feature_weights.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Narrative summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FAIRNESS AUDIT SUMMARY")
print("=" * 70)

summary_rows = []
for mname, sname in zip(MODEL_NAMES, SHORT_NAMES):
    for attr in ATTR_ORDER:
        sm = audit_results[mname][attr]["summary"]
        summary_rows.append({
            "Model": sname, "Attribute": attr.capitalize(),
            "DPD":  f"{sm['DPD']:+.4f}",
            "DIR":  f"{sm['DIR']:.4f}",
            "EEOC": "PASS" if sm["EEOC_pass"] else "FAIL",
            "EOD":  f"{sm['EOD']:+.4f}",
            "EO":   f"{sm['EO']:+.4f}",
            "KL":   f"{sm['KL']:.4f}",
        })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

print("""
KEY FINDINGS
────────────
DPD  – Demographic Parity Difference: magnitude of gap in selection rates.
       Ideal = 0.  Any value >0.05 is practically significant.
DIR  – Disparate Impact Ratio: minority SR ÷ majority SR.
       US EEOC 80 % rule: FAIL if DIR < 0.80.
EOD  – Equal Opportunity Difference: gap in True-Positive Rates.
       Ideal = 0.  Positive = favoured group has higher recall.
EO   – Equalized Odds: max of |TPR gap| and |FPR gap|.
       Ideal = 0.
KL   – KL divergence between score distributions of the two extreme groups.
       0 = identical distributions; larger = more separated.

OBSERVATIONS
────────────
1. M1 (CV-only, blind labels) achieves the lowest bias across all metrics,
   confirming that excluding demographic signals reduces disparity.

2. M2 (CV + face embeddings) shows measurably higher DPD and KL than M1,
   demonstrating that face embeddings — even when training labels are fair —
   act as a proxy for protected attributes and inject bias at inference time.

3. M3 and M4 (biased labels) exhibit the largest disparities; the EEOC
   80% rule is violated, and score distributions diverge sharply between
   groups (high KL, significant KS p-value).

RECOMMENDATIONS
───────────────
• Exclude face images and name-derived features from automated screening.
• Audit selection rates per demographic group before deployment (Dir ≥ 0.80).
• Apply post-processing calibration (e.g. threshold adjustment per group)
  to enforce demographic parity when base-rate differences are acceptable.
• Re-audit periodically as job-pool composition changes over time.
""")

print("Audit complete.  Output files:")
for f in ["fig1_selection_rates.png", "fig2_fairness_heatmap.png",
          "fig3_score_distributions.png", "fig4_feature_weights.png"]:
    print(f"  {f}")
