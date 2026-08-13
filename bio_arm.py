"""
Bio / text arm — does demographic signal leak through language?
================================================================
The final untested modality of FairCVdb: the free-text `Bios` field
(original = col 0, "blind"/redacted = col 1). The numeric audit
(faircv_audit_v2.py) and the face embedding cover cols 0-30 only; the
Bios, Names and Image List keys were declared out of scope there.

Questions
---------
A. Leakage: how well can a simple text model predict gender / ethnicity
   from (i) the ORIGINAL bios, (ii) the BLIND (redacted) bios, and
   (iii, control) the Names field?
B. Hiring: does a text-only model trained on the blind bios (what a
   deployed system would actually see) reproduce the blind hiring label,
   and how does its demographic disparity compare to M1 (CV7 numeric)?

Method
------
- TF-IDF (word 1-2 grams, min_df=5, max_df=0.9, sublinear_tf) + L2
  logistic regression (seed 42), fitted on train bios, evaluated on test.
- Names control: char_wb 2-4 grams on the Names field.
- Hiring labels: blind label binarised at the train median (identical to
  faircv_audit_v2.py M1) so performance/fairness are comparable.
- Fairness audit ports v2 definitions: SR from predicted class, DPD,
  DIR (worst/best SR), EOD, EO, KL (extreme SR pair), pairwise KS with
  Holm correction / Kruskal-Wallis, chi2(group x hired), bootstrap 95%
  CIs (n=1000, fixed extreme pair for KL).

Outputs: results/bio_leakage.csv, results/bio_metrics.csv,
         results/bio_report.txt, results/fig7_bio_leakage.png
"""

import os
import sys
import re
import atexit
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from scipy.stats import ks_2samp, kruskal, chi2_contingency
from scipy.special import rel_entr

warnings.filterwarnings("ignore")

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH   = "FairCVdb.npy"
RNG_SEED    = 42
N_BOOT      = 1000
EEOC_RULE   = 0.80
ALPHA       = 0.05
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

_report_file = open(os.path.join(RESULTS_DIR, "bio_report.txt"), "w", encoding="utf-8")
atexit.register(_report_file.close)
sys.stdout = _Tee(sys.__stdout__, _report_file)

GENDER_LABELS    = {0: "Male", 1: "Female"}
ETHNICITY_LABELS = {0: "G1", 1: "G2", 2: "G3"}

# ── Helpers (ported from faircv_audit_v2.py for identical definitions) ───────
def binarise(arr, thr):
    return (arr >= thr).astype(int)


def kl_divergence(p_scores, q_scores, bins=50, edges=None):
    if edges is None:
        lo = min(p_scores.min(), q_scores.min())
        hi = max(p_scores.max(), q_scores.max()) + 1e-9
        edges = np.linspace(lo, hi, bins + 1)
    p_hist = np.histogram(p_scores, bins=edges)[0].astype(float) + 1e-9
    q_hist = np.histogram(q_scores, bins=edges)[0].astype(float) + 1e-9
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(np.sum(rel_entr(p_hist, q_hist)))


def holm_correct(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        val = min(1.0, p[idx] * (m - rank + 1))
        running = max(running, val)
        adj[idx] = running
    return adj


def per_group_stats(y_true, y_pred, y_score, group_vec, groups):
    ng = len(groups)
    N = np.zeros(ng, dtype=int)
    SR = np.zeros(ng)
    TPR = np.full(ng, np.nan)
    FPR = np.full(ng, np.nan)
    PPV = np.full(ng, np.nan)
    scores = []
    for gi, g in enumerate(groups):
        mask = group_vec == g
        N[gi] = int(mask.sum())
        yt, yp = y_true[mask], y_pred[mask]
        SR[gi] = yp.mean()
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        if tp + fn > 0:
            TPR[gi] = tp / (tp + fn)
        if fp + tn > 0:
            FPR[gi] = fp / (fp + tn)
        if tp + fp > 0:
            PPV[gi] = tp / (tp + fp)
        scores.append(y_score[mask])
    return N, SR, TPR, FPR, PPV, scores


def _agg(SR, TPR, FPR, s_hi, s_lo):
    dpd = SR.max() - SR.min()
    dir_ = (SR.min() / SR.max()) if SR.max() > 0 else np.nan
    tpr_f = TPR[np.isfinite(TPR)]
    fpr_f = FPR[np.isfinite(FPR)]
    eod = (tpr_f.max() - tpr_f.min()) if len(tpr_f) else np.nan
    eo = max(tpr_f.max() - tpr_f.min(), fpr_f.max() - fpr_f.min()) if (len(tpr_f) and len(fpr_f)) else np.nan
    kl = kl_divergence(s_hi, s_lo)
    return dpd, dir_, eod, eo, kl


def bootstrap_cis(y_true, y_pred, y_score, group_vec, groups, hi, lo, rng, n_boot=N_BOOT):
    n = len(y_true)
    ng = len(groups)
    cols = ("DPD", "DIR", "EOD", "EO", "KL")
    out = {k: np.empty(n_boot) for k in cols}
    yt1 = (y_true == 1)
    yp1 = (y_pred == 1)
    edges = None
    s_hi_full = y_score[group_vec == groups[hi]]
    s_lo_full = y_score[group_vec == groups[lo]]
    edges = np.linspace(min(s_hi_full.min(), s_lo_full.min()),
                        max(s_hi_full.max(), s_lo_full.max()) + 1e-9, 51)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        gv = group_vec[idx]
        yt, yp, ys = yt1[idx], yp1[idx], y_score[idx]
        SR = np.zeros(ng)
        TPR = np.full(ng, np.nan)
        FPR = np.full(ng, np.nan)
        scores = []
        for gi, g in enumerate(groups):
            m = gv == g
            sel, t = yp[m], yt[m]
            n_g = int(m.sum())
            scores.append(ys[m])
            if n_g == 0:
                continue
            SR[gi] = sel.mean()
            tp = int((t & sel).sum())
            fn = int((t & ~sel).sum())
            fp = int((~t & sel).sum())
            tn = n_g - tp - fn - fp
            if tp + fn > 0:
                TPR[gi] = tp / (tp + fn)
            if fp + tn > 0:
                FPR[gi] = fp / (fp + tn)
        dpd = SR.max() - SR.min()
        dir_ = (SR.min() / SR.max()) if SR.max() > 0 else np.nan
        tf = TPR[np.isfinite(TPR)]
        ff = FPR[np.isfinite(FPR)]
        eod = tf.max() - tf.min() if len(tf) else np.nan
        eo = max(tf.max() - tf.min(), ff.max() - ff.min()) if (len(tf) and len(ff)) else np.nan
        kl = kl_divergence(scores[hi], scores[lo], edges=edges)
        out["DPD"][b] = dpd
        out["DIR"][b] = dir_
        out["EOD"][b] = eod
        out["EO"][b] = eo
        out["KL"][b] = kl
    cis = {}
    for k in cols:
        a = out[k][np.isfinite(out[k])]
        cis[k] = (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (np.nan, np.nan)
    return cis


def run_audit(y_true, y_pred, y_score, group_vec, groups, group_labels, rng):
    N, SR, TPR, FPR, PPV, scores = per_group_stats(y_true, y_pred, y_score, group_vec, groups)
    hi, lo = int(SR.argmax()), int(SR.argmin())
    dpd, dir_, eod, eo, kl = _agg(SR, TPR, FPR, scores[hi], scores[lo])
    pairs = [(i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))]
    ks_rows = []
    for i, j in pairs:
        st, p = ks_2samp(scores[i], scores[j])
        ks_rows.append({"comparison": f"{group_labels[groups[i]]} vs {group_labels[groups[j]]}",
                        "stat": float(st), "p": float(p)})
    p_adj = holm_correct([r["p"] for r in ks_rows]) if ks_rows else []
    for r, pa in zip(ks_rows, p_adj):
        r["p_adj"] = float(pa)
    kw = None
    if len(groups) >= 3:
        st, p = kruskal(*scores)
        kw = (float(st), float(p))
    sel = y_pred.astype(int)
    cont = np.array([[int(((group_vec == g) & (sel == 1)).sum()),
                      int(((group_vec == g) & (sel == 0)).sum())] for g in groups])
    chi2_stat, chi2_p, _, _ = chi2_contingency(cont)
    cis = bootstrap_cis(y_true, y_pred, y_score, group_vec, groups, hi, lo, rng)
    return {
        "groups": groups, "labels": group_labels,
        "N": N, "SR": SR, "TPR": TPR, "FPR": FPR, "PPV": PPV, "scores": scores,
        "idx_hi": hi, "idx_lo": lo,
        "DPD": dpd, "DIR": dir_, "EOD": eod, "EO": eo, "KL": kl,
        "ks_rows": ks_rows, "kw": kw, "chi2_p": float(chi2_p), "cis": cis,
    }


def clean_text(txt):
    """Underscore is the redaction marker; drop it (token-level, as a text
    classifier would). Also drop stray "'s" fragments left by redaction
    (e.g. "Lu's" -> "_'s" -> " 's") while keeping legitimate possessives."""
    t = txt.replace("_", " ")
    t = re.sub(r"\s+'s\b", "", t)
    return re.sub(r"\s+", " ", t).strip()


# ── 1. Load ──────────────────────────────────────────────────────────────────
print("=" * 74)
print("Bio / text arm — does demographic signal leak through language?")
print("=" * 74)

db = np.load(DATA_PATH, allow_pickle=True).item()
P_tr, P_te = db["Profiles Train"], db["Profiles Test"]
B_tr, B_te = db["Bios Train"], db["Bios Test"]
N_tr, N_te = db["Names Train"], db["Names Test"]
y_blind_tr, y_blind_te = db["Blind Labels Train"], db["Blind Labels Test"]

g_tr, g_te = P_tr[:, 1].astype(int), P_te[:, 1].astype(int)
e_tr, e_te = P_tr[:, 0].astype(int), P_te[:, 0].astype(int)

BLIND_THR = float(np.median(y_blind_tr))
yb_tr = binarise(y_blind_tr, BLIND_THR)
yb_te = binarise(y_blind_te, BLIND_THR)
print(f"  bios train {B_tr.shape} | test {B_te.shape} | blind-label threshold {BLIND_THR:.4f}")

bio_orig_tr = [clean_text(t) for t in B_tr[:, 0]]
bio_orig_te = [clean_text(t) for t in B_te[:, 0]]
bio_blind_tr = [clean_text(t) for t in B_tr[:, 1]]
bio_blind_te = [clean_text(t) for t in B_te[:, 1]]
names_tr = [str(t).strip() for t in N_tr]
names_te = [str(t).strip() for t in N_te]

# ── 2. Residual-cue scan (what survives redaction?) ──────────────────────────
print("\n[2/7] Residual gender cues in the BLIND bios (word-boundary counts):")
CUES = {"wife": None, "husband": None, "son": None, "daughter": None,
        "brother": None, "sister": None, "mother": None, "father": None,
        "ms": None, "mr": None, "mrs": None}
for cue in CUES:
    pat = re.compile(rf"\b{re.escape(cue)}s?\b")
    hits = np.array([bool(pat.search(t.lower())) for t in bio_blind_tr])
    CUES[cue] = (int(hits.sum()),
                 int((hits & (g_tr == 0)).sum()),
                 int((hits & (g_tr == 1)).sum()))
    print(f"  {cue:<9s} present in {CUES[cue][0]:>5d} rows  "
          f"(gender0={CUES[cue][1]:>5d}, gender1={CUES[cue][2]:>5d})")
asym = sorted(((c, n, max(g0, g1) / max(1, min(g0, g1)))
                for c, (n, g0, g1) in CUES.items() if n > 50), key=lambda t: -t[2])
print("  -> most asymmetric residual cues (n > 50; ratio = stronger side / weaker side):")
for c, n, ratio in asym[:5]:
    print(f"    {c:<9s} n={n:>5d} ratio={ratio:>6.1f}")

# ── 3. Leakage classifiers ───────────────────────────────────────────────────
print("\n[3/7] Demographic leakage classifiers (TF-IDF + logistic regression):")


def make_text_clf(analyzer="word", ngram=(1, 2), min_df=5):
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=min_df,
                          max_df=0.9, sublinear_tf=True, lowercase=True)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=RNG_SEED)
    return vec, clf


def eval_clf(Xtr, ytr, Xte, yte, n_classes, analyzer="word", ngram=(1, 2), min_df=5):
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=min_df,
                          max_df=0.9, sublinear_tf=True, lowercase=True)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=RNG_SEED)
    Xtr_v = vec.fit_transform(Xtr)
    Xte_v = vec.transform(Xte)
    clf.fit(Xtr_v, ytr)
    proba = clf.predict_proba(Xte_v)
    pred = clf.predict(Xte_v)
    acc = accuracy_score(yte, pred)
    if n_classes == 2:
        auc = roc_auc_score(yte, proba[:, 1])
    else:
        auc = roc_auc_score(yte, proba, multi_class="ovr", average="macro")
    return dict(acc=acc, auc=auc, pred=pred, proba=proba, vec=vec, clf=clf)


leakage_rows = []
results_leak = {}


def add_leakage(name, Xtr, ytr, Xte, yte, n_classes, **kwargs):
    r = eval_clf(Xtr, ytr, Xte, yte, n_classes, **kwargs)
    results_leak[name] = r
    leakage_rows.append({"channel": name, "n_classes": n_classes,
                         "accuracy": r["acc"], "auc": r["auc"]})
    print(f"  {name:<34s} n_class={n_classes}  acc={r['acc']:.4f}  AUC={r['auc']:.4f}")


add_leakage("gender | original bios", bio_orig_tr, g_tr, bio_orig_te, g_te, 2)
add_leakage("gender | blind bios", bio_blind_tr, g_tr, bio_blind_te, g_te, 2)
add_leakage("gender | names (control)", names_tr, g_tr, names_te, g_te, 2,
           analyzer="char_wb", ngram=(2, 4), min_df=2)
add_leakage("ethnicity | original bios", bio_orig_tr, e_tr, bio_orig_te, e_te, 3)
add_leakage("ethnicity | blind bios", bio_blind_tr, e_tr, bio_blind_te, e_te, 3)
add_leakage("ethnicity | names (control)", names_tr, e_tr, names_te, e_te, 3,
           analyzer="char_wb", ngram=(2, 4), min_df=2)

print("\n  Top gender-discriminating tokens in the BLIND bios (female-side coefs):")
vec = results_leak["gender | blind bios"]["vec"]
coef = results_leak["gender | blind bios"]["clf"].coef_[0]
feats = np.array(vec.get_feature_names_out())
idx = np.argsort(-coef)[:12]
for i in idx:
    print(f"    {feats[i]:<16s} coef={coef[i]:+.3f}")

# ── 4. Hiring models on bios ─────────────────────────────────────────────────
print("\n[4/7] Hiring models (blind label, median threshold):")


# M1 reference: re-fit the frozen audit's exact pipeline (CV7, blind labels,
# seed 42) so the comparison accuracy is derived, not hardcoded.
CV7_COLS = list(range(4, 11))
X_cv7_tr = P_tr[:, CV7_COLS].astype(float)
X_cv7_te = P_te[:, CV7_COLS].astype(float)
m1_pipe = Pipeline([("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                                random_state=RNG_SEED))])
m1_pipe.fit(X_cv7_tr, yb_tr)
M1_ACC = accuracy_score(yb_te, m1_pipe.predict(X_cv7_te))
print(f"  M1 (CV7 numeric, re-fitted) acc={M1_ACC:.4f}  (frozen audit: 0.793)")


def bio_hiring(Xtr, Xte, tag):
    vec, clf = make_text_clf()
    Xtr_v = vec.fit_transform(Xtr)
    Xte_v = vec.transform(Xte)
    clf.fit(Xtr_v, yb_tr)
    y_pred = clf.predict(Xte_v)
    y_scr = clf.predict_proba(Xte_v)[:, 1]
    acc = accuracy_score(yb_te, y_pred)
    auc = roc_auc_score(yb_te, y_scr)
    print(f"  {tag:<30s} acc={acc:.4f}  AUC={auc:.4f}  (vs M1 acc={M1_ACC:.4f})")
    return y_pred, y_scr


rng = np.random.default_rng(RNG_SEED)
bio_models = {}
for tag, Xtr, Xte in [("BioBlind", bio_blind_tr, bio_blind_te),
                      ("BioOriginal", bio_orig_tr, bio_orig_te)]:
    y_pred, y_scr = bio_hiring(Xtr, Xte, tag)
    bio_models[tag] = (y_pred, y_scr)

# ── 5. Fairness audit (identical definitions to v2) ──────────────────────────
print("\n[5/7] Fairness audit (v2 conventions):")

metric_rows = []
for tag, (y_pred, y_scr) in bio_models.items():
    for attr, gvec, glabels in [("gender", g_te, GENDER_LABELS),
                                ("ethnicity", e_te, ETHNICITY_LABELS)]:
        res = run_audit(yb_te, y_pred, y_scr, gvec,
                        np.unique(gvec).astype(int), glabels, rng)
        hi_name = glabels[int(res["groups"][res["idx_hi"]])]
        lo_name = glabels[int(res["groups"][res["idx_lo"]])]
        kw_p = res["kw"][1] if res["kw"] else np.nan
        ks_min = min((r["p_adj"] for r in res["ks_rows"]), default=np.nan)
        print(f"  {tag:<12s} {attr:<10s} DPD={res['DPD']:+.4f} DIR={res['DIR']:.4f} "
              f"({lo_name}/{hi_name}) {'PASS' if res['DIR'] >= EEOC_RULE else 'FAIL'}  "
              f"EOD={res['EOD']:+.4f} KL={res['KL']:.4f}  KW_p={kw_p:.2e} chi2_p={res['chi2_p']:.2e}")
        metric_rows.append({
            "model": f"BIO-{tag}", "features": "bio-text", "attribute": attr,
            "DPD": res["DPD"], "DPD_ci_lo": res["cis"]["DPD"][0], "DPD_ci_hi": res["cis"]["DPD"][1],
            "DIR": res["DIR"], "DIR_ci_lo": res["cis"]["DIR"][0], "DIR_ci_hi": res["cis"]["DIR"][1],
            "EOD": res["EOD"], "EOD_ci_lo": res["cis"]["EOD"][0], "EOD_ci_hi": res["cis"]["EOD"][1],
            "EO": res["EO"], "KL": res["KL"], "KL_ci_lo": res["cis"]["KL"][0], "KL_ci_hi": res["cis"]["KL"][1],
            "KW_p": kw_p, "KS_min_p_adj": ks_min, "chi2_p": res["chi2_p"],
            "EEOC_pass": bool(res["DIR"] >= EEOC_RULE),
            "worst_group": lo_name, "best_group": hi_name,
        })

# ── 6. Compare vs M1 (numeric CV7) ───────────────────────────────────────────
print("\n[6/7] Comparison vs M1-Fair (CV7, numeric) from the frozen audit:")
med = pd.read_csv(os.path.join(RESULTS_DIR, "metrics.csv"))
m1 = med[med.model == "M1-Fair (CV7)"].set_index("attribute")
print(f"  {'model':<16s} {'attr':<10s} {'acc':>6s} {'DPD':>7s} {'DIR':>7s} {'EOD':>7s} {'KL':>6s}  EEOC")
for tag, (y_pred, y_scr) in bio_models.items():
    acc = accuracy_score(yb_te, y_pred)
    for attr in ("gender", "ethnicity"):
        r = next(x for x in metric_rows if x["model"] == f"BIO-{tag}" and x["attribute"] == attr)
        print(f"  {'BIO-'+tag:<16s} {attr:<10s} {acc:>6.3f} {r['DPD']:>7.4f} {r['DIR']:>7.4f} "
              f"{r['EOD']:>7.4f} {r['KL']:>6.4f}  {'PASS' if r['EEOC_pass'] else 'FAIL'}")
for attr in ("gender", "ethnicity"):
    r = m1.loc[attr]
    print(f"  {'M1-Fair (CV7)':<16s} {attr:<10s} {M1_ACC:>6.3f} {r['DPD']:>7.4f} {r['DIR']:>7.4f} "
          f"{r['EOD']:>7.4f} {r['KL_extreme']:>6.4f}  {'PASS' if r['EEOC_pass'] else 'FAIL'}")

# ── 7. Exports ───────────────────────────────────────────────────────────────
print("\n[7/7] Exporting …")
pd.DataFrame(leakage_rows).to_csv(os.path.join(RESULTS_DIR, "bio_leakage.csv"), index=False)
pd.DataFrame(metric_rows).to_csv(os.path.join(RESULTS_DIR, "bio_metrics.csv"), index=False)
print("  wrote results/bio_leakage.csv, results/bio_metrics.csv")

fig, ax = plt.subplots(figsize=(10, 5))
rows = pd.DataFrame(leakage_rows)
order = rows.sort_values(["n_classes", "auc"], ascending=[True, False])
colors = {"gender": "#4878CF", "ethnicity": "#5CB85C"}
labels = list(order["channel"])
vals = order["auc"].values
colors_v = [colors["gender"] if r["n_classes"] == 2 else colors["ethnicity"] for _, r in order.iterrows()]
ax.barh(np.arange(len(order)), vals, color=colors_v, alpha=0.85)
ax.set_yticks(np.arange(len(order)))
ax.set_yticklabels(labels, fontsize=9)
ax.axvline(0.5, color="gray", lw=0.8, ls="--", label="chance / weak")
ax.set_xlabel("Test AUC (macro for 3-class)")
ax.set_title("Bio/text arm — demographic leakage through language", fontsize=12)
for i, (_, r) in enumerate(order.iterrows()):
    ax.text(r["auc"] + 0.005, i, f"{r['auc']:.3f}", va="center", fontsize=8)
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "fig7_bio_leakage.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved results/fig7_bio_leakage.png")

print("\nBio/text arm complete. Outputs in results/.")
_report_file.flush()
_report_file.close()
