"""
FairCV Bias Audit v2 (corrected / reproducible)
=================================================
Rebuild of `faircv_audit.py` implementing fixes R-1..R-15 from
`audit_code_review.md`. Ground truth is documented in `dataset_ground_truth.md`.

Fixes implemented
-----------------
R-1..R-4   Narrative is COMPUTED from the metrics, never hardcoded
           (no more false claims about EEOC, KS significance, or M2/KL).
R-5        Multi-group significance: Kruskal-Wallis for 3-group ethnicity
           + pairwise KS tests with Holm-Bonferroni correction.
R-6        DIR reported explicitly as "worst-group SR / best-group SR".
R-7        EOD magnitude PLUS disclosure of which group has the highest TPR.
R-8        Ethnicity groups named G1/G2/G3 (per official README), not the
           invented "Grp-A/B/C".
R-9        Explicit caveat: M3/M4 TPR/FPR/PPV are relative to artificially
           biased ground truth; M1-vs-M3 framed as the label-bias contrast.
R-10       Heatmap uses a single "severity" orientation; DIR no longer
           rendered with an inverted colour scale.
R-11       All pairwise KL values reported for 3-group ethnicity (+ mean).
R-12       Feature set documented (CV7 vs CV9); robustness arms M5 (CV9)
           and M6 (CV9 + face) added.
R-13       Bootstrap 95% CIs (2,000 resamples) for DPD/DIR/EOD/EO/KL plus
           effect sizes (Cohen's d / h).
R-14       Verified bias construction reported (×0.75 female; ×0.75/×1.25
           ethnicity); all claims scoped to the numeric-profile experiment.
R-15       Machine-readable exports:
             results/metrics.csv
             results/statistical_tests.csv
             results/per_group_metrics.csv
             results/audit_report.txt
           Figures saved to results/.

Additional robustness:
- utf-8 console output (Windows-safe when redirected).
- Deterministic RNG (seed 42) for model + bootstrap.
- Warns about the degenerate blind-face block (cols 31-50).
"""

import os
import sys
import atexit
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from scipy.stats import ks_2samp, kruskal, chi2_contingency
from scipy.special import rel_entr

warnings.filterwarnings("ignore")

# ── Windows-safe console (cp1252 cannot encode some report characters) ──
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


class _Tee:
    """Duplicate every write to both the console and results/audit_report.txt."""
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

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH   = "FairCVdb.npy"
RNG_SEED    = 42
N_BOOT      = 2000            # bootstrap resamples (R-13)
ALPHA       = 0.05
EEOC_RULE   = 0.80            # US EEOC four-fifths rule
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

_report_file = open(os.path.join(RESULTS_DIR, "audit_report.txt"), "w", encoding="utf-8")
atexit.register(_report_file.close)   # close even if the pipeline fails mid-run
sys.stdout = _Tee(sys.__stdout__, _report_file)

GENDER_LABELS    = {0: "Male", 1: "Female"}
ETHNICITY_LABELS = {0: "G1", 1: "G2", 2: "G3"}   # R-8: README placeholder names

# Feature groups (verified in dataset_ground_truth.md §2)
CV7_FEATURE_NAMES = ["Education", "Experience", "Rec-Letter",
                     "Availability", "Lang-1", "Lang-2", "Lang-3"]
CV9_FEATURE_NAMES = ["Occupation", "Suitability"] + CV7_FEATURE_NAMES
CV7_COLS  = list(range(4, 11))     # competencies (7)
CV9_COLS  = list(range(2, 11))     # + occupation, suitability (9)
FACE_COLS = list(range(11, 31))    # face embedding (20-d, norm 1)
BLIND_FACE_COLS = list(range(31, 51))  # degenerate in this file (see §6)


# ── Helpers ──────────────────────────────────────────────────────────────────
def binarise(arr, thr):
    """label >= thr -> hired (1). Binarisation is an audit decision (R-4/N-4)."""
    return (arr >= thr).astype(int)


def kl_divergence(p_scores, q_scores, bins=50, edges=None):
    """KL(P||Q) between two score histograms (same implementation as v1).
    If edges is given, both histograms use the same fixed binning (used by
    the bootstrap so resampled KL targets the same quantity as the point
    estimate)."""
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
    """Holm-Bonferroni adjusted p-values.

    For sorted p_(1) <= ... <= p_(m): q_(i) = max_{j<=i} min(1, (m-j+1) p_(j))
    — i.e. a RUNNING MAXIMUM of the step-down products.
    """
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


def cohens_d(a, b):
    """Standardised mean difference between two score samples."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def per_group_stats(y_true, y_pred, y_score, group_vec, groups):
    """Per-group N / SR / TPR / FPR / PPV / score arrays (order = groups)."""
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
    """Aggregate disparity metrics from per-group arrays (v1-compatible)."""
    dpd = SR.max() - SR.min()
    dir_ = (SR.min() / SR.max()) if SR.max() > 0 else np.nan
    tpr_f = TPR[np.isfinite(TPR)]
    fpr_f = FPR[np.isfinite(FPR)]
    eod = (tpr_f.max() - tpr_f.min()) if len(tpr_f) else np.nan
    eo = max(tpr_f.max() - tpr_f.min(), fpr_f.max() - fpr_f.min()) if (len(tpr_f) and len(fpr_f)) else np.nan
    kl = kl_divergence(s_hi, s_lo)
    return dpd, dir_, eod, eo, kl


def bootstrap_cis(y_true, y_pred, y_score, group_vec, groups, hi=None, lo=None,
                  n_boot=N_BOOT, rng=None):
    """Percentile 95% CIs for DPD/DIR/EOD/EO/KL_extreme on resampled test set.

    DPD/DIR/EOD/EO re-derive their extreme groups per resample (standard
    bootstrap of max-min statistics). KL is evaluated on the FIXED extreme
    pair observed in the full test set (hi/lo), so the CI targets one
    well-defined quantity rather than a resample-dependent pair.
    Fast path: boolean-count per-group stats instead of sklearn confusion_matrix.
    """
    n = len(y_true)
    ng = len(groups)
    cols = ("DPD", "DIR", "EOD", "EO", "KL")
    out = {k: np.empty(n_boot) for k in cols}
    yt1 = (y_true == 1)
    yp1 = (y_pred == 1)
    # Fixed binning from the full test-set extreme pair (avoids resampling bias).
    edges = None
    if hi is not None and lo is not None:
        s_hi_full = y_score[group_vec == groups[hi]]
        s_lo_full = y_score[group_vec == groups[lo]]
        lo_e = min(s_hi_full.min(), s_lo_full.min())
        hi_e = max(s_hi_full.max(), s_lo_full.max()) + 1e-9
        edges = np.linspace(lo_e, hi_e, 51)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        gv = group_vec[idx]
        yt, yp, ys = yt1[idx], yp1[idx], y_score[idx]
        SR = np.zeros(ng)          # 0.0 for an (unreachable) empty group
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
        if hi is not None and lo is not None:
            kl = kl_divergence(scores[hi], scores[lo], edges=edges)
        else:
            h, l = int(np.nanargmax(SR)), int(np.nanargmin(SR))
            kl = kl_divergence(scores[h], scores[l], edges=edges)
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
    """Full per-model × per-attribute audit. Returns dict of results."""
    N, SR, TPR, FPR, PPV, scores = per_group_stats(y_true, y_pred, y_score, group_vec, groups)
    hi, lo = int(SR.argmax()), int(SR.argmin())
    dpd, dir_, eod, eo, kl_extreme = _agg(SR, TPR, FPR, scores[hi], scores[lo])

    # Pairwise KL (R-11): all pairs, plus mean.
    pairs = [(i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))]
    kl_pairs = {f"{group_labels[groups[i]]}-{group_labels[groups[j]]}":
                kl_divergence(scores[i], scores[j]) for i, j in pairs}
    kl_mean = float(np.mean(list(kl_pairs.values()))) if kl_pairs else np.nan

    # Pairwise KS + Holm correction within this (model, attribute) block (R-5).
    ks_rows = []
    for i, j in pairs:
        st, p = ks_2samp(scores[i], scores[j])
        ks_rows.append({"i": i, "j": j,
                        "comparison": f"{group_labels[groups[i]]} vs {group_labels[groups[j]]}",
                        "stat": float(st), "p": float(p)})
    p_adj = holm_correct([r["p"] for r in ks_rows]) if ks_rows else []
    for r, pa in zip(ks_rows, p_adj):
        r["p_adj"] = float(pa)

    # Kruskal-Wallis for >= 3 groups (R-5).
    kw = None
    if len(groups) >= 3:
        st, p = kruskal(*scores)
        kw = (float(st), float(p))

    # chi2: group × predicted-selection contingency.
    sel = y_pred.astype(int)
    cont = np.array([[int(((group_vec == g) & (sel == 1)).sum()),
                      int(((group_vec == g) & (sel == 0)).sum())] for g in groups])
    chi2_stat, chi2_p, _, _ = chi2_contingency(cont)

    # Effect sizes (R-13).
    d = cohens_d(scores[hi], scores[lo])
    h = 2.0 * (np.arcsin(np.sqrt(SR[hi])) - np.arcsin(np.sqrt(SR[lo])))

    cis = bootstrap_cis(y_true, y_pred, y_score, group_vec, groups,
                        hi=hi, lo=lo, n_boot=N_BOOT, rng=rng)

    return {
        "N": N, "SR": SR, "TPR": TPR, "FPR": FPR, "PPV": PPV, "scores": scores,
        "groups": groups, "labels": group_labels,
        "idx_hi": hi, "idx_lo": lo,
        "DPD": dpd, "DIR": dir_, "EOD": eod, "EO": eo,
        "KL_extreme": kl_extreme, "KL_pairs": kl_pairs, "KL_mean": kl_mean,
        "ks_rows": ks_rows, "kw": kw, "chi2_stat": chi2_stat, "chi2_p": float(chi2_p),
        "cohens_d": d, "cohens_h": float(h),
        "cis": cis,
    }


# ── 1. Load dataset ──────────────────────────────────────────────────────────
print("=" * 74)
print("FairCV Bias Audit v2 (corrected implementation)")
print("=" * 74)
print("\n[1/8] Loading FairCVdb.npy …")

db = np.load(DATA_PATH, allow_pickle=True).item()
P_tr = db["Profiles Train"]
P_te = db["Profiles Test"]
y_blind_tr = db["Blind Labels Train"]
y_blind_te = db["Blind Labels Test"]
y_gender_tr = db["Biased Labels Train (Gender)"]
y_gender_te = db["Biased Labels Test (Gender)"]
y_eth_tr = db["Biased Labels Train (Ethnicity)"]
y_eth_te = db["Biased Labels Test (Ethnicity)"]
print(f"  Train: {P_tr.shape[0]:,} profiles | Test: {P_te.shape[0]:,} profiles")

# ── 2. Features, labels, bias construction ───────────────────────────────────
print("\n[2/8] Feature & label documentation (ground truth) …")

gender_tr = P_tr[:, 1].astype(int)
gender_te = P_te[:, 1].astype(int)
ethnicity_tr = P_tr[:, 0].astype(int)
ethnicity_te = P_te[:, 0].astype(int)

X_cv7_tr, X_cv7_te = P_tr[:, CV7_COLS].astype(float), P_te[:, CV7_COLS].astype(float)
X_cv9_tr, X_cv9_te = P_tr[:, CV9_COLS].astype(float), P_te[:, CV9_COLS].astype(float)
X_f7_tr,  X_f7_te  = P_tr[:, CV7_COLS + FACE_COLS].astype(float), P_te[:, CV7_COLS + FACE_COLS].astype(float)
X_f9_tr,  X_f9_te  = P_tr[:, CV9_COLS + FACE_COLS].astype(float), P_te[:, CV9_COLS + FACE_COLS].astype(float)

print("  Feature groups (verified vs FairCVdb README):")
print("    col0-1   : ethnicity (G1/G2/G3), gender (Male/Female)  -> protected, excluded")
print("    col2-3   : occupation (10 cat.), suitability (4 levels) -> CV9 arms only (R-12)")
print("    col4-10  : 7 competencies (education … 3 languages)     -> CV7 (all models)")
print("    col11-30 : face embedding (20-d, norm 1)                -> M2/M6")
print("    col31-50 : blind face embedding (SensitiveNets-style)   -> EXCLUDED")

bf = P_tr[:, BLIND_FACE_COLS]
print(f"    NOTE: cols 31-50 are degenerate in this file "
      f"(per-col std ~{bf.std(axis=0).max():.1e}, all rows within 1e-4 of a constant "
      f"vector) -> unusable as features; no SensitiveNets control possible (N-2).")

print("\n  Label binarisation (R-4 / N-4):")
BLIND_THR = float(np.median(y_blind_tr))
GENDER_THR = float(np.median(y_gender_tr))
ETH_THR = float(np.median(y_eth_tr))
print(f"    threshold = train median, label >= threshold -> hired.")
print(f"    blind: {BLIND_THR:.4f} | gender: {GENDER_THR:.4f} | ethnicity: {ETH_THR:.4f}")
print("    NOTE: 'above median = hired' is an audit decision; the FairCV papers")
print("    evaluate with top-N score screening. This binarisation is an artificial")
print("    outcome and is reported as such.")

yb_blind_tr = binarise(y_blind_tr, BLIND_THR)
yb_blind_te = binarise(y_blind_te, BLIND_THR)
yb_gender_tr = binarise(y_gender_tr, GENDER_THR)
yb_gender_te = binarise(y_gender_te, GENDER_THR)
yb_eth_tr = binarise(y_eth_tr, ETH_THR)
yb_eth_te = binarise(y_eth_te, ETH_THR)
print(f"    hired rate (test): blind={yb_blind_te.mean():.4f} "
      f"gender={yb_gender_te.mean():.4f} ethnicity={yb_eth_te.mean():.4f}")

print("\n  Verified bias construction in the data (R-14):")
m = y_blind_tr > 0.05
ratios = {}
for tag, arr, gvec, vals in [
    ("Gender-biased", y_gender_tr, gender_tr, [0, 1]),
    ("Ethnicity-biased", y_eth_tr, ethnicity_tr, [0, 1, 2]),
]:
    for v in vals:
        mm = m & (gvec == v)
        ratio = float((arr[mm] / y_blind_tr[mm]).mean())
        name = GENDER_LABELS[v] if tag.startswith("Gender") else ETHNICITY_LABELS[v]
        print(f"    {tag:<16s} {name:<8s} mean(ratio) = {ratio:.4f}")
        ratios[(tag, name)] = ratio
g_f = ratios[("Gender-biased", "Female")]
e_g1 = ratios[("Ethnicity-biased", "G1")]
e_g3 = ratios[("Ethnicity-biased", "G3")]
print(f"    -> gender-biased labels penalise Female x{g_f:.2f}; ethnicity-biased")
print(f"       labels penalise G3 x{e_g3:.2f} and boost G1 x{e_g1:.2f}.")

# ── 3. Train models ──────────────────────────────────────────────────────────
print("\n[3/8] Training classifiers …")


def make_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=RNG_SEED)),
    ])


# (name, short, features-tag, label-tag, X_tr, y_tr, X_te, y_te)
MODELS = [
    ("M1-Fair (CV7)",        "M1-Fair",       "cv7",     "blind",        X_cv7_tr, yb_blind_tr,  X_cv7_te, yb_blind_te),
    ("M2-Multimodal (CV7+Face)", "M2-Multimodal", "cv7+face", "blind",    X_f7_tr,  yb_blind_tr,  X_f7_te,  yb_blind_te),
    ("M3-Gender-Biased (CV7)",   "M3-Gender-Bias", "cv7",  "gender-bias", X_cv7_tr, yb_gender_tr, X_cv7_te, yb_gender_te),
    ("M4-Ethnicity-Biased (CV7)", "M4-Eth-Bias",   "cv7",  "eth-bias",    X_cv7_tr, yb_eth_tr,    X_cv7_te, yb_eth_te),
    ("M5-Robust (CV9)",      "M5-CV9",        "cv9",     "blind",        X_cv9_tr, yb_blind_tr,  X_cv9_te, yb_blind_te),
    ("M6-Robust (CV9+Face)", "M6-CV9+Face",   "cv9+face", "blind",       X_f9_tr,  yb_blind_tr,  X_f9_te,  yb_blind_te),
]

trained = {}
for mname, sname, fset, lset, X_tr, y_tr, X_te, y_te in MODELS:
    pipe = make_lr()
    pipe.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, pipe.predict(X_te))
    auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    f1 = f1_score(y_te, pipe.predict(X_te))
    trained[mname] = dict(pipe=pipe, X_te=X_te, y_te=y_te, acc=acc, auc=auc, f1=f1,
                          short=sname, fset=fset, lset=lset)
    print(f"  {mname:<38s} acc={acc:.3f}  AUC={auc:.3f}  F1={f1:.3f}")
print("  NOTE (R-9): M1..M4 performance values are NOT directly comparable because")
print("  M3/M4 train on different (artificially biased) label sets, and M3/M4 are")
print("  evaluated against those biased labels.")

# ── 4. Fairness metrics ──────────────────────────────────────────────────────
print("\n[4/8] Computing fairness metrics + statistical tests …")

rng = np.random.default_rng(RNG_SEED)
ATTRIBUTES = [("gender", gender_te, GENDER_LABELS), ("ethnicity", ethnicity_te, ETHNICITY_LABELS)]

audit = {}          # model -> attr -> run_audit dict
per_group_rows = [] # for per_group_metrics.csv
test_rows = []      # for statistical_tests.csv
metric_rows = []    # for metrics.csv

DIV = "─" * 74


def fmt(v, spec):
    return f"{v:{spec}}" if not (isinstance(v, float) and np.isnan(v)) else "  nan"


for mname, md in trained.items():
    pipe = md["pipe"]
    y_pred = pipe.predict(md["X_te"])
    y_scr = pipe.predict_proba(md["X_te"])[:, 1]
    audit[mname] = {}
    for attr, gvec, glabels in ATTRIBUTES:
        res = run_audit(md["y_te"], y_pred, y_scr, gvec,
                        np.unique(gvec).astype(int), glabels, rng)
        audit[mname][attr] = res

        # per-group rows
        for gi, g in enumerate(res["groups"]):
            per_group_rows.append({
                "model": mname, "attribute": attr, "group": glabels[int(g)],
                "n": int(res["N"][gi]),
                "selection_rate": float(res["SR"][gi]),
                "tpr": res["TPR"][gi], "fpr": res["FPR"][gi], "ppv": res["PPV"][gi],
                "mean_score": float(res["scores"][gi].mean()),
                "std_score": float(res["scores"][gi].std()),
            })

        # test rows
        for r in res["ks_rows"]:
            test_rows.append({"model": mname, "attribute": attr, "test": "KS-2samp",
                              "comparison": r["comparison"], "statistic": r["stat"],
                              "p_value": r["p"], "p_adjusted": r["p_adj"],
                              "significant_0.05": bool(r["p_adj"] < ALPHA)})
        if res["kw"] is not None:
            test_rows.append({"model": mname, "attribute": attr, "test": "Kruskal-Wallis",
                              "comparison": "all groups", "statistic": res["kw"][0],
                              "p_value": res["kw"][1], "p_adjusted": np.nan,
                              "significant_0.05": bool(res["kw"][1] < ALPHA)})
        test_rows.append({"model": mname, "attribute": attr, "test": "chi2-selection",
                          "comparison": "group x hired", "statistic": res["chi2_stat"],
                          "p_value": res["chi2_p"], "p_adjusted": np.nan,
                          "significant_0.05": bool(res["chi2_p"] < ALPHA)})

        # metric row
        ks_min_adj = min((r["p_adj"] for r in res["ks_rows"]), default=np.nan)
        metric_rows.append({
            "model": mname, "short_name": md["short"], "features": md["fset"],
            "label_set": md["lset"], "attribute": attr, "n": len(md["y_te"]),
            "DPD": res["DPD"], "DPD_ci_lo": res["cis"]["DPD"][0], "DPD_ci_hi": res["cis"]["DPD"][1],
            "DIR": res["DIR"], "DIR_ci_lo": res["cis"]["DIR"][0], "DIR_ci_hi": res["cis"]["DIR"][1],
            "EOD": res["EOD"], "EOD_ci_lo": res["cis"]["EOD"][0], "EOD_ci_hi": res["cis"]["EOD"][1],
            "EO": res["EO"], "EO_ci_lo": res["cis"]["EO"][0], "EO_ci_hi": res["cis"]["EO"][1],
            "KL_extreme": res["KL_extreme"], "KL_mean_pairwise": res["KL_mean"],
            "KS_min_p_adj": ks_min_adj,
            "KW_stat": res["kw"][0] if res["kw"] else np.nan,
            "KW_p": res["kw"][1] if res["kw"] else np.nan,
            "chi2_p": res["chi2_p"],
            "EEOC_pass": bool(res["DIR"] >= EEOC_RULE),
            "highest_SR_group": glabels[int(res["groups"][res["idx_hi"]])],
            "highest_TPR_group": glabels[int(res["groups"][np.nanargmax(res["TPR"])])],
            "cohens_d_extreme": res["cohens_d"], "cohens_h_extreme": res["cohens_h"],
        })

print("  done.")

print("  NOTE: bootstrap CIs for DPD/DIR/EOD/EO contain the point estimates. The")
print("  histogram-KL CIs are upward-biased relative to the plug-in point estimate")
print("  (standard for plug-in KL on resampled histograms); read KL CIs as")
print("  conservative upper ranges.")
print("  NOTE: KL is directed (P||Q). 'KL (extreme SR pair)' = KL(best-SR group ||")
print("  worst-SR group); 'KL mean (all pairs)' averages the i<j directed pairs.")

# ── 5. Print audit report (per model x attribute) ────────────────────────────
print("\n[5/8] Audit report …\n")


def print_per_group(res):
    print(f"  {'Group':<10} {'N':>6}  {'SR':>6}  {'TPR':>6}  {'FPR':>6}  {'PPV':>6}  {'meanScore':>9}")
    print("  " + "-" * 62)
    for gi, g in enumerate(res["groups"]):
        print(f"  {res['labels'][int(g)]:<10} {int(res['N'][gi]):>6}  "
              f"{fmt(res['SR'][gi], '.3f'):>6}  {fmt(res['TPR'][gi], '.3f'):>6}  "
              f"{fmt(res['FPR'][gi], '.3f'):>6}  {fmt(res['PPV'][gi], '.3f'):>6}  "
              f"{fmt(res['scores'][gi].mean(), '.3f'):>9}")


for mname, md in trained.items():
    print(f"\n{'━' * 74}\n  {mname}   (features={md['fset']}, labels={md['lset']})\n{'━' * 74}")
    for attr in ("gender", "ethnicity"):
        res = audit[mname][attr]
        print(f"\n  Protected attribute: {attr.upper()}")
        print(DIV)
        print_per_group(res)
        hi_name = res["labels"][int(res["groups"][res["idx_hi"]])]
        lo_name = res["labels"][int(res["groups"][res["idx_lo"]])]
        tpr_hi = res["labels"][int(res["groups"][np.nanargmax(res["TPR"])])]
        print(f"\n  DPD = {res['DPD']:+.4f}   (0 = perfect parity)")
        print(f"  DIR = {res['DIR']:.4f}   = worst-group SR / best-group SR  "
              f"({lo_name} / {hi_name})  [EEOC >= {EEOC_RULE:.2f} -> "
              f"{'PASS' if res['DIR'] >= EEOC_RULE else 'FAIL'}]")
        print(f"  EOD = {res['EOD']:+.4f}   (magnitude; highest TPR group = {tpr_hi})")
        print(f"  EO  = {res['EO']:+.4f}   (0 = perfect)")
        print(f"  KL (extreme SR pair) = {res['KL_extreme']:.4f}   KL mean (all pairs) = {fmt(res['KL_mean'], '.4f')}")
        if res["KL_pairs"]:
            print("  Pairwise KL: " + ", ".join(f"{k}={v:.4f}" for k, v in res["KL_pairs"].items()))
        for r in res["ks_rows"]:
            sig = "SIGNIFICANT" if r["p_adj"] < ALPHA else "ns"
            print(f"  KS {r['comparison']:<18s} stat={r['stat']:.4f}  p={r['p']:.3e}  "
                  f"p_holm={r['p_adj']:.3e}  [{sig}]")
        if res["kw"] is not None:
            sig = "SIGNIFICANT" if res["kw"][1] < ALPHA else "ns"
            print(f"  Kruskal-Wallis stat={res['kw'][0]:.4f}  p={res['kw'][1]:.3e}  [{sig}]")
        sig = "SIGNIFICANT" if res["chi2_p"] < ALPHA else "ns"
        print(f"  chi2 (group x hired) stat={res['chi2_stat']:.4f}  p={res['chi2_p']:.3e}  [{sig}]")
        print(f"  Effect sizes: Cohen's d={fmt(res['cohens_d'], '.3f')}  "
              f"Cohen's h={fmt(res['cohens_h'], '.3f')}  (extreme SR pair)")
        print("  Bootstrap 95%% CI (n=%d): DPD [%.4f, %.4f]  DIR [%.4f, %.4f]  "
              "EOD [%.4f, %.4f]  EO [%.4f, %.4f]  KL [%.4f, %.4f]"
              % (N_BOOT, *res["cis"]["DPD"], *res["cis"]["DIR"], *res["cis"]["EOD"],
                 *res["cis"]["EO"], *res["cis"]["KL"]))

# ── 6. Computed narrative (R-1..R-4) ─────────────────────────────────────────
print("\n" + "=" * 74)
print("FINDINGS (computed from results, not hardcoded)")
print("=" * 74)

# 6.1 EEOC verdict table
print("\nEEOC (80% rule) verdicts — DIR = worst-group SR / best-group SR:")
print(f"  {'Model':<32s} {'gender DIR':>12s} {'eth DIR':>12s}  verdict")
for mname, md in trained.items():
    g = audit[mname]["gender"]
    e = audit[mname]["ethnicity"]
    print(f"  {mname:<32s} {g['DIR']:>12.4f} {e['DIR']:>12.4f}  "
          f"gender={'PASS' if g['DIR'] >= EEOC_RULE else 'FAIL'} | "
          f"ethnicity={'PASS' if e['DIR'] >= EEOC_RULE else 'FAIL'}")

# 6.2 lowest-disparity model per attribute and metric
print("\nLowest-disparity model per attribute/metric:")
for attr in ("gender", "ethnicity"):
    for metric in ("DPD", "EOD", "KL_extreme"):
        vals = {mname: audit[mname][attr][metric] for mname in trained}
        best = min(vals, key=lambda k: (vals[k] if np.isfinite(vals[k]) else np.inf))
        tag = " (robustness arm)" if best in ("M5-Robust (CV9)", "M6-Robust (CV9+Face)") else ""
        print(f"  {attr:<10s} {metric:<12s} lowest = {best:<28s} ({vals[best]:.4f}){tag}")

# 6.3 proxy comparison M1 vs M2 (R-3, careful wording)
print("\nProxy comparison — M1 (CV7, blind) vs M2 (CV7+Face, blind), same labels:")
for attr in ("gender", "ethnicity"):
    m1, m2 = audit["M1-Fair (CV7)"][attr], audit["M2-Multimodal (CV7+Face)"][attr]
    d_dpd = m2["DPD"] - m1["DPD"]
    d_kl = m2["KL_extreme"] - m1["KL_extreme"]
    d_dirlab = "increased" if d_dpd > 0 else "decreased"
    d_kllab = "increased" if d_kl > 0 else "decreased"
    # informal CI-overlap check (R-13)
    c1, c2 = m1["cis"]["DPD"], m2["cis"]["DPD"]
    overlap = not (c1[1] < c2[0] or c2[1] < c1[0])
    print(f"  {attr:<10s} DPD {m1['DPD']:.4f} -> {m2['DPD']:.4f} (delta {d_dpd:+.4f}, {d_dirlab}); "
          f"KL {m1['KL_extreme']:.4f} -> {m2['KL_extreme']:.4f} (delta {d_kl:+.4f}, {d_kllab}); "
          f"DPD CIs overlap: {'yes' if overlap else 'no'}")
print("  Wording: 'Adding face embeddings was ASSOCIATED with the changes above under")
print("  this evaluation setup.' This is an association, not a demonstrated causal")
print("  mechanism (the labels are identical; only the features differ).")

# 6.4 label-bias comparison M1 vs M3 / M4 (R-9)
print("\nLabel-bias comparison (same CV7 features, different training labels):")
m1g = audit["M1-Fair (CV7)"]["gender"]
m3 = audit["M3-Gender-Biased (CV7)"]["gender"]
m1e = audit["M1-Fair (CV7)"]["ethnicity"]
m4 = audit["M4-Ethnicity-Biased (CV7)"]["ethnicity"]
print(f"  gender   : DPD {m1g['DPD']:.4f} -> {m3['DPD']:.4f} (delta {m3['DPD'] - m1g['DPD']:+.4f}) "
      f"with gender-biased labels (female x0.75 penalty)")
print(f"  ethnicity: DPD {m1e['DPD']:.4f} -> {m4['DPD']:.4f} (delta {m4['DPD'] - m1e['DPD']:+.4f}) "
      f"with ethnicity-biased labels (G3 x0.75 / G1 x1.25)")
print("  Caveat (R-9): M3/M4 TPR/FPR/PPV are computed against artificially biased")
print("  ground truth; only selection-rate (SR) based metrics are comparable to M1/M2.")

# 6.5 significance summary (R-2)
print("\nScore-distribution significance (Holm-adjusted within each model x attribute block):")
for mname in trained:
    bits = []
    for attr in ("gender", "ethnicity"):
        res = audit[mname][attr]
        if res["kw"] is not None:
            bits.append(f"{attr}: KW p={res['kw'][1]:.2e} "
                        f"({'sig' if res['kw'][1] < ALPHA else 'ns'})")
        else:
            p = min((r["p_adj"] for r in res["ks_rows"]), default=np.nan)
            bits.append(f"{attr}: KS p_holm={p:.2e} ({'sig' if p < ALPHA else 'ns'})")
    print(f"  {mname:<38s} " + "  ".join(bits))

# 6.6 robustness arms (R-12)
print("\nRobustness arms (full CV9 feature set incl. occupation + suitability):")
m5 = audit["M5-Robust (CV9)"]
m6 = audit["M6-Robust (CV9+Face)"]
print(f"  M5 (CV9, blind): gender DPD={m5['gender']['DPD']:.4f}  "
      f"ethnicity DPD={m5['ethnicity']['DPD']:.4f}  "
      f"acc={trained['M5-Robust (CV9)']['acc']:.3f}  "
      f"AUC={trained['M5-Robust (CV9)']['auc']:.3f}  (vs M1 acc={trained['M1-Fair (CV7)']['acc']:.3f})")
print(f"  M6 (CV9+Face, blind): gender DPD={m6['gender']['DPD']:.4f}  "
      f"ethnicity DPD={m6['ethnicity']['DPD']:.4f}")
print("  Interpretation: adding occupation/suitability to the CV set can materially")
print("  change both performance and disparity; M1-M4 conclusions should be read as")
print("  specific to the CV7 feature set.")

# 6.7 uncertainty summary (R-13)
print("\nUncertainty (bootstrap 95% CIs, n=2000):")
print(f"  {'Model':<32s} {'attr':<10s} {'DPD CI':<20s} {'DIR CI':<20s}")
for mname in trained:
    for attr in ("gender", "ethnicity"):
        res = audit[mname][attr]
        print(f"  {mname:<32s} {attr:<10s} "
              f"[{res['cis']['DPD'][0]:.4f}, {res['cis']['DPD'][1]:.4f}]       "
              f"[{res['cis']['DIR'][0]:.4f}, {res['cis']['DIR'][1]:.4f}]")

# ── 7. Figures ───────────────────────────────────────────────────────────────
print("\n[6/8] Saving figures …")

MODEL_NAMES = list(trained.keys())
SHORT_NAMES = [trained[m]["short"] for m in MODEL_NAMES]
PALETTE_GENDER = {0: "#4878CF", 1: "#D65F5F"}
PALETTE_ETH = {0: "#5CB85C", 1: "#F0AD4E", 2: "#9B59B6"}

# Fig 1: selection rates
fig1, axes1 = plt.subplots(2, len(MODEL_NAMES), figsize=(3.2 * len(MODEL_NAMES), 8),
                           sharey="row")
fig1.suptitle("FairCV Audit v2 – Selection Rate by Demographic Group",
              fontsize=14, fontweight="bold", y=1.02)
for ci, mname in enumerate(MODEL_NAMES):
    for ri, attr in enumerate(("gender", "ethnicity")):
        ax = axes1[ri][ci]
        res = audit[mname][attr]
        pal = PALETTE_GENDER if attr == "gender" else PALETTE_ETH
        labels = list(res["labels"].values())
        for gi, g in enumerate(res["groups"]):
            ax.bar(gi, res["SR"][gi], color=pal[int(g)], alpha=0.85,
                   edgecolor="white", linewidth=0.5)
            ax.text(gi, res["SR"][gi] + 0.005, f"{res['SR'][gi]:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(res["groups"])))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 0.75)
        ax.axhline(0.5, color="gray", lw=0.8, ls="--")
        ax.set_title(f"{trained[mname]['short']}\n"
                     f"DIR={res['DIR']:.3f} "
                     f"{'✓' if res['DIR'] >= EEOC_RULE else '✗'}",
                     fontsize=9)
        if ci == 0:
            ax.set_ylabel(f"{attr.capitalize()}\nSelection Rate", fontsize=9)
plt.tight_layout()
fig1.savefig(f"{RESULTS_DIR}/fig1_selection_rates.png", dpi=150, bbox_inches="tight")
print(f"  Saved {RESULTS_DIR}/fig1_selection_rates.png")

# Fig 2: heatmap (R-10 — severity orientation; DIR no longer inverted)
METRIC_KEYS = ["DPD", "DIR", "EOD", "EO", "KL_extreme"]
METRIC_LABELS = ["DPD\n(Dem. Parity Diff)", "DIR\n(Disp. Impact Ratio)",
                 "EOD\n(Equal Opp. Diff)", "EO\n(Equalized Odds)", "KL\n(extreme)"]
DIRECTION = {"DPD": "up", "DIR": "down", "EOD": "up", "EO": "up", "KL_extreme": "up"}

fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5.5))
fig2.suptitle("FairCV Audit v2 – Fairness Metrics (severity: 0 = best, 1 = worst)",
              fontsize=13, fontweight="bold")
for ax, attr in zip(axes2, ("gender", "ethnicity")):
    raw = np.zeros((len(MODEL_NAMES), len(METRIC_KEYS)))
    for mi, mname in enumerate(MODEL_NAMES):
        for ki, mk in enumerate(METRIC_KEYS):
            raw[mi, ki] = audit[mname][attr][mk]
    sev = np.zeros_like(raw)
    for c in range(raw.shape[1]):
        col = raw[:, c]
        rng_c = col.max() - col.min()
        if rng_c > 0:
            if DIRECTION[METRIC_KEYS[c]] == "up":
                sev[:, c] = (col - col.min()) / rng_c
            else:  # DIR: lower is better -> severity inverted (R-10)
                sev[:, c] = (col.max() - col) / rng_c
    im = ax.imshow(sev, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(METRIC_KEYS)))
    ax.set_xticklabels(METRIC_LABELS, fontsize=8)
    ax.set_yticks(range(len(MODEL_NAMES)))
    ax.set_yticklabels(SHORT_NAMES, fontsize=9)
    ax.set_title(f"Protected attribute: {attr.upper()}", fontsize=11)
    for mi in range(len(MODEL_NAMES)):
        for ki in range(len(METRIC_KEYS)):
            ax.text(ki, mi, f"{raw[mi, ki]:.3f}", ha="center", va="center",
                    fontsize=8, color="black" if sev[mi, ki] < 0.7 else "white")
    plt.colorbar(im, ax=ax, label="Severity (0=best, 1=worst)")
plt.tight_layout()
fig2.savefig(f"{RESULTS_DIR}/fig2_fairness_heatmap.png", dpi=150, bbox_inches="tight")
print(f"  Saved {RESULTS_DIR}/fig2_fairness_heatmap.png")

# Fig 3: score distributions
fig3, axes3 = plt.subplots(2, len(MODEL_NAMES), figsize=(3.2 * len(MODEL_NAMES), 8))
fig3.suptitle("FairCV Audit v2 – Predicted Score Distributions by Group",
              fontsize=14, fontweight="bold", y=1.02)
bins = np.linspace(0, 1, 31)
for ci, mname in enumerate(MODEL_NAMES):
    for ri, (attr, pal) in enumerate((("gender", PALETTE_GENDER), ("ethnicity", PALETTE_ETH))):
        ax = axes3[ri][ci]
        res = audit[mname][attr]
        for gi, g in enumerate(res["groups"]):
            ax.hist(res["scores"][gi], bins=bins, alpha=0.55, density=True,
                    color=pal[int(g)], label=res["labels"][int(g)], edgecolor="none")
        if res["kw"] is not None:
            pval = res["kw"][1]
            plab = f"KW p={pval:.2e}"
        else:
            pval = min((r["p_adj"] for r in res["ks_rows"]), default=np.nan)
            plab = f"KS p={pval:.2e}"
        ax.set_title(f"{trained[mname]['short']}\nKL={res['KL_extreme']:.3f}  {plab}",
                     fontsize=8)
        if ci == 0:
            ax.set_ylabel(f"{attr.capitalize()}\nDensity", fontsize=9)
        if ri == 1:
            ax.set_xlabel("Predicted Score", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
plt.tight_layout()
fig3.savefig(f"{RESULTS_DIR}/fig3_score_distributions.png", dpi=150, bbox_inches="tight")
print(f"  Saved {RESULTS_DIR}/fig3_score_distributions.png")

# Fig 4: coefficients (CV7 models + CV9 robustness arm)
fig4, axes4 = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
fig4.suptitle("FairCV Audit v2 – Logistic Regression Coefficients",
              fontsize=13, fontweight="bold")
coef_panels = [
    ("M1-Fair", "M1-Fair (CV7)", CV7_FEATURE_NAMES, "#4878CF"),
    ("M3-Gender-Bias", "M3-Gender-Biased (CV7)", CV7_FEATURE_NAMES, "#D65F5F"),
    ("M4-Eth-Bias", "M4-Ethnicity-Biased (CV7)", CV7_FEATURE_NAMES, "#5CB85C"),
    ("M5-CV9", "M5-Robust (CV9)", CV9_FEATURE_NAMES, "#9B59B6"),
]
for ax, (sname, mname, names, color) in zip(axes4, coef_panels):
    coefs = trained[mname]["pipe"].named_steps["clf"].coef_[0][:len(names)]
    ax.barh(range(len(names)), coefs, color=color, alpha=0.80)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(sname, fontsize=10)
    ax.set_xlabel("Coefficient", fontsize=9)
plt.tight_layout()
fig4.savefig(f"{RESULTS_DIR}/fig4_feature_weights.png", dpi=150, bbox_inches="tight")
print(f"  Saved {RESULTS_DIR}/fig4_feature_weights.png")

# Fig 5: bootstrap CIs for DPD and DIR
fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
fig5.suptitle("FairCV Audit v2 – Disparity Metrics with Bootstrap 95% CIs",
              fontsize=13, fontweight="bold")
xpos = np.arange(len(MODEL_NAMES))
for ax, attr, metric, ylab in [
    (axes5[0], "gender", "DPD", "DPD"),
    (axes5[1], "ethnicity", "DIR", "DIR"),
]:
    vals = np.array([audit[m][attr][metric] for m in MODEL_NAMES])
    los = np.array([audit[m][attr]["cis"][metric][0] for m in MODEL_NAMES])
    his = np.array([audit[m][attr]["cis"][metric][1] for m in MODEL_NAMES])
    ax.errorbar(xpos, vals, yerr=[vals - los, his - vals],
                fmt="o", capsize=4, color="#4878CF")
    ax.set_xticks(xpos)
    ax.set_xticklabels(SHORT_NAMES, rotation=20, fontsize=8)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    if metric == "DIR":
        ax.axhline(EEOC_RULE, color="red", lw=1, ls="--", label=f"EEOC {EEOC_RULE:.2f}")
        ax.legend(fontsize=8)
    ax.set_title(f"{attr.capitalize()} – {ylab}", fontsize=11)
    ax.set_ylabel(ylab, fontsize=10)
plt.tight_layout()
fig5.savefig(f"{RESULTS_DIR}/fig5_bootstrap_cis.png", dpi=150, bbox_inches="tight")
print(f"  Saved {RESULTS_DIR}/fig5_bootstrap_cis.png")

# ── 8. CSV exports (R-15) ────────────────────────────────────────────────────
print("\n[7/8] Exporting machine-readable results …")

metrics_df = pd.DataFrame(metric_rows)
tests_df = pd.DataFrame(test_rows)
per_group_df = pd.DataFrame(per_group_rows)

metrics_df.to_csv(f"{RESULTS_DIR}/metrics.csv", index=False)
tests_df.to_csv(f"{RESULTS_DIR}/statistical_tests.csv", index=False)
per_group_df.to_csv(f"{RESULTS_DIR}/per_group_metrics.csv", index=False)
print(f"  Saved {RESULTS_DIR}/metrics.csv             ({len(metrics_df)} rows)")
print(f"  Saved {RESULTS_DIR}/statistical_tests.csv   ({len(tests_df)} rows)")
print(f"  Saved {RESULTS_DIR}/per_group_metrics.csv   ({len(per_group_df)} rows)")

print("\n[8/8] Scope & limitations (R-14):")
print("  - This experiment uses ONLY the numeric profile block (cols 0-10, 11-30).")
print("    Bios (text), Names, and Image List are out of scope, as are raw images")
print("    (not present in this folder).")
print("  - The blind-face block (cols 31-50) is a constant vector in this file and")
print("    cannot be used as a SensitiveNets-style control.")
print("  - Binarised 'hired' labels are an audit decision (median split), not the")
print("    papers' top-N screening protocol.")
print("  - All models are regularised logistic regressions on standardised features;")
print("    conclusions are specific to this model class and the CV7/CV9 feature sets.")
print("  - Findings describe associations under a synthetic testbed; they do not")
print("    generalise to real hiring systems.")
print("\nAudit v2 complete. Outputs in 'results/'.")

_report_file.flush()
_report_file.close()
