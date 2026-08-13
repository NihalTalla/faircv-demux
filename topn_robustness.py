"""
Top-N robustness analysis — repository selection protocol (companion script)
============================================================================
Re-tests the frozen audit conclusions (faircv_audit_v2.py) under the hiring
rule DOCUMENTED in the official FairCVtest repository (`FairCV.py`, BiDAlab/
FairCVtest) instead of the audit's invented median split:

  Protocol A — TOP-1000 SELECTION  (FairCV.py `computeTopScore` /
              `testDemographicParity`): "hired" = the 1000 highest predicted
              scores out of the 4,800 test profiles (top ~20.8%).
  Protocol B — 75TH-PERCENTILE THRESHOLD (FairCV.py
              `testEqualityOfOpportunity(..., p=75)`): threshold = 75th
              percentile of the training labels; "qualified" = label >=
              threshold, "predicted qualified" = predicted score >= the same
              threshold. We report both the resulting SELECTION rates (score
              >= threshold) and the repo's equality-of-opportunity TPR test.

Design
------
- M1-M6 are trained EXACTLY as in faircv_audit_v2.py (same features, same
  median-binarised training labels, same seed 42), so predicted scores are
  identical to the frozen audit. Only the SELECTION rule changes.
- Bootstrap 95% CIs (percentile method, seed 42, n = 2,000, matching the
  frozen audit's N_BOOT) for DPD / DIR / EOD under each protocol.
- For Protocol B the 75th-percentile threshold is taken from each model's OWN
  training label set (blind for M1/M2/M5/M6, gender-biased for M3,
  ethnicity-biased for M4), mirroring the frozen audit's per-label-set
  binarisation so M3/M4 stay directly comparable (R-9 caveat applies).
- Every metric is compared against the frozen median-split results in
  results/metrics.csv and results/per_group_metrics.csv; verdict flips and
  worst-off-group changes are flagged.

Frozen-audit validation built in
--------------------------------
Before trusting the new rules, the script recomputes the median-split metrics
(hired = predicted probability >= 0.5) from the freshly trained models and
compares them EXACTLY against results/metrics.csv / per_group_metrics.csv.
Any drift would mean the models differ from the frozen ones.

Outputs (results/robustness/ — does NOT touch results/metrics.csv or any
other frozen artifact)
  top1000_metrics.csv, top1000_per_group.csv
  p75_metrics.csv,     p75_per_group.csv
  fig9_robustness_dir.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import chi2_contingency

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

# ── Configuration (mirrors faircv_audit_v2.py) ───────────────────────────────
DATA_PATH   = "FairCVdb.npy"
RNG_SEED    = 42
N_BOOT      = 2000            # same as the frozen audit
ALPHA       = 0.05
EEOC_RULE   = 0.80            # US EEOC four-fifths rule
RESULTS_DIR = "results"
ROB_DIR     = os.path.join(RESULTS_DIR, "robustness")
N_TOP       = 1000            # repo protocol A: top-1000 of 4,800 test scores
P_THR       = 75              # repo protocol B: 75th-percentile threshold

FROZEN_METRICS = os.path.join(RESULTS_DIR, "metrics.csv")
FROZEN_PG      = os.path.join(RESULTS_DIR, "per_group_metrics.csv")

GENDER_LABELS    = {0: "Male", 1: "Female"}
ETHNICITY_LABELS = {0: "G1", 1: "G2", 2: "G3"}

CV7_COLS  = list(range(4, 11))
CV9_COLS  = list(range(2, 11))
FACE_COLS = list(range(11, 31))

os.makedirs(ROB_DIR, exist_ok=True)

print("=" * 74)
print("Top-N robustness analysis — repo selection protocol (top-1000, p75)")
print("=" * 74)

# ── 1. Load data & build features/labels EXACTLY as the frozen audit ────────
db = np.load(DATA_PATH, allow_pickle=True).item()
P_tr, P_te = db["Profiles Train"], db["Profiles Test"]
y_blind_tr, y_blind_te = db["Blind Labels Train"], db["Blind Labels Test"]
y_gender_tr, y_gender_te = db["Biased Labels Train (Gender)"], db["Biased Labels Test (Gender)"]
y_eth_tr, y_eth_te = db["Biased Labels Train (Ethnicity)"], db["Biased Labels Test (Ethnicity)"]

gender_te = P_te[:, 1].astype(int)
ethnicity_te = P_te[:, 0].astype(int)

X_cv7_tr, X_cv7_te = P_tr[:, CV7_COLS].astype(float), P_te[:, CV7_COLS].astype(float)
X_cv9_tr, X_cv9_te = P_tr[:, CV9_COLS].astype(float), P_te[:, CV9_COLS].astype(float)
X_f7_tr,  X_f7_te  = P_tr[:, CV7_COLS + FACE_COLS].astype(float), P_te[:, CV7_COLS + FACE_COLS].astype(float)
X_f9_tr,  X_f9_te  = P_tr[:, CV9_COLS + FACE_COLS].astype(float), P_te[:, CV9_COLS + FACE_COLS].astype(float)


def binarise(arr, thr):
    return (arr >= thr).astype(int)


BLIND_THR   = float(np.median(y_blind_tr))
GENDER_THR  = float(np.median(y_gender_tr))
ETH_THR     = float(np.median(y_eth_tr))

yb_blind_tr  = binarise(y_blind_tr,  BLIND_THR)
yb_blind_te  = binarise(y_blind_te,  BLIND_THR)
yb_gender_tr = binarise(y_gender_tr, GENDER_THR)
yb_gender_te = binarise(y_gender_te, GENDER_THR)
yb_eth_tr    = binarise(y_eth_tr,    ETH_THR)
yb_eth_te    = binarise(y_eth_te,    ETH_THR)


# ── 2. Train the same six models (identical to the frozen audit) ─────────────
def make_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=RNG_SEED)),
    ])


# (name, features-tag, label-tag, X_tr, y_tr, X_te, y_te, y_true_te)
MODELS = [
    ("M1-Fair (CV7)",          "cv7",     "blind",        X_cv7_tr, yb_blind_tr,  X_cv7_te, yb_blind_te,  y_blind_te),
    ("M2-Multimodal (CV7+Face)", "cv7+face", "blind",     X_f7_tr,  yb_blind_tr,  X_f7_te,  yb_blind_te,  y_blind_te),
    ("M3-Gender-Biased (CV7)", "cv7",     "gender-bias",  X_cv7_tr, yb_gender_tr, X_cv7_te, yb_gender_te, y_gender_te),
    ("M4-Ethnicity-Biased (CV7)", "cv7",  "eth-bias",     X_cv7_tr, yb_eth_tr,    X_cv7_te, yb_eth_te,    y_eth_te),
    ("M5-Robust (CV9)",        "cv9",     "blind",        X_cv9_tr, yb_blind_tr,  X_cv9_te, yb_blind_te,  y_blind_te),
    ("M6-Robust (CV9+Face)",   "cv9+face", "blind",       X_f9_tr,  yb_blind_tr,  X_f9_te,  yb_blind_te,  y_blind_te),
]

trained = {}
for mname, fset, lset, X_tr, y_tr, X_te, y_te, y_true_te in MODELS:
    pipe = make_lr()
    pipe.fit(X_tr, y_tr)
    trained[mname] = dict(pipe=pipe, fset=fset, lset=lset, y_true_te=y_true_te,
                          y_te=y_te, X_te=X_te)
    print(f"  trained {mname:<38s} ({fset}, {lset})")

rng = np.random.default_rng(RNG_SEED)

# ── 3. Frozen-median reproduction check (models must be identical) ───────────
print("\n[1/5] Validation: recompute median-split metrics vs frozen results …")

med = pd.read_csv(FROZEN_METRICS)
med_pg = pd.read_csv(FROZEN_PG)
max_drift = 0.0
for mname, md in trained.items():
    y_pred = md["pipe"].predict(md["X_te"])
    y_scr = md["pipe"].predict_proba(md["X_te"])[:, 1]
    for attr, gvec in (("gender", gender_te), ("ethnicity", ethnicity_te)):
        groups = np.unique(gvec).astype(int)
        SR = np.array([y_pred[gvec == g].mean() for g in groups])
        row = med[(med.model == mname) & (med.attribute == attr)].iloc[0]
        max_drift = max(max_drift,
                        abs(SR.max() - SR.min() - row["DPD"]),
                        abs(SR.min() / SR.max() - row["DIR"]))
print(f"  max |drift| vs frozen metrics.csv (DPD/DIR): {max_drift:.3e}")
if max_drift < 1e-9:
    print("  OK — models reproduce the frozen audit exactly; scores are identical.")
else:
    print("  WARNING — drift detected; results below are NOT comparable to the frozen audit.")
    sys.exit(1)

# ── 4. Metric helpers ────────────────────────────────────────────────────────
def group_rows(group_vec, groups, labels_dict, sel):
    """Per-group n / selected / selection rate / share of selected."""
    n_sel = int(sel.sum())
    rows = []
    for g in groups:
        m = group_vec == g
        n_g = int(m.sum())
        s_g = int((sel & m).sum())
        rows.append(dict(group=labels_dict[int(g)], n=n_g, selected=s_g,
                         selection_rate=s_g / n_g if n_g else np.nan,
                         share=s_g / n_sel if n_sel else np.nan))
    return rows


def agg_metrics(rows, groups, group_vec, sel):
    """DPD / DIR / EEOC / chi2 / worst-best groups from selection rows."""
    SR = np.array([r["selection_rate"] for r in rows], dtype=float)
    dpd = float(np.nanmax(SR) - np.nanmin(SR))
    dir_ = float(np.nanmin(SR) / np.nanmax(SR)) if np.nanmax(SR) > 0 else np.nan
    cont = np.array([[r["selected"], r["n"] - r["selected"]] for r in rows])
    chi2_p = float(chi2_contingency(cont)[1])
    hi = int(np.nanargmax(SR))
    lo = int(np.nanargmin(SR))
    return dict(dpd=dpd, dir_=dir_, eeoc=bool(dir_ >= EEOC_RULE), chi2_p=chi2_p,
                hi=hi, lo=lo)


def min_share_ratio(rows):
    """Repo testDemographicParity 'P-value': min over pairs of min(share_i/share_j, share_j/share_i)."""
    shares = np.array([r["share"] for r in rows], dtype=float)
    best = np.inf
    for i in range(len(shares)):
        for j in range(i + 1, len(shares)):
            if shares[i] > 0 and shares[j] > 0:
                best = min(best, shares[i] / shares[j], shares[j] / shares[i])
    return float(best) if np.isfinite(best) else np.nan


def bootstrap_sel_ci(y_score, group_vec, groups, n_top, rng, n_boot=N_BOOT):
    """Percentile 95% CIs for DPD/DIR under resampled 'hired = top n_top scores'."""
    n = len(y_score)
    g_idx = np.searchsorted(groups, group_vec)
    ng = len(groups)
    dpds, dirs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scr, gv = y_score[idx], g_idx[idx]
        top = np.argpartition(-scr, n_top - 1)[:n_top]
        sel = np.zeros(n, dtype=bool)
        sel[top] = True
        n_g = np.bincount(gv, minlength=ng).astype(float)
        s_g = np.bincount(gv, weights=sel.astype(float), minlength=ng)
        SR = np.divide(s_g, n_g, out=np.full(ng, np.nan), where=n_g > 0)
        mx, mn = np.nanmax(SR), np.nanmin(SR)
        if np.isfinite(mx) and np.isfinite(mn):
            dpds.append(mx - mn)
            if mx > 0:
                dirs.append(mn / mx)
    dpds, dirs = np.asarray(dpds), np.asarray(dirs)
    d_ci = (float(np.percentile(dpds, 2.5)), float(np.percentile(dpds, 97.5))) if len(dpds) else (np.nan, np.nan)
    r_ci = (float(np.percentile(dirs, 2.5)), float(np.percentile(dirs, 97.5))) if len(dirs) else (np.nan, np.nan)
    return d_ci, r_ci


def bootstrap_thr_sel_ci(y_score, group_vec, groups, thr, rng, n_boot=N_BOOT):
    """Percentile 95% CIs for DPD/DIR under fixed-threshold hiring (score >= thr)."""
    n = len(y_score)
    g_idx = np.searchsorted(groups, group_vec)
    ng = len(groups)
    sel0 = (y_score >= thr)
    dpds, dirs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        gv = g_idx[idx]
        s = sel0[idx]
        n_g = np.bincount(gv, minlength=ng).astype(float)
        s_g = np.bincount(gv, weights=s.astype(float), minlength=ng)
        SR = np.divide(s_g, n_g, out=np.full(ng, np.nan), where=n_g > 0)
        mx, mn = np.nanmax(SR), np.nanmin(SR)
        if np.isfinite(mx) and np.isfinite(mn):
            dpds.append(mx - mn)
            if mx > 0:
                dirs.append(mn / mx)
    dpds, dirs = np.asarray(dpds), np.asarray(dirs)
    d_ci = (float(np.percentile(dpds, 2.5)), float(np.percentile(dpds, 97.5))) if len(dpds) else (np.nan, np.nan)
    r_ci = (float(np.percentile(dirs, 2.5)), float(np.percentile(dirs, 97.5))) if len(dirs) else (np.nan, np.nan)
    return d_ci, r_ci


def bootstrap_eod_ci(y_score, y_true, group_vec, groups, thr, rng, n_boot=N_BOOT):
    """Percentile 95% CI for EOD (max-min TPR) under the repo EEO test (fixed threshold)."""
    n = len(y_score)
    g_idx = np.searchsorted(groups, group_vec)
    ng = len(groups)
    pred_q = (y_score >= thr)
    true_q = (y_true >= thr)
    eods = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        gv = g_idx[idx]
        p, t = pred_q[idx], true_q[idx]
        tpr = np.full(ng, np.nan)
        for gi in range(ng):
            m = gv == gi
            tp_all = int((t & m).sum())
            if tp_all > 0:
                tpr[gi] = int((t & p & m).sum()) / tp_all
        tf = tpr[np.isfinite(tpr)]
        if len(tf):
            eods.append(float(tf.max() - tf.min()))
    eods = np.asarray(eods)
    return (float(np.percentile(eods, 2.5)), float(np.percentile(eods, 97.5))) if len(eods) else (np.nan, np.nan)


ATTRIBUTES = [("gender", gender_te, GENDER_LABELS), ("ethnicity", ethnicity_te, ETHNICITY_LABELS)]

# ── 5. Protocol A: top-1000 selection ────────────────────────────────────────
print("\n[2/5] Protocol A — top-1000 selection (repo computeTopScore / testDemographicParity) …")

t1_rows, t1_pg_rows = [], []
for mname, md in trained.items():
    y_scr = md["pipe"].predict_proba(md["X_te"])[:, 1]
    order = np.argsort(-y_scr)[:N_TOP]
    sel = np.zeros(len(y_scr), dtype=bool)
    sel[order] = True
    for attr, gvec, glabels in ATTRIBUTES:
        groups = np.unique(gvec).astype(int)
        rows = group_rows(gvec, groups, glabels, sel)
        agg = agg_metrics(rows, groups, gvec, sel)
        d_ci, r_ci = bootstrap_sel_ci(y_scr, gvec, groups, N_TOP, rng)
        med_row = med[(med.model == mname) & (med.attribute == attr)].iloc[0]
        # worst-off group under median split from per-group CSV
        mpg = med_pg[(med_pg.model == mname) & (med_pg.attribute == attr)]
        med_worst_grp = mpg.loc[mpg["selection_rate"].idxmin(), "group"]
        best = glabels[int(groups[agg["hi"]])]
        worst = glabels[int(groups[agg["lo"]])]
        verdict_flip = agg["eeoc"] != bool(med_row["EEOC_pass"])
        t1_rows.append({
            "model": mname, "features": md["fset"], "label_set": md["lset"],
            "attribute": attr, "protocol": "top1000", "n_selected": N_TOP,
            "DPD": agg["dpd"], "DPD_ci_lo": d_ci[0], "DPD_ci_hi": d_ci[1],
            "DIR": agg["dir_"], "DIR_ci_lo": r_ci[0], "DIR_ci_hi": r_ci[1],
            "EEOC_pass": agg["eeoc"], "chi2_p": agg["chi2_p"],
            "repo_pvalue": min_share_ratio(rows),
            "best_group": best, "worst_group": worst,
            "median_worst_group": med_worst_grp,
            "worst_group_flip": worst != med_worst_grp,
            "median_split_DPD": float(med_row["DPD"]),
            "median_split_DPD_ci_lo": float(med_row["DPD_ci_lo"]),
            "median_split_DPD_ci_hi": float(med_row["DPD_ci_hi"]),
            "median_split_DIR": float(med_row["DIR"]),
            "median_split_DIR_ci_lo": float(med_row["DIR_ci_lo"]),
            "median_split_DIR_ci_hi": float(med_row["DIR_ci_hi"]),
            "median_split_EEOC": bool(med_row["EEOC_pass"]),
            "verdict_change": verdict_flip,
        })
        for r in rows:
            mpg_r = mpg[mpg.group == r["group"]]
            t1_pg_rows.append({
                "model": mname, "attribute": attr, "protocol": "top1000",
                "group": r["group"], "n": r["n"], "selected": r["selected"],
                "selection_rate": r["selection_rate"], "share_of_top1000": r["share"],
                "median_selection_rate": float(mpg_r["selection_rate"].iloc[0]) if len(mpg_r) else np.nan,
            })
print("  done.")

# ── 6. Protocol B: 75th-percentile threshold (repo testEqualityOfOpportunity) ─
print("\n[3/5] Protocol B — 75th-percentile threshold (repo testEqualityOfOpportunity, p=75) …")

p75_rows, p75_pg_rows = [], []
for mname, md in trained.items():
    # model's OWN training label set, mirroring the frozen audit's binarisation
    label_tr = {"blind": y_blind_tr, "gender-bias": y_gender_tr, "eth-bias": y_eth_tr}[md["lset"]]
    thr = float(np.percentile(label_tr, P_THR))
    y_scr = md["pipe"].predict_proba(md["X_te"])[:, 1]
    sel = (y_scr >= thr)
    true_q = (md["y_true_te"] >= thr)
    for attr, gvec, glabels in ATTRIBUTES:
        groups = np.unique(gvec).astype(int)
        rows = group_rows(gvec, groups, glabels, sel)
        agg = agg_metrics(rows, groups, gvec, sel)
        d_ci, r_ci = bootstrap_thr_sel_ci(y_scr, gvec, groups, thr, rng)
        # repo EEO: per-group TPR
        tpr = []
        for g in groups:
            m = gvec == g
            tp_all = int((true_q & m).sum())
            tpr.append(float((true_q & sel & m).sum()) / tp_all if tp_all else np.nan)
        tpr = np.array(tpr, dtype=float)
        eod = float(np.nanmax(tpr) - np.nanmin(tpr)) if np.any(np.isfinite(tpr)) else np.nan
        eod_ci = bootstrap_eod_ci(y_scr, md["y_true_te"], gvec, groups, thr, rng)
        med_row = med[(med.model == mname) & (med.attribute == attr)].iloc[0]
        mpg = med_pg[(med_pg.model == mname) & (med_pg.attribute == attr)]
        med_worst_grp = mpg.loc[mpg["selection_rate"].idxmin(), "group"]
        worst = glabels[int(groups[agg["lo"]])]
        best = glabels[int(groups[agg["hi"]])]
        verdict_flip = agg["eeoc"] != bool(med_row["EEOC_pass"])
        p75_rows.append({
            "model": mname, "features": md["fset"], "label_set": md["lset"],
            "attribute": attr, "protocol": "p75", "threshold": thr,
            "n_selected": int(sel.sum()),
            "DPD": agg["dpd"], "DPD_ci_lo": d_ci[0], "DPD_ci_hi": d_ci[1],
            "DIR": agg["dir_"], "DIR_ci_lo": r_ci[0], "DIR_ci_hi": r_ci[1],
            "EEOC_pass": agg["eeoc"], "chi2_p": agg["chi2_p"],
            "EOD": eod, "EOD_ci_lo": eod_ci[0], "EOD_ci_hi": eod_ci[1],
            "best_group": best,
            "worst_group": worst,
            "median_worst_group": med_worst_grp,
            "worst_group_flip": worst != med_worst_grp,
            "median_split_DPD": float(med_row["DPD"]),
            "median_split_DPD_ci_lo": float(med_row["DPD_ci_lo"]),
            "median_split_DPD_ci_hi": float(med_row["DPD_ci_hi"]),
            "median_split_DIR": float(med_row["DIR"]),
            "median_split_DIR_ci_lo": float(med_row["DIR_ci_lo"]),
            "median_split_DIR_ci_hi": float(med_row["DIR_ci_hi"]),
            "median_split_EOD": float(med_row["EOD"]),
            "median_split_EOD_ci_lo": float(med_row["EOD_ci_lo"]),
            "median_split_EOD_ci_hi": float(med_row["EOD_ci_hi"]),
            "median_split_EEOC": bool(med_row["EEOC_pass"]),
            "verdict_change": verdict_flip,
        })
        for ri, r in enumerate(rows):
            mpg_r = mpg[mpg.group == r["group"]]
            tprv = tpr[ri]
            p75_pg_rows.append({
                "model": mname, "attribute": attr, "protocol": "p75",
                "group": r["group"], "n": r["n"], "selected": r["selected"],
                "selection_rate": r["selection_rate"],
                "tpr": float(tprv) if np.isfinite(tprv) else np.nan,
                "median_selection_rate": float(mpg_r["selection_rate"].iloc[0]) if len(mpg_r) else np.nan,
                "median_tpr": float(mpg_r["tpr"].iloc[0]) if len(mpg_r) else np.nan,
            })
print("  done.")

# ── 7. Persist CSVs under results/robustness/ ────────────────────────────────
print("\n[4/5] Writing results/robustness/*.csv …")
df_t1 = pd.DataFrame(t1_rows)
df_t1_pg = pd.DataFrame(t1_pg_rows)
df_p75 = pd.DataFrame(p75_rows)
df_p75_pg = pd.DataFrame(p75_pg_rows)
df_t1.to_csv(os.path.join(ROB_DIR, "top1000_metrics.csv"), index=False)
df_t1_pg.to_csv(os.path.join(ROB_DIR, "top1000_per_group.csv"), index=False)
df_p75.to_csv(os.path.join(ROB_DIR, "p75_metrics.csv"), index=False)
df_p75_pg.to_csv(os.path.join(ROB_DIR, "p75_per_group.csv"), index=False)
print(f"  wrote {ROB_DIR}/top1000_metrics.csv     ({len(df_t1)} rows)")
print(f"  wrote {ROB_DIR}/top1000_per_group.csv   ({len(df_t1_pg)} rows)")
print(f"  wrote {ROB_DIR}/p75_metrics.csv         ({len(df_p75)} rows)")
print(f"  wrote {ROB_DIR}/p75_per_group.csv       ({len(df_p75_pg)} rows)")

# ── 8. Summary: DIR + EEOC verdicts vs the frozen median split ───────────────
print("\n[5/5] EEOC verdict comparison (P = pass DIR >= 0.80, X = fail):")
print(f"  {'Model':<38s} {'attr':<10s} {'median':>7s} {'top1000':>8s} {'p75':>5s}  notes")
changes = []
for mname in trained:
    for attr in ("gender", "ethnicity"):
        t1 = df_t1[(df_t1.model == mname) & (df_t1.attribute == attr)].iloc[0]
        p75 = df_p75[(df_p75.model == mname) & (df_p75.attribute == attr)].iloc[0]
        m_s = " P " if t1["median_split_EEOC"] else " X "
        a_s = " P " if t1["EEOC_pass"] else " X "
        b_s = " P " if p75["EEOC_pass"] else " X "
        notes = []
        if t1["verdict_change"]:
            notes.append("VERDICT FLIP (top1000)")
            changes.append((mname, attr, "top1000"))
        if p75["verdict_change"]:
            notes.append("VERDICT FLIP (p75)")
            changes.append((mname, attr, "p75"))
        if t1["worst_group_flip"]:
            notes.append(f"worst-off {t1['worst_group']} vs {t1['median_worst_group']} (top1000)")
        print(f"  {mname:<38s} {attr:<10s} {m_s:>7s} {a_s:>8s} {b_s:>5s}  {'; '.join(notes)}")

print()
if changes:
    print("  Verdict changes under the repo protocol:")
    for c in changes:
        print(f"    - {c[0]} / {c[1]}  ({c[2]})")
else:
    print("  No EEOC verdict changes under top-1000 or p75 selection.")
print("  (Full numbers incl. bootstrap 95% CIs in results/robustness/*.csv)")

# ── 9. Figure: DIR under the three hiring rules with 95% CIs ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(trained))
COLORS = {"M1-Fair (CV7)": "#1f77b4", "M2-Multimodal (CV7+Face)": "#ff7f0e",
          "M3-Gender-Biased (CV7)": "#2ca02c", "M4-Ethnicity-Biased (CV7)": "#d62728",
          "M5-Robust (CV9)": "#9467bd", "M6-Robust (CV9+Face)": "#8c564b"}
for ax, attr in zip(axes, ("gender", "ethnicity")):
    for mi, mname in enumerate(trained):
        med_row = med[(med.model == mname) & (med.attribute == attr)].iloc[0]
        t1 = df_t1[(df_t1.model == mname) & (df_t1.attribute == attr)].iloc[0]
        p75 = df_p75[(df_p75.model == mname) & (df_p75.attribute == attr)].iloc[0]
        col = COLORS[mname]
        ax.errorbar([mi - 0.22, mi, mi + 0.22],
                    [med_row["DIR"], t1["DIR"], p75["DIR"]],
                    yerr=[[med_row["DIR"] - med_row["DIR_ci_lo"], t1["DIR"] - t1["DIR_ci_lo"], p75["DIR"] - p75["DIR_ci_lo"]],
                          [med_row["DIR_ci_hi"] - med_row["DIR"], t1["DIR_ci_hi"] - t1["DIR"], p75["DIR_ci_hi"] - p75["DIR"]]],
                    fmt="o", ms=5, capsize=3, color=col, label=mname)
    ax.axhline(EEOC_RULE, ls="--", color="black", lw=1)
    ax.axhline(1.0, color="gray", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split(" (")[0] for m in trained], rotation=20, fontsize=8)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("DIR (worst/best selection rate)")
    ax.set_title(f"{attr.capitalize()} — DIR by hiring rule (median / top-1000 / p75)")
    ax.grid(alpha=0.3)
    if attr == "gender":
        ax.legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig_path = os.path.join(ROB_DIR, "fig9_robustness_dir.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {fig_path}")

print("\nTop-N robustness analysis complete. Results in results/robustness/.")
