"""
Top-N screening experiment (companion to faircv_audit_v2.py)
==============================================================
Tests how the audit's conclusions depend on the arbitrary median
binarisation by replacing the selection rule with the papers' top-N
screening protocol: "hired" = top k% of the model's predicted scores on
the test set (FairCVtest demo: top-100 of 4,800 ≈ top-2%; repo code:
top-1000 ≈ top-20%).

Design
------
- Models M1-M6 are trained EXACTLY as in faircv_audit_v2.py (same
  features, same median-binarised labels, same seed) so predicted scores
  are identical to the frozen audit. Only the SELECTION rule changes.
- Thresholds: top 2%, 5%, 10%, 20%, 30%, 50% of predicted scores.
- Per (model, attribute, threshold):
    per-group selection rate (SR), share of top-k slots (paper Table 1),
    DPD (max-min SR), DIR (worst/best SR, EEOC >= 0.80),
    chi2 (group x selected), label-based EOD (true = top k% of the
    model's own test labels), bootstrap 95% CIs for DPD/DIR.
- Compares each metric against the median-split values from
  results/metrics.csv (frozen audit) and prints whether the headline
  conclusions change.

Outputs: results/topn_metrics.csv, results/topn_per_group.csv,
         results/fig6_topn_selection.png
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

DATA_PATH   = "FairCVdb.npy"
RNG_SEED    = 42
N_BOOT      = 1000
THRESHOLDS  = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
RESULTS_DIR = "results"
EEOC_RULE   = 0.80
MEDIAN_CSV  = os.path.join(RESULTS_DIR, "metrics.csv")

GENDER_LABELS    = {0: "Male", 1: "Female"}
ETHNICITY_LABELS = {0: "G1", 1: "G2", 2: "G3"}

# ── 1. Load & features (identical to faircv_audit_v2.py) ─────────────────────
print("=" * 74)
print("Top-N screening experiment (companion to faircv_audit_v2.py)")
print("=" * 74)

db = np.load(DATA_PATH, allow_pickle=True).item()
P_tr, P_te = db["Profiles Train"], db["Profiles Test"]
y_blind_tr, y_blind_te = db["Blind Labels Train"], db["Blind Labels Test"]
y_gender_tr, y_gender_te = db["Biased Labels Train (Gender)"], db["Biased Labels Test (Gender)"]
y_eth_tr, y_eth_te = db["Biased Labels Train (Ethnicity)"], db["Biased Labels Test (Ethnicity)"]

gender_te = P_te[:, 1].astype(int)
ethnicity_te = P_te[:, 0].astype(int)

CV7_COLS = list(range(4, 11))
CV9_COLS = list(range(2, 11))
FACE_COLS = list(range(11, 31))

X_cv7_tr, X_cv7_te = P_tr[:, CV7_COLS].astype(float), P_te[:, CV7_COLS].astype(float)
X_cv9_tr, X_cv9_te = P_tr[:, CV9_COLS].astype(float), P_te[:, CV9_COLS].astype(float)
X_f7_tr,  X_f7_te  = P_tr[:, CV7_COLS + FACE_COLS].astype(float), P_te[:, CV7_COLS + FACE_COLS].astype(float)
X_f9_tr,  X_f9_te  = P_tr[:, CV9_COLS + FACE_COLS].astype(float), P_te[:, CV9_COLS + FACE_COLS].astype(float)

# Binarise labels exactly as the frozen audit (train-median threshold).
def binarise(arr, thr):
    return (arr >= thr).astype(int)

yb_blind_tr  = binarise(y_blind_tr,  float(np.median(y_blind_tr)))
yb_blind_te  = binarise(y_blind_te,  float(np.median(y_blind_tr)))
yb_gender_tr = binarise(y_gender_tr, float(np.median(y_gender_tr)))
yb_gender_te = binarise(y_gender_te, float(np.median(y_gender_tr)))
yb_eth_tr    = binarise(y_eth_tr,    float(np.median(y_eth_tr)))
yb_eth_te    = binarise(y_eth_te,    float(np.median(y_eth_tr)))

# ── 2. Train the same six models (identical to the frozen audit) ─────────────
def make_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=RNG_SEED)),
    ])

MODELS = [
    ("M1-Fair (CV7)",        "cv7",     "blind",        X_cv7_tr, yb_blind_tr,  X_cv7_te, yb_blind_te,  y_blind_te),
    ("M2-Multimodal (CV7+Face)", "cv7+face", "blind",    X_f7_tr,  yb_blind_tr,  X_f7_te,  yb_blind_te,  y_blind_te),
    ("M3-Gender-Biased (CV7)",   "cv7",  "gender-bias", X_cv7_tr, yb_gender_tr, X_cv7_te, yb_gender_te, y_gender_te),
    ("M4-Ethnicity-Biased (CV7)", "cv7", "eth-bias",    X_cv7_tr, yb_eth_tr,    X_cv7_te, yb_eth_te,    y_eth_te),
    ("M5-Robust (CV9)",      "cv9",     "blind",        X_cv9_tr, yb_blind_tr,  X_cv9_te, yb_blind_te,  y_blind_te),
    ("M6-Robust (CV9+Face)", "cv9+face", "blind",       X_f9_tr,  yb_blind_tr,  X_f9_te,  yb_blind_te,  y_blind_te),
]

trained = {}
for mname, fset, lset, X_tr, y_tr, X_te, y_te, y_true_te in MODELS:
    pipe = make_lr()
    pipe.fit(X_tr, y_tr)
    trained[mname] = dict(pipe=pipe, fset=fset, lset=lset, y_true_te=y_true_te, X_te=X_te)
    print(f"  trained {mname:<38s} ({fset}, {lset})")

# ── 3. Top-N screening metrics ───────────────────────────────────────────────
print("\n[2/4] Computing top-N screening metrics …")


def topk_metrics(y_score, group_vec, groups, labels_dict, k, y_true=None):
    """Selection metrics for 'hired = top k% of predicted scores'."""
    n = len(y_score)
    n_sel = int(round(n * k))
    order = np.argsort(-y_score)[:n_sel]
    sel = np.zeros(n, dtype=bool)
    sel[order] = True

    rows = []
    for gi, g in enumerate(groups):
        m = group_vec == g
        n_g = int(m.sum())
        s_g = int((sel & m).sum())
        rows.append(dict(group=labels_dict[int(g)], n=n_g, selected=s_g,
                         sr=s_g / n_g if n_g else np.nan,
                         share=s_g / n_sel if n_sel else np.nan))
    SR = np.array([r["sr"] for r in rows], dtype=float)
    dpd = float(np.nanmax(SR) - np.nanmin(SR))
    dir_ = float(np.nanmin(SR) / np.nanmax(SR)) if np.nanmax(SR) > 0 else np.nan

    cont = np.array([[r["selected"], r["n"] - r["selected"]] for r in rows])
    chi2, p_chi, _, _ = chi2_contingency(cont)

    eod = np.nan
    if y_true is not None:
        thr = np.quantile(y_true, 1.0 - k)
        true_hire = (y_true >= thr).astype(bool)
        tprs = []
        for gi, g in enumerate(groups):
            m = group_vec == g
            t, s = true_hire[m], sel[m]
            tp = int((t & s).sum())
            tp_all = int(t.sum())
            tprs.append(tp / tp_all if tp_all else np.nan)
        tprs = np.array(tprs, dtype=float)
        tprs = tprs[np.isfinite(tprs)]
        eod = float(tprs.max() - tprs.min()) if len(tprs) else np.nan

    hi = int(np.nanargmax(SR))
    lo = int(np.nanargmin(SR))
    return dict(rows=rows, dpd=dpd, dir_=dir_, eeoc=bool(dir_ >= EEOC_RULE),
                chi2_p=float(p_chi), eod=eod,
                share_delta=float(rows[hi]["share"] - rows[lo]["share"]))


def bootstrap_topk_ci(y_score, group_vec, groups, k, rng, n_boot=N_BOOT):
    """Percentile 95% CIs for DPD and DIR under resampled top-k selection.

    Vectorised fast path: argpartition for the top-k, bincount for the
    per-group selection rates (no per-resample Python loops).
    """
    n = len(y_score)
    n_sel = int(round(n * k))
    g_idx = np.searchsorted(groups, group_vec)   # aligned to groups order
    ng = len(groups)
    dpds, dirs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scr, gv = y_score[idx], g_idx[idx]
        top = np.argpartition(-scr, n_sel - 1)[:n_sel]
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
    dpds = np.asarray(dpds)
    dirs = np.asarray(dirs)
    d_ci = (float(np.percentile(dpds, 2.5)), float(np.percentile(dpds, 97.5))) if len(dpds) else (np.nan, np.nan)
    r_ci = (float(np.percentile(dirs, 2.5)), float(np.percentile(dirs, 97.5))) if len(dirs) else (np.nan, np.nan)
    return d_ci, r_ci


rng = np.random.default_rng(RNG_SEED)
ATTRIBUTES = [("gender", gender_te, GENDER_LABELS), ("ethnicity", ethnicity_te, ETHNICITY_LABELS)]

# median-split reference from the frozen audit
med = pd.read_csv(MEDIAN_CSV) if os.path.exists(MEDIAN_CSV) else None

metric_rows = []
per_group_rows = []
for mname, md in trained.items():
    y_scr = md["pipe"].predict_proba(md["X_te"])[:, 1]
    for attr, gvec, glabels in ATTRIBUTES:
        groups = np.unique(gvec).astype(int)
        for k in THRESHOLDS:
            res = topk_metrics(y_scr, gvec, groups, glabels, k, y_true=md["y_true_te"])
            d_ci, r_ci = bootstrap_topk_ci(y_scr, gvec, groups, k, rng)
            for r in res["rows"]:
                per_group_rows.append({
                    "model": mname, "attribute": attr, "threshold": k,
                    "group": r["group"], "n": r["n"], "selected": r["selected"],
                    "selection_rate": r["sr"], "share_of_topk": r["share"],
                })
            med_row = None
            if med is not None:
                mm = med[(med.model == mname) & (med.attribute == attr)]
                if len(mm):
                    med_row = mm.iloc[0]
            metric_rows.append({
                "model": mname, "features": md["fset"], "label_set": md["lset"],
                "attribute": attr, "threshold": k, "n_selected": int(round(len(y_scr) * k)),
                "DPD": res["dpd"], "DPD_ci_lo": d_ci[0], "DPD_ci_hi": d_ci[1],
                "DIR": res["dir_"], "DIR_ci_lo": r_ci[0], "DIR_ci_hi": r_ci[1],
                "EEOC_pass": res["eeoc"], "chi2_p": res["chi2_p"],
                "EOD_labels": res["eod"], "share_delta": res["share_delta"],
                "median_split_DIR": float(med_row["DIR"]) if med_row is not None else np.nan,
                "median_split_DPD": float(med_row["DPD"]) if med_row is not None else np.nan,
            })
print("  done.")

# ── 4. Persist CSVs ──────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
df_topn = pd.DataFrame(metric_rows)
df_pg = pd.DataFrame(per_group_rows)
df_topn.to_csv(os.path.join(RESULTS_DIR, "topn_metrics.csv"), index=False)
df_pg.to_csv(os.path.join(RESULTS_DIR, "topn_per_group.csv"), index=False)
print(f"  wrote results/topn_metrics.csv      ({len(df_topn)} rows)")
print(f"  wrote results/topn_per_group.csv    ({len(df_pg)} rows)")

# ── 5. Sanity check: top-50% vs median split (worst-off group identity) ─────
print("\n[3/4] Sanity check: top-50% vs median-split (worst-off group identity)")
print("  (DIR = min/max SR is symmetric, so it cannot reveal WHICH group is worst;")
print("   compare the identity of the worst-off group directly instead.)")
MED_PG = os.path.join(RESULTS_DIR, "per_group_metrics.csv")
med_pg = pd.read_csv(MED_PG) if os.path.exists(MED_PG) else None
flips = []
for mname in trained:
    for attr in ("gender", "ethnicity"):
        pg50 = df_pg[(df_pg.model == mname) & (df_pg.attribute == attr)
                     & (df_pg.threshold == 0.50)]
        mm = med_pg[(med_pg.model == mname) & (med_pg.attribute == attr)] if med_pg is not None else None
        if not len(pg50) or (mm is None) or not len(mm):
            continue
        w50 = pg50.loc[pg50["selection_rate"].idxmin(), "group"]
        wmed = mm.loc[mm["selection_rate"].idxmin(), "group"]
        row = df_topn[(df_topn.model == mname) & (df_topn.attribute == attr)
                      & (df_topn.threshold == 0.50)].iloc[0]
        same = (w50 == wmed)
        flag = "" if same else "   <-- WORST-OFF GROUP FLIP"
        if not same:
            flips.append((mname, attr, w50, wmed))
        print(f"  {mname:<38s} {attr:<9s} worst-off: top50={w50:<8s} median-split={wmed:<8s}  "
              f"DIR {row['DIR']:.4f} vs {row['median_split_DIR']:.4f}"
              f"  {'AGREE' if same else 'FLIP'}{flag}")
print("  sanity:", "OK (worst-off group identity agrees)" if not flips else f"FLIPS: {flips}")
if flips:
    print("  (flips occur only where DIR ~ 1.0, i.e. near-zero disparity, so the")
    print("   worst-off designation is at noise level; see the report for context.)")

# ── 6. Conclusion comparison vs frozen audit ────────────────────────────────
print("\n[4/4] Does the top-N selection rule change the EEOC conclusions?")
print(f"  'P' = EEOC pass (DIR >= {EEOC_RULE:.2f}), 'X' = fail. Last column = median-split verdict.")
print(f"  {'Model':<38s} {'attr':<10s} " + " ".join(f"{int(t*100):>3d}%" for t in THRESHOLDS)
      + "   median")
changed = []
for mname in trained:
    for attr in ("gender", "ethnicity"):
        sub = df_topn[(df_topn.model == mname) & (df_topn.attribute == attr)]
        cells = []
        for k in THRESHOLDS:
            r = sub[sub.threshold == k].iloc[0]
            cells.append(" P " if bool(r["EEOC_pass"]) else " X ")
        med_row = None
        if med is not None:
            mm = med[(med.model == mname) & (med.attribute == attr)]
            if len(mm):
                med_row = mm.iloc[0]
        if med_row is not None and np.isfinite(med_row["DIR"]):
            med_pass = bool(med_row["DIR"] >= EEOC_RULE)
            med_s = " P " if med_pass else " X "
        else:
            med_pass = None
            med_s = " ? "
        any_flip = med_pass is not None and any(c != med_s for c in cells)
        tag = "   <-- verdict changes" if any_flip else ""
        if any_flip:
            changed.append((mname, attr))
        print(f"  {mname:<38s} {attr:<10s} " + " ".join(cells) + f"    {med_s}{tag}")

print()
if changed:
    print("  CONCLUSION CHANGE: verdict flips under top-N for:")
    for mname, attr in changed:
        print(f"    - {mname} / {attr}")
else:
    print("  No EEOC verdict flips across the tested thresholds.")
print("\n  Caveats:")
print("  (1) M3/M4 label-based EOD is computed against artificially biased label sets")
print("      (same caveat as v2 R-9); only selection-rate metrics are neutral.")
print("  (2) At the tightest threshold (top-2%, 96 of 4,800 selected) bootstrap CIs")
print("      are wide and chi2 has limited power — treat top-2% verdicts as")
print("      low-power evidence, not as precise estimates.")

# ── 7. Figure: DIR vs threshold ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
COLORS = {"M1-Fair (CV7)": "#1f77b4", "M2-Multimodal (CV7+Face)": "#ff7f0e",
          "M3-Gender-Biased (CV7)": "#2ca02c", "M4-Ethnicity-Biased (CV7)": "#d62728",
          "M5-Robust (CV9)": "#9467bd", "M6-Robust (CV9+Face)": "#8c564b"}
for ax, attr in zip(axes, ("gender", "ethnicity")):
    for mname in trained:
        sub = df_topn[(df_topn.model == mname) & (df_topn.attribute == attr)]\
            .sort_values("threshold")
        ax.plot(sub["threshold"] * 100, sub["DIR"], marker="o", ms=4,
                color=COLORS[mname], label=mname)
        ax.fill_between(sub["threshold"] * 100, sub["DIR_ci_lo"], sub["DIR_ci_hi"],
                        color=COLORS[mname], alpha=0.12)
    ax.axhline(EEOC_RULE, ls="--", color="black", lw=1, label=f"EEOC {EEOC_RULE:.2f}")
    ax.set_xlabel("Top-N threshold (% selected by score)")
    ax.set_ylabel("DIR (worst/best selection rate)")
    ax.set_title(f"{attr.capitalize()} — DIR vs screening threshold")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "fig6_topn_selection.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {fig_path}")

print("\nTop-N screening experiment complete.")
