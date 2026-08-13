"""Audit core — the verified fairness/statistical math, extracted verbatim.

Every function in this module is copied UNCHANGED from the frozen audit
script ``faircv_audit_v2.py`` (lines ~113-340 and ~411-415). The frozen
script is never imported or modified; this module is the product-facing
copy of the verified math. The equivalence test
(``tests/test_equivalence.py``) proves that running this module with the
frozen FairCV configuration (``faircv_spec.py``) reproduces the frozen
result CSVs byte-for-byte.

Reference: ``faircv_audit_v2.py`` — R-1..R-15 implementation, seed 42,
n_boot 2000. Definitions are documented in ``BASELINE_MANIFEST.md`` §5.
"""

import numpy as np
from scipy.stats import ks_2samp, kruskal, chi2_contingency
from scipy.special import rel_entr
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Frozen configuration defaults (same as faircv_audit_v2.py lines 90-97)
RNG_SEED = 42
N_BOOT = 2000            # bootstrap resamples (R-13)
ALPHA = 0.05
EEOC_RULE = 0.80         # US EEOC four-fifths rule


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


def make_lr():
    """Model pipeline identical to the frozen audit (faircv_audit_v2.py L411)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=RNG_SEED)),
    ])
