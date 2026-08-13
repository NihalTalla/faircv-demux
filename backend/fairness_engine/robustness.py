"""Robustness protocols for the product layer.

Replicates the frozen repository-protocol analysis (``topn_robustness.py``,
``topn_screening.py``) so that demo-mode robustness numbers match the frozen
``results/robustness/*.csv`` artifacts:

  Protocol A — TOP-N selection: "hired" = the N highest predicted scores.
  Protocol B — P75 threshold: threshold = 75th percentile of the training
               labels; "qualified" = label >= threshold, "predicted qualified"
               = predicted score >= the same threshold.

Both protocols report selection-rate disparity (DPD/DIR + EEOC verdict) with
percentile bootstrap 95% CIs (n_boot resamples, seeded), exactly as the frozen
scripts do. ``compute_topn`` / ``compute_p75`` are the two protocol
implementations; callers assemble the "median" rows from the primary metrics.

These functions accept *freshly trained* models — they never touch the frozen
results or re-run the frozen scripts.
"""

import numpy as np
from scipy.stats import chi2_contingency

from . import audit_core
from . import faircv_spec as spec

EEOC_RULE = audit_core.EEOC_RULE
GENDER_LABELS = spec.GENDER_LABELS
ETHNICITY_LABELS = spec.ETHNICITY_LABELS


def _groups_of(group_vec):
    return np.unique(group_vec).astype(int)


def _selection_rows(group_vec, groups, labels, sel):
    """Per-group n / selected / selection rate / share of selected."""
    n_sel = int(sel.sum())
    rows = []
    for g in groups:
        m = group_vec == g
        n_g = int(m.sum())
        s_g = int((sel & m).sum())
        rows.append({
            "group": labels[int(g)], "n": n_g, "selected": s_g,
            "selection_rate": float(s_g / n_g) if n_g else None,
            "share_of_selected": float(s_g / n_sel) if n_sel else None,
        })
    return rows


def _agg_selection(rows, groups):
    """DPD / DIR / EEOC verdict / chi2 / best-worst groups from selection rows."""
    sr = np.array([r["selection_rate"] for r in rows], dtype=float)
    dpd = float(np.nanmax(sr) - np.nanmin(sr))
    mx, mn = np.nanmax(sr), np.nanmin(sr)
    dir_ = float(mn / mx) if mx > 0 else None
    cont = np.array([[r["selected"], r["n"] - r["selected"]] for r in rows])
    # A threshold can select nothing (or everything) in a small/synthetic
    # sample -> zero-cell contingency tables; degrade to None instead of
    # crashing (the frozen demo data never hits this, but user CSVs can).
    try:
        chi2_p = float(chi2_contingency(cont)[1])
    except ValueError:
        chi2_p = None
    hi, lo = int(np.nanargmax(sr)), int(np.nanargmin(sr))
    return {
        "dpd": dpd, "dir": dir_, "eeoc_pass": bool(dir_ >= EEOC_RULE) if dir_ is not None else None,
        "chi2_p": chi2_p, "hi": hi, "lo": lo,
    }


def _bootstrap_sel_ci(y_score, group_vec, groups, n_top, rng, n_boot):
    """Percentile 95% CIs for DPD/DIR under 'hired = top n_top scores'."""
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
        sr = np.divide(s_g, n_g, out=np.full(ng, np.nan), where=n_g > 0)
        mx, mn = np.nanmax(sr), np.nanmin(sr)
        if np.isfinite(mx) and np.isfinite(mn):
            dpds.append(mx - mn)
            if mx > 0:
                dirs.append(mn / mx)
    dpds, dirs = np.asarray(dpds), np.asarray(dirs)
    d_ci = (float(np.percentile(dpds, 2.5)), float(np.percentile(dpds, 97.5))) if len(dpds) else (None, None)
    r_ci = (float(np.percentile(dirs, 2.5)), float(np.percentile(dirs, 97.5))) if len(dirs) else (None, None)
    return d_ci, r_ci


def _bootstrap_thr_sel_ci(y_score, group_vec, groups, thr, rng, n_boot):
    """Percentile 95% CIs for DPD/DIR under fixed-threshold hiring (score >= thr)."""
    n = len(y_score)
    g_idx = np.searchsorted(groups, group_vec)
    ng = len(groups)
    sel0 = y_score >= thr
    dpds, dirs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        gv, s = g_idx[idx], sel0[idx]
        n_g = np.bincount(gv, minlength=ng).astype(float)
        s_g = np.bincount(gv, weights=s.astype(float), minlength=ng)
        sr = np.divide(s_g, n_g, out=np.full(ng, np.nan), where=n_g > 0)
        mx, mn = np.nanmax(sr), np.nanmin(sr)
        if np.isfinite(mx) and np.isfinite(mn):
            dpds.append(mx - mn)
            if mx > 0:
                dirs.append(mn / mx)
    dpds, dirs = np.asarray(dpds), np.asarray(dirs)
    d_ci = (float(np.percentile(dpds, 2.5)), float(np.percentile(dpds, 97.5))) if len(dpds) else (None, None)
    r_ci = (float(np.percentile(dirs, 2.5)), float(np.percentile(dirs, 97.5))) if len(dirs) else (None, None)
    return d_ci, r_ci


def compute_topn(model, short, features, label_set, y_score, group_vec, groups,
                 labels, n_top, rng, n_boot=audit_core.N_BOOT, protocol="topN",
                 median_row=None):
    """Protocol A — hire the top ``n_top`` scores. Returns (row, per-group rows).

    ``median_row`` (optional dict with DPD/DIR/EEOC_pass from the primary
    metrics) enables the ``verdict_change`` flag used by the frozen scripts.
    """
    order = np.argsort(-y_score)[:n_top]
    sel = np.zeros(len(y_score), dtype=bool)
    sel[order] = True

    rows = _selection_rows(group_vec, groups, labels, sel)
    agg = _agg_selection(rows, groups)
    d_ci, r_ci = _bootstrap_sel_ci(y_score, group_vec, groups, n_top, rng, n_boot)
    best = labels[int(groups[agg["hi"]])]
    worst = labels[int(groups[agg["lo"]])]

    verdict_change = None
    if median_row is not None:
        verdict_change = bool(agg["eeoc_pass"]) != bool(median_row["EEOC_pass"])

    row = {
        "model": model, "short_name": short, "features": features, "label_set": label_set,
        "attribute": None, "protocol": protocol, "n_selected": int(n_top),
        "DPD": agg["dpd"], "DPD_ci_lo": d_ci[0], "DPD_ci_hi": d_ci[1],
        "DIR": agg["dir"], "DIR_ci_lo": r_ci[0], "DIR_ci_hi": r_ci[1],
        "EEOC_pass": agg["eeoc_pass"], "chi2_p": agg["chi2_p"],
        "best_group": best, "worst_group": worst,
        "verdict_change": verdict_change,
    }
    return row, rows


def compute_p75(model, short, features, label_set, y_score, y_true_te, group_vec,
                groups, labels, train_labels, rng, n_boot=audit_core.N_BOOT,
                protocol="p75", median_row=None, p=75):
    """Protocol B — 75th-percentile threshold of the training labels.

    hired = score >= thr; qualified = label >= thr. Selection disparity rows
    plus an EOD (max-min TPR among qualified) row with bootstrap CI, mirroring
    the official repo's equality-of-opportunity test.
    """
    thr = float(np.percentile(train_labels, p))
    hired = y_score >= thr
    qualified = y_true_te >= thr

    rows = _selection_rows(group_vec, groups, labels, hired)
    agg = _agg_selection(rows, groups)
    d_ci, r_ci = _bootstrap_thr_sel_ci(y_score, group_vec, groups, thr, rng, n_boot)
    best = labels[int(groups[agg["hi"]])]
    worst = labels[int(groups[agg["lo"]])]

    # TPR among qualified candidates per group (official EEO test), + bootstrap CI
    g_idx = np.searchsorted(groups, group_vec)
    ng = len(groups)
    pred_q, true_q = hired, qualified
    for r, g in zip(rows, groups):
        m = group_vec == g
        n_q = int((true_q & m).sum())
        r["tpr_qualified"] = float(int((true_q & pred_q & m).sum()) / n_q) if n_q else None
    eods = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_score), size=len(y_score))
        gv, pr, tq = g_idx[idx], pred_q[idx], true_q[idx]
        tpr = np.full(ng, np.nan)
        for gi in range(ng):
            m = gv == gi
            n_q = int((tq & m).sum())
            if n_q > 0:
                tpr[gi] = int((tq & pr & m).sum()) / n_q
        tf = tpr[np.isfinite(tpr)]
        if len(tf):
            eods.append(float(tf.max() - tf.min()))
    eods = np.asarray(eods)
    eod_ci = (float(np.percentile(eods, 2.5)), float(np.percentile(eods, 97.5))) if len(eods) else (None, None)

    verdict_change = None
    if median_row is not None:
        verdict_change = bool(agg["eeoc_pass"]) != bool(median_row["EEOC_pass"])

    row = {
        "model": model, "short_name": short, "features": features, "label_set": label_set,
        "attribute": None, "protocol": protocol, "threshold": thr,
        "DPD": agg["dpd"], "DPD_ci_lo": d_ci[0], "DPD_ci_hi": d_ci[1],
        "DIR": agg["dir"], "DIR_ci_lo": r_ci[0], "DIR_ci_hi": r_ci[1],
        "EEOC_pass": agg["eeoc_pass"], "chi2_p": agg["chi2_p"],
        "EOD_qualified": (float(np.nanmax([r["tpr_qualified"] for r in rows]))
                           - float(np.nanmin([r["tpr_qualified"] for r in rows]))
                           if any(r["tpr_qualified"] is not None for r in rows) else None),
        "EOD_ci_lo": eod_ci[0], "EOD_ci_hi": eod_ci[1],
        "best_group": best, "worst_group": worst,
        "verdict_change": verdict_change,
    }
    return row, rows


def demo_robustness(runtime, n_top=1000, n_boot=audit_core.N_BOOT, seed=audit_core.RNG_SEED,
                    median_lookup=None):
    """Run top-N and p75 protocols over the frozen demo models.

    ``median_lookup``: dict mapping (model, attribute) -> frozen metrics row
    (from the primary audit run) so verdict changes can be flagged.

    Returns {"rows": [...], "per_group": [...]} where each row carries an
    ``attribute`` field (gender/ethnicity) for the dashboard.
    """
    rng = np.random.default_rng(seed)
    out_rows, out_pg = [], []

    for mname, md in runtime["trained"].items():
        y_scr = md["pipe"].predict_proba(md["X_te"])[:, 1]
        for attr, gvec, glabels in (
            ("gender", runtime["gender_te"], GENDER_LABELS),
            ("ethnicity", runtime["ethnicity_te"], ETHNICITY_LABELS),
        ):
            groups = _groups_of(gvec)
            med_row = (median_lookup or {}).get((mname, attr))

            row, pg = compute_topn(
                mname, md["short"], md["fset"], md["lset"], y_scr, gvec, groups,
                glabels, n_top, rng, n_boot=n_boot, protocol="top1000",
                median_row=med_row,
            )
            row["attribute"] = attr
            for r in pg:
                r["model"], r["attribute"], r["protocol"] = mname, attr, "top1000"
            out_rows.append(row)
            out_pg.extend(pg)

            p75_row, p75_pg = compute_p75(
                mname, md["short"], md["fset"], md["lset"], y_scr, md["y_true_te"],
                gvec, groups, glabels, runtime["p75_thresholds"][md["lset"]],
                rng, n_boot=n_boot, protocol="p75", median_row=med_row,
            )
            p75_row["attribute"] = attr
            for r in p75_pg:
                r["model"], r["attribute"], r["protocol"] = mname, attr, "p75"
            out_rows.append(p75_row)
            out_pg.extend(p75_pg)

    return {"rows": out_rows, "per_group": out_pg}
