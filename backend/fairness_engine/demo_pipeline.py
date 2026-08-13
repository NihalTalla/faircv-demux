"""FairCV demo pipeline — replicates the frozen orchestration of
``faircv_audit_v2.py`` (steps 1-4, the CSV row assembly) so that the
equivalence test can prove the extracted engine reproduces the frozen
result CSVs byte-for-byte.

The metric/statistic/per-group ROW DICTS below mirror the frozen script
verbatim (same keys, same value conversions, same order) — this is what
makes ``to_csv`` output byte-identical to ``results/*.csv``.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from . import audit_core
from . import faircv_spec as spec


def run_faircv_audit(db_path, n_boot=audit_core.N_BOOT, seed=audit_core.RNG_SEED):
    """Run the full frozen-equivalent FairCV audit on ``db_path``.

    Returns a dict with DataFrames: ``metrics``, ``statistics``,
    ``per_group``, plus ``performance`` and ``dataset_summary``.
    """
    db = np.load(db_path, allow_pickle=True).item()
    P_tr, P_te = db["Profiles Train"], db["Profiles Test"]

    gender_tr = P_tr[:, spec.GENDER_COL].astype(int)
    gender_te = P_te[:, spec.GENDER_COL].astype(int)
    ethnicity_tr = P_tr[:, spec.ETHNICITY_COL].astype(int)
    ethnicity_te = P_te[:, spec.ETHNICITY_COL].astype(int)

    # Feature matrices (same slices as the frozen script)
    feats = {}
    for key, cols in spec.FEATURE_SLICES.items():
        feats[(key, "tr")] = P_tr[:, cols].astype(float)
        feats[(key, "te")] = P_te[:, cols].astype(float)

    # Binarisation: train median, label >= median -> hired (R-4/N-4)
    y_raw = {ls: (db[spec.LABEL_KEYS[ls][0]], db[spec.LABEL_KEYS[ls][1]]) for ls in spec.LABEL_KEYS}
    thrs = {ls: float(np.median(y_raw[ls][0])) for ls in spec.LABEL_KEYS}
    y_bin = {ls: (audit_core.binarise(y_raw[ls][0], thrs[ls]),
                  audit_core.binarise(y_raw[ls][1], thrs[ls])) for ls in spec.LABEL_KEYS}

    # Train models (identical to the frozen MODELS loop)
    trained = {}
    for mname, sname, fkey, lset in spec.MODELS:
        X_tr, X_te = feats[(fkey, "tr")], feats[(fkey, "te")]
        y_tr, y_te = y_bin[lset]
        pipe = audit_core.make_lr()
        pipe.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, pipe.predict(X_te))
        auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        f1 = f1_score(y_te, pipe.predict(X_te))
        trained[mname] = dict(pipe=pipe, X_te=X_te, y_te=y_te, acc=float(acc),
                              auc=float(auc), f1=float(f1), short=sname,
                              fset=fkey, lset=lset)

    # Audit loop (identical row assembly to faircv_audit_v2.py L468-540)
    rng = np.random.default_rng(seed)
    alpha = audit_core.ALPHA
    eeoc = audit_core.EEOC_RULE

    per_group_rows = []
    test_rows = []
    metric_rows = []

    for mname, md in trained.items():
        pipe = md["pipe"]
        y_pred = pipe.predict(md["X_te"])
        y_scr = pipe.predict_proba(md["X_te"])[:, 1]
        for attr, gcol, glabels in spec.ATTRIBUTES:
            gvec = gender_te if attr == "gender" else ethnicity_te
            res = audit_core.run_audit(md["y_te"], y_pred, y_scr, gvec,
                                       np.unique(gvec).astype(int), glabels, rng)

            for gi, g in enumerate(res["groups"]):
                per_group_rows.append({
                    "model": mname, "attribute": attr, "group": glabels[int(g)],
                    "n": int(res["N"][gi]),
                    "selection_rate": float(res["SR"][gi]),
                    "tpr": res["TPR"][gi], "fpr": res["FPR"][gi], "ppv": res["PPV"][gi],
                    "mean_score": float(res["scores"][gi].mean()),
                    "std_score": float(res["scores"][gi].std()),
                })

            for r in res["ks_rows"]:
                test_rows.append({"model": mname, "attribute": attr, "test": "KS-2samp",
                                  "comparison": r["comparison"], "statistic": r["stat"],
                                  "p_value": r["p"], "p_adjusted": r["p_adj"],
                                  "significant_0.05": bool(r["p_adj"] < alpha)})
            if res["kw"] is not None:
                test_rows.append({"model": mname, "attribute": attr, "test": "Kruskal-Wallis",
                                  "comparison": "all groups", "statistic": res["kw"][0],
                                  "p_value": res["kw"][1], "p_adjusted": np.nan,
                                  "significant_0.05": bool(res["kw"][1] < alpha)})
            test_rows.append({"model": mname, "attribute": attr, "test": "chi2-selection",
                              "comparison": "group x hired", "statistic": res["chi2_stat"],
                              "p_value": res["chi2_p"], "p_adjusted": np.nan,
                              "significant_0.05": bool(res["chi2_p"] < alpha)})

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
                "EEOC_pass": bool(res["DIR"] >= eeoc),
                "highest_SR_group": glabels[int(res["groups"][res["idx_hi"]])],
                "highest_TPR_group": glabels[int(res["groups"][np.nanargmax(res["TPR"])])],
                "cohens_d_extreme": res["cohens_d"], "cohens_h_extreme": res["cohens_h"],
            })

    metrics_df = pd.DataFrame(metric_rows)
    tests_df = pd.DataFrame(test_rows)
    per_group_df = pd.DataFrame(per_group_rows)

    # Runtime artifacts for the product layer (robustness protocols). This key
    # is NOT part of the equivalence contract — the three CSVs above are — so
    # adding it cannot invalidate the byte-identical reproduction.
    raw_test_labels = {ls: db[spec.LABEL_KEYS[ls][1]] for ls in spec.LABEL_KEYS}
    runtime = {
        "trained": {
            mname: {"pipe": md["pipe"], "X_te": md["X_te"],
                    "y_true_te": raw_test_labels[md["lset"]],
                    "short": md["short"], "fset": md["fset"], "lset": md["lset"]}
            for mname, md in trained.items()
        },
        "gender_te": gender_te,
        "ethnicity_te": ethnicity_te,
        "median_thresholds": thrs,
        "p75_thresholds": {ls: float(np.percentile(y_raw[ls][0], 75))
                            for ls in spec.LABEL_KEYS},
        "test_size": int(len(gender_te)),
    }

    return {
        "metrics": metrics_df,
        "statistics": tests_df,
        "per_group": per_group_df,
        "performance": {m: {"acc": md["acc"], "auc": md["auc"], "f1": md["f1"],
                            "features": md["fset"], "label_set": md["lset"]}
                        for m, md in trained.items()},
        "dataset_summary": {
            "train_rows": int(P_tr.shape[0]), "test_rows": int(P_te.shape[0]),
            "columns": int(P_tr.shape[1]),
            "gender": {spec.GENDER_LABELS[int(v)]: int((gender_te == v).sum())
                       for v in np.unique(gender_te)},
            "ethnicity": {spec.ETHNICITY_LABELS[int(v)]: int((ethnicity_te == v).sum())
                          for v in np.unique(ethnicity_te)},
            "thresholds": {f"{k}_median_train": v for k, v in thrs.items()},
            "blind_face_block_std_max": float(P_tr[:, spec.BLIND_FACE_COLS].std(axis=0).max()),
        },
        "runtime": runtime,
    }
