"""Generic CSV audit — the same verified fairness math applied to any dataset.

Design (mirrors the frozen demo pipeline where it matters):
  - input: DataFrame with a numeric label column, protected attribute column(s),
    and numeric feature columns
  - 80/20 random split (seed 42) on the test set used for audit
  - train-median binarisation of the label (the frozen audit's hiring rule)
  - one StandardScaler -> LogisticRegression(C=1.0, seed 42) model (the frozen
    pipeline; generic mode has one model rather than the six FairCV variants)
  - per-group selection rates, DPD/DIR/EOD/EO/KL + extreme-pair KL, pairwise KS
    with Holm, Kruskal-Wallis (>=3 groups), chi-square, Cohen's d/h, and
    percentile bootstrap 95% CIs (2,000 resamples, seed 42)
  - robustness protocols: p75 threshold and top-N selection

The metric row schema matches the demo pipeline, so the report builder and
dashboard are shared between demo and csv modes.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from . import audit_core
from . import robustness


def _groups_of(group_vec):
    return np.unique(group_vec).astype(int)


def run_generic_audit(
    df: pd.DataFrame,
    label_column: str,
    protected_attributes: list[str],
    feature_columns: list[str] | None = None,
    test_ratio: float = 0.2,
    seed: int = 42,
    n_boot: int = 2000,
    model_name: str = "Model-1",
    top_n_pct: float = 0.10,
):
    """Run a full fairness audit on a user CSV. Returns the same result dict
    shape as ``demo_pipeline.run_faircv_audit`` (minus the runtime artifacts
    only where irrelevant: runtime IS included for robustness)."""
    columns = [str(c) for c in df.columns]
    if label_column not in columns:
        raise ValueError(f"Label column '{label_column}' not found")
    for attr in protected_attributes:
        if attr not in columns:
            raise ValueError(f"Protected attribute '{attr}' not found")

    excluded = {label_column, *protected_attributes}
    if feature_columns is None:
        feature_columns = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])
                           and c not in excluded]
    feature_columns = [c for c in feature_columns if c not in excluded]

    X = df[feature_columns].astype(float).to_numpy()
    y_raw = df[label_column].astype(float).to_numpy()
    # group vectors are numeric-encoded via the natural sort of their values
    group_vectors = {
        attr: _encode_groups(df[attr].to_numpy()) for attr in protected_attributes
    }

    # Train/test split (seeded, deterministic)
    idx = np.arange(len(df))
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y_raw, idx, test_size=test_ratio, random_state=seed
    )

    thr = float(np.median(y_tr))
    yb_tr = audit_core.binarise(y_tr, thr)
    yb_te = audit_core.binarise(y_te, thr)

    pipe = audit_core.make_lr()
    pipe.fit(X_tr, yb_tr)
    y_pred = pipe.predict(X_te)
    y_scr = pipe.predict_proba(X_te)[:, 1]

    acc = float(accuracy_score(yb_te, y_pred))
    auc = float(roc_auc_score(yb_te, y_scr))
    f1 = float(f1_score(yb_te, y_pred))

    rng = np.random.default_rng(seed)
    alpha = audit_core.ALPHA
    eeoc = audit_core.EEOC_RULE

    metric_rows, test_rows, per_group_rows = [], [], []
    runtime_group_vectors = {}

    for attr in protected_attributes:
        gvec = group_vectors[attr][idx_te]
        groups = _groups_of(gvec)
        # decode numeric group code -> original label via the stable encoding map
        label_of = {code: orig for orig, code in _encoding(df[attr].to_numpy()).items()}
        label_of = {int(k): str(v) for k, v in label_of.items()}
        runtime_group_vectors[attr] = (gvec, label_of)

        res = audit_core.run_audit(yb_te, y_pred, y_scr, gvec, groups, label_of, rng)

        for gi, g in enumerate(res["groups"]):
            per_group_rows.append({
                "model": model_name, "attribute": attr, "group": label_of[int(g)],
                "n": int(res["N"][gi]),
                "selection_rate": float(res["SR"][gi]),
                "tpr": res["TPR"][gi], "fpr": res["FPR"][gi], "ppv": res["PPV"][gi],
                "mean_score": float(res["scores"][gi].mean()),
                "std_score": float(res["scores"][gi].std()),
            })

        for r in res["ks_rows"]:
            test_rows.append({"model": model_name, "attribute": attr, "test": "KS-2samp",
                              "comparison": r["comparison"], "statistic": r["stat"],
                              "p_value": r["p"], "p_adjusted": r["p_adj"],
                              "significant_0.05": bool(r["p_adj"] < alpha)})
        if res["kw"] is not None:
            test_rows.append({"model": model_name, "attribute": attr, "test": "Kruskal-Wallis",
                              "comparison": "all groups", "statistic": res["kw"][0],
                              "p_value": res["kw"][1], "p_adjusted": np.nan,
                              "significant_0.05": bool(res["kw"][1] < alpha)})
        test_rows.append({"model": model_name, "attribute": attr, "test": "chi2-selection",
                          "comparison": "group x hired", "statistic": res["chi2_stat"],
                          "p_value": res["chi2_p"], "p_adjusted": np.nan,
                          "significant_0.05": bool(res["chi2_p"] < alpha)})

        ks_min_adj = min((r["p_adj"] for r in res["ks_rows"]), default=np.nan)
        metric_rows.append({
            "model": model_name, "short_name": model_name,
            "features": "|".join(feature_columns), "label_set": label_column,
            "attribute": attr, "n": len(yb_te),
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
            "highest_SR_group": label_of[int(res["groups"][res["idx_hi"]])],
            "highest_TPR_group": label_of[int(res["groups"][np.nanargmax(res["TPR"])])],
            "cohens_d_extreme": res["cohens_d"], "cohens_h_extreme": res["cohens_h"],
        })

    # Robustness protocols (p75 + topN) for each protected attribute
    n_test = len(yb_te)
    n_top = max(1, int(round(top_n_pct * n_test)))
    rob_rows, rob_pg = [], []
    median_lookup = {}
    for attr in protected_attributes:
        med = next((r for r in metric_rows if r["attribute"] == attr), None)
        median_lookup[(model_name, attr)] = med
    for attr in protected_attributes:
        gvec, label_of = runtime_group_vectors[attr]
        groups = _groups_of(gvec)
        med = median_lookup[(model_name, attr)]

        row, pg = robustness.compute_topn(
            model_name, model_name, "|".join(feature_columns), label_column,
            y_scr, gvec, groups, label_of, n_top, rng, n_boot=n_boot,
            protocol="topN", median_row=med,
        )
        row["attribute"] = attr
        for r in pg:
            r["model"], r["attribute"], r["protocol"] = model_name, attr, "topN"
        rob_rows.append(row)
        rob_pg.extend(pg)

        p75_row, p75_pg = robustness.compute_p75(
            model_name, model_name, "|".join(feature_columns), label_column,
            y_scr, y_te, gvec, groups, label_of, y_tr, rng, n_boot=n_boot,
            protocol="p75", median_row=med,
        )
        p75_row["attribute"] = attr
        for r in p75_pg:
            r["model"], r["attribute"], r["protocol"] = model_name, attr, "p75"
        rob_rows.append(p75_row)
        rob_pg.extend(p75_pg)

    # Dataset summary
    group_counts = {}
    for attr in protected_attributes:
        vals_str = df[attr].astype(str).to_numpy()
        uniq = sorted(set(vals_str))
        group_counts[attr] = {v: int((vals_str == v).sum()) for v in uniq}

    metrics_df = pd.DataFrame(metric_rows)
    tests_df = pd.DataFrame(test_rows)
    per_group_df = pd.DataFrame(per_group_rows)

    return {
        "metrics": metrics_df,
        "statistics": tests_df,
        "per_group": per_group_df,
        "performance": {
            model_name: {"acc": acc, "auc": auc, "f1": f1,
                         "features": "|".join(feature_columns), "label_set": label_column}
        },
        "dataset_summary": {
            "train_rows": int(len(idx_tr)), "test_rows": int(len(idx_te)),
            "columns": len(columns), "columns_list": columns,
            "label_column": label_column, "threshold": thr,
            "protected_counts": group_counts,
            "top_n_selected": n_top,
        },
        "robustness": {"rows": rob_rows, "per_group": rob_pg},
        "runtime": {
            "trained": {model_name: {"pipe": pipe, "X_te": X_te, "y_true_te": y_te,
                                     "short": model_name, "fset": "|".join(feature_columns),
                                     "lset": label_column}},
            "gender_te": None, "ethnicity_te": None,
            "median_thresholds": {label_column: thr},
            "p75_thresholds": {label_column: float(np.percentile(y_tr, 75))},
            "test_size": n_test,
        },
    }


def _encoding(values) -> dict:
    """Map each unique original value -> numeric code (sorted, stable)."""
    uniq = sorted({str(v) for v in values})
    return {v: i for i, v in enumerate(uniq)}


def _encode_groups(values) -> np.ndarray:
    enc = _encoding(values)
    return np.array([enc[str(v)] for v in values], dtype=int)
