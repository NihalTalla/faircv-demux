"""Unit tests for the extracted audit math.

Cross-checks the pure functions against hand-computed examples and against
values pinned in the frozen report (``results/metrics.csv``).
"""

import numpy as np
import pytest

from fairness_engine import audit_core


def test_binarise_threshold():
    arr = np.array([0.2, 0.5, 0.6, 0.5, 0.9])
    assert list(audit_core.binarise(arr, 0.5)) == [0, 1, 1, 1, 1]  # >= threshold


def test_kl_divergence_identical_is_zero():
    s = np.random.default_rng(0).normal(0.5, 0.1, 500)
    assert audit_core.kl_divergence(s, s) < 1e-9


def test_kl_divergence_separated_is_positive():
    a = np.zeros(500)
    b = np.ones(500)
    assert audit_core.kl_divergence(a, b) > 1e-3


def test_kl_divergence_fixed_edges():
    a = np.random.default_rng(1).uniform(0, 1, 300)
    b = np.random.default_rng(2).uniform(0, 1, 300)
    edges = np.linspace(0, 1, 51)
    k1 = audit_core.kl_divergence(a, b, edges=edges)
    k2 = audit_core.kl_divergence(a, b, edges=edges)
    assert k1 == k2  # deterministic given fixed edges


def test_holm_correct_known_values():
    # m=3, p = [0.01, 0.02, 0.9] -> products [0.03, 0.04, 0.9] -> running max
    adj = audit_core.holm_correct(np.array([0.01, 0.02, 0.9]))
    assert np.allclose(adj, [0.03, 0.04, 0.9])


def test_holm_correct_clips_at_one():
    adj = audit_core.holm_correct(np.array([0.5, 0.6, 0.7]))
    assert adj.max() <= 1.0 + 1e-12


def test_cohens_d_hand_computed():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])
    # pooled sd = 1, d = (2-3)/1 = -1
    assert abs(audit_core.cohens_d(a, b) - (-1.0)) < 1e-9


def test_agg_metrics_hand_computed():
    # two groups: SR 0.5 vs 0.25 -> DPD 0.25, DIR 0.5
    SR = np.array([0.5, 0.25])
    TPR = np.array([0.6, 0.4])
    FPR = np.array([0.2, 0.1])
    s_hi = np.random.default_rng(0).uniform(0.4, 0.6, 200)
    s_lo = np.random.default_rng(1).uniform(0.2, 0.4, 200)
    dpd, dir_, eod, eo, _ = audit_core._agg(SR, TPR, FPR, s_hi, s_lo)
    assert abs(dpd - 0.25) < 1e-12
    assert abs(dir_ - 0.5) < 1e-12
    assert abs(eod - 0.2) < 1e-12
    assert abs(eo - 0.2) < 1e-12  # TPR spread (0.2) >= FPR spread (0.1)


def test_run_audit_structure_tiny():
    rng = np.random.default_rng(3)
    n = 120
    y_true = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    y_score = rng.random(n)
    gvec = rng.integers(0, 3, n)
    groups = np.array([0, 1, 2])
    res = audit_core.run_audit(y_true, y_pred, y_score, gvec, groups,
                               {0: "G1", 1: "G2", 2: "G3"}, rng)
    for key in ("DPD", "DIR", "EOD", "EO", "KL_extreme", "cis", "ks_rows",
                "kw", "chi2_p", "cohens_d", "cohens_h"):
        assert key in res
    assert res["kw"] is not None          # 3 groups -> Kruskal-Wallis runs
    assert len(res["ks_rows"]) == 3       # 3 pairwise comparisons
    assert all("p_adj" in r for r in res["ks_rows"])
    for metric in ("DPD", "DIR", "EOD", "EO"):
        lo, hi = res["cis"][metric]
        assert lo is not None and hi is not None and lo <= hi


def test_bootstrap_cis_fixed_extreme_pair():
    """KL bootstrap CI must target the FIXED full-sample extreme pair: with a
    deterministic score structure the resampled KL stays close to the point
    estimate."""
    rng = np.random.default_rng(5)
    n = 400
    gvec = np.repeat([0, 1], n // 2)
    y_true = np.repeat([1, 0], n // 2)
    y_pred = y_true.copy()
    y_score = np.concatenate([rng.uniform(0.6, 1.0, n // 2),
                              rng.uniform(0.0, 0.4, n // 2)])
    groups = np.array([0, 1])
    res = audit_core.run_audit(y_true, y_pred, y_score, gvec, groups,
                               {0: "hi", 1: "lo"}, rng)
    lo, hi = res["cis"]["KL"]
    assert hi > 0 and lo < hi


def test_metrics_match_frozen_headline():
    """The frozen M4/ethnicity DIR point estimate is reproducible from the
    frozen per-group selection rates (results/per_group_metrics.csv):
    G1 0.5429467084639499, G2 0.4552896725440806, G3 0.4186765615337044.
    DPD = max-min = 0.12427, DIR = min/max = 0.77112 — matching
    results/metrics.csv. (The byte-identical end-to-end check lives in
    test_equivalence.py; this test pins the arithmetic.)"""
    frozen_sr = np.array([0.5429467084639499, 0.4552896725440806, 0.4186765615337044])
    dpd = frozen_sr.max() - frozen_sr.min()
    dir_ = frozen_sr.min() / frozen_sr.max()
    assert abs(dpd - 0.12427) < 1e-4
    assert abs(dir_ - 0.77112) < 1e-4
