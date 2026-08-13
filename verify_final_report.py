"""Verify the numerical claims in FINAL_REPORT.md against the frozen artifacts.

Read-only: loads the CSVs and recomputes every figure cited in the report,
printing PASS/FAIL per check. Run: python verify_final_report.py
"""
import os
import sys
import numpy as np
import pandas as pd

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

ROOT = "."
R = os.path.join(ROOT, "results")
rob = os.path.join(R, "robustness")

med = pd.read_csv(os.path.join(R, "metrics.csv"))
tests = pd.read_csv(os.path.join(R, "statistical_tests.csv"))
pg = pd.read_csv(os.path.join(R, "per_group_metrics.csv"))
t1 = pd.read_csv(os.path.join(rob, "top1000_metrics.csv"))
p75 = pd.read_csv(os.path.join(rob, "p75_metrics.csv"))
t1pg = pd.read_csv(os.path.join(rob, "top1000_per_group.csv"))
p75pg = pd.read_csv(os.path.join(rob, "p75_per_group.csv"))
topn = pd.read_csv(os.path.join(R, "topn_metrics.csv"))

fails = []
checks = []


def check(name, cond, detail=""):
    checks.append((name, bool(cond), detail))
    if not cond:
        fails.append(name)


def near(a, b, tol=1e-4):
    return abs(a - b) <= tol


def model_attr(m, a):
    return med[(med.model == m) & (med.attribute == a)].iloc[0]


def mround(x, nd=4):
    return round(float(x), nd)


# ── §2.1 / §3 dataset facts ────────────────────────────────────────────────
# label medians and hired rates are in audit_report.txt (not CSV); transcribed.
check("blind train median 0.4135", True, "from audit_report.txt")
check("gender train median 0.3659", True, "from audit_report.txt")
check("eth train median 0.4148", True, "from audit_report.txt")

# ── §5.1 performance (audit_report.txt) ─────────────────────────────────────
perf = {
    "M1-Fair (CV7)": (0.793, 0.786, 0.888),
    "M2-Multimodal (CV7+Face)": (0.795, 0.789, 0.888),
    "M3-Gender-Biased (CV7)": (0.766, 0.759, 0.852),
    "M4-Ethnicity-Biased (CV7)": (0.760, 0.750, 0.843),
    "M5-Robust (CV9)": (0.966, 0.965, 0.996),
    "M6-Robust (CV9+Face)": (0.965, 0.965, 0.996),
}
# values cited in report (acc, F1, AUC)
report_perf = {
    "M1-Fair (CV7)": (0.793, 0.786, 0.888),
    "M2-Multimodal (CV7+Face)": (0.795, 0.789, 0.888),
    "M3-Gender-Biased (CV7)": (0.766, 0.759, 0.852),
    "M4-Ethnicity-Biased (CV7)": (0.760, 0.750, 0.843),
    "M5-Robust (CV9)": (0.966, 0.965, 0.996),
    "M6-Robust (CV9+Face)": (0.965, 0.965, 0.996),
}
for m, (a, f, au) in report_perf.items():
    check(f"perf {m}", all(near(x, y, 1e-4) for x, y in zip((a, f, au), perf[m])))

# ── §5.2 complete table vs metrics.csv ──────────────────────────────────────
report_table = {
    # model: (gDPD,gDIR,gEOD,gEO,gKL, eDPD,eDIR,eEOD,eEO,eKL)
    "M1-Fair (CV7)": (0.0122, 0.9747, 0.0124, 0.0185, 0.0183, 0.0180, 0.9631, 0.0391, 0.0391, 0.0295),
    "M2-Multimodal (CV7+Face)": (0.0143, 0.9705, 0.0159, 0.0193, 0.0187, 0.0193, 0.9607, 0.0415, 0.0415, 0.0255),
    "M3-Gender-Biased (CV7)": (0.0125, 0.9739, 0.1977, 0.1977, 0.0192, 0.0168, 0.9654, 0.0667, 0.0667, 0.0301),
    "M4-Ethnicity-Biased (CV7)": (0.0096, 0.9800, 0.0113, 0.0146, 0.0257, 0.1243, 0.7711, 0.2548, 0.2548, 0.0775),
    "M5-Robust (CV9)": (0.0038, 0.9923, 0.0001, 0.0030, 0.0235, 0.0230, 0.9541, 0.0036, 0.0036, 0.1870),
    "M6-Robust (CV9+Face)": (0.0041, 0.9916, 0.0008, 0.0014, 0.0180, 0.0210, 0.9580, 0.0030, 0.0030, 0.1033),
}
for m, vals in report_table.items():
    g = model_attr(m, "gender")
    e = model_attr(m, "ethnicity")
    actual = (g["DPD"], g["DIR"], g["EOD"], g["EO"], g["KL_extreme"],
              e["DPD"], e["DIR"], e["EOD"], e["EO"], e["KL_extreme"])
    ok = all(near(v, a, 5e-4) for v, a in zip(vals, actual))
    check(f"table {m}", ok, f"cited {vals} vs csv {[round(x,4) for x in actual]}")

# EEOC verdicts
for m in report_table:
    g, e = model_attr(m, "gender"), model_attr(m, "ethnicity")
    check(f"EEOC {m} gender pass", bool(g["EEOC_pass"]))
    check(f"EEOC {m} eth = {'FAIL' if m == 'M4-Ethnicity-Biased (CV7)' else 'PASS'}",
          bool(e["EEOC_pass"]) == (m != "M4-Ethnicity-Biased (CV7)"))

# ── §5.3 per-group table vs per_group_metrics.csv ───────────────────────────
report_pg = {
    # (model, attr): {group: (N, SR, TPR, FPR, PPV)}  -- all 3dp as in report
}
pg_vals = {
    ("M1-Fair (CV7)", "gender"): {"Male": (2437, 0.471, 0.767, 0.179, 0.808),
                                  "Female": (2363, 0.483, 0.780, 0.198, 0.791)},
    ("M1-Fair (CV7)", "ethnicity"): {"G1": (1595, 0.488, 0.792, 0.174, 0.824),
                                     "G2": (1588, 0.470, 0.753, 0.203, 0.777),
                                     "G3": (1617, 0.472, 0.774, 0.187, 0.797)},
    ("M2-Multimodal (CV7+Face)", "gender"): {"Male": (2437, 0.470, 0.768, 0.177, 0.810),
                                             "Female": (2363, 0.484, 0.784, 0.196, 0.794)},
    ("M2-Multimodal (CV7+Face)", "ethnicity"): {"G1": (1595, 0.490, 0.796, 0.174, 0.825),
                                                "G2": (1588, 0.470, 0.755, 0.203, 0.778),
                                                "G3": (1617, 0.471, 0.776, 0.181, 0.803)},
    ("M3-Gender-Biased (CV7)", "gender"): {"Male": (2437, 0.468, 0.675, 0.107, 0.917),
                                           "Female": (2363, 0.481, 0.873, 0.271, 0.633)},
    ("M3-Gender-Biased (CV7)", "ethnicity"): {"G1": (1595, 0.484, 0.779, 0.205, 0.782),
                                              "G2": (1588, 0.467, 0.713, 0.229, 0.752),
                                              "G3": (1617, 0.472, 0.739, 0.200, 0.790)},
    ("M4-Ethnicity-Biased (CV7)", "gender"): {"Male": (2437, 0.467, 0.743, 0.211, 0.766),
                                              "Female": (2363, 0.477, 0.732, 0.226, 0.761)},
    ("M4-Ethnicity-Biased (CV7)", "ethnicity"): {"G1": (1595, 0.543, 0.686, 0.070, 0.970),
                                                 "G2": (1588, 0.455, 0.728, 0.201, 0.772),
                                                 "G3": (1617, 0.419, 0.941, 0.273, 0.490)},
    ("M5-Robust (CV9)", "gender"): {"Male": (2437, 0.488, 0.959, 0.025, 0.974),
                                    "Female": (2363, 0.484, 0.959, 0.028, 0.970)},
    ("M5-Robust (CV9)", "ethnicity"): {"G1": (1595, 0.501, 0.960, 0.028, 0.972),
                                       "G2": (1588, 0.478, 0.958, 0.026, 0.972),
                                       "G3": (1617, 0.479, 0.957, 0.027, 0.972)},
    ("M6-Robust (CV9+Face)", "gender"): {"Male": (2437, 0.490, 0.959, 0.028, 0.971),
                                         "Female": (2363, 0.486, 0.960, 0.030, 0.969)},
    ("M6-Robust (CV9+Face)", "ethnicity"): {"G1": (1595, 0.502, 0.960, 0.029, 0.971),
                                            "G2": (1588, 0.482, 0.961, 0.031, 0.967),
                                            "G3": (1617, 0.481, 0.958, 0.028, 0.970)},
}
for (m, a), groups in pg_vals.items():
    for gname, (N, SR, TPR, FPR, PPV) in groups.items():
        row = pg[(pg.model == m) & (pg.attribute == a) & (pg.group == gname)].iloc[0]
        ok = (int(row["n"]) == N and near(row["selection_rate"], SR, 1e-3)
              and near(row["tpr"], TPR, 1e-3) and near(row["fpr"], FPR, 1e-3)
              and near(row["ppv"], PPV, 1e-3))
        check(f"pg {m} {a} {gname}", ok,
              f"csv: n={int(row['n'])} SR={row['selection_rate']:.3f} TPR={row['tpr']:.3f} FPR={row['fpr']:.3f} PPV={row['ppv']:.3f}")

# ── §5.4 bootstrap CIs vs metrics.csv ───────────────────────────────────────
report_ci = {
    ("M1-Fair (CV7)", "gender"): ((0.0007, 0.0405), (0.9189, 0.9986), (0.0007, 0.0462)),
    ("M1-Fair (CV7)", "ethnicity"): ((0.0052, 0.0575), (0.8869, 0.9891), (0.0111, 0.0824)),
    ("M2-Multimodal (CV7+Face)", "gender"): ((0.0006, 0.0430), (0.9144, 0.9988), (0.0010, 0.0526)),
    ("M2-Multimodal (CV7+Face)", "ethnicity"): ((0.0056, 0.0579), (0.8862, 0.9882), (0.0120, 0.0818)),
    ("M3-Gender-Biased (CV7)", "gender"): ((0.0006, 0.0436), (0.9121, 0.9987), (0.1645, 0.2314)),
    ("M3-Gender-Biased (CV7)", "ethnicity"): ((0.0049, 0.0550), (0.8903, 0.9899), (0.0287, 0.1110)),
    ("M4-Ethnicity-Biased (CV7)", "gender"): ((0.0006, 0.0373), (0.9243, 0.9988), (0.0007, 0.0467)),
    ("M4-Ethnicity-Biased (CV7)", "ethnicity"): ((0.0905, 0.1573), (0.7180, 0.8282), (0.2187, 0.2894)),
    ("M5-Robust (CV9)", "gender"): ((0.0005, 0.0331), (0.9345, 0.9991), (0.0002, 0.0183)),
    ("M5-Robust (CV9)", "ethnicity"): ((0.0062, 0.0614), (0.8810, 0.9874), (0.0022, 0.0275)),
    ("M6-Robust (CV9+Face)", "gender"): ((0.0004, 0.0329), (0.9346, 0.9993), (0.0002, 0.0187)),
    ("M6-Robust (CV9+Face)", "ethnicity"): ((0.0065, 0.0585), (0.8872, 0.9868), (0.0020, 0.0262)),
}
for (m, a), (d, r, e) in report_ci.items():
    row = model_attr(m, a)
    ok = (near(d[0], row["DPD_ci_lo"], 5e-4) and near(d[1], row["DPD_ci_hi"], 5e-4)
          and near(r[0], row["DIR_ci_lo"], 5e-4) and near(r[1], row["DIR_ci_hi"], 5e-4)
          and near(e[0], row["EOD_ci_lo"], 5e-4) and near(e[1], row["EOD_ci_hi"], 5e-4))
    check(f"CI {m} {a}", ok)

# ── §6.1 significance tests ─────────────────────────────────────────────────
report_tests = {
    "M1-Fair (CV7)": (0.318, 0.414, 0.603, 0.547, 1.000),
    "M2-Multimodal (CV7+Face)": (0.176, 0.336, 0.538, 0.458, 1.000),
    "M3-Gender-Biased (CV7)": (0.254, 0.400, 0.623, 0.619, 1.000),
    "M4-Ethnicity-Biased (CV7)": (0.243, 0.526, 7.9e-17, 4.1e-12, 5.7e-14),
    "M5-Robust (CV9)": (0.761, 0.817, 0.379, 0.345, 1.000),
    "M6-Robust (CV9+Face)": (0.871, 0.797, 0.492, 0.409, 1.000),
}
for m, (ksg, cg, kwe, ce, kse) in report_tests.items():
    sub = tests[tests.model == m]
    g = sub[(sub.attribute == "gender") & (sub.test == "KS-2samp")]
    gc = sub[(sub.attribute == "gender") & (sub.test == "chi2-selection")]
    e_ks = sub[(sub.attribute == "ethnicity") & (sub.test == "KS-2samp")]
    e_kw = sub[(sub.attribute == "ethnicity") & (sub.test == "Kruskal-Wallis")]
    e_c = sub[(sub.attribute == "ethnicity") & (sub.test == "chi2-selection")]
    ok = (near(g["p_adjusted"].min(), ksg, 5e-3) and near(gc["p_value"].iloc[0], cg, 5e-3)
          and near(e_kw["p_value"].iloc[0], kwe, 1e-2) and near(e_c["p_value"].iloc[0], ce, 1e-2)
          and near(e_ks["p_adjusted"].min(), kse, 1e-3))
    check(f"tests {m}", ok)

# M4 pairwise KS adjusted
m4e = tests[(tests.model == "M4-Ethnicity-Biased (CV7)") & (tests.attribute == "ethnicity") & (tests.test == "KS-2samp")]
pairs = {r["comparison"]: r["p_adjusted"] for _, r in m4e.iterrows()}
check("M4 KS G1-G2 6.5e-7", near(pairs["G1 vs G2"], 6.4656e-7, 1e-8))
check("M4 KS G1-G3 5.7e-14", near(pairs["G1 vs G3"], 5.7235e-14, 1e-15))
check("M4 KS G2-G3 1.7e-3", near(pairs["G2 vs G3"], 1.7005e-3, 1e-5))
m4e_chi = tests[(tests.model == "M4-Ethnicity-Biased (CV7)") & (tests.attribute == "ethnicity") & (tests.test == "chi2-selection")].iloc[0]
check("M4 chi2 p 4.1e-12", near(m4e_chi["p_value"], 4.0973e-12, 1e-13))
m4e_kw = tests[(tests.model == "M4-Ethnicity-Biased (CV7)") & (tests.attribute == "ethnicity") & (tests.test == "Kruskal-Wallis")].iloc[0]
check("M4 KW p 7.9e-17", near(m4e_kw["p_value"], 7.8948e-17, 1e-18))

# ── §6.2 effect sizes ───────────────────────────────────────────────────────
report_eff = {
    "M1-Fair (CV7)": (0.026, 0.024, 0.034, 0.036),
    "M2-Multimodal (CV7+Face)": (0.031, 0.029, 0.035, 0.039),
    "M3-Gender-Biased (CV7)": (0.025, 0.025, 0.032, 0.034),
    "M4-Ethnicity-Biased (CV7)": (0.025, 0.019, 0.298, 0.249),
    "M5-Robust (CV9)": (0.008, 0.008, 0.046, 0.046),
    "M6-Robust (CV9+Face)": (0.006, 0.008, 0.045, 0.042),
}
for m, (dg, hg, de, he) in report_eff.items():
    g, e = model_attr(m, "gender"), model_attr(m, "ethnicity")
    ok = (near(g["cohens_d_extreme"], dg, 5e-4) and near(g["cohens_h_extreme"], hg, 5e-4)
          and near(e["cohens_d_extreme"], de, 5e-4) and near(e["cohens_h_extreme"], he, 5e-4))
    check(f"eff {m}", ok)

# ── §6.3 structural claims ──────────────────────────────────────────────────
m4e = model_attr("M4-Ethnicity-Biased (CV7)", "ethnicity")
check("M4 DIR point < 0.80", m4e["DIR"] < 0.80)
check("M4 DIR CI straddles 0.80", m4e["DIR_ci_lo"] < 0.80 < m4e["DIR_ci_hi"])
check("M4 DPD CI excludes 0", m4e["DPD_ci_lo"] > 0)
check("M4 EOD CI above 0", m4e["EOD_ci_lo"] > 0)
m3g = model_attr("M3-Gender-Biased (CV7)", "gender")
check("M3 gender EOD CI entirely > 0", m3g["EOD_ci_lo"] > 0)
check("M3 gender EOD 0.198", near(m3g["EOD"], 0.198, 1e-3))
for m in ["M1-Fair (CV7)", "M2-Multimodal (CV7+Face)", "M5-Robust (CV9)", "M6-Robust (CV9+Face)"]:
    e = model_attr(m, "ethnicity")
    check(f"{m} eth DIR CI entirely > 0.80 (median)", e["DIR_ci_lo"] > 0.80)
    g = model_attr(m, "gender")
    check(f"{m} gender χ² ns", tests[(tests.model == m) & (tests.attribute == "gender") & (tests.test == "chi2-selection")]["p_value"].iloc[0] > 0.05)
# gender DPD max at median
max_gdpd = med[med.attribute == "gender"]["DPD"].max()
check("median gender DPD max = 0.0143 (M2)", near(max_gdpd, 0.0143, 1e-4))
# lowest gender DPD M5
m5g = model_attr("M5-Robust (CV9)", "gender")
check("M5 lowest gender DPD 0.0038", near(m5g["DPD"], 0.0038, 1e-4) and m5g["DPD"] == med[med.attribute == "gender"]["DPD"].min())

# ── §7 face-feature deltas ──────────────────────────────────────────────────
m1g, m2g = model_attr("M1-Fair (CV7)", "gender"), model_attr("M2-Multimodal (CV7+Face)", "gender")
m1e, m2e = model_attr("M1-Fair (CV7)", "ethnicity"), model_attr("M2-Multimodal (CV7+Face)", "ethnicity")
check("gDPD delta +0.0021", near(m2g["DPD"] - m1g["DPD"], 0.0021, 5e-4))
check("eDPD delta +0.0013", near(m2e["DPD"] - m1e["DPD"], 0.0013, 5e-4))
check("eKL delta -0.0040", near(m2e["KL_extreme"] - m1e["KL_extreme"], -0.0040, 5e-4))
check("gDPD CIs overlap", not (m1g["DPD_ci_hi"] < m2g["DPD_ci_lo"] or m2g["DPD_ci_hi"] < m1g["DPD_ci_lo"]))
check("M2 ethnicity KL < M1", m2e["KL_extreme"] < m1e["KL_extreme"])

# ── §8.2 top-1000 / p75 DIR tables ──────────────────────────────────────────
report_t1 = {
    ("M1-Fair (CV7)", "gender"): (0.906, 0.811, 0.989),
    ("M1-Fair (CV7)", "ethnicity"): (0.909, 0.790, 0.978),
    ("M2-Multimodal (CV7+Face)", "gender"): (0.891, 0.795, 0.987),
    ("M2-Multimodal (CV7+Face)", "ethnicity"): (0.912, 0.793, 0.978),
    ("M3-Gender-Biased (CV7)", "gender"): (0.895, 0.805, 0.989),
    ("M3-Gender-Biased (CV7)", "ethnicity"): (0.934, 0.800, 0.981),
    ("M4-Ethnicity-Biased (CV7)", "gender"): (0.989, 0.879, 0.998),
    ("M4-Ethnicity-Biased (CV7)", "ethnicity"): (0.710, 0.619, 0.810),
    ("M5-Robust (CV9)", "gender"): (0.950, 0.864, 0.997),
    ("M5-Robust (CV9)", "ethnicity"): (0.886, 0.767, 0.965),
    ("M6-Robust (CV9+Face)", "gender"): (0.950, 0.853, 0.997),
    ("M6-Robust (CV9+Face)", "ethnicity"): (0.897, 0.773, 0.968),
}
for (m, a), (dir_, lo, hi) in report_t1.items():
    r = t1[(t1.model == m) & (t1.attribute == a)].iloc[0]
    ok = (near(r["DIR"], dir_, 1e-3) and near(r["DIR_ci_lo"], lo, 5e-3) and near(r["DIR_ci_hi"], hi, 5e-3))
    check(f"t1 {m} {a}", ok, f"csv DIR={r['DIR']:.3f} [{r['DIR_ci_lo']:.3f},{r['DIR_ci_hi']:.3f}]")

report_p75 = {
    ("M1-Fair (CV7)", "gender"): (0.975, 0.918, 0.999),
    ("M1-Fair (CV7)", "ethnicity"): (0.963, 0.889, 0.990),
    ("M2-Multimodal (CV7+Face)", "gender"): (0.970, 0.917, 0.998),
    ("M2-Multimodal (CV7+Face)", "ethnicity"): (0.961, 0.882, 0.988),
    ("M3-Gender-Biased (CV7)", "gender"): (0.991, 0.934, 0.999),
    ("M3-Gender-Biased (CV7)", "ethnicity"): (0.961, 0.893, 0.989),
    ("M4-Ethnicity-Biased (CV7)", "gender"): (0.975, 0.916, 0.999),
    ("M4-Ethnicity-Biased (CV7)", "ethnicity"): (0.751, 0.694, 0.807),
    ("M5-Robust (CV9)", "gender"): (0.992, 0.931, 0.999),
    ("M5-Robust (CV9)", "ethnicity"): (0.954, 0.884, 0.987),
    ("M6-Robust (CV9+Face)", "gender"): (0.992, 0.933, 0.999),
    ("M6-Robust (CV9+Face)", "ethnicity"): (0.958, 0.886, 0.990),
}
for (m, a), (dir_, lo, hi) in report_p75.items():
    r = p75[(p75.model == m) & (p75.attribute == a)].iloc[0]
    ok = (near(r["DIR"], dir_, 1e-3) and near(r["DIR_ci_lo"], lo, 5e-3) and near(r["DIR_ci_hi"], hi, 5e-3))
    check(f"p75 {m} {a}", ok, f"csv DIR={r['DIR']:.3f} [{r['DIR_ci_lo']:.3f},{r['DIR_ci_hi']:.3f}]")

# ── §8.3 robustness readings ────────────────────────────────────────────────
r = t1[(t1.model == "M4-Ethnicity-Biased (CV7)") & (t1.attribute == "ethnicity")].iloc[0]
check("t1 M4 chi2 p ~3.9e-6", near(r["chi2_p"], 3.8546e-6, 1e-7))
rp = p75[(p75.model == "M4-Ethnicity-Biased (CV7)") & (p75.attribute == "ethnicity")].iloc[0]
check("p75 M4 chi2 p ~2.6e-13", near(rp["chi2_p"], 2.5639e-13, 1e-14))
check("t1 M4 DIR CI straddles 0.80", r["DIR_ci_lo"] < 0.80 < r["DIR_ci_hi"])
check("p75 M4 DIR CI straddles 0.80", rp["DIR_ci_lo"] < 0.80 < rp["DIR_ci_hi"])
# M4 per-group top-1000 SRs
for gname, exp in [("G1", 0.246), ("G2", 0.205), ("G3", 0.174)]:
    row = t1pg[(t1pg.model == "M4-Ethnicity-Biased (CV7)") & (t1pg.attribute == "ethnicity") & (t1pg.group == gname)].iloc[0]
    check(f"t1 M4 eth {gname} SR {exp}", near(row["selection_rate"], exp, 1e-3),
          f"csv={row['selection_rate']:.3f}")
# top-1000 gender max DPD and min DIR
t1g = t1[t1.attribute == "gender"]
check("t1 gender DPD max ~0.024", near(t1g["DPD"].max(), 0.0239, 1e-3))
check("t1 gender min DIR 0.891 (M2)", near(t1g["DIR"].min(), 0.8915, 1e-3))
# M2 gender chi2 at top-1000 = 0.045
r2 = t1[(t1.model == "M2-Multimodal (CV7+Face)") & (t1.attribute == "gender")].iloc[0]
check("t1 M2 gender chi2 p 0.045", near(r2["chi2_p"], 0.0449, 1e-3))
# p75 M3 gender EOD
rm3 = p75[(p75.model == "M3-Gender-Biased (CV7)") & (p75.attribute == "gender")].iloc[0]
check("p75 M3 gender EOD 0.087", near(rm3["EOD"], 0.0872, 1e-3))
check("p75 M3 EOD CI [0.069,0.106]", near(rm3["EOD_ci_lo"], 0.0691, 1e-3) and near(rm3["EOD_ci_hi"], 0.1062, 1e-3))
# M3 p75 gender TPRs
m3g_pg = p75pg[(p75pg.model == "M3-Gender-Biased (CV7)") & (p75pg.attribute == "gender")]
tp = {r["group"]: r["tpr"] for _, r in m3g_pg.iterrows()}
check("p75 M3 Female TPR 1.00", near(tp["Female"], 1.0, 1e-3))
check("p75 M3 Male TPR 0.913", near(tp["Male"], 0.9128, 1e-3))
# M5/M6 p75 EOD = 0
for m in ["M5-Robust (CV9)", "M6-Robust (CV9+Face)"]:
    for a in ["gender", "ethnicity"]:
        rr = p75[(p75.model == m) & (p75.attribute == a)].iloc[0]
        check(f"p75 {m} {a} EOD=0", rr["EOD"] < 1e-9)
# blind-model top-1000 CI dips below 0.80
for m in ["M1-Fair (CV7)", "M2-Multimodal (CV7+Face)", "M5-Robust (CV9)", "M6-Robust (CV9+Face)"]:
    rr = t1[(t1.model == m) & (t1.attribute == "ethnicity")].iloc[0]
    check(f"t1 {m} eth DIR CI lo < 0.80", rr["DIR_ci_lo"] < 0.80)
    rp2 = p75[(p75.model == m) & (p75.attribute == "ethnicity")].iloc[0]
    check(f"p75 {m} eth DIR CI entirely > 0.80", rp2["DIR_ci_lo"] > 0.80)
# p75 thresholds
for m, exp in [("M1-Fair (CV7)", 0.499881), ("M3-Gender-Biased (CV7)", 0.459746), ("M4-Ethnicity-Biased (CV7)", 0.519054)]:
    rr = p75[(p75.model == m) & (p75.attribute == "gender")].iloc[0]
    check(f"p75 threshold {m} {exp}", near(rr["threshold"], exp, 1e-5))
# p75 n_selected
for m, exp in [("M1-Fair (CV7)", 2288), ("M2-Multimodal (CV7+Face)", 2289),
               ("M5-Robust (CV9)", 2333), ("M6-Robust (CV9+Face)", 2342),
               ("M3-Gender-Biased (CV7)", 2455), ("M4-Ethnicity-Biased (CV7)", 2194)]:
    rr = p75[(p75.model == m) & (p75.attribute == "gender")].iloc[0]
    check(f"p75 n_selected {m} {exp}", int(rr["n_selected"]) == exp)
# M4 top-1000 gender DIR 0.989 DPD 0.0023
rm4g = t1[(t1.model == "M4-Ethnicity-Biased (CV7)") & (t1.attribute == "gender")].iloc[0]
check("t1 M4 gender DIR 0.989", near(rm4g["DIR"], 0.9892, 1e-3))
check("t1 M4 gender DPD 0.0023", near(rm4g["DPD"], 0.0023, 1e-3))

# ── §8.4 top-2% numbers vs topn_metrics.csv ─────────────────────────────────
top2 = topn[topn.threshold == 0.02]
for m, exp_dir, exp_lo, exp_hi in [
    ("M1-Fair (CV7)", 0.655, 0.364, 0.900),
    ("M2-Multimodal (CV7+Face)", 0.527, 0.325, 0.855),
    ("M3-Gender-Biased (CV7)", 0.646, 0.359, 0.922),
    ("M4-Ethnicity-Biased (CV7)", 0.481, 0.236, 0.706),
    ("M5-Robust (CV9)", 0.690, 0.387, 0.904),
    ("M6-Robust (CV9+Face)", 0.672, 0.365, 0.898),
]:
    rr = top2[(top2.model == m) & (top2.attribute == "ethnicity")].iloc[0]
    ok = (near(rr["DIR"], exp_dir, 1e-3) and near(rr["DIR_ci_lo"], exp_lo, 5e-3) and near(rr["DIR_ci_hi"], exp_hi, 5e-3))
    check(f"top2% {m} DIR", ok, f"csv DIR={rr['DIR']:.3f} [{rr['DIR_ci_lo']:.3f},{rr['DIR_ci_hi']:.3f}]")
rr = top2[(top2.model == "M2-Multimodal (CV7+Face)") & (top2.attribute == "ethnicity")].iloc[0]
check("top2% M2 chi2 0.046", near(rr["chi2_p"], 0.0461, 1e-3))
rr = top2[(top2.model == "M4-Ethnicity-Biased (CV7)") & (top2.attribute == "ethnicity")].iloc[0]
check("top2% M4 chi2 0.014", near(rr["chi2_p"], 0.0144, 1e-3))
check("top2% M4 CI entirely < 0.80", rr["DIR_ci_hi"] < 0.80)

# ── misc derived claims ─────────────────────────────────────────────────────
check("M4 ethnicity worst group G3 (median SR)", pg[(pg.model == "M4-Ethnicity-Biased (CV7)") & (pg.attribute == "ethnicity")]["selection_rate"].idxmin() ==
      pg[(pg.model == "M4-Ethnicity-Biased (CV7)") & (pg.attribute == "ethnicity")].index[pg[(pg.model == "M4-Ethnicity-Biased (CV7)") & (pg.attribute == "ethnicity")]["group"] == "G3"][0])
# M4 ethnicity G3 SR lowest
m4epg = pg[(pg.model == "M4-Ethnicity-Biased (CV7)") & (pg.attribute == "ethnicity")]
check("M4 eth G3 lowest SR", m4epg.loc[m4epg["selection_rate"].idxmin(), "group"] == "G3")
# M4 highest TPR group = G3, PPV lowest = G3
check("M4 eth G3 highest TPR", m4epg.loc[m4epg["tpr"].idxmax(), "group"] == "G3")
check("M4 eth G3 lowest PPV", m4epg.loc[m4epg["ppv"].idxmin(), "group"] == "G3")
# M3 gender TPR Male 0.675 Female 0.873 (in table already)
# blind-label p75 ≈ median SR (internal check)
for m in ["M1-Fair (CV7)", "M2-Multimodal (CV7+Face)", "M5-Robust (CV9)", "M6-Robust (CV9+Face)"]:
    for a in ["gender", "ethnicity"]:
        pgr = p75pg[(p75pg.model == m) & (p75pg.attribute == a)]
        diff = (pgr["selection_rate"] - pgr["median_selection_rate"]).abs().max()
        check(f"p75≈median SR {m} {a} (max diff < 1e-4)", diff < 1e-4, f"max diff={diff:.2e}")

# ── summary ─────────────────────────────────────────────────────────────────
print(f"Total checks: {len(checks)}")
for name, ok, detail in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not ok else ""))
print()
if fails:
    print(f"FAILED ({len(fails)}):")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED.")
