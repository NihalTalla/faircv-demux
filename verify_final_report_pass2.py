"""Second verification pass for FINAL_REPORT.md.

Cross-checks claims that were sourced from PROSE artifacts (audit_report.txt,
dataset_ground_truth.md, audit_code_review.md, faircv_audit_v2_validation.md,
topn_screening.md) rather than from the metric CSVs:

  A. audit_report.txt per-model blocks must agree with results/metrics.csv
     (point estimates + all bootstrap CIs).
  B. audit_report.txt dataset section: medians, hired rates, bias ratios.
  C. dataset_ground_truth.md numeric claims referenced by the report.
  D. audit_code_review.md issue/claim inventory referenced by the report.
  E. faircv_audit_v2_validation.md claims referenced by the report.
  F. topn_screening.md tables vs results/topn_metrics.csv + topn_per_group.csv.

Read-only. Run: python verify_final_report_pass2.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

R = "results"
fails = []
checks = []


def check(name, cond, detail=""):
    checks.append((name, bool(cond)))
    if not cond:
        fails.append((name, detail))


def near(a, b, tol=1e-3):
    return abs(float(a) - float(b)) <= tol


med = pd.read_csv(os.path.join(R, "metrics.csv"))
report = open(os.path.join(R, "audit_report.txt"), encoding="utf-8").read()
gt = open("dataset_ground_truth.md", encoding="utf-8").read()
review = open("audit_code_review.md", encoding="utf-8").read()
val = open("faircv_audit_v2_validation.md", encoding="utf-8").read()
topn_md = open("topn_screening.md", encoding="utf-8").read()
topn = pd.read_csv(os.path.join(R, "topn_metrics.csv"))
topn_pg = pd.read_csv(os.path.join(R, "topn_per_group.csv"))

# ── A. audit_report.txt blocks vs metrics.csv ───────────────────────────────
# split report into per-model sections
model_names = [m for m in med.model.unique()]
sections = {}
for m in model_names:
    start = report.index(f"{m}   (features=")
    # the model-name line is followed by a heavy separator that belongs to this
    # section's header; the section ends at the NEXT heavy separator (next model)
    first = report.find("━━━", start + 10)
    run_end = first
    while run_end < len(report) and report[run_end] == "━":
        run_end += 1
    nxt = report.find("━━━", run_end) if first != -1 else -1
    end = report.find("FINDINGS (computed")
    candidates = [i for i in [nxt, end] if i > start]
    sections[m] = report[start:min(candidates)]


def parse_block(block, attr):
    """Pull DPD/DIR/EOD/EO/KL points + bootstrap CI from one attr section."""
    pats = {
        "DPD": r"DPD = ([+-][\d.]+)",
        "DIR": r"DIR = ([\d.]+)",
        "EOD": r"EOD = ([+-][\d.]+)",
        "EO": r"EO  = ([+-][\d.]+)",
        "KL": r"KL \(extreme SR pair\) = ([\d.]+)",
    }
    out = {}
    for k, p in pats.items():
        m = re.search(p, block)
        if m:
            out[k] = float(m.group(1))
    ci = re.search(r"Bootstrap 95% CI \(n=2000\): DPD \[([\d.]+), ([\d.]+)\]  "
                   r"DIR \[([\d.]+), ([\d.]+)\]  EOD \[([\d.]+), ([\d.]+)\]  "
                   r"EO \[([\d.]+), ([\d.]+)\]  KL \[([\d.]+), ([\d.]+)\]", block)
    if ci:
        out["cis"] = {k: (float(ci.group(i)), float(ci.group(i + 1)))
                      for k, i in [("DPD", 1), ("DIR", 3), ("EOD", 5), ("EO", 7), ("KL", 9)]}
    return out


for m in model_names:
    block = sections[m]
    for attr in ("gender", "ethnicity"):
        seg = block
        # split gender/ethnicity halves
        gi = seg.index("Protected attribute: GENDER")
        ei = seg.index("Protected attribute: ETHNICITY")
        if attr == "gender":
            sub = seg[gi:ei]
        else:
            sub = seg[ei:]
        parsed = parse_block(sub, attr)
        row = med[(med.model == m) & (med.attribute == attr)].iloc[0]
        for k in ("DPD", "DIR", "EOD", "EO", "KL"):
            col = k if k != "KL" else "KL_extreme"
            check(f"report-txt {m} {attr} {k} point",
                  near(parsed[k], row[col], 5e-4),
                  f"txt={parsed[k]} csv={row[col]}")
        for k in ("DPD", "DIR", "EOD", "EO"):  # these have CI columns in metrics.csv
            cl = {"DPD": "DPD_ci_lo", "DIR": "DIR_ci_lo", "EOD": "EOD_ci_lo", "EO": "EO_ci_lo"}[k]
            ch = {"DPD": "DPD_ci_hi", "DIR": "DIR_ci_hi", "EOD": "EOD_ci_hi", "EO": "EO_ci_hi"}[k]
            check(f"report-txt {m} {attr} {k} CI",
                  near(parsed["cis"][k][0], row[cl], 5e-4) and near(parsed["cis"][k][1], row[ch], 5e-4),
                  f"txt={parsed['cis'][k]} csv=({row[cl]},{row[ch]})")
        # KL CIs exist only in prose. The bootstrap KL distribution is
        # upward-biased (documented in audit_report.txt), so percentile CIs may
        # sit entirely above the plug-in point estimate. Sanity check: the CI
        # upper bound never lies below the point estimate.
        klo, khi = parsed["cis"]["KL"]
        check(f"report-txt {m} {attr} KL CI sane",
              khi >= parsed["KL"] - 1e-6 and klo >= 0,
              f"CI=[{klo},{khi}] point={parsed['KL']}")

# KL CIs from the validation doc §3 table vs audit_report.txt
val_kl = {}
for line in val.splitlines():
    mrow = re.match(r"\| (M\d[^|]*?) \| ([a-z]+) \|.*?KL \[([\d.]+), ([\d.]+)\] \|", line)
    if mrow:
        mname = {"M1": "M1-Fair (CV7)", "M2": "M2-Multimodal (CV7+Face)",
                 "M3": "M3-Gender-Biased (CV7)", "M4": "M4-Ethnicity-Biased (CV7)",
                 "M5": "M5-Robust (CV9)", "M6": "M6-Robust (CV9+Face)"}[mrow.group(1)]
        val_kl[(mname, mrow.group(2))] = (float(mrow.group(3)), float(mrow.group(4)))
for m in model_names:
    for attr in ("gender", "ethnicity"):
        if (m, attr) in val_kl:
            seg = sections[m]
            gi = seg.index("Protected attribute: GENDER")
            ei = seg.index("Protected attribute: ETHNICITY")
            sub = seg[gi:ei] if attr == "gender" else seg[ei:]
            parsed = parse_block(sub, attr)
            klo, khi = parsed["cis"]["KL"]
            vlo, vhi = val_kl[(m, attr)]
            check(f"valdoc KL CI {m} {attr}", near(klo, vlo, 5e-4) and near(khi, vhi, 5e-4),
                  f"txt=({klo},{khi}) valdoc=({vlo},{vhi})")

# ── B. dataset section of audit_report.txt ──────────────────────────────────
check("txt medians blind 0.4135", "blind: 0.4135" in report)
check("txt medians gender 0.3659", "gender: 0.3659" in report)
check("txt medians ethnicity 0.4148", "ethnicity: 0.4148" in report)
check("txt hired rates", "blind=0.4929 gender=0.4946 ethnicity=0.4885" in report)
check("txt bias ratios", all(s in report for s in
      ["Female   mean(ratio) = 0.7502", "G1       mean(ratio) = 1.2481", "G3       mean(ratio) = 0.7478"]))
check("txt cols31-50 std ~3.2e-6", "3.2e-06" in report or "3.2e-6" in report)
check("txt performance M1 acc=0.793", "M1-Fair (CV7)                          acc=0.793  AUC=0.888  F1=0.786" in report)
check("txt performance M5 acc=0.966", "M5-Robust (CV9)                        acc=0.966  AUC=0.996  F1=0.965" in report)

# ── C. dataset_ground_truth.md claims referenced by FINAL_REPORT ────────────
gt_checks = {
    "train/test 19,200/4,800": "19,200" in gt and "4,800" in gt,
    "gender balance 49.8/50.2": "49.8/50.2" in gt,
    "ethnicity ~33/33/33": "33.4/33.4/33.2" in gt,
    "suitability corr ~0.48": "0.48" in gt,
    "occupation 10 / suitability 4 levels": "0.25, 0.5, 0.75, 1.0" in gt,
    "bias ratio 0.7502": "0.7502" in gt,
    "bias ratio 1.2481": "1.2481" in gt,
    "bias ratio 0.7478": "0.7478" in gt,
    "no missing values": "No missing values anywhere" in gt,
    "0 row overlap": "0 of 4,800" in gt,
    "1,162 names": "1,162 names" in gt,
    "blind face std ~2-3e-6": "2–3e-6" in gt or "2-3e-6" in gt,
    "face corr gender 0.456": "0.456" in gt,
    "cols 31-50 constant": "constant vector" in gt,
    "median blind 0.4135": "0.4135" in gt,
    "median gender 0.3659": "0.3659" in gt,
    "median eth 0.4148": "0.4148" in gt,
    "label agreement 0.933/0.855/0.798": "0.933" in gt and "0.855" in gt and "0.798" in gt,
    "bias penalty factor paper": "penalty factor" in gt,
}
for k, v in gt_checks.items():
    check(f"gt: {k}", v)

# ── D. audit_code_review.md claims referenced by FINAL_REPORT ───────────────
review_checks = {
    "reproduce within 5e-4": "5e-4" in review,
    "M1 gender KS 0.318": "0.318" in review,
    "M3 passes both attributes": "M3 passes both attributes" in review,
    "only M4/ethnicity fails DIR 0.771": "0.771" in review,
    "ethnicity KL 0.0255 < 0.0295": "0.0255" in review and "0.0295" in review,
    "M4 gender EOD 0.011 vs M1 0.012": "0.011" in review and "0.012" in review,
    "BUG-1 present": "BUG-1" in review,
    "BUG-11 present": "BUG-11" in review,
    "N-1 present": "N-1" in review,
    "N-10 present": "N-10" in review,
    "R-1..R-15 listed": all(f"R-{i}" in review for i in range(1, 16)),
    "median binarisation artificial (N-4)": "N-4" in review,
    "suitability strongest predictor": "strongest" in review,
}
for k, v in review_checks.items():
    check(f"review: {k}", v)

# ── E. validation doc claims referenced by FINAL_REPORT ─────────────────────
val_checks = {
    "gender AUC 0.93 (face block)": "0.93" in val,
    "sha256 provenance": "sha256" in val,
    "same size as LFS object": "203,041,354" in val,
    "cols 31-50 verdict": "constant block" in val,
    "v2 reproduces v1 numerics": "identical" in val,
    "M4 DIR CI straddles 0.80": "0.718, 0.828" in val,
    "M3 gender EOD 0.198": "0.198" in val,
    "bootstrap n=2,000": "2,000" in val,
}
for k, v in val_checks.items():
    check(f"valdoc: {k}", v)

# ── F. topn_screening.md tables vs topn CSVs ────────────────────────────────
# M4 ethnicity DIR per threshold (from the md table)
m4dir = {0.02: 0.481, 0.05: 0.691, 0.10: 0.752, 0.20: 0.702, 0.30: 0.762, 0.50: 0.767}
for k, exp in m4dir.items():
    row = topn[(topn.model == "M4-Ethnicity-Biased (CV7)") & (topn.attribute == "ethnicity") & (topn.threshold == k)].iloc[0]
    check(f"topn-md M4 eth DIR {int(k*100)}%", near(row["DIR"], exp, 1e-3), f"csv={row['DIR']:.3f}")
# M4 per-group SRs at 2% / 20% / 50%
m4_sr = {0.02: {"G1": 0.0245, "G2": 0.0239, "G3": 0.0118},
         0.20: {"G1": 0.2389, "G2": 0.1940, "G3": 0.1676},
         0.50: {"G1": 0.5730, "G2": 0.4880, "G3": 0.4397}}
for k, grps in m4_sr.items():
    for g, exp in grps.items():
        row = topn_pg[(topn_pg.model == "M4-Ethnicity-Biased (CV7)") & (topn_pg.attribute == "ethnicity")
                      & (topn_pg.threshold == k) & (topn_pg.group == g)].iloc[0]
        check(f"topn-md M4 eth {g} SR @ {int(k*100)}%", near(row["selection_rate"], exp, 5e-4),
              f"csv={row['selection_rate']:.4f}")
# M3 gender EOD at 50% = 0.178
r = topn[(topn.model == "M3-Gender-Biased (CV7)") & (topn.attribute == "gender") & (topn.threshold == 0.50)].iloc[0]
check("topn-md M3 gender EOD @50% = 0.178", near(r["EOD_labels"], 0.178, 1e-3), f"csv={r['EOD_labels']:.4f}")
# top-2% ethnicity DIRs (md table) vs csv
for m, exp in [("M1-Fair (CV7)", 0.655), ("M2-Multimodal (CV7+Face)", 0.527),
               ("M3-Gender-Biased (CV7)", 0.646), ("M4-Ethnicity-Biased (CV7)", 0.481),
               ("M5-Robust (CV9)", 0.690), ("M6-Robust (CV9+Face)", 0.672)]:
    row = topn[(topn.model == m) & (topn.attribute == "ethnicity") & (topn.threshold == 0.02)].iloc[0]
    check(f"topn-md {m} DIR @2%", near(row["DIR"], exp, 1e-3), f"csv={row['DIR']:.3f}")
# validation table in topn_screening.md: top-50% DIR vs median-split DIR
t50 = topn[topn.threshold == 0.50]
for m in ["M1-Fair (CV7)", "M2-Multimodal (CV7+Face)", "M3-Gender-Biased (CV7)",
          "M4-Ethnicity-Biased (CV7)", "M5-Robust (CV9)", "M6-Robust (CV9+Face)"]:
    for a in ("gender", "ethnicity"):
        row = t50[(t50.model == m) & (t50.attribute == a)].iloc[0]
        check(f"topn-md {m} {a} @50% dir==median_dir",
              near(row["DIR"], row["median_split_DIR"], 2e-2),
              f"dir={row['DIR']:.4f} med={row['median_split_DIR']:.4f}")

# ── G. bio companion claims (Appendix A of FINAL_REPORT.md) ───────────────
bio_metrics = pd.read_csv(os.path.join(R, "bio_metrics.csv"))
bio_leak = pd.read_csv(os.path.join(R, "bio_leakage.csv"))
bio_strong = pd.read_csv(os.path.join(R, "bio_leak_strong.csv"))
bio_ft = pd.read_csv(os.path.join(R, "bio_leak_finetune.csv"))

bio_hiring = {  # (model, attr): (DPD, DIR, EOD, KL, chi2, KW)
    ("BIO-BioBlind", "gender"): (0.0039, 0.9923, 0.0075, 0.0222, 0.8077, None),
    ("BIO-BioBlind", "ethnicity"): (0.0344, 0.9347, 0.0102, 0.0778, 0.1276, 0.1801),
    ("BIO-BioOriginal", "gender"): (0.0121, 0.9761, 0.0168, 0.0401, 0.4173, None),
    ("BIO-BioOriginal", "ethnicity"): (0.0436, 0.9170, 0.0315, 0.0538, 0.0403, 0.1433),
}
for (m, a), (d, dir_, eod, kl, c, kw) in bio_hiring.items():
    row = bio_metrics[(bio_metrics.model == m) & (bio_metrics.attribute == a)].iloc[0]
    ok = (near(row["DPD"], d, 5e-4) and near(row["DIR"], dir_, 5e-4)
          and near(row["EOD"], eod, 5e-4) and near(row["KL"], kl, 5e-4)
          and near(row["chi2_p"], c, 5e-3))
    if kw is not None:
        ok = ok and near(row["KW_p"], kw, 5e-3)
    check(f"bio-hiring {m} {a}", ok)
    check(f"bio-hiring {m} {a} EEOC pass", bool(row["EEOC_pass"]))

bio_leak_expected = {
    "gender | original bios": (0.9975, 0.99998),
    "gender | blind bios": (0.6452, 0.7091),
    "gender | names (control)": (0.9165, 0.9711),
    "ethnicity | original bios": (0.3229, 0.4945),
    "ethnicity | blind bios": (0.3246, 0.4969),
    "ethnicity | names (control)": (0.3235, 0.4954),
}
for ch, (acc, auc) in bio_leak_expected.items():
    row = bio_leak[bio_leak.channel == ch].iloc[0]
    check(f"bio-leak {ch}", near(row["accuracy"], acc, 5e-4) and near(row["auc"], auc, 5e-4))

ladder = {t: a for t, a in zip(bio_strong["tier"], bio_strong["auc"])}
check("bio-ladder T0 0.7091", near(ladder["T0"], 0.7091, 5e-4))
check("bio-ladder T1 0.7037", near(ladder["T1"], 0.7037, 5e-4))
check("bio-ladder T2 0.6589", near(ladder["T2"], 0.6589, 5e-4))
check("bio-ladder T3 0.6799", near(ladder["T3"], 0.6799, 5e-4))
check("bio-ladder: no frozen tier exceeds T0", max(ladder.values()) == ladder["T0"])

ft = bio_ft.iloc[0]
check("bio-finetune epoch 1 test AUC 0.7592", near(ft["test_auc"], 0.7592, 5e-4))
check("bio-finetune exceeds word-level baseline", ft["test_auc"] > 0.7091)
check("bio-finetune train AUC 0.8217", near(ft["train_auc"], 0.8217, 5e-4))
check("bio-finetune test acc 0.6846", near(ft["test_acc"], 0.6846, 5e-4))

# ── summary ─────────────────────────────────────────────────────────────────
print(f"Total checks: {len(checks)}")
for (name, ok) in checks:
    if not ok:
        print(f"  FAIL  {name}")
nfail = len(fails)
print(f"FAILED: {nfail}")
if nfail:
    for name, detail in fails:
        print(f"  - {name}: {detail}")
    raise SystemExit(1)
print("ALL PASS-2 CHECKS PASSED.")
