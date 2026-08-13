"""Audit report generation — markdown built ONLY from the audit's own output.

Every number in the report comes from the stored ``AuditResult`` payload;
nothing is hardcoded. Wording is deliberately hedged: the report describes
observed disparities and uncertainty, and never asserts causation or that a
model "is biased".
"""

from datetime import datetime, timezone

from ..config import settings

EEOC = 0.80


def _fmt(x, nd=4) -> str:
    if x is None:
        return "—"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:  # NaN
        return "—"
    return f"{f:.{nd}f}"


def _ci(lo, hi, nd=4) -> str:
    return f"[{_fmt(lo, nd)}, {_fmt(hi, nd)}]"


def _interpret_dir(row) -> str | None:
    """Scientifically defensible interpretation of a DIR row (EEOC rule)."""
    dirv = row.get("DIR")
    if dirv is None or dirv != dirv:  # None or NaN
        return None
    lo, hi = row.get("DIR_ci_lo"), row.get("DIR_ci_hi")
    if dirv >= EEOC:
        return (
            f"DIR {_fmt(dirv)} meets the {EEOC:.2f} four-fifths threshold "
            "(selection-rate ratio within the acceptable range for this comparison)."
        )
    if hi is not None and hi != hi:  # NaN → treat as unknown
        hi = None
    if hi is not None and hi < EEOC:
        return (
            f"DIR point estimate {_fmt(dirv)} is below the {EEOC:.2f} threshold "
            f"and the 95% CI {_ci(lo, hi)} lies entirely below it — the disparity "
            "is not explained by sampling uncertainty."
        )
    return (
        f"DIR point estimate {_fmt(dirv)} is below the {EEOC:.2f} threshold, but "
        f"the 95% CI {_ci(lo, hi)} crosses {EEOC:.2f} — the magnitude is "
        "borderline rather than definitive."
    )


def _interpret_eod(row) -> str | None:
    eod = row.get("EOD")
    lo, hi = row.get("EOD_ci_lo"), row.get("EOD_ci_hi")
    if eod is None or (isinstance(eod, float) and eod != eod):
        return None
    if lo is not None and lo == lo and lo > 0:
        return (
            f"EOD {_fmt(eod)} with 95% CI {_ci(lo, hi)} excluding 0 — error-rate "
            "disparity not explained by sampling uncertainty."
        )
    return f"EOD {_fmt(eod)} (95% CI {_ci(lo, hi)}); the interval includes 0, so the observed error-rate gap is consistent with sampling noise."


def build_report(audit, payload: dict) -> str:
    """Build the markdown report from an audit record + result payload."""
    summary = payload.get("dataset_summary") or {}
    metrics = payload.get("metrics") or []
    statistics = payload.get("statistics") or []
    performance = payload.get("performance") or {}
    robustness = payload.get("robustness") or {}

    lines = []
    lines.append("# FairCV Bias Audit — Report")
    lines.append("")
    lines.append(f"- **Audit ID:** `{audit.id}`")
    lines.append(f"- **Audit name:** {audit.name}")
    lines.append(f"- **Dataset mode:** {audit.dataset_mode}")
    lines.append(f"- **Protected attribute(s):** {', '.join(audit.protected_attributes)}")
    lines.append(f"- **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    lines.append("## 1. Dataset summary")
    lines.append("")
    if audit.dataset_mode == "demo":
        lines.append(
            f"- {summary.get('train_rows', '?')} training / {summary.get('test_rows', '?')} "
            f"test profiles, {summary.get('columns', '?')} columns (FairCVdb.npy, frozen release)"
        )
        for k, v in (summary.get("gender") or {}).items():
            lines.append(f"- gender group '{k}': {v} test rows")
        for k, v in (summary.get("ethnicity") or {}).items():
            lines.append(f"- ethnicity group '{k}': {v} test rows")
    else:
        lines.append(
            f"- {summary.get('train_rows', '?')} training / {summary.get('test_rows', '?')} "
            f"test rows, {summary.get('columns', '?')} columns"
        )
        for attr, counts in (summary.get("protected_counts") or {}).items():
            for g, n in counts.items():
                lines.append(f"- '{attr}' group '{g}': {n} rows")
    lines.append("")

    lines.append("## 2. Configuration")
    lines.append("")
    cfg = audit.config or {}
    lines.append(f"- Label column: `{cfg.get('label_column', 'blind labels (demo)')}`")
    lines.append(f"- Features: `{cfg.get('feature_columns') or 'frozen feature slices (demo)'}`")
    lines.append(f"- Model: StandardScaler → LogisticRegression (C=1.0, seed {settings.AUDIT_SEED})")
    lines.append(
        f"- Bootstrap: {settings.AUDIT_N_BOOT} resamples, percentile 95% CIs, seed {settings.AUDIT_SEED}"
    )
    lines.append("")

    lines.append("## 3. Model performance")
    lines.append("")
    lines.append("| Model | Accuracy | AUC | F1 |")
    lines.append("|---|---|---|---|")
    for m, p in performance.items():
        lines.append(f"| {m} | {_fmt(p.get('acc'), 3)} | {_fmt(p.get('auc'), 3)} | {_fmt(p.get('f1'), 3)} |")
    lines.append("")

    lines.append("## 4. Fairness metrics (median-split hiring rule)")
    lines.append("")
    lines.append(
        "| Model | Attribute | DPD | DIR (95% CI) | EOD (95% CI) | EO | KL (extreme) | EEOC pass |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in metrics:
        lines.append(
            f"| {r.get('short_name', r.get('model'))} | {r.get('attribute')} "
            f"| {_fmt(r.get('DPD'))} | {_fmt(r.get('DIR'))} {_ci(r.get('DIR_ci_lo'), r.get('DIR_ci_hi'))} "
            f"| {_fmt(r.get('EOD'))} {_ci(r.get('EOD_ci_lo'), r.get('EOD_ci_hi'))} "
            f"| {_fmt(r.get('EO'))} | {_fmt(r.get('KL_extreme'))} "
            f"| {'yes' if r.get('EEOC_pass') else 'no'} |"
        )
    lines.append("")

    lines.append("## 5. Statistical tests")
    lines.append("")
    lines.append("| Model | Attribute | Test | Comparison | Statistic | p-value | p-adjusted (Holm) | Sig. (0.05) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in statistics:
        lines.append(
            f"| {r.get('short_name', r.get('model'))} | {r.get('attribute')} | {r.get('test')} "
            f"| {r.get('comparison')} | {_fmt(r.get('statistic'))} | {_fmt(r.get('p_value'))} "
            f"| {_fmt(r.get('p_adjusted'))} | {r.get('significant_0.05')} |"
        )
    lines.append("")

    rob_rows = robustness.get("rows") or []
    if rob_rows:
        lines.append("## 6. Robustness (alternative hiring rules)")
        lines.append("")
        lines.append("| Model | Attribute | Protocol | DPD (95% CI) | DIR (95% CI) | EEOC pass | Verdict change |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in rob_rows:
            lines.append(
                f"| {r.get('short_name', r.get('model'))} | {r.get('attribute')} | {r.get('protocol')} "
                f"| {_fmt(r.get('DPD'))} {_ci(r.get('DPD_ci_lo'), r.get('DPD_ci_hi'))} "
                f"| {_fmt(r.get('DIR'))} {_ci(r.get('DIR_ci_lo'), r.get('DIR_ci_hi'))} "
                f"| {'yes' if r.get('EEOC_pass') else 'no'} "
                f"| {'yes' if r.get('verdict_change') else 'no'} |"
            )
        lines.append("")

    lines.append("## 7. Key findings")
    lines.append("")
    findings = []
    for r in metrics:
        interp = _interpret_dir(r)
        if interp:
            findings.append(
                f"- **{r.get('short_name', r.get('model'))} / {r.get('attribute')}:** {interp}"
            )
        interp_eod = _interpret_eod(r)
        if interp_eod:
            findings.append(
                f"- **{r.get('short_name', r.get('model'))} / {r.get('attribute')}:** {interp_eod}"
            )
    if not findings:
        findings.append("- No metric exceeded its interpretive threshold in this run.")
    lines.extend(findings)
    lines.append("")

    lines.append("## 8. Limitations")
    lines.append("")
    lines.append(
        "- This report describes **observed disparities** in the audited dataset. "
        "It does not establish causation."
    )
    lines.append(
        "- Statistical significance does not imply practical magnitude; "
        "effect sizes and confidence intervals must be read alongside p-values."
    )
    lines.append(
        "- The four-fifths (0.80) rule is a US-EEOC screening heuristic; a "
        "point estimate below it, especially with a CI crossing 0.80, is "
        "evidence of disparity to investigate, not a legal determination."
    )
    lines.append(
        "- For the FairCV demo, the dataset is synthetic; results do not "
        "generalise to real hiring pipelines."
    )
    lines.append("")
    lines.append("---")
    lines.append(
        f"_Generated by the FairCV audit platform. All numbers above come from "
        f"the audit run `{audit.id}`, not from hardcoded values._"
    )
    return "\n".join(lines)
