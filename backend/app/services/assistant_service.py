"""AI assistant service — Groq-backed explanation layer for FairCV audits.

Architecture:
  React frontend  →  POST /api/assistant/chat  →  this service  →  Groq API
                                                        ↑
                                               AuditResult DB (real data)

SECURITY: GROQ_API_KEY is read from settings (backend env only).
This module never touches the frontend and never exposes the key.

The service NEVER fabricates metric values, p-values, CIs, or any computed
number. It only cites values present in the AuditResult record. If data is
unavailable the system prompt instructs the model to say so explicitly.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from sqlalchemy.orm import Session

from ..config import settings
from ..models import Audit, AuditResult
from ..schemas import AssistantChatRequest, AssistantChatResponse, AssistantMessage

log = logging.getLogger(__name__)

# ─── Lazy Groq client singleton ───────────────────────────────────────────────

_client = None


def _get_client():
    global _client
    if _client is None:
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not configured on this server. "
                "Add it to the backend .env file. "
                "Never put the Groq API key in the React/Vite frontend."
            )
        try:
            from groq import Groq  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "The 'groq' package is not installed. "
                "Run: pip install groq>=0.11"
            ) from exc
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


# ─── System prompt ────────────────────────────────────────────────────────────

_BASE_SYSTEM = """\
You are the FairCV AI Assistant — an expert explanation layer for the FairCV bias-audit platform.
FairCV audits automated CV/résumé screening models for demographic fairness using a rigorous,
peer-reviewed statistical methodology. Your role is to help users understand audit methodology,
metrics, statistical tests, and — when audit results are loaded — to interpret real computed results.

## What you can explain (general knowledge, no audit data needed)
- The FairCV project: purpose, dataset (recruitment CV dataset, ~24,000 samples, train/test split),
  six logistic regression models M1–M6 that vary by feature set and hiring-label definition
- M1: full features, strict label | M2: full features, lenient label | M3: reduced features, strict label
  M4: reduced features, lenient label | M5: minimal features, strict label | M6: minimal features, lenient label
- Fairness metrics:
    DPD (Demographic Parity Difference): SR_privileged − SR_disadvantaged; 0 = parity
    DIR (Disparate Impact Ratio): SR_disadvantaged / SR_privileged; EEOC threshold ≥ 0.80
    EOD (Equal Opportunity Difference): TPR_privileged − TPR_disadvantaged
    EO (Equalised Odds): combined false-positive and false-negative rate disparity
    KL divergence: measures distributional divergence between score distributions of groups
- Statistical tests: two-sample KS test, Kruskal-Wallis H-test, Pearson chi-square test
- Holm-Bonferroni correction: a step-down multiple-comparison procedure applied across all pairwise tests
- Bootstrap confidence intervals: 2,000 resamples, 95th-percentile (percentile method) CIs;
  wider CI = more uncertainty; CI crossing 0 (for DPD) or 0.80 (for DIR) = ambiguous evidence
- EEOC four-fifths rule: DIR ≥ 0.80 passes; DIR < 0.80 signals potential disparate impact
- Cohen's d (score-distribution effect size) and Cohen's h (selection-rate effect size)
- Robustness analysis: three decision rules — Median threshold, 75th-percentile threshold, Top-N hiring;
  a finding is "robust" if it holds across all three rules
- Evidence strength vocabulary: strong, moderate, limited, inconclusive

## What you MUST NOT do
- NEVER fabricate, estimate, or guess any numerical result (metric values, p-values,
  confidence intervals, bootstrap results, dataset statistics, group selection rates, etc.)
- If a specific number is not present in the audit context provided below, say clearly:
  "I don't have that value in the current audit context."
- Do NOT invent model performance figures (accuracy, AUC, F1) unless they appear below
- Do NOT replace the FairCV statistical engine — you are an explanation and interpretation layer only
- Do NOT suggest the user change the audit methodology, frozen results, or Python audit code

## Response format
- Use markdown: headers, bullet lists, bold for key terms, code spans for metric symbols
- Be precise and concise. Cite exact figures from the audit context when available.
- Distinguish real audit data from general explanations using phrases like:
  "According to the audit results..." vs "In general, ..."
"""

_NO_AUDIT_CONTEXT = """
## Audit Context
No audit is currently loaded. You can answer general questions about FairCV methodology,
metrics, and statistics. For audit-specific questions, tell the user to select a completed
audit in the assistant panel so you can cite real computed values.
"""

_AUDIT_CONTEXT_TEMPLATE = """
## Audit Context — REAL DATA FROM COMPLETED AUDIT
IMPORTANT: Cite ONLY the figures listed below. Do not invent any additional numbers.

Audit ID: {audit_id}
Audit Name: {audit_name}
Status: {status}
Protected attributes audited: {protected_attributes}
Dataset mode: {dataset_mode}
Completed: {completed_at}

### Model Performance (accuracy / AUC / F1 on held-out test set)
{performance_text}

### Fairness Metrics per model × attribute
DPD = Demographic Parity Difference | DIR = Disparate Impact Ratio | CIs = 95% bootstrap percentile
EOD = Equal Opportunity Difference | EO = Equalised Odds | KL = KL divergence (extreme pair)
EEOC: PASS = DIR ≥ 0.80 | FAIL = DIR < 0.80

{metrics_text}

### Statistical Tests (Holm-Bonferroni corrected at α=0.05)
{statistics_text}

### Dataset Summary
{dataset_summary_text}

### Robustness Analysis
{robustness_text}
"""


# ─── Format helpers ───────────────────────────────────────────────────────────

def _f(v: object) -> str:
    """Format a float to 4 decimal places, or '?' if missing."""
    if v is None:
        return "?"
    try:
        return f"{float(v):.4f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(v)


def _fmt_performance(performance: dict) -> str:
    if not performance:
        return "  (not available)"
    lines = []
    for model, perf in sorted(performance.items()):
        if isinstance(perf, dict):
            acc = _f(perf.get("acc"))
            auc = _f(perf.get("auc"))
            f1 = _f(perf.get("f1"))
            feat = perf.get("features", "")
            lset = perf.get("label_set", "")
            lines.append(
                f"  {model}: acc={acc}  auc={auc}  f1={f1}"
                + (f"  features={feat}" if feat else "")
                + (f"  label_set={lset}" if lset else "")
            )
    return "\n".join(lines) if lines else "  (not available)"


def _fmt_metrics(metrics: list) -> str:
    if not metrics:
        return "  (not available)"
    lines = []
    for row in metrics:
        if not isinstance(row, dict):
            continue
        m = row.get("model", "?")
        attr = row.get("attribute", "?")
        n = row.get("n", "?")
        dpd = _f(row.get("DPD"))
        dpd_lo = _f(row.get("DPD_ci_lo"))
        dpd_hi = _f(row.get("DPD_ci_hi"))
        dir_ = _f(row.get("DIR"))
        dir_lo = _f(row.get("DIR_ci_lo"))
        dir_hi = _f(row.get("DIR_ci_hi"))
        eod = _f(row.get("EOD"))
        eo = _f(row.get("EO"))
        kl = _f(row.get("KL_extreme"))
        eeoc = row.get("EEOC_pass")
        hi_sr = row.get("highest_SR_group", "?")
        hi_tpr = row.get("highest_TPR_group", "?")
        eeoc_str = "PASS" if eeoc is True else "FAIL" if eeoc is False else "?"
        lines.append(
            f"  {m} | attr={attr} | n={n}\n"
            f"    DPD={dpd} [{dpd_lo}–{dpd_hi}]  "
            f"DIR={dir_} [{dir_lo}–{dir_hi}]  EEOC={eeoc_str}\n"
            f"    EOD={eod}  EO={eo}  KL_extreme={kl}\n"
            f"    Highest-SR group: {hi_sr} | Highest-TPR group: {hi_tpr}"
        )
    return "\n".join(lines) if lines else "  (not available)"


def _fmt_statistics(statistics: list) -> str:
    if not statistics:
        return "  (not available)"
    all_lines = []
    sig_lines = []
    for row in statistics:
        if not isinstance(row, dict):
            continue
        m = row.get("model", "?")
        attr = row.get("attribute", "?")
        test = row.get("test", "?")
        comp = row.get("comparison", "?")
        stat = _f(row.get("statistic"))
        p = _f(row.get("p_value"))
        p_adj_raw = row.get("p_adjusted")
        p_adj = _f(p_adj_raw) if p_adj_raw is not None else "N/A"
        sig = row.get("significant_0.05", False)
        sig_mark = " *SIGNIFICANT*" if sig else ""
        line = (
            f"  {m} | {attr} | {test} | {comp}: "
            f"stat={stat}  p={p}  p_adj={p_adj}{sig_mark}"
        )
        all_lines.append(line)
        if sig:
            sig_lines.append(line)

    if len(all_lines) > 24:
        total = len(all_lines)
        sig_count = len(sig_lines)
        header = (
            f"  [{total} total tests; {sig_count} significant after Holm-Bonferroni at α=0.05]\n"
            f"  Significant tests:\n"
        )
        body = "\n".join(sig_lines) if sig_lines else "  (none)"
        return header + body

    return "\n".join(all_lines) if all_lines else "  (not available)"


def _fmt_dataset_summary(ds: dict) -> str:
    if not ds:
        return "  (not available)"
    try:
        return "  " + json.dumps(ds, indent=2).replace("\n", "\n  ")
    except Exception:
        return "  (not serializable)"


def _fmt_robustness(robustness: dict) -> str:
    if not robustness:
        return "  (not available)"
    try:
        lines = []
        for key, val in list(robustness.items())[:20]:  # cap at 20 keys
            snippet = json.dumps(val)[:300] if not isinstance(val, str) else val[:300]
            lines.append(f"  {key}: {snippet}")
        return "\n".join(lines) if lines else "  (empty)"
    except Exception:
        return "  (not serializable)"


# ─── Audit context builder ────────────────────────────────────────────────────

def build_audit_context(db: Session, audit_id: str) -> tuple[str, bool]:
    """Return (context_block, has_completed_data)."""
    audit = db.get(Audit, audit_id)
    if audit is None:
        return (
            _NO_AUDIT_CONTEXT
            + f"\n(Audit ID {audit_id!r} was not found in the database.)\n",
            False,
        )

    result: Optional[AuditResult] = db.get(AuditResult, audit_id)
    if result is None:
        return (
            _NO_AUDIT_CONTEXT
            + f"\nAudit '{audit.name}' (ID: {audit_id}) has not completed yet "
            f"(current status: {audit.status}). No metric results are available.\n",
            False,
        )

    completed = getattr(audit, "completed_at", None)
    completed_str = (
        completed.isoformat()
        if hasattr(completed, "isoformat")
        else str(completed or "unknown")
    )

    context = _AUDIT_CONTEXT_TEMPLATE.format(
        audit_id=audit_id,
        audit_name=audit.name,
        status=audit.status,
        protected_attributes=", ".join(audit.protected_attributes),
        dataset_mode=audit.dataset_mode,
        completed_at=completed_str,
        performance_text=_fmt_performance(result.performance or {}),
        metrics_text=_fmt_metrics(result.metrics or []),
        statistics_text=_fmt_statistics(result.statistics or []),
        dataset_summary_text=_fmt_dataset_summary(result.dataset_summary or {}),
        robustness_text=_fmt_robustness(result.robustness or {}),
    )
    return context, True


# ─── Chat entrypoint ──────────────────────────────────────────────────────────

_MAX_HISTORY = 10  # keep last N turns to avoid token overflow


def chat(db: Session, request: AssistantChatRequest) -> AssistantChatResponse:
    client = _get_client()

    has_ctx = False
    audit_id_used: Optional[str] = None
    if request.audit_id:
        audit_ctx, has_ctx = build_audit_context(db, request.audit_id)
        audit_id_used = request.audit_id
    else:
        audit_ctx = _NO_AUDIT_CONTEXT

    system_prompt = _BASE_SYSTEM + audit_ctx

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Include trimmed conversation history (oldest → newest, capped)
    history: list[AssistantMessage] = (request.history or [])[-_MAX_HISTORY:]
    for msg in history:
        if msg.role in ("user", "assistant"):
            messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": request.message})

    log.info(
        "assistant chat audit_id=%s has_ctx=%s model=%s history=%d",
        request.audit_id,
        has_ctx,
        settings.GROQ_MODEL,
        len(history),
    )

    completion = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=messages,
        max_tokens=2048,
        temperature=0.5,
    )
    reply: str = completion.choices[0].message.content or ""

    return AssistantChatResponse(
        reply=reply,
        has_audit_context=has_ctx,
        audit_id=audit_id_used,
        model_used=settings.GROQ_MODEL,
    )
