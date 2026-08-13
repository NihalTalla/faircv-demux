import { ShieldCheck, Info } from "lucide-react"
import { SectionCard } from "@/components/ui/SectionCard"
import { EmptyState } from "@/components/ui/EmptyState"
import { ChartPlaceholder } from "@/components/ui/ChartPlaceholder"
import type { FairnessMetricKey } from "@/types"

// ─── Metric reference data ────────────────────────────────────────────────────

interface MetricDef {
  key: FairnessMetricKey
  name: string
  formula: string
  interpretation: string
  direction: string
  threshold?: string
  thresholdNote?: string
}

const METRIC_DEFS: MetricDef[] = [
  {
    key: "DPD",
    name: "Demographic Parity Difference",
    formula: "P(Ŷ=1 | A=a) − P(Ŷ=1 | A=b)",
    interpretation:
      "Measures the absolute difference in selection (positive prediction) rates between two demographic groups. A value of 0 means both groups are selected at equal rates, regardless of qualification.",
    direction: "Closer to 0 is fairer.",
    threshold: "< 0.05",
    thresholdNote: "Rule-of-thumb for minimal disparity",
  },
  {
    key: "DIR",
    name: "Disparate Impact Ratio",
    formula: "P(Ŷ=1 | A=minority) / P(Ŷ=1 | A=majority)",
    interpretation:
      "Ratio of selection rates: the least-favoured group over the most-favoured group. A value of 1.0 indicates perfect parity. The four-fifths (80%) rule is a widely-used legal/regulatory threshold.",
    direction: "Higher is fairer. Closer to 1.0 is ideal.",
    threshold: "≥ 0.80",
    thresholdNote: "Four-fifths rule (EEOC / UGESP guideline)",
  },
  {
    key: "EOD",
    name: "Equalized Odds Difference",
    formula: "max |TPR_a − TPR_b|, |FPR_a − FPR_b|",
    interpretation:
      "Captures both true positive rate (qualified candidates selected) and false positive rate (unqualified candidates selected) differences between groups. Jointly constrains opportunity and error parity.",
    direction: "Closer to 0 is fairer.",
    threshold: "< 0.05",
    thresholdNote: "Both TPR and FPR gaps must be below threshold",
  },
  {
    key: "EO",
    name: "Equal Opportunity Difference",
    formula: "TPR_a − TPR_b",
    interpretation:
      "Focuses exclusively on true positive rates — whether qualified candidates from each demographic group are selected at equal rates. Does not constrain false positive rates.",
    direction: "Closer to 0 is fairer.",
    threshold: "< 0.05",
  },
  {
    key: "KL",
    name: "KL Divergence",
    formula: "Σ P(x) · log(P(x) / Q(x))",
    interpretation:
      "Kullback-Leibler divergence between score distributions of two demographic groups. Captures the information-theoretic distance between full score distributions, not just decision-boundary differences.",
    direction: "Closer to 0 means more similar distributions.",
    threshold: "< 0.1",
    thresholdNote: "Indicative threshold; context-dependent",
  },
]

// ─── Metric card ──────────────────────────────────────────────────────────────

function MetricCard({ def, hasData }: { def: MetricDef; hasData: boolean }) {
  return (
    <div className="rounded-xl border border-border bg-card">
      {/* Header */}
      <div className="flex items-start justify-between gap-3 border-b border-border px-5 py-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="rounded bg-muted px-2 py-0.5 font-mono text-xs font-bold text-foreground">
              {def.key}
            </span>
            <span className="text-sm font-semibold text-foreground">{def.name}</span>
          </div>
          <p className="mt-1 text-xs font-mono text-muted-foreground/70">{def.formula}</p>
        </div>
        {def.threshold && (
          <div className="shrink-0 text-right">
            <p className="text-[0.65rem] font-semibold uppercase tracking-wider text-muted-foreground/50">
              Threshold
            </p>
            <p className="font-mono text-sm font-bold text-foreground">{def.threshold}</p>
            {def.thresholdNote && (
              <p className="mt-0.5 max-w-[14rem] text-[0.65rem] text-muted-foreground">
                {def.thresholdNote}
              </p>
            )}
          </div>
        )}
      </div>

      {/* Interpretation */}
      <div className="flex items-start gap-2 border-b border-border/60 px-5 py-3">
        <Info className="mt-0.5 size-3.5 shrink-0 text-muted-foreground/50" />
        <p className="text-xs leading-relaxed text-muted-foreground">{def.interpretation}</p>
      </div>

      {/* Data area */}
      {hasData ? (
        <div className="grid grid-cols-1 gap-4 p-5 sm:grid-cols-2">
          <div className="space-y-1">
            <p className="text-[0.68rem] font-semibold uppercase tracking-wider text-muted-foreground">
              Point Estimate
            </p>
            <p className="text-2xl font-bold tabular-nums text-foreground">—</p>
            <p className="text-xs text-muted-foreground">95% CI: — to —</p>
          </div>
          <ChartPlaceholder label="Group comparison" height={100} />
        </div>
      ) : (
        <div className="flex items-center gap-2.5 px-5 py-4">
          <div className="size-2 rounded-full bg-muted-foreground/20" />
          <p className="text-xs text-muted-foreground/60">
            Awaiting audit data · value, confidence interval, and group breakdown will appear here
          </p>
        </div>
      )}

      {/* Direction footer */}
      <div className="border-t border-border/60 px-5 py-2.5">
        <p className="text-[0.68rem] font-medium text-muted-foreground/60">
          ↳ {def.direction}
        </p>
      </div>
    </div>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export function FairnessPage() {
  const hasResults = false

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight text-foreground">Fairness Analysis</h2>
        <p className="text-sm text-muted-foreground max-w-2xl">
          Five group-fairness metrics evaluated per model and demographic attribute pair. Each metric captures a different dimension of algorithmic disparity.
        </p>
      </div>

      {/* Status banner */}
      {!hasResults && (
        <div className="flex items-start gap-3 rounded-lg border border-dashed border-border bg-muted/20 px-4 py-3.5">
          <ShieldCheck className="mt-0.5 size-4 shrink-0 text-muted-foreground/50" />
          <div>
            <p className="text-sm font-medium text-foreground">No audit results loaded.</p>
            <p className="mt-0.5 text-xs text-muted-foreground">
              Metric values, confidence intervals, and group comparisons will populate once an audit run completes and the backend API is connected.
            </p>
          </div>
        </div>
      )}

      {/* Model selector — shown when data is available */}
      {hasResults && (
        <SectionCard title="Model" description="Select a model to view its fairness results">
          <p className="text-sm text-muted-foreground">Model selector will render here.</p>
        </SectionCard>
      )}

      {/* Metric cards */}
      <div>
        <p className="mb-3 text-[0.68rem] font-semibold uppercase tracking-widest text-muted-foreground/50">
          Metrics — {hasResults ? "Results loaded" : "Awaiting data"}
        </p>
        <div className="space-y-4">
          {METRIC_DEFS.map((def) => (
            <MetricCard key={def.key} def={def} hasData={hasResults} />
          ))}
        </div>
      </div>

      {/* Score distribution overview — only with data */}
      {hasResults ? (
        <SectionCard
          title="Score Distributions by Group"
          description="Full score distributions per demographic group per model"
        >
          <ChartPlaceholder label="Score distribution chart" height={220} />
        </SectionCard>
      ) : (
        <SectionCard
          title="Score Distributions by Group"
          description="Score distributions will appear here — grouped by demographic attribute and model"
        >
          <EmptyState
            icon={ShieldCheck}
            title="Awaiting score distribution data."
            description="The backend will return per-group score distributions for chart rendering. No data will be synthesised."
          />
        </SectionCard>
      )}

      {/* Interpreting uncertainty */}
      <div className="rounded-lg border border-border bg-card p-5">
        <h4 className="mb-2 text-sm font-semibold text-foreground">Interpreting Uncertainty</h4>
        <div className="space-y-2 text-xs leading-relaxed text-muted-foreground">
          <p>
            All metric estimates are accompanied by 95% bootstrap confidence intervals (2,000 resamples). A confidence interval that crosses a fairness threshold indicates <span className="font-semibold text-foreground">statistical uncertainty about the direction of the finding</span> — the result will be flagged accordingly.
          </p>
          <p>
            The audit does not reduce findings to a single PASS/FAIL label when evidence is ambiguous. Direction, magnitude, and uncertainty are reported together.
          </p>
          <p>
            Thresholds (e.g. DIR ≥ 0.80) are reference points from regulatory guidelines and research literature. They inform but do not replace domain judgement.
          </p>
        </div>
      </div>
    </div>
  )
}
