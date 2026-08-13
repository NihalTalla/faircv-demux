import { Scale, AlertTriangle, Info } from "lucide-react"
import { EvidenceStrengthBadge } from "@/components/ui/EvidenceStrengthBadge"
import { EmptyState } from "@/components/ui/EmptyState"
import { cn } from "@/lib/utils"
import type { EvidenceStrength } from "@/types"

// ─── Verdict dimension structure ──────────────────────────────────────────────

interface DimensionDef {
  key: string
  label: string
  question: string
  sources: string[]
}

const DIMENSIONS: DimensionDef[] = [
  {
    key: "selection_disparity",
    label: "Selection Disparity",
    question: "Do different demographic groups get selected at different rates?",
    sources: ["DPD", "DIR"],
  },
  {
    key: "error_disparity",
    label: "Error Disparity",
    question: "Are error rates (false positives / false negatives) equal across groups?",
    sources: ["EOD", "EO"],
  },
  {
    key: "statistical_evidence",
    label: "Statistical Evidence",
    question: "Are observed differences statistically significant after multiple-comparison correction?",
    sources: ["KS", "Kruskal-Wallis", "χ²", "Holm-Bonferroni"],
  },
  {
    key: "robustness",
    label: "Robustness",
    question: "Do findings persist across different decision rules (Median, P75, Top-N)?",
    sources: ["Robustness analysis"],
  },
]

// ─── Placeholder dimension card ───────────────────────────────────────────────

function DimensionCard({
  def,
  strength,
  summary,
}: {
  def: DimensionDef
  strength?: EvidenceStrength
  summary?: string
}) {
  const hasData = !!strength

  return (
    <div
      className={cn(
        "rounded-xl border border-border bg-card transition-colors",
        hasData && strength === "strong" && "border-emerald-200 dark:border-emerald-900/60",
        hasData && strength === "moderate" && "border-blue-200 dark:border-blue-900/60",
        hasData && strength === "limited" && "border-amber-200 dark:border-amber-900/60",
        hasData && strength === "inconclusive" && ""
      )}
    >
      <div className="flex items-start justify-between gap-3 border-b border-border/70 px-5 py-4">
        <div className="space-y-1">
          <h3 className="text-sm font-semibold text-foreground">{def.label}</h3>
          <p className="text-xs italic text-muted-foreground">{def.question}</p>
        </div>
        {hasData && strength ? (
          <EvidenceStrengthBadge strength={strength} className="shrink-0" />
        ) : (
          <span className="shrink-0 rounded-full border border-dashed border-border px-2 py-0.5 text-[0.68rem] text-muted-foreground/50">
            Awaiting
          </span>
        )}
      </div>

      {/* Sources */}
      <div className="flex flex-wrap gap-1.5 border-b border-border/60 px-5 py-3">
        <span className="text-[0.68rem] text-muted-foreground/50 self-center">Sources:</span>
        {def.sources.map((s) => (
          <span
            key={s}
            className="rounded bg-muted px-1.5 py-0.5 font-mono text-[0.68rem] text-muted-foreground"
          >
            {s}
          </span>
        ))}
      </div>

      {/* Summary / placeholder */}
      <div className="px-5 py-4">
        {hasData && summary ? (
          <p className="text-sm text-foreground">{summary}</p>
        ) : (
          <p className="text-xs text-muted-foreground/50">
            Summary will be generated from audit results once backend is connected.
          </p>
        )}
      </div>
    </div>
  )
}

// ─── Overall verdict block ────────────────────────────────────────────────────

function OverallVerdictBlock({ interpretation }: { interpretation?: string }) {
  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      <div className="bg-muted/40 px-6 py-5 border-b border-border">
        <div className="flex items-center gap-2 mb-1">
          <Scale className="size-5 text-foreground/60" />
          <h2 className="text-base font-bold text-foreground">Overall Interpretation</h2>
        </div>
        <p className="text-xs text-muted-foreground">
          Synthesised from all fairness metrics, statistical tests, and robustness analysis
        </p>
      </div>

      <div className="px-6 py-6">
        {interpretation ? (
          <p className="text-sm leading-relaxed text-foreground">{interpretation}</p>
        ) : (
          <div className="flex items-start gap-3">
            <div className="flex size-10 shrink-0 items-center justify-center rounded-full border border-dashed border-border bg-muted/20">
              <Scale className="size-5 text-muted-foreground/30" />
            </div>
            <div>
              <p className="text-sm font-medium text-foreground">No verdict available.</p>
              <p className="mt-1 text-xs text-muted-foreground max-w-lg">
                The overall audit interpretation will appear here once all four dimensions — selection disparity, error disparity, statistical evidence, and robustness — have been evaluated by the backend audit engine.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Uncertainty notes */}
      <div className="border-t border-border/60 bg-muted/20 px-6 py-4">
        <div className="flex items-start gap-2">
          <Info className="mt-0.5 size-3.5 shrink-0 text-muted-foreground/50" />
          <p className="text-[0.72rem] leading-relaxed text-muted-foreground">
            The verdict reflects the weight of evidence across the full audit. It does not provide a binary BIASED / NOT BIASED judgment. Where evidence is limited or ambiguous, that uncertainty is stated explicitly.
          </p>
        </div>
      </div>
    </div>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export function VerdictPage() {
  const hasVerdict = false

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight text-foreground">Final Verdict</h2>
        <p className="text-sm text-muted-foreground max-w-2xl">
          The audit verdict synthesises all evidence across fairness metrics, statistical tests, and robustness analysis into a structured, evidence-grounded interpretation.
        </p>
      </div>

      {/* Warning if no data */}
      {!hasVerdict && (
        <div className="flex items-start gap-3 rounded-lg border border-dashed border-border bg-muted/20 px-4 py-3.5">
          <AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-500" />
          <div>
            <p className="text-sm font-medium text-foreground">Verdict not yet generated.</p>
            <p className="mt-0.5 text-xs text-muted-foreground">
              All four verdict dimensions must be populated by the backend before an overall interpretation is issued. No partial verdicts are displayed.
            </p>
          </div>
        </div>
      )}

      {/* Overall verdict block */}
      <OverallVerdictBlock />

      {/* Four dimensions */}
      <div>
        <p className="mb-3 text-[0.68rem] font-semibold uppercase tracking-widest text-muted-foreground/50">
          Evidence Dimensions
        </p>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {DIMENSIONS.map((def) => (
            <DimensionCard key={def.key} def={def} />
          ))}
        </div>
      </div>

      {/* No data fallback */}
      {!hasVerdict && (
        <EmptyState
          icon={Scale}
          title="Awaiting completed audit."
          description="The final verdict is computed server-side by the audit engine and cannot be generated from the frontend alone. Connect the backend API after completing an audit run."
        />
      )}

      {/* Methodology note */}
      <div className="rounded-xl border border-border bg-card p-5 space-y-3">
        <h3 className="text-sm font-semibold text-foreground">Verdict Methodology</h3>
        <div className="space-y-3 text-xs leading-relaxed text-muted-foreground">
          <p>
            The verdict is not a single score or label. It is a structured summary across four dimensions, each graded by evidence strength:
            <span className="font-semibold text-emerald-600 dark:text-emerald-400"> Strong</span>,
            <span className="font-semibold text-blue-600 dark:text-blue-400"> Moderate</span>,
            <span className="font-semibold text-amber-600 dark:text-amber-400"> Limited</span>, or
            <span className="font-semibold text-foreground"> Inconclusive</span>.
          </p>
          <p>
            Evidence strength is based on: (1) the magnitude and direction of the fairness metric, (2) whether confidence intervals cross threshold boundaries, (3) whether findings replicate across multiple models and demographic attributes, and (4) whether findings are robust to decision rule changes.
          </p>
          <p>
            The FairCV audit does not fabricate verdicts. If the evidence is insufficient to draw a reliable conclusion, the verdict states that explicitly. A verdict of "Inconclusive" is a legitimate and important outcome.
          </p>
        </div>
      </div>

      {/* Uncertainty notes */}
      <div className="rounded-lg border border-amber-200 bg-amber-50/60 dark:border-amber-900/40 dark:bg-amber-950/20 px-5 py-4">
        <div className="flex items-start gap-2">
          <AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-600 dark:text-amber-400" />
          <div className="space-y-1 text-xs text-muted-foreground">
            <p className="font-semibold text-foreground">Uncertainty is not hidden.</p>
            <p>
              Confidence intervals that cross fairness thresholds, statistically non-significant differences, and findings that are not robust across threshold strategies are all surfaced — not suppressed. Decisions made from this audit must account for the stated uncertainty.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
