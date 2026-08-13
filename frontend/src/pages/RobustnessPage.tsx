import { LineChart, Layers } from "lucide-react"
import { SectionCard } from "@/components/ui/SectionCard"
import { EmptyState } from "@/components/ui/EmptyState"
import { ChartPlaceholder } from "@/components/ui/ChartPlaceholder"

// ─── Decision rules ───────────────────────────────────────────────────────────

interface DecisionRule {
  key: string
  label: string
  description: string
  implication: string
}

const RULES: DecisionRule[] = [
  {
    key: "median",
    label: "Median Threshold",
    description:
      "All candidates scoring at or above the median score are forwarded. Selects approximately 50% of the pool.",
    implication:
      "Tests fairness under a mid-point decision boundary — a common baseline rule.",
  },
  {
    key: "p75",
    label: "P75 Threshold",
    description:
      "Only candidates scoring at or above the 75th percentile are forwarded. Selects approximately 25% of the pool.",
    implication:
      "Stricter selection. Fairness metrics often worsen at higher cut-points if group score distributions diverge in the tails.",
  },
  {
    key: "top_n",
    label: "Top-N Selection",
    description:
      "A fixed number of top-scoring candidates are forwarded. Evaluated across multiple N values (e.g. Top-500, Top-1000).",
    implication:
      "Removes threshold ambiguity. Enables robustness analysis across selection volumes.",
  },
]

// ─── Metric headers for the comparison table ─────────────────────────────────

const TABLE_COLUMNS = ["Decision Rule", "Metric", "Model", "Point Estimate", "95% CI", "Status"]

// ─── Page ─────────────────────────────────────────────────────────────────────

export function RobustnessPage() {
  const hasResults = false

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight text-foreground">Robustness Analysis</h2>
        <p className="text-sm text-muted-foreground max-w-2xl">
          Does the fairness finding persist when the screening rule changes? Metrics are evaluated under three decision strategies to assess threshold sensitivity.
        </p>
      </div>

      {/* Status */}
      {!hasResults && (
        <div className="flex items-start gap-3 rounded-lg border border-dashed border-border bg-muted/20 px-4 py-3.5">
          <LineChart className="mt-0.5 size-4 shrink-0 text-muted-foreground/50" />
          <div>
            <p className="text-sm font-medium text-foreground">No robustness results loaded.</p>
            <p className="mt-0.5 text-xs text-muted-foreground">
              Results across Median, P75, and Top-N strategies will populate once an audit run completes.
            </p>
          </div>
        </div>
      )}

      {/* Decision rules */}
      <div>
        <p className="mb-3 text-[0.68rem] font-semibold uppercase tracking-widest text-muted-foreground/50">
          Decision Strategies
        </p>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          {RULES.map(({ key, label, description, implication }) => (
            <div key={key} className="rounded-xl border border-border bg-card p-5 space-y-3">
              <div>
                <span className="rounded bg-muted px-2 py-0.5 font-mono text-xs font-bold text-foreground">
                  {key}
                </span>
                <p className="mt-2 text-sm font-semibold text-foreground">{label}</p>
              </div>
              <p className="text-xs leading-relaxed text-muted-foreground">{description}</p>
              <div className="border-t border-border/60 pt-3">
                <p className="text-[0.7rem] font-semibold text-muted-foreground/60 uppercase tracking-wider mb-1">
                  Why it matters
                </p>
                <p className="text-xs text-muted-foreground">{implication}</p>
              </div>
              <div className="rounded-md border border-dashed border-border bg-muted/20 p-2 text-center">
                <p className="text-[0.68rem] text-muted-foreground/50">
                  Metrics awaiting data
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Sensitivity chart */}
      <SectionCard
        title="Metric Sensitivity Across Thresholds"
        description="How each fairness metric changes as the decision rule shifts — per model and demographic attribute"
      >
        {hasResults ? (
          <ChartPlaceholder label="Threshold sensitivity line chart" height={240} />
        ) : (
          <EmptyState
            icon={LineChart}
            title="Awaiting robustness data."
            description="Line charts showing metric trajectories across Median → P75 → Top-N will render here once backend data is available."
          />
        )}
      </SectionCard>

      {/* Results table */}
      <SectionCard
        title="Results Table"
        description="Point estimates and confidence intervals per decision rule, metric, and model"
      >
        {hasResults ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  {TABLE_COLUMNS.map((h) => (
                    <th
                      key={h}
                      className="px-3 py-2.5 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="px-3 py-3 text-muted-foreground text-xs" colSpan={TABLE_COLUMNS.length}>
                    Results will render here.
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            icon={Layers}
            title="Table awaiting data."
            description="Decision Rule · Metric · Model · Point Estimate · CI · Status will populate here."
          />
        )}
      </SectionCard>

      {/* Interpretation */}
      <div className="rounded-xl border border-border bg-card p-5 space-y-3">
        <h3 className="text-sm font-semibold text-foreground">How to Interpret Robustness</h3>
        <div className="space-y-3 text-xs leading-relaxed text-muted-foreground">
          <p>
            A model is considered <span className="font-semibold text-foreground">robust</span> if its fairness metrics remain within acceptable bounds across all three strategies. A finding that holds at Median, P75, and Top-N is a much stronger claim than one that appears only under a single threshold.
          </p>
          <p>
            <span className="font-semibold text-foreground">Threshold sensitivity</span> — large metric variation across strategies — indicates that the model's fairness behaviour depends strongly on where the selection cut-point is drawn. This warrants disclosure and additional scrutiny before deployment.
          </p>
          <p>
            Confidence intervals are included for every robustness estimate. Intervals that cross a threshold boundary indicate genuine uncertainty and must not be treated as unambiguous pass or fail.
          </p>
        </div>
      </div>

    </div>
  )
}
