import { BookOpen, CheckCircle2, Clock, AlertCircle } from "lucide-react"
import { SectionCard } from "@/components/ui/SectionCard"
import { EmptyState } from "@/components/ui/EmptyState"
import { cn } from "@/lib/utils"

// ─── Evidence trail columns ───────────────────────────────────────────────────

const COLUMNS = [
  "Model",
  "Metric",
  "Attribute",
  "Decision Rule",
  "Result",
  "95% CI",
  "Threshold",
  "Source Artifact",
  "Verified",
]

const verificationIcon = {
  verified: <CheckCircle2 className="size-3.5 text-emerald-500" />,
  pending: <Clock className="size-3.5 text-amber-500" />,
  unverified: <AlertCircle className="size-3.5 text-muted-foreground/40" />,
}

// ─── What this page will contain when connected ───────────────────────────────

const FINDING_DIMENSIONS = [
  {
    label: "Model",
    description: "Which of M1–M6 produced this finding",
    color: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
  },
  {
    label: "Metric",
    description: "Which fairness metric (DPD, DIR, EOD, EO, KL)",
    color: "bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-400",
  },
  {
    label: "Demographic Attribute",
    description: "The protected group attribute (e.g. gender, ethnicity)",
    color: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400",
  },
  {
    label: "Decision Rule",
    description: "Median, P75, or Top-N threshold strategy",
    color: "bg-muted text-muted-foreground",
  },
  {
    label: "Result",
    description: "Point estimate of the metric",
    color: "bg-muted text-muted-foreground",
  },
  {
    label: "Confidence Interval",
    description: "95% bootstrap CI from 2,000 resamples",
    color: "bg-muted text-muted-foreground",
  },
  {
    label: "Source Artifact",
    description: "Frozen result file or report the finding traces to",
    color: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400",
  },
  {
    label: "Verification Status",
    description: "Whether the finding has been independently verified",
    color: "bg-muted text-muted-foreground",
  },
]

// ─── Page ─────────────────────────────────────────────────────────────────────

export function EvidencePage() {
  const hasFindings = false

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight text-foreground">Evidence & Audit Trail</h2>
        <p className="text-sm text-muted-foreground max-w-2xl">
          Every finding in the audit is traceable to its source artifact. This page records the full chain from raw computation to reported result.
        </p>
      </div>

      {/* Status */}
      {!hasFindings && (
        <div className="flex items-start gap-3 rounded-lg border border-dashed border-border bg-muted/20 px-4 py-3.5">
          <BookOpen className="mt-0.5 size-4 shrink-0 text-muted-foreground/50" />
          <div>
            <p className="text-sm font-medium text-foreground">No findings loaded.</p>
            <p className="mt-0.5 text-xs text-muted-foreground">
              Audit findings with full provenance will appear here once the backend API is connected and an audit run has completed.
            </p>
          </div>
        </div>
      )}

      {/* Finding structure */}
      <SectionCard
        title="Finding Structure"
        description="Each audit finding is recorded with the following dimensions"
      >
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {FINDING_DIMENSIONS.map(({ label, description, color }) => (
            <div key={label} className="rounded-md border border-border bg-card p-3 space-y-1.5">
              <span className={cn("inline-flex rounded-full px-2 py-0.5 text-[0.68rem] font-medium", color)}>
                {label}
              </span>
              <p className="text-[0.72rem] text-muted-foreground leading-relaxed">{description}</p>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Findings table */}
      <SectionCard
        title="Audit Findings"
        description="Complete record of all findings from the latest audit run"
      >
        {hasFindings ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  {COLUMNS.map((h) => (
                    <th
                      key={h}
                      className="px-3 py-2.5 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground whitespace-nowrap"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td colSpan={COLUMNS.length} className="px-3 py-6 text-center text-xs text-muted-foreground">
                    Findings will render here from the backend API.
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            icon={BookOpen}
            title="No audit findings recorded."
            description="Run a completed audit and connect the backend API to populate the evidence table."
          />
        )}
      </SectionCard>

      {/* Provenance note */}
      <div className="rounded-xl border border-border bg-card">
        <div className="border-b border-border px-5 py-4">
          <h3 className="text-sm font-semibold text-foreground">Source of Truth</h3>
        </div>
        <div className="p-5 space-y-3 text-xs leading-relaxed text-muted-foreground">
          <p>
            The FairCV audit engine produces frozen result artifacts (CSV, JSON) that serve as the primary evidence record. This frontend reads from those artifacts via the backend API — it does <span className="font-semibold text-foreground">not</span> independently recalculate any values.
          </p>
          <p>
            Each finding links to the specific frozen artifact it was derived from. If a finding cannot be traced to a source artifact, its verification status is marked <span className="inline-flex items-center gap-1 font-medium text-foreground">unverified <AlertCircle className="size-3 text-muted-foreground" /></span>.
          </p>
          <p>
            The backend audit pipeline must not be modified to accommodate the frontend. Any frontend-audit discrepancy must be reported and investigated before being surfaced as a finding.
          </p>
        </div>
        {/* Verification legend */}
        <div className="border-t border-border/60 flex flex-wrap gap-5 px-5 py-3">
          {Object.entries(verificationIcon).map(([status, icon]) => (
            <span key={status} className="flex items-center gap-1.5 text-xs text-muted-foreground capitalize">
              {icon} {status}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}
