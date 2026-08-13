import {
  Database,
  BrainCircuit,
  ShieldCheck,
  BarChart3,
  LineChart,
  Scale,
  ArrowRight,
  Clock,
} from "lucide-react"
import { Link } from "react-router-dom"
import { cn } from "@/lib/utils"

// ─── Pipeline diagram ─────────────────────────────────────────────────────────

interface PipelineNode {
  label: string
  sublabel: string
  to: string
  icon: typeof Database
  status: "pending" | "ready" | "awaiting"
}

const pipeline: PipelineNode[] = [
  { label: "Dataset", sublabel: "FairCVdb.npy", to: "/datasets", icon: Database, status: "pending" },
  { label: "Models", sublabel: "M1 – M6", to: "/models", icon: BrainCircuit, status: "pending" },
  { label: "Fairness", sublabel: "DPD · DIR · EOD", to: "/fairness", icon: ShieldCheck, status: "awaiting" },
  { label: "Statistics", sublabel: "KS · χ² · HB", to: "/statistics", icon: BarChart3, status: "awaiting" },
  { label: "Robustness", sublabel: "Median · P75 · Top-N", to: "/robustness", icon: LineChart, status: "awaiting" },
  { label: "Verdict", sublabel: "Final interpretation", to: "/verdict", icon: Scale, status: "awaiting" },
]

const nodeStatusStyle: Record<PipelineNode["status"], string> = {
  ready: "border-emerald-300 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-950/30",
  pending: "border-border bg-card",
  awaiting: "border-dashed border-border bg-muted/30",
}

const nodeIconStyle: Record<PipelineNode["status"], string> = {
  ready: "text-emerald-600 dark:text-emerald-400",
  pending: "text-muted-foreground",
  awaiting: "text-muted-foreground/40",
}

function PipelineDiagram() {
  return (
    <div className="overflow-x-auto">
      <div className="flex min-w-max items-center gap-0 py-4">
        {pipeline.map((node, i) => (
          <div key={node.label} className="flex items-center">
            <Link
              to={node.to}
              className={cn(
                "group flex w-36 flex-col items-center gap-2 rounded-xl border p-4 text-center transition-all duration-150 hover:shadow-sm",
                nodeStatusStyle[node.status]
              )}
            >
              <node.icon
                className={cn("size-6 transition-colors", nodeIconStyle[node.status])}
              />
              <div>
                <p className="text-xs font-semibold text-foreground">{node.label}</p>
                <p className="text-[0.68rem] text-muted-foreground">{node.sublabel}</p>
              </div>
              {node.status === "pending" && (
                <span className="inline-flex items-center gap-1 text-[0.65rem] text-amber-600 dark:text-amber-400">
                  <Clock className="size-2.5" />
                  Awaiting API
                </span>
              )}
              {node.status === "awaiting" && (
                <span className="text-[0.65rem] text-muted-foreground/50">—</span>
              )}
            </Link>
            {i < pipeline.length - 1 && (
              <ArrowRight className="mx-2 size-4 shrink-0 text-muted-foreground/30" />
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Status cards ─────────────────────────────────────────────────────────────

interface StatusItem {
  label: string
  state: string
  icon: typeof Database
  to: string
}

const statusItems: StatusItem[] = [
  { label: "Dataset", state: "Pending backend", icon: Database, to: "/datasets" },
  { label: "Models", state: "M1 – M6", icon: BrainCircuit, to: "/models" },
  { label: "Fairness Results", state: "Awaiting audit", icon: ShieldCheck, to: "/fairness" },
  { label: "Statistical Tests", state: "Awaiting audit", icon: BarChart3, to: "/statistics" },
  { label: "Robustness", state: "Awaiting audit", icon: LineChart, to: "/robustness" },
  { label: "Final Verdict", state: "Awaiting audit", icon: Scale, to: "/verdict" },
]

// ─── Page ─────────────────────────────────────────────────────────────────────

export function OverviewPage() {
  return (
    <div className="space-y-8">

      {/* Heading */}
      <div className="space-y-1">
        <h2 className="text-2xl font-semibold tracking-tight text-foreground">
          FairCV Bias Audit
        </h2>
        <p className="text-sm text-muted-foreground max-w-2xl">
          Evidence-based fairness auditing for automated CV/résumé screening models. Connect the backend API to populate results across the pipeline below.
        </p>
      </div>

      {/* Pipeline */}
      <div className="rounded-xl border border-border bg-card p-6">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-foreground">Audit Pipeline</h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              End-to-end flow from raw dataset to fairness verdict
            </p>
          </div>
          <span className="rounded-full border border-border bg-muted px-2.5 py-1 text-[0.7rem] font-medium text-muted-foreground">
            No active run
          </span>
        </div>
        <PipelineDiagram />
      </div>

      {/* Status grid */}
      <div>
        <h3 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground/60">
          Component Status
        </h3>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          {statusItems.map(({ label, state, icon: Icon, to }) => (
            <Link
              key={label}
              to={to}
              className="group flex flex-col gap-3 rounded-lg border border-border bg-card p-4 transition-colors hover:bg-muted/40"
            >
              <div className="flex size-8 items-center justify-center rounded-md bg-muted">
                <Icon className="size-4 text-muted-foreground" />
              </div>
              <div>
                <p className="text-xs font-semibold text-foreground">{label}</p>
                <p className="mt-0.5 text-[0.7rem] text-muted-foreground">{state}</p>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* Audit methodology */}
      <div className="rounded-xl border border-border bg-card">
        <div className="border-b border-border px-6 py-4">
          <h3 className="text-sm font-semibold text-foreground">Audit Methodology</h3>
          <p className="mt-0.5 text-xs text-muted-foreground">
            The FairCV audit pipeline applies the following approach
          </p>
        </div>
        <div className="grid grid-cols-1 gap-0 divide-y divide-border sm:grid-cols-2 sm:divide-x sm:divide-y-0">
          {[
            {
              step: "01",
              title: "Fairness Metrics",
              body: "Five group-fairness metrics are computed per model and demographic attribute: Demographic Parity Difference (DPD), Disparate Impact Ratio (DIR), Equalized Odds Difference (EOD), Equal Opportunity (EO), and KL Divergence (KL).",
            },
            {
              step: "02",
              title: "Statistical Tests",
              body: "Kolmogorov-Smirnov, Kruskal-Wallis, and Chi-Square tests are applied to score distributions and selection outcomes. Multiple-comparison corrections use the Holm-Bonferroni procedure.",
            },
            {
              step: "03",
              title: "Bootstrap Uncertainty",
              body: "Point estimates are accompanied by bootstrap confidence intervals (2,000 resamples, 95% CI). Findings that cross threshold boundaries are flagged with explicit uncertainty.",
            },
            {
              step: "04",
              title: "Robustness Analysis",
              body: "Fairness metrics are evaluated under multiple decision rules: median threshold, P75 threshold, and Top-N selection. Sensitivity across rules is reported before any verdict is issued.",
            },
          ].map(({ step, title, body }) => (
            <div key={step} className="p-6">
              <div className="mb-2 text-[0.65rem] font-bold uppercase tracking-widest text-muted-foreground/40">
                Step {step}
              </div>
              <h4 className="mb-1.5 text-sm font-semibold text-foreground">{title}</h4>
              <p className="text-xs leading-relaxed text-muted-foreground">{body}</p>
            </div>
          ))}
        </div>
      </div>

      {/* No results notice */}
      <div className="flex items-center gap-3 rounded-lg border border-dashed border-border bg-muted/20 px-5 py-4">
        <Clock className="size-4 shrink-0 text-muted-foreground/50" />
        <p className="text-xs text-muted-foreground">
          <span className="font-semibold text-foreground">No audit results are loaded.</span>{" "}
          Connect the FairCV backend API at{" "}
          <code className="rounded bg-muted px-1 font-mono text-[0.7rem]">VITE_API_BASE_URL</code>{" "}
          to populate the analysis sections.
        </p>
      </div>

    </div>
  )
}
