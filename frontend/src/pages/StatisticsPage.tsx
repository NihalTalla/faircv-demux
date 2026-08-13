import { BarChart3, AlertTriangle } from "lucide-react"
import { SectionCard } from "@/components/ui/SectionCard"
import { EmptyState } from "@/components/ui/EmptyState"
import { ChartPlaceholder } from "@/components/ui/ChartPlaceholder"

// ─── Test definitions ─────────────────────────────────────────────────────────

interface TestDef {
  key: string
  name: string
  type: "nonparametric" | "parametric" | "correction"
  hypothesis: string
  outputs: string[]
  notes?: string
}

const TEST_DEFS: TestDef[] = [
  {
    key: "KS",
    name: "Kolmogorov-Smirnov Test",
    type: "nonparametric",
    hypothesis:
      "H₀: Two score distributions are drawn from the same underlying distribution.",
    outputs: ["KS statistic (D)", "p-value"],
    notes:
      "Applied pairwise to score distributions across demographic groups. Sensitive to any difference in distribution shape, not just means.",
  },
  {
    key: "Kruskal-Wallis",
    name: "Kruskal-Wallis H Test",
    type: "nonparametric",
    hypothesis:
      "H₀: Score distributions across k groups have the same median (non-parametric ANOVA).",
    outputs: ["H statistic", "p-value", "η² (effect size)"],
    notes:
      "Used when three or more demographic groups are compared simultaneously. η² quantifies the proportion of variance explained by group membership.",
  },
  {
    key: "χ²",
    name: "Chi-Square Test of Independence",
    type: "nonparametric",
    hypothesis:
      "H₀: Group membership and selection outcome are statistically independent.",
    outputs: ["χ² statistic", "p-value", "df", "Cramér's V"],
    notes:
      "Tests whether selection rates differ significantly across groups. Cramér's V provides a normalised effect size between 0 and 1.",
  },
  {
    key: "Holm-Bonferroni",
    name: "Holm-Bonferroni Correction",
    type: "correction",
    hypothesis: "Controls the family-wise error rate (FWER) across the full test battery.",
    outputs: ["Adjusted p-values", "Revised significance decisions"],
    notes:
      "Applied sequentially across all pairwise tests. More powerful than the standard Bonferroni correction while still controlling FWER at α = 0.05.",
  },
]

const typeLabel: Record<TestDef["type"], { text: string; className: string }> = {
  nonparametric: { text: "Non-parametric", className: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400" },
  parametric: { text: "Parametric", className: "bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-400" },
  correction: { text: "Correction", className: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400" },
}

function TestCard({ def, hasData }: { def: TestDef; hasData: boolean }) {
  return (
    <div className="rounded-xl border border-border bg-card">
      <div className="border-b border-border px-5 py-4">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="rounded bg-muted px-2 py-0.5 font-mono text-xs font-bold text-foreground">
              {def.key}
            </span>
            <span className="text-sm font-semibold text-foreground">{def.name}</span>
          </div>
          <span
            className={`rounded-full px-2 py-0.5 text-[0.68rem] font-medium ${typeLabel[def.type].className}`}
          >
            {typeLabel[def.type].text}
          </span>
        </div>
        <p className="mt-2 text-xs italic text-muted-foreground">{def.hypothesis}</p>
      </div>

      <div className="grid grid-cols-1 gap-4 p-5 sm:grid-cols-2">
        {/* Outputs */}
        <div>
          <p className="mb-2 text-[0.68rem] font-semibold uppercase tracking-wider text-muted-foreground">
            Output Statistics
          </p>
          <ul className="space-y-1">
            {def.outputs.map((o) => (
              <li key={o} className="flex items-center gap-2 text-xs text-foreground">
                <span className="size-1.5 rounded-full bg-muted-foreground/30 shrink-0" />
                {o}
              </li>
            ))}
          </ul>
        </div>

        {/* Results placeholder */}
        <div className="rounded-md border border-dashed border-border bg-muted/20 p-3">
          {hasData ? (
            <p className="text-sm">Result will render here.</p>
          ) : (
            <div className="flex h-full items-center justify-center text-center">
              <div>
                <p className="text-[0.7rem] font-medium text-muted-foreground/60">
                  Awaiting audit data
                </p>
                <p className="text-[0.65rem] text-muted-foreground/40">
                  test statistic · p-value · effect size
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {def.notes && (
        <div className="border-t border-border/60 px-5 py-3">
          <p className="text-xs text-muted-foreground">{def.notes}</p>
        </div>
      )}
    </div>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export function StatisticsPage() {
  const hasResults = false

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight text-foreground">Statistical Analysis</h2>
        <p className="text-sm text-muted-foreground max-w-2xl">
          Hypothesis tests for distributional differences and independence, with multiple-comparison correction and effect size reporting.
        </p>
      </div>

      {/* Status */}
      {!hasResults && (
        <div className="flex items-start gap-3 rounded-lg border border-dashed border-border bg-muted/20 px-4 py-3.5">
          <BarChart3 className="mt-0.5 size-4 shrink-0 text-muted-foreground/50" />
          <div>
            <p className="text-sm font-medium text-foreground">No statistical results loaded.</p>
            <p className="mt-0.5 text-xs text-muted-foreground">
              Results populate once an audit run completes. Test statistics, p-values, corrected p-values, and effect sizes will appear below per model and attribute.
            </p>
          </div>
        </div>
      )}

      {/* Test battery */}
      <div>
        <p className="mb-3 text-[0.68rem] font-semibold uppercase tracking-widest text-muted-foreground/50">
          Test Battery
        </p>
        <div className="space-y-4">
          {TEST_DEFS.map((def) => (
            <TestCard key={def.key} def={def} hasData={hasResults} />
          ))}
        </div>
      </div>

      {/* Summary table placeholder */}
      {hasResults ? (
        <SectionCard title="Results Summary" description="All tests across models and demographic attributes">
          <p className="text-sm text-muted-foreground">Table will render here.</p>
        </SectionCard>
      ) : (
        <SectionCard
          title="Results Summary Table"
          description="Corrected p-values and effect sizes across all tests, models, and attributes"
        >
          <EmptyState
            icon={BarChart3}
            title="Awaiting test results."
            description="The backend will return a complete table of test statistics, raw and Holm-corrected p-values, and effect sizes."
          />
        </SectionCard>
      )}

      {/* Significance distinction */}
      <div className="rounded-xl border border-border bg-card">
        <div className="border-b border-border px-5 py-4">
          <div className="flex items-center gap-2">
            <AlertTriangle className="size-4 text-amber-500" />
            <h3 className="text-sm font-semibold text-foreground">
              Statistical vs. Practical Significance
            </h3>
          </div>
        </div>
        <div className="grid grid-cols-1 gap-0 divide-y divide-border sm:grid-cols-2 sm:divide-x sm:divide-y-0">
          <div className="p-5">
            <h4 className="mb-2 text-xs font-bold uppercase tracking-wider text-foreground">
              Statistical Significance
            </h4>
            <p className="text-xs leading-relaxed text-muted-foreground">
              A statistically significant result (p &lt; 0.05 after Holm correction) means the observed difference is unlikely to have arisen by random chance, given the dataset size. It does not by itself indicate that the difference is meaningful in practice.
            </p>
            <div className="mt-3 space-y-1 text-xs">
              <p className="font-medium text-foreground">Key outputs:</p>
              <ul className="ml-3 space-y-0.5 text-muted-foreground">
                <li>• p-value (raw and corrected)</li>
                <li>• FWER-controlled significance decision</li>
              </ul>
            </div>
          </div>
          <div className="p-5">
            <h4 className="mb-2 text-xs font-bold uppercase tracking-wider text-foreground">
              Practical Significance (Effect Size)
            </h4>
            <p className="text-xs leading-relaxed text-muted-foreground">
              Effect sizes (η², Cramér's V) quantify the magnitude of the difference independently of sample size. A large sample can make a trivially small disparity statistically significant. Effect size anchors the finding to real-world impact.
            </p>
            <div className="mt-3 space-y-1 text-xs">
              <p className="font-medium text-foreground">Effect size benchmarks:</p>
              <ul className="ml-3 space-y-0.5 text-muted-foreground">
                <li>• Small: η² ≈ 0.01 / V ≈ 0.1</li>
                <li>• Medium: η² ≈ 0.06 / V ≈ 0.3</li>
                <li>• Large: η² ≈ 0.14 / V ≈ 0.5</li>
              </ul>
            </div>
          </div>
        </div>
        <div className="border-t border-border/60 bg-muted/20 px-5 py-3">
          <p className="text-xs text-muted-foreground">
            The FairCV audit reports both dimensions. A finding requires <span className="font-semibold text-foreground">both</span> statistical and practical significance to be considered strong evidence of algorithmic bias.
          </p>
        </div>
      </div>

      {/* Distribution comparison placeholder */}
      <SectionCard title="Score Distribution Comparison" description="Pairwise KS test visualisation per attribute">
        {hasResults ? (
          <ChartPlaceholder label="Distribution comparison" height={180} />
        ) : (
          <EmptyState
            icon={BarChart3}
            title="Awaiting distribution data."
            description="Pairwise distribution charts will render here once audit results are loaded from the backend."
          />
        )}
      </SectionCard>

    </div>
  )
}
