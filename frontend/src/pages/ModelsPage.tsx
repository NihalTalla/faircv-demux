import { BrainCircuit, Info } from "lucide-react"
import { SectionCard } from "@/components/ui/SectionCard"

// ─── FairCV audit model descriptions (from the experiment design) ─────────────
// These describe the six Logistic Regression configurations evaluated in the audit.
// Feature sets and label definitions are part of the frozen methodology, not results.

interface ModelSpec {
  id: string
  short: string
  featureSet: string
  labelSet: string
  description: string
}

const MODEL_SPECS: ModelSpec[] = [
  {
    id: "M1",
    short: "Skills",
    featureSet: "Skill-based features only",
    labelSet: "Recruiter rating (binarised at train median)",
    description:
      "Logistic Regression trained on skill-related profile features. Tests whether skills alone introduce demographic disparities when used for automated screening.",
  },
  {
    id: "M2",
    short: "Skills + Experience",
    featureSet: "Skills and experience features",
    labelSet: "Recruiter rating (binarised at train median)",
    description:
      "Extends M1 with experience-related features (e.g. years of experience, number of positions). Tests whether adding experience narrows or widens fairness gaps.",
  },
  {
    id: "M3",
    short: "Full Profile",
    featureSet: "All structured profile features",
    labelSet: "Recruiter rating (binarised at train median)",
    description:
      "Includes all structured (non-image) features. Represents a complete structured-profile screening system.",
  },
  {
    id: "M4",
    short: "Image",
    featureSet: "Photo-derived features only",
    labelSet: "Recruiter rating (binarised at train median)",
    description:
      "Trained exclusively on features extracted from profile photos. Isolates the fairness impact of visual appearance in automated screening.",
  },
  {
    id: "M5",
    short: "Full + Image",
    featureSet: "All structured features + photo features",
    labelSet: "Recruiter rating (binarised at train median)",
    description:
      "Combines all structured profile features with photo-derived features. Represents a multimodal screening system.",
  },
  {
    id: "M6",
    short: "Full + Image (Blind)",
    featureSet: "All features with face region zeroed out",
    labelSet: "Recruiter rating (binarised at train median)",
    description:
      "Same as M5 but with face-related image columns zeroed out. Tests whether face-blinding mitigates visual appearance bias.",
  },
]

// ─── Page ─────────────────────────────────────────────────────────────────────

export function ModelsPage() {
  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight text-foreground">Models</h2>
        <p className="text-sm text-muted-foreground max-w-2xl">
          The FairCV audit evaluates six Logistic Regression configurations (M1–M6) differing in feature set and decision label. Model configurations are part of the frozen audit methodology.
        </p>
      </div>

      {/* Info callout */}
      <div className="flex items-start gap-3 rounded-lg border border-dashed border-border bg-muted/20 px-4 py-3.5">
        <Info className="mt-0.5 size-4 shrink-0 text-muted-foreground/50" />
        <div>
          <p className="text-sm font-medium text-foreground">Per-model performance metrics come from the backend.</p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Accuracy, AUC, and F1 are computed server-side during an audit run and returned by the metrics endpoint. No performance numbers are shown here until an audit completes.
          </p>
        </div>
      </div>

      {/* Model cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {MODEL_SPECS.map((m) => (
          <div key={m.id} className="rounded-xl border border-border bg-card p-5 space-y-3">
            <div className="flex items-center gap-2.5">
              <div className="flex size-9 shrink-0 items-center justify-center rounded-md bg-muted">
                <BrainCircuit className="size-4 text-muted-foreground" />
              </div>
              <div>
                <p className="text-sm font-bold text-foreground">{m.id}</p>
                <p className="text-[0.72rem] text-muted-foreground">{m.short}</p>
              </div>
            </div>

            <p className="text-xs leading-relaxed text-muted-foreground">{m.description}</p>

            <div className="space-y-1.5">
              <div className="rounded-md bg-muted/50 px-3 py-2">
                <p className="text-[0.65rem] font-semibold uppercase tracking-wider text-muted-foreground/60 mb-0.5">
                  Feature Set
                </p>
                <p className="text-xs text-foreground">{m.featureSet}</p>
              </div>
              <div className="rounded-md bg-muted/50 px-3 py-2">
                <p className="text-[0.65rem] font-semibold uppercase tracking-wider text-muted-foreground/60 mb-0.5">
                  Label
                </p>
                <p className="text-xs text-foreground">{m.labelSet}</p>
              </div>
            </div>

            {/* Performance placeholder */}
            <div className="rounded-md border border-dashed border-border bg-muted/20 p-2.5 text-center">
              <p className="text-[0.68rem] font-medium text-muted-foreground/60">
                Acc · AUC · F1
              </p>
              <p className="text-[0.65rem] text-muted-foreground/40">
                Awaiting audit run
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Common settings */}
      <SectionCard
        title="Common Audit Settings"
        description="Applied uniformly across all six models"
      >
        <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2 text-sm">
          {[
            { label: "Algorithm", value: "Logistic Regression (scikit-learn)" },
            { label: "Train/Test Split", value: "Fixed — defined by FairCVdb partition" },
            { label: "Binarisation Threshold", value: "Train-set median of the label score" },
            { label: "Protected Attributes", value: "Gender, Ethnicity" },
            { label: "Bootstrap Resamples", value: "2,000 (seed = fixed)" },
            { label: "Confidence Level", value: "95% percentile interval" },
            { label: "EEOC Rule", value: "DIR ≥ 0.80 (four-fifths rule)" },
            { label: "α (significance)", value: "0.05 (Holm-corrected)" },
          ].map(({ label, value }) => (
            <div key={label} className="flex flex-col gap-0.5 rounded-md border border-border/60 bg-muted/20 px-3 py-2.5">
              <dt className="text-[0.68rem] font-semibold uppercase tracking-wider text-muted-foreground">
                {label}
              </dt>
              <dd className="text-xs text-foreground">{value}</dd>
            </div>
          ))}
        </dl>
      </SectionCard>
    </div>
  )
}
