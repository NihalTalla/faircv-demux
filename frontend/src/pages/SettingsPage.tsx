import { PageHeader } from "@/components/ui/PageHeader"
import { SectionCard } from "@/components/ui/SectionCard"
import { Button } from "@/components/ui/button"
import { useTheme } from "@/components/theme-provider"

export function SettingsPage() {
  const { theme, setTheme } = useTheme()

  return (
    <div className="space-y-6">
      <PageHeader
        title="Settings"
        description="Application preferences and backend configuration."
      />

      {/* Appearance */}
      <SectionCard title="Appearance" description="Visual theme of the application.">
        <div className="flex flex-wrap gap-2">
          {(["light", "dark", "system"] as const).map((t) => (
            <Button
              key={t}
              variant={theme === t ? "default" : "outline"}
              size="sm"
              onClick={() => setTheme(t)}
              className="capitalize"
            >
              {t}
            </Button>
          ))}
        </div>
        <p className="mt-3 text-xs text-muted-foreground">
          Shortcut: press{" "}
          <kbd className="rounded border border-border bg-muted px-1 py-0.5 font-mono text-xs">d</kbd>{" "}
          to toggle dark / light.
        </p>
      </SectionCard>

      {/* API */}
      <SectionCard
        title="Backend API"
        description="Connection point for the FairCV audit engine."
      >
        <div className="space-y-3">
          <label className="block space-y-1.5 text-sm">
            <span className="font-medium text-foreground">API Base URL</span>
            <input
              type="text"
              defaultValue={import.meta.env.VITE_API_BASE_URL ?? "/api"}
              disabled
              className="block w-full max-w-sm rounded-md border border-border bg-muted px-3 py-2 font-mono text-sm text-muted-foreground disabled:cursor-not-allowed"
              aria-label="API Base URL"
            />
          </label>
          <p className="text-xs text-muted-foreground">
            Set{" "}
            <code className="rounded bg-muted px-1 font-mono text-[0.7rem]">
              VITE_API_BASE_URL
            </code>{" "}
            in <code className="rounded bg-muted px-1 font-mono text-[0.7rem]">.env</code> to point
            to the FairCV backend.
          </p>
        </div>
      </SectionCard>

      {/* About */}
      <SectionCard title="About" description="Platform information.">
        <dl className="space-y-4 text-sm">
          {[
            { label: "Product", value: "FairCV — Bias Audit Platform" },
            {
              label: "Purpose",
              value:
                "Evidence-based fairness auditing for automated CV/résumé screening models",
            },
            {
              label: "Audit Engine",
              value:
                "Python-based pipeline. All calculations are server-side. The frontend displays, never computes, audit results.",
            },
            {
              label: "Frontend Stack",
              value: "React 19 · Vite 8 · TypeScript · Tailwind CSS v4 · shadcn · Base UI · Geist",
            },
          ].map(({ label, value }) => (
            <div key={label} className="grid grid-cols-[10rem_1fr] gap-2">
              <dt className="text-xs font-semibold uppercase tracking-wider text-muted-foreground pt-0.5">
                {label}
              </dt>
              <dd className="text-sm text-foreground">{value}</dd>
            </div>
          ))}
        </dl>
      </SectionCard>
    </div>
  )
}
