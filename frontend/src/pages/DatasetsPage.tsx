import { Database, Users, List } from "lucide-react"
import { SectionCard } from "@/components/ui/SectionCard"
import { EmptyState } from "@/components/ui/EmptyState"
import { LoadingState } from "@/components/ui/LoadingState"
import { ErrorState } from "@/components/ui/ErrorState"
import { useApi } from "@/hooks/useApi"
import { datasetsApi } from "@/api/datasets"

export function DatasetsPage() {
  const { data: demo, loading, error, refetch } = useApi(() => datasetsApi.getDemo())

  return (
    <div className="space-y-6">

      {/* Page intro */}
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight text-foreground">Dataset Inspection</h2>
        <p className="text-sm text-muted-foreground">
          Structure and demographics of the FairCV audit dataset (FairCVdb.npy). Values are sourced from the backend — nothing is hardcoded here.
        </p>
      </div>

      {loading && <LoadingState label="Loading dataset…" />}
      {!loading && error && <ErrorState message={error} onRetry={refetch} />}

      {!loading && !error && !demo && (
        <EmptyState
          icon={Database}
          title="No dataset information returned."
          description="The backend API did not return dataset metadata. Ensure the backend is running and the demo dataset (FairCVdb.npy) is present."
        />
      )}

      {demo && (
        <>
          {/* Stat tiles */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[
              {
                label: "Rows",
                value: demo.row_count != null ? demo.row_count.toLocaleString() : "—",
              },
              {
                label: "Columns",
                value: demo.column_count != null ? String(demo.column_count) : "—",
              },
              {
                label: "Source",
                value: demo.source,
              },
              {
                label: "Label Column",
                value: demo.label_column ?? "—",
              },
            ].map(({ label, value }) => (
              <div key={label} className="rounded-lg border border-border bg-card p-4">
                <p className="text-[0.68rem] font-semibold uppercase tracking-wider text-muted-foreground">
                  {label}
                </p>
                <div className="mt-1.5 text-lg font-semibold text-foreground font-mono tabular-nums">
                  {value}
                </div>
              </div>
            ))}
          </div>

          {/* Demographic groups */}
          {demo.groups && Object.keys(demo.groups).length > 0 ? (
            <SectionCard
              title="Demographic Attributes"
              description="Protected attributes and group sizes used in the fairness audit"
              action={<Users className="size-4 text-muted-foreground" />}
            >
              <div className="space-y-4">
                {Object.entries(demo.groups).map(([attr, counts]) => (
                  <div key={attr}>
                    <p className="mb-1.5 text-xs font-semibold capitalize text-foreground">{attr}</p>
                    <div className="flex flex-wrap gap-1.5">
                      {Object.entries(counts).map(([group, count]) => (
                        <span
                          key={group}
                          className="inline-flex items-center gap-1.5 rounded-full border border-border bg-muted px-2.5 py-0.5 text-xs text-foreground"
                        >
                          {group}
                          <span className="text-muted-foreground">({count.toLocaleString()})</span>
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </SectionCard>
          ) : (
            <SectionCard
              title="Demographic Attributes"
              description="Protected attributes from the backend API"
            >
              <EmptyState title="No group data returned." />
            </SectionCard>
          )}

          {/* Column list */}
          {demo.columns && demo.columns.length > 0 ? (
            <SectionCard
              title="Column Names"
              description={`${demo.columns.length} columns in the dataset`}
              action={<List className="size-4 text-muted-foreground" />}
            >
              <div className="flex flex-wrap gap-1.5">
                {demo.columns.map((col) => (
                  <span
                    key={col}
                    className="rounded-md border border-border bg-muted px-2 py-0.5 font-mono text-[0.7rem] text-muted-foreground"
                  >
                    {col}
                  </span>
                ))}
              </div>
            </SectionCard>
          ) : (
            <SectionCard title="Columns" description="Column names returned by the backend">
              <EmptyState title="No column names returned." />
            </SectionCard>
          )}
        </>
      )}
    </div>
  )
}
