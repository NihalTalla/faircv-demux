import { Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"

interface LoadingStateProps {
  label?: string
  className?: string
}

export function LoadingState({ label = "Loading…", className }: LoadingStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3 py-16 text-center",
        className
      )}
      role="status"
      aria-label={label}
    >
      <Loader2 className="size-6 animate-spin text-muted-foreground" />
      <p className="text-xs text-muted-foreground">{label}</p>
    </div>
  )
}

export function LoadingRows({ rows = 5 }: { rows?: number }) {
  return (
    <div className="space-y-2" role="status" aria-label="Loading data">
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          className="h-9 animate-pulse rounded-md bg-muted"
          style={{ opacity: 1 - i * 0.12 }}
        />
      ))}
    </div>
  )
}
