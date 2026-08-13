import { BarChart3 } from "lucide-react"
import { cn } from "@/lib/utils"

interface ChartPlaceholderProps {
  label?: string
  description?: string
  height?: number
  className?: string
}

export function ChartPlaceholder({
  label = "Chart",
  description = "Awaiting audit data from backend",
  height = 200,
  className,
}: ChartPlaceholderProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-2 rounded-md border border-dashed border-border bg-muted/30",
        className
      )}
      style={{ minHeight: height }}
      role="img"
      aria-label={`${label} — ${description}`}
    >
      <BarChart3 className="size-8 text-muted-foreground/30" />
      <div className="space-y-0.5 text-center">
        <p className="text-xs font-medium text-muted-foreground/60">{label}</p>
        <p className="text-[0.7rem] text-muted-foreground/40">{description}</p>
      </div>
    </div>
  )
}
