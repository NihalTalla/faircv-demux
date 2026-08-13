import { cn } from "@/lib/utils"
import type { EvidenceStrength } from "@/types"

const config: Record<EvidenceStrength, { label: string; className: string; dot: string }> = {
  strong: {
    label: "Strong",
    className: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-400",
    dot: "bg-emerald-500",
  },
  moderate: {
    label: "Moderate",
    className: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400",
    dot: "bg-blue-500",
  },
  limited: {
    label: "Limited",
    className: "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400",
    dot: "bg-amber-500",
  },
  inconclusive: {
    label: "Inconclusive",
    className: "bg-muted text-muted-foreground",
    dot: "bg-muted-foreground/40",
  },
}

interface EvidenceStrengthBadgeProps {
  strength: EvidenceStrength
  className?: string
}

export function EvidenceStrengthBadge({ strength, className }: EvidenceStrengthBadgeProps) {
  const c = config[strength]
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium",
        c.className,
        className
      )}
    >
      <span className={cn("size-1.5 rounded-full", c.dot)} />
      {c.label}
    </span>
  )
}
