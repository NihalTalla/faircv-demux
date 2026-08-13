import { cn } from "@/lib/utils"

interface SectionCardProps {
  title?: string
  description?: string
  action?: React.ReactNode
  children?: React.ReactNode
  className?: string
  contentClassName?: string
}

export function SectionCard({
  title,
  description,
  action,
  children,
  className,
  contentClassName,
}: SectionCardProps) {
  return (
    <div
      className={cn(
        "rounded-lg border border-border bg-card text-card-foreground",
        className
      )}
    >
      {(title || action) && (
        <div className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
          <div>
            {title && (
              <h3 className="text-sm font-semibold text-foreground">{title}</h3>
            )}
            {description && (
              <p className="mt-0.5 text-xs text-muted-foreground">
                {description}
              </p>
            )}
          </div>
          {action && <div className="shrink-0">{action}</div>}
        </div>
      )}
      <div className={cn("p-5", contentClassName)}>{children}</div>
    </div>
  )
}
