import { Moon, Sun, Monitor, PanelLeftClose, PanelLeft } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useTheme } from "@/components/theme-provider"

interface HeaderProps {
  title: string
  subtitle?: string
  sidebarCollapsed: boolean
  onToggleSidebar: () => void
}

export function Header({ title, subtitle, sidebarCollapsed, onToggleSidebar }: HeaderProps) {
  const { theme, setTheme } = useTheme()

  const cycleTheme = () => {
    if (theme === "light") setTheme("dark")
    else if (theme === "dark") setTheme("system")
    else setTheme("light")
  }

  const ThemeIcon = theme === "light" ? Sun : theme === "dark" ? Moon : Monitor
  const themeLabel =
    theme === "light"
      ? "Switch to dark mode"
      : theme === "dark"
        ? "Switch to system theme"
        : "Switch to light mode"

  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b border-border bg-background px-4">
      <Button
        variant="ghost"
        size="icon-sm"
        onClick={onToggleSidebar}
        aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        {sidebarCollapsed ? (
          <PanelLeft className="size-4" />
        ) : (
          <PanelLeftClose className="size-4" />
        )}
      </Button>

      <div className="flex flex-1 items-baseline gap-2 min-w-0">
        <h1 className="text-sm font-semibold text-foreground truncate">{title}</h1>
        {subtitle && (
          <span className="hidden text-xs text-muted-foreground sm:inline truncate">
            — {subtitle}
          </span>
        )}
      </div>

      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={cycleTheme}
          aria-label={themeLabel}
          title={themeLabel}
        >
          <ThemeIcon className="size-4" />
        </Button>
      </div>
    </header>
  )
}
