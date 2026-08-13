import { NavLink } from "react-router-dom"
import {
  LayoutDashboard,
  Database,
  BrainCircuit,
  ShieldCheck,
  BarChart3,
  LineChart,
  BookOpen,
  Scale,
  Bot,
  Settings,
} from "lucide-react"
import { cn } from "@/lib/utils"

interface NavGroup {
  label?: string
  items: {
    to: string
    label: string
    icon: typeof LayoutDashboard
    end?: boolean
  }[]
}

const navGroups: NavGroup[] = [
  {
    items: [
      { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
    ],
  },
  {
    label: "Data",
    items: [
      { to: "/datasets", label: "Dataset", icon: Database },
      { to: "/models", label: "Models", icon: BrainCircuit },
    ],
  },
  {
    label: "Analysis",
    items: [
      { to: "/fairness", label: "Fairness", icon: ShieldCheck },
      { to: "/statistics", label: "Statistics", icon: BarChart3 },
      { to: "/robustness", label: "Robustness", icon: LineChart },
    ],
  },
  {
    label: "Findings",
    items: [
      { to: "/evidence", label: "Evidence", icon: BookOpen },
      { to: "/verdict", label: "Verdict", icon: Scale },
    ],
  },
  {
    label: "AI",
    items: [
      { to: "/assistant", label: "Assistant", icon: Bot },
    ],
  },
]

interface SidebarProps {
  collapsed?: boolean
}

const linkClass = (isActive: boolean, collapsed: boolean) =>
  cn(
    "flex items-center gap-2.5 rounded-md px-2.5 py-2 text-[0.8125rem] font-medium transition-colors duration-100",
    isActive
      ? "bg-sidebar-accent text-sidebar-accent-foreground"
      : "text-sidebar-foreground/60 hover:bg-sidebar-accent/50 hover:text-sidebar-foreground",
    collapsed && "justify-center px-2"
  )

export function Sidebar({ collapsed = false }: SidebarProps) {
  return (
    <aside
      className={cn(
        "flex h-full flex-col bg-sidebar border-r border-sidebar-border transition-[width] duration-200",
        collapsed ? "w-[3.25rem]" : "w-[13.5rem]"
      )}
      aria-label="Application navigation"
    >
      {/* Brand */}
      <div
        className={cn(
          "flex h-14 shrink-0 items-center border-b border-sidebar-border",
          collapsed ? "justify-center px-0" : "gap-2.5 px-4"
        )}
      >
        <div className="flex size-7 shrink-0 items-center justify-center rounded-md bg-sidebar-primary">
          <span className="text-[11px] font-bold tracking-tight text-sidebar-primary-foreground">
            FCV
          </span>
        </div>
        {!collapsed && (
          <div className="min-w-0">
            <p className="truncate text-[0.8125rem] font-semibold text-sidebar-foreground">
              FairCV
            </p>
            <p className="truncate text-[0.65rem] text-sidebar-foreground/50">
              Bias Audit Platform
            </p>
          </div>
        )}
      </div>

      {/* Nav groups */}
      <nav className="flex flex-1 flex-col gap-1 overflow-y-auto p-2">
        {navGroups.map((group, gi) => (
          <div key={gi} className={cn("flex flex-col gap-0.5", gi > 0 && "mt-3")}>
            {group.label && !collapsed && (
              <p className="mb-0.5 px-2.5 text-[0.65rem] font-semibold uppercase tracking-widest text-sidebar-foreground/35">
                {group.label}
              </p>
            )}
            {group.items.map(({ to, label, icon: Icon, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                title={collapsed ? label : undefined}
                className={({ isActive }) => linkClass(isActive, collapsed)}
              >
                <Icon className="size-[1.0625rem] shrink-0" />
                {!collapsed && <span>{label}</span>}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      {/* Settings */}
      <div className="border-t border-sidebar-border p-2">
        <NavLink
          to="/settings"
          title={collapsed ? "Settings" : undefined}
          className={({ isActive }) => linkClass(isActive, collapsed)}
        >
          <Settings className="size-[1.0625rem] shrink-0" />
          {!collapsed && <span>Settings</span>}
        </NavLink>
      </div>
    </aside>
  )
}
