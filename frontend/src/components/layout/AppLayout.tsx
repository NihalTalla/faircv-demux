import { useState } from "react"
import { Outlet, useLocation } from "react-router-dom"
import { Sidebar } from "./Sidebar"
import { Header } from "./Header"

const pageMeta: Record<string, { title: string; subtitle?: string }> = {
  "/": { title: "Overview", subtitle: "Audit pipeline status" },
  "/datasets": { title: "Dataset", subtitle: "Inspection & validation" },
  "/models": { title: "Models", subtitle: "M1 – M6 configuration & evaluation" },
  "/fairness": { title: "Fairness Analysis", subtitle: "DPD · DIR · EOD · EO · KL" },
  "/statistics": { title: "Statistical Analysis", subtitle: "KS · Kruskal-Wallis · χ² · Holm" },
  "/robustness": { title: "Robustness", subtitle: "Median · P75 · Top-N sensitivity" },
  "/evidence": { title: "Evidence", subtitle: "Audit trail & finding provenance" },
  "/verdict": { title: "Final Verdict", subtitle: "Overall audit interpretation" },
  "/assistant": { title: "AI Assistant", subtitle: "Audit explainer · Groq-powered" },
  "/settings": { title: "Settings" },
}

function resolveMeta(pathname: string) {
  if (pageMeta[pathname]) return pageMeta[pathname]
  const segment = `/${pathname.split("/")[1]}`
  return pageMeta[segment] ?? { title: "FairCV" }
}

export function AppLayout() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const { pathname } = useLocation()
  const meta = resolveMeta(pathname)

  return (
    <div className="flex h-svh overflow-hidden bg-background">
      {/* Sidebar — desktop only */}
      <div className="hidden md:flex md:shrink-0">
        <Sidebar collapsed={sidebarCollapsed} />
      </div>

      {/* Main area */}
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <Header
          title={meta.title}
          subtitle={meta.subtitle}
          sidebarCollapsed={sidebarCollapsed}
          onToggleSidebar={() => setSidebarCollapsed((v) => !v)}
        />
        <main
          id="main-content"
          className="flex-1 overflow-y-auto focus-visible:outline-none"
          tabIndex={-1}
        >
          <div className="mx-auto max-w-6xl p-6 lg:p-8">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}
