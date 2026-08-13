import { BrowserRouter, Routes, Route } from "react-router-dom"
import { AppLayout } from "@/components/layout/AppLayout"
import { OverviewPage } from "@/pages/OverviewPage"
import { DatasetsPage } from "@/pages/DatasetsPage"
import { ModelsPage } from "@/pages/ModelsPage"
import { FairnessPage } from "@/pages/FairnessPage"
import { StatisticsPage } from "@/pages/StatisticsPage"
import { RobustnessPage } from "@/pages/RobustnessPage"
import { EvidencePage } from "@/pages/EvidencePage"
import { VerdictPage } from "@/pages/VerdictPage"
import { SettingsPage } from "@/pages/SettingsPage"
import { AssistantPage } from "@/pages/AssistantPage"

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<OverviewPage />} />
          <Route path="datasets" element={<DatasetsPage />} />
          <Route path="models" element={<ModelsPage />} />
          <Route path="fairness" element={<FairnessPage />} />
          <Route path="statistics" element={<StatisticsPage />} />
          <Route path="robustness" element={<RobustnessPage />} />
          <Route path="evidence" element={<EvidencePage />} />
          <Route path="verdict" element={<VerdictPage />} />
          <Route path="assistant" element={<AssistantPage />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
