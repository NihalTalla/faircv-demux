import { apiClient } from "./client"
import type { DatasetOut } from "@/types"

export const datasetsApi = {
  getDemo: () => apiClient.get<DatasetOut>("/api/datasets/demo"),
  list: () => apiClient.get<DatasetOut[]>("/api/datasets"),
  upload: (form: FormData) =>
    fetch(`${import.meta.env.VITE_API_BASE_URL ?? "/api"}/api/datasets`, {
      method: "POST",
      body: form,
    }).then((r) => r.json() as Promise<DatasetOut>),
}
