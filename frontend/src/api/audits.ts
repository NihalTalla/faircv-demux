import { apiClient } from "./client"
import type {
  AuditCreate,
  AuditOut,
  AuditRunOut,
  MetricsResponse,
  StatisticsResponse,
  RobustnessResponse,
} from "@/types"

export const auditsApi = {
  list: () => apiClient.get<AuditOut[]>("/api/audits"),
  get: (id: string) => apiClient.get<AuditOut>(`/api/audits/${id}`),
  create: (body: AuditCreate) => apiClient.post<AuditOut>("/api/audits", body),
  run: (id: string) => apiClient.post<AuditRunOut>(`/api/audits/${id}/run`, {}),
  metrics: (id: string) => apiClient.get<MetricsResponse>(`/api/audits/${id}/metrics`),
  statistics: (id: string) => apiClient.get<StatisticsResponse>(`/api/audits/${id}/statistics`),
  robustness: (id: string) => apiClient.get<RobustnessResponse>(`/api/audits/${id}/robustness`),
}
