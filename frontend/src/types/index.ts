// ─── Shared primitives ────────────────────────────────────────────────────────

export type EvidenceStrength = "strong" | "moderate" | "limited" | "inconclusive"

export type FairnessMetricKey = "DPD" | "DIR" | "EOD" | "EO" | "KL"

// ─── API error ────────────────────────────────────────────────────────────────

export interface ApiError {
  code: string
  message: string
  details?: unknown
}

// ─── Dataset (mirrors DatasetOut from backend/app/schemas.py) ─────────────────

export interface DatasetOut {
  id: string
  name: string
  source: string
  filename?: string
  row_count?: number
  column_count?: number
  columns?: string[]
  groups?: Record<string, Record<string, number>>
  label_column?: string
  created_at?: string
}

// ─── Audit (mirrors AuditOut, AuditCreate, AuditRunOut from schemas.py) ───────

export interface AuditCreate {
  name: string
  dataset_mode: "demo" | "csv"
  dataset_id?: string
  protected_attributes: string[]
  config?: Record<string, unknown>
}

export interface AuditOut {
  id: string
  name: string
  status: string
  stage?: string
  progress?: number
  dataset_mode: "demo" | "csv"
  dataset_id?: string
  protected_attributes: string[]
  config?: Record<string, unknown>
  error?: string
  created_at?: string
  completed_at?: string
}

export interface AuditRunOut {
  id: string
  status: string
  stage?: string
  progress?: number
  message: string
}

// ─── Metric rows (from demo_pipeline.py metric_rows dict structure) ───────────

export interface MetricRow {
  model: string
  short_name: string
  features: string
  label_set: string
  attribute: string
  n: number
  DPD: number
  DPD_ci_lo: number
  DPD_ci_hi: number
  DIR: number
  DIR_ci_lo: number
  DIR_ci_hi: number
  EOD: number
  EOD_ci_lo: number
  EOD_ci_hi: number
  EO: number
  EO_ci_lo: number
  EO_ci_hi: number
  KL_extreme: number
  KL_mean_pairwise: number
  KS_min_p_adj: number
  KW_stat: number
  KW_p: number
  chi2_p: number
  EEOC_pass: boolean
  highest_SR_group: string
  highest_TPR_group: string
  cohens_d_extreme: number
  cohens_h_extreme: number
}

// ─── Per-group rows (from demo_pipeline.py per_group_rows dict structure) ─────

export interface PerGroupRow {
  model: string
  attribute: string
  group: string
  n: number
  selection_rate: number
  tpr: number
  fpr: number
  ppv: number
  mean_score: number
  std_score: number
}

// ─── Statistical test rows (from demo_pipeline.py test_rows dict structure) ───

export interface StatRow {
  model: string
  attribute: string
  test: string
  comparison: string
  statistic: number
  p_value: number
  p_adjusted: number | null
  "significant_0.05": boolean
}

// ─── API response envelopes ────────────────────────────────────────────────────

export interface MetricsResponse {
  audit_id: string
  metrics: MetricRow[]
  per_group: PerGroupRow[]
  performance: Record<string, { acc: number; auc: number; f1: number; features: string; label_set: string }>
}

export interface StatisticsResponse {
  audit_id: string
  statistics: StatRow[]
}

export interface RobustnessResponse {
  audit_id: string
  robustness: Record<string, unknown>
}

// ─── StatusBadge display values ───────────────────────────────────────────────

export type StatusKey =
  | "idle"
  | "pending"
  | "validating"
  | "running"
  | "completed"
  | "failed"
  | "valid"
  | "invalid"
  | "unknown"
  | "ready"
  | "generating"

// ─── Assistant (mirrors AssistantMessage/Request/Response from schemas.py) ────

export interface AssistantMessage {
  role: "user" | "assistant"
  content: string
}

export interface AssistantChatRequest {
  message: string
  audit_id?: string
  context_page?: string
  history?: AssistantMessage[]
}

export interface AssistantChatResponse {
  reply: string
  has_audit_context: boolean
  audit_id?: string
  model_used: string
}
