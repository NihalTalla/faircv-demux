import type { ApiError } from "@/types"

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api"

export class ApiRequestError extends Error {
  readonly status: number
  readonly error: ApiError

  constructor(status: number, error: ApiError) {
    super(error.message)
    this.name = "ApiRequestError"
    this.status = status
    this.error = error
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  })

  if (!res.ok) {
    let error: ApiError
    try {
      error = await res.json()
    } catch {
      error = { code: "UNKNOWN", message: `HTTP ${res.status}` }
    }
    throw new ApiRequestError(res.status, error)
  }

  return res.json() as Promise<T>
}

export const apiClient = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "POST", body: JSON.stringify(body) }),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
}
