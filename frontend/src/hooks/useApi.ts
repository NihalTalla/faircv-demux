import { useCallback, useEffect, useRef, useState } from "react"
import { ApiRequestError } from "@/api/client"

type AsyncFn<T> = () => Promise<T>

interface UseApiState<T> {
  data: T | null
  loading: boolean
  error: string | null
}

export function useApi<T>(fn: AsyncFn<T>, deps: unknown[] = []) {
  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    loading: true,
    error: null,
  })
  const fnRef = useRef(fn)
  fnRef.current = fn

  const execute = useCallback(async () => {
    setState((s) => ({ ...s, loading: true, error: null }))
    try {
      const data = await fnRef.current()
      setState({ data, loading: false, error: null })
    } catch (err) {
      const message =
        err instanceof ApiRequestError
          ? err.error.message
          : err instanceof Error
            ? err.message
            : "An unexpected error occurred"
      setState({ data: null, loading: false, error: message })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(() => {
    execute()
  }, [execute])

  return { ...state, refetch: execute }
}
