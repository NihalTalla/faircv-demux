import { useState, useEffect, useRef, useCallback } from "react"
import {
  Bot,
  Send,
  Trash2,
  AlertCircle,
  Loader2,
  MessageSquare,
  ChevronDown,
} from "lucide-react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { assistantApi } from "@/api/assistant"
import { auditsApi } from "@/api/audits"
import type { AssistantMessage, AuditOut } from "@/types"

// ─── Local message type (with UI-only id) ─────────────────────────────────────

interface LocalMessage {
  id: string
  role: "user" | "assistant"
  content: string
}

// ─── Quick-action prompts ──────────────────────────────────────────────────────

const QUICK_ACTIONS_GENERAL = [
  {
    label: "What is DPD?",
    prompt:
      "What is Demographic Parity Difference (DPD) and how is it calculated in FairCV?",
  },
  {
    label: "DIR & four-fifths rule",
    prompt:
      "Explain the Disparate Impact Ratio (DIR) and the EEOC four-fifths rule used in FairCV.",
  },
  {
    label: "Bootstrap CIs",
    prompt:
      "Explain the bootstrap confidence intervals used in FairCV audits: how they are computed, what the width indicates, and how to interpret them.",
  },
  {
    label: "M1–M6 models",
    prompt:
      "What are the M1 through M6 models in FairCV and how do they differ from each other in terms of features and label definitions?",
  },
  {
    label: "Holm-Bonferroni",
    prompt:
      "Explain the Holm-Bonferroni multiple-comparison correction used in FairCV statistical tests.",
  },
  {
    label: "KL divergence",
    prompt:
      "What is KL divergence and how is it used as a fairness metric in FairCV?",
  },
]

const QUICK_ACTIONS_AUDIT = [
  {
    label: "Summarise audit",
    prompt: "Please give an overall summary of this audit's fairness findings.",
  },
  {
    label: "Key findings",
    prompt:
      "What are the most important fairness findings from this audit? Highlight the strongest evidence.",
  },
  {
    label: "Compare models",
    prompt:
      "Compare the fairness performance of all models in this audit. Which model is most equitable and why?",
  },
  {
    label: "Statistical significance",
    prompt:
      "Which findings in this audit are statistically significant after Holm-Bonferroni correction?",
  },
  {
    label: "Explain CIs",
    prompt:
      "Explain the confidence intervals for the key fairness metrics in this audit. What do the bounds imply?",
  },
  {
    label: "Non-technical summary",
    prompt:
      "Summarise this audit's findings in plain language suitable for a non-technical stakeholder or hiring manager.",
  },
  {
    label: "Robustness results",
    prompt:
      "Explain the robustness analysis results from this audit and what they mean for the strength of the findings.",
  },
  {
    label: "EEOC compliance",
    prompt:
      "Which models in this audit pass or fail the EEOC four-fifths rule and what does that mean?",
  },
]

// ─── Markdown renderer (assistant responses only) ─────────────────────────────

function MarkdownMessage({ content }: { content: string }) {
  return (
    <div
      className={cn(
        "text-sm leading-relaxed text-foreground",
        "[&_p]:mb-2 [&_p:last-child]:mb-0",
        "[&_ul]:mb-2 [&_ul]:ml-4 [&_ul]:list-disc",
        "[&_ol]:mb-2 [&_ol]:ml-4 [&_ol]:list-decimal",
        "[&_li]:mb-0.5",
        "[&_strong]:font-semibold",
        "[&_em]:italic",
        "[&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-xs",
        "[&_pre]:mb-2 [&_pre]:overflow-x-auto [&_pre]:rounded-md [&_pre]:border [&_pre]:border-border [&_pre]:bg-muted [&_pre]:p-3",
        "[&_pre_code]:bg-transparent [&_pre_code]:p-0",
        "[&_h1]:mb-2 [&_h1]:text-base [&_h1]:font-bold",
        "[&_h2]:mb-1.5 [&_h2]:mt-3 [&_h2]:text-sm [&_h2]:font-bold",
        "[&_h3]:mb-1 [&_h3]:mt-2 [&_h3]:text-sm [&_h3]:font-semibold",
        "[&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-3 [&_blockquote]:italic [&_blockquote]:text-muted-foreground",
        "[&_table]:mb-2 [&_table]:w-full [&_table]:border-collapse [&_table]:text-xs",
        "[&_th]:border [&_th]:border-border [&_th]:bg-muted [&_th]:px-2 [&_th]:py-1 [&_th]:text-left [&_th]:font-semibold",
        "[&_td]:border [&_td]:border-border [&_td]:px-2 [&_td]:py-1",
        "[&_hr]:my-2 [&_hr]:border-border"
      )}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  )
}

// ─── Message bubble ───────────────────────────────────────────────────────────

function MessageBubble({ msg }: { msg: LocalMessage }) {
  const isUser = msg.role === "user"
  return (
    <div className={cn("flex gap-3", isUser ? "flex-row-reverse" : "flex-row")}>
      <div
        className={cn(
          "flex size-7 shrink-0 items-center justify-center rounded-full text-[0.6875rem] font-bold",
          isUser
            ? "bg-sidebar-primary text-sidebar-primary-foreground"
            : "border border-border bg-muted text-muted-foreground"
        )}
      >
        {isUser ? "U" : <Bot className="size-3.5" />}
      </div>
      <div
        className={cn(
          "max-w-[78%] rounded-xl px-4 py-3",
          isUser
            ? "rounded-tr-sm bg-sidebar-primary text-sidebar-primary-foreground"
            : "rounded-tl-sm border border-border bg-card"
        )}
      >
        {isUser ? (
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
        ) : (
          <MarkdownMessage content={msg.content} />
        )}
      </div>
    </div>
  )
}

// ─── Audit context selector ───────────────────────────────────────────────────

function AuditSelector({
  audits,
  loading,
  auditId,
  onChange,
}: {
  audits: AuditOut[]
  loading: boolean
  auditId: string
  onChange: (id: string) => void
}) {
  const completed = audits.filter((a) => a.status === "completed")
  return (
    <div className="flex items-center gap-2 min-w-0">
      <span className="shrink-0 text-xs text-muted-foreground">Audit context:</span>
      <div className="relative min-w-[200px]">
        <select
          value={auditId}
          onChange={(e) => onChange(e.target.value)}
          disabled={loading || completed.length === 0}
          className="w-full appearance-none rounded-md border border-border bg-background px-3 py-1.5 pr-8 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
          aria-label="Select audit for assistant context"
        >
          <option value="">No audit — general questions only</option>
          {completed.map((a) => (
            <option key={a.id} value={a.id}>
              {a.name}
            </option>
          ))}
        </select>
        <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
      </div>
      {loading && <Loader2 className="size-3.5 animate-spin text-muted-foreground" />}
      {!loading && auditId && (
        <span className="shrink-0 text-[0.68rem] text-emerald-600 dark:text-emerald-400">
          ● Audit context loaded
        </span>
      )}
      {!loading && !auditId && completed.length === 0 && (
        <span className="shrink-0 text-[0.68rem] text-muted-foreground/50">
          No completed audits found
        </span>
      )}
    </div>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export function AssistantPage() {
  const [messages, setMessages] = useState<LocalMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [auditId, setAuditId] = useState("")
  const [audits, setAudits] = useState<AuditOut[]>([])
  const [auditsLoading, setAuditsLoading] = useState(true)
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    setAuditsLoading(true)
    auditsApi
      .list()
      .then((list) => {
        setAudits(list)
        const first = list.find((a) => a.status === "completed")
        if (first) setAuditId(first.id)
      })
      .catch(() => {
        // Non-fatal: audit list is best-effort
      })
      .finally(() => setAuditsLoading(false))
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isLoading])

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim()
      if (!trimmed || isLoading) return

      const userMsg: LocalMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: trimmed,
      }
      setMessages((prev) => [...prev, userMsg])
      setInput("")
      setError(null)
      setIsLoading(true)

      // History excludes the new user message — service appends it
      const history: AssistantMessage[] = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }))

      try {
        const response = await assistantApi.chat({
          message: trimmed,
          audit_id: auditId || undefined,
          history,
        })
        setMessages((prev) => [
          ...prev,
          { id: crypto.randomUUID(), role: "assistant", content: response.reply },
        ])
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : "Failed to get a response from the assistant."
        setError(msg)
        setMessages((prev) => prev.filter((m) => m.id !== userMsg.id))
      } finally {
        setIsLoading(false)
        textareaRef.current?.focus()
      }
    },
    [messages, auditId, isLoading]
  )

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      void sendMessage(input)
    }
  }

  const handleAuditChange = (id: string) => {
    setAuditId(id)
    if (messages.length > 0) {
      setMessages([])
      setError(null)
    }
  }

  const clearConversation = () => {
    setMessages([])
    setError(null)
  }

  const quickActions = auditId ? QUICK_ACTIONS_AUDIT : QUICK_ACTIONS_GENERAL
  const hasMessages = messages.length > 0

  return (
    <div className="space-y-4">
      {/* Header row */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-foreground">AI Assistant</h2>
          <p className="mt-0.5 max-w-xl text-sm text-muted-foreground">
            FairCV concept explainer and audit interpreter. Answers are grounded in real computed
            audit results — no fabricated numbers.
          </p>
        </div>
        {(hasMessages || error) && (
          <Button variant="outline" size="sm" onClick={clearConversation} className="shrink-0">
            <Trash2 className="mr-1.5 size-3.5" />
            Clear
          </Button>
        )}
      </div>

      {/* Audit context selector */}
      <div className="flex items-center gap-3 rounded-lg border border-border bg-card px-4 py-2.5">
        <AuditSelector
          audits={audits}
          loading={auditsLoading}
          auditId={auditId}
          onChange={handleAuditChange}
        />
      </div>

      {/* Quick actions */}
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="mr-0.5 self-center text-[0.68rem] font-semibold uppercase tracking-widest text-muted-foreground/40">
          Quick:
        </span>
        {quickActions.map(({ label, prompt }) => (
          <button
            key={label}
            onClick={() => void sendMessage(prompt)}
            disabled={isLoading}
            className="rounded-full border border-border bg-card px-2.5 py-1 text-[0.72rem] text-muted-foreground transition-colors hover:border-foreground/30 hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
          >
            {label}
          </button>
        ))}
      </div>

      {/* Message area */}
      <div className="flex min-h-64 flex-col rounded-xl border border-border bg-card">
        {/* Empty state */}
        {!hasMessages && (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 py-16 text-center">
            <div className="flex size-12 items-center justify-center rounded-full border border-dashed border-border bg-muted/30">
              <MessageSquare className="size-5 text-muted-foreground/40" />
            </div>
            <div>
              <p className="text-sm font-medium text-foreground">Ask anything about FairCV</p>
              <p className="mt-1 max-w-xs text-xs text-muted-foreground">
                {auditId
                  ? "An audit is loaded — ask about specific metrics, model comparisons, or statistical findings."
                  : "No audit loaded. Ask about fairness concepts, metrics, statistical tests, or methodology."}
              </p>
            </div>
          </div>
        )}

        {/* Messages */}
        {hasMessages && (
          <div className="flex flex-col gap-4 overflow-y-auto p-4">
            {messages.map((msg) => (
              <MessageBubble key={msg.id} msg={msg} />
            ))}
            {isLoading && (
              <div className="flex gap-3">
                <div className="flex size-7 shrink-0 items-center justify-center rounded-full border border-border bg-muted">
                  <Bot className="size-3.5 text-muted-foreground" />
                </div>
                <div className="flex items-center gap-2 rounded-xl rounded-tl-sm border border-border bg-card px-4 py-3">
                  <Loader2 className="size-3.5 animate-spin text-muted-foreground" />
                  <span className="text-xs text-muted-foreground">Thinking…</span>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        )}

        {!hasMessages && <div ref={bottomRef} />}
      </div>

      {/* Error banner */}
      {error && (
        <div className="flex items-start gap-2.5 rounded-lg border border-red-200 bg-red-50/60 px-4 py-3 dark:border-red-900/40 dark:bg-red-950/20">
          <AlertCircle className="mt-0.5 size-4 shrink-0 text-red-500" />
          <p className="flex-1 text-xs text-red-700 dark:text-red-400">{error}</p>
          <button
            onClick={() => setError(null)}
            className="shrink-0 text-xs text-red-500 hover:text-red-700"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Input area */}
      <div className="rounded-xl border border-border bg-card px-4 py-3">
        <div className="flex items-end gap-3">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              auditId
                ? "Ask about this audit's findings, metrics, or statistical tests…"
                : "Ask about fairness metrics, statistical tests, or FairCV methodology…"
            }
            rows={2}
            disabled={isLoading}
            className="flex-1 resize-none bg-transparent text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
            aria-label="Chat input"
          />
          <Button
            onClick={() => void sendMessage(input)}
            disabled={!input.trim() || isLoading}
            size="sm"
            className="mb-0.5 shrink-0"
            aria-label="Send message"
          >
            {isLoading ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Send className="size-4" />
            )}
          </Button>
        </div>
        <p className="mt-1.5 text-center text-[0.65rem] text-muted-foreground/40">
          Enter to send · Shift+Enter for new line · Answers cite only computed audit data
        </p>
      </div>

      {/* Methodology note */}
      <div className="rounded-lg border border-border bg-muted/20 px-4 py-3">
        <p className="text-[0.72rem] leading-relaxed text-muted-foreground">
          <span className="font-semibold text-foreground">About the assistant:</span> This AI
          assistant explains FairCV methodology and interprets completed audit results. It is
          grounded in the real computed data from the selected audit and will not invent metric
          values, p-values, or confidence intervals. If a value is not in the audit record, it
          will say so. The assistant does not replace or modify the FairCV audit engine.
        </p>
      </div>
    </div>
  )
}
