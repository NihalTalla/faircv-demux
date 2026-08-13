import { apiClient } from "./client"
import type { AssistantChatRequest, AssistantChatResponse } from "@/types"

export const assistantApi = {
  chat: (body: AssistantChatRequest) =>
    apiClient.post<AssistantChatResponse>("/api/assistant/chat", body),
}
