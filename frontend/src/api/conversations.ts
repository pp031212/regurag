import client from './client'
import type { Citation, ChatDebug } from './chat'

// 会话是聊天容器，不强绑定当前知识库；每条消息会记录实际使用的知识库。
export interface Conversation {
  id: string
  default_knowledge_base_id?: string | null
  title: string
  created_at: string
  updated_at: string
}

export interface ConversationListResponse {
  items: Conversation[]
  total: number
}

export interface ConversationMessage {
  id: string
  conversation_id: string
  knowledge_base_id?: string | null
  sequence: number
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  debug?: ChatDebug | null
  created_at: string
}

export interface ConversationMessageListResponse {
  items: ConversationMessage[]
  total: number
}

export interface ConversationDeleteResponse {
  id: string
  deleted: boolean
}

export const conversationApi = {
  // knowledgeBaseId 只作为列表筛选条件，不会改变会话历史消息。
  list: (knowledgeBaseId?: string): Promise<ConversationListResponse> =>
    client.get('/conversations', { params: knowledgeBaseId ? { knowledge_base_id: knowledgeBaseId } : undefined }),
  create: (payload?: { title?: string; default_knowledge_base_id?: string | null }): Promise<Conversation> =>
    client.post('/conversations', payload || {}),
  listMessages: (conversationId: string): Promise<ConversationMessageListResponse> => client.get(`/conversations/${conversationId}/messages`),
  delete: (conversationId: string): Promise<ConversationDeleteResponse> => client.delete(`/conversations/${conversationId}`),
}
