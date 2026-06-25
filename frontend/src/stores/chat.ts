import { defineStore } from 'pinia'
import { chatApi, type Citation, type ChatDebug } from '../api/chat'
import { conversationApi, type Conversation } from '../api/conversations'
import { useKBStore } from './kb'

// 会话选择只记在 sessionStorage，刷新当前标签页可恢复，但不会跨浏览器会话长期保留。
const isDev = import.meta.env.DEV
const ACTIVE_CONVERSATION_KEY = 'regurag.active_conversation_id'

const createMessageId = () => {
  if (typeof globalThis.crypto !== 'undefined' && typeof globalThis.crypto.randomUUID === 'function') {
    return globalThis.crypto.randomUUID()
  }

  return `msg-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  debug?: ChatDebug
  sequence?: number
  streaming?: boolean
  timestamp: number
}

const compareMessages = (left: Message, right: Message) => {
  // 后端历史消息按 sequence 排序；本地新消息还没有 sequence 时退回 timestamp。
  if (left.sequence !== undefined && right.sequence !== undefined && left.sequence !== right.sequence) {
    return left.sequence - right.sequence
  }
  if (left.timestamp !== right.timestamp) {
    return left.timestamp - right.timestamp
  }
  if (left.role === right.role) {
    return 0
  }
  return left.role === 'user' ? -1 : 1
}

const readActiveConversationId = (): string | null => {
  if (typeof window === 'undefined') return null
  try {
    return window.sessionStorage.getItem(ACTIVE_CONVERSATION_KEY)
  } catch {
    return null
  }
}

const writeActiveConversationId = (value: string | null) => {
  if (typeof window === 'undefined') return
  if (value) {
    window.sessionStorage.setItem(ACTIVE_CONVERSATION_KEY, value)
    return
  }
  window.sessionStorage.removeItem(ACTIVE_CONVERSATION_KEY)
}

export const useChatStore = defineStore('chat', {
  state: () => ({
    messages: [] as Message[],
    conversations: [] as Conversation[],
    currentConversationId: null as string | null,
    loading: false,
    error: null as string | null
  }),

  actions: {
    _updateMessage(messageId: string, patch: Partial<Message>) {
      const target = this.messages.find((item) => item.id === messageId)
      if (!target) return
      Object.assign(target, patch)
    },

    async _syncResolvedKnowledgeBase(targetKnowledgeBaseId: string | null | undefined) {
      // 自动路由可能切到别的知识库，前端要把当前 KB 同步到后端最终解析结果。
      if (!targetKnowledgeBaseId) return

      const kbStore = useKBStore()
      if (kbStore.currentKB?.id === targetKnowledgeBaseId) {
        return
      }

      const matchedKB = kbStore.knowledgeBases.find((item) => item.id === targetKnowledgeBaseId) || null
      if (matchedKB) {
        kbStore.setCurrentKB(matchedKB)
        return
      }

      await kbStore.fetchKBs()
      const refreshedKB = kbStore.knowledgeBases.find((item) => item.id === targetKnowledgeBaseId) || null
      if (refreshedKB) {
        kbStore.setCurrentKB(refreshedKB)
      }
    },

    async _loadConversationMessages(conversationId: string) {
      const messageResponse = await conversationApi.listMessages(conversationId)
      this.error = null
      this.messages = messageResponse.items
        .map((item) => ({
          id: item.id,
          role: item.role,
          content: item.content,
          citations: item.citations,
          debug: item.debug || undefined,
          sequence: item.sequence,
          streaming: false,
          timestamp: new Date(item.created_at).getTime(),
        }))
        .sort(compareMessages)
    },

    async _refreshConversations() {
      const response = await conversationApi.list()
      this.conversations = response.items
    },

    _rememberConversation(conversationId: string | null) {
      writeActiveConversationId(conversationId)
    },

    async hydrateConversation(options?: { restoreStoredSelection?: boolean }) {
      // 进入聊天页时先刷新会话列表，再决定恢复当前会话、sessionStorage 会话或空会话。
      await this._refreshConversations()

      const restoreStoredSelection = options?.restoreStoredSelection ?? false
      const storedConversationId = restoreStoredSelection ? readActiveConversationId() : null
      const currentConversation =
        this.currentConversationId
          ? this.conversations.find((item) => item.id === this.currentConversationId) || null
          : null
      const storedConversation =
        storedConversationId
          ? this.conversations.find((item) => item.id === storedConversationId) || null
          : null
      const activeConversation = currentConversation || storedConversation || null

      this.currentConversationId = activeConversation?.id || null
      this._rememberConversation(this.currentConversationId)

      if (!this.currentConversationId) {
        this.error = null
        this.messages = []
        return
      }

      await this._loadConversationMessages(this.currentConversationId)
    },

    async selectConversation(conversationId: string) {
      if (this.currentConversationId === conversationId) {
        return
      }

      this.error = null
      this.currentConversationId = conversationId
      this._rememberConversation(conversationId)
      await this._loadConversationMessages(conversationId)
    },

    async startNewConversation() {
      this.error = null
      this.currentConversationId = null
      this.messages = []
      this._rememberConversation(null)
      await this._refreshConversations()
    },

    async deleteConversation(conversationId: string) {
      this.error = null
      await conversationApi.delete(conversationId)
      await this._refreshConversations()

      if (this.currentConversationId !== conversationId) {
        return
      }

      const nextConversationId = this.conversations[0]?.id || null
      this.currentConversationId = nextConversationId
      this._rememberConversation(nextConversationId)

      if (!nextConversationId) {
        this.messages = []
        return
      }

      await this._loadConversationMessages(nextConversationId)
    },

    async clearAllConversations() {
      this.error = null
      const conversationIds = this.conversations.map((item) => item.id)
      if (conversationIds.length === 0) {
        this.currentConversationId = null
        this.messages = []
        this._rememberConversation(null)
        return
      }

      await Promise.all(conversationIds.map((conversationId) => conversationApi.delete(conversationId)))
      this.conversations = []
      this.currentConversationId = null
      this.messages = []
      this._rememberConversation(null)
    },

    async sendMessage(kbId: string, query: string) {
      // 先乐观插入用户消息，后续 start/end 事件会带回真正的 conversation_id。
      const userMessage: Message = {
        id: createMessageId(),
        role: 'user',
        content: query,
        streaming: false,
        timestamp: Date.now()
      }
      this.messages.push(userMessage)

      this.loading = true
      this.error = null
      let assistantMessageId: string | null = null
      let pendingDelta = ''
      let flushScheduled = false

      const flushPendingDelta = () => {
        // token 先累积再批量写入，避免每个 token 都触发一次 Vue 渲染。
        if (!assistantMessageId || !pendingDelta) return

        const currentMessage = this.messages.find((item) => item.id === assistantMessageId)
        this._updateMessage(assistantMessageId, {
          content: `${currentMessage?.content || ''}${pendingDelta}`,
        })
        pendingDelta = ''
      }

      const scheduleDeltaFlush = () => {
        // requestAnimationFrame 让流式输出按帧刷新，兼顾顺滑和性能。
        if (flushScheduled) return
        flushScheduled = true
        requestAnimationFrame(() => {
          flushScheduled = false
          flushPendingDelta()
        })
      }

      try {
        await chatApi.streamQuery({
          knowledge_base_id: kbId,
          query,
          conversation_id: this.currentConversationId,
          enable_auto_route: true,
          debug: isDev,
          debug_chunks: isDev
        }, {
          onStart: (payload) => {
            // start 事件最早告诉前端后端创建/解析出的会话 id。
            this.currentConversationId = payload.conversation_id
            this._rememberConversation(payload.conversation_id)
          },
          onToken: (delta) => {
            if (!assistantMessageId) {
              assistantMessageId = createMessageId()
              this.messages.push({
                id: assistantMessageId,
                role: 'assistant',
                content: '',
                streaming: true,
                timestamp: Date.now()
              })
            }

            pendingDelta += delta
            scheduleDeltaFlush()
          },
          onEnd: (response) => {
            // end 事件包含最终 answer、citations 和 debug，覆盖临时流式内容。
            flushPendingDelta()
            if (!assistantMessageId) {
              assistantMessageId = createMessageId()
              this.messages.push({
                id: assistantMessageId,
                role: 'assistant',
                content: response.answer,
                streaming: false,
                timestamp: Date.now()
              })
            }

            this._updateMessage(assistantMessageId, {
              content: response.answer,
              citations: response.citations,
              debug: response.debug,
              streaming: false,
              timestamp: Date.now(),
            })

            this.currentConversationId = response.conversation_id
            this._rememberConversation(response.conversation_id)
            void this._syncResolvedKnowledgeBase(response.knowledge_base_id)
          }
        })

        this.loading = false
        await this._refreshConversations()
      } catch (err: any) {
        if (assistantMessageId) {
          flushPendingDelta()
          this._updateMessage(assistantMessageId, { streaming: false })
        }
        this.error = err.message || '提问失败'
        const lastMessage = this.messages[this.messages.length - 1]
        if (lastMessage?.role === 'assistant' && !lastMessage.content.trim()) {
          this.messages.pop()
        }
        this.loading = false
      } finally {
        if (this.loading) {
          this.loading = false
        }
      }
    },

    clearMessages() {
      this.error = null
      this.messages = []
      this.currentConversationId = null
      this._rememberConversation(null)
    }
  }
})
