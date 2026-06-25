import client, { resolveApiBaseUrl, type ApiError } from './client'

// chat API 同时支持普通 JSON 响应和 SSE 流式响应；页面默认走 streamQuery。
export interface Citation {
  document_id: string
  chunk_id: string
  content: string
  score: number
  source_type?: string | null
  page_number?: number | null
  block_index?: number | null
  metadata?: Record<string, any>
}

export interface DebugChunk {
  chunk_id: string
  parent_id: string
  child_text?: string | null
  parent_text?: string | null
  distance?: number | null
  rerank_score?: number | null
  source_type?: string | null
  page_number?: number | null
  block_index?: number | null
}

export interface StageTimings {
  history_rewrite_ms?: number | null
  rewrite_ms?: number | null
  retrieve_ms?: number | null
  rerank_ms?: number | null
  context_build_ms?: number | null
  generate_ms?: number | null
  llm_first_token_ms?: number | null
  llm_after_first_token_ms?: number | null
  service_overhead_ms?: number | null
  [key: string]: number | null | undefined
}

export interface ChatDebug {
  intent_name?: string | null
  intent_source?: string | null
  intent_classifier_source?: string | null
  intent_classifier_mode?: string | null
  intent_classifier_score?: number | null
  intent_classifier_margin?: number | null
  rewritten_query: string
  retrieved_count: number
  reranked_count: number
  latency_ms: number
  rewrite_ms?: number
  retrieve_ms?: number
  rerank_ms?: number
  context_build_ms?: number
  generate_ms?: number
  llm_model?: string
  llm_prompt_tokens?: number
  llm_completion_tokens?: number
  llm_total_tokens?: number
  llm_first_token_ms?: number | null
  stage_timings_ms?: StageTimings | null
  retrieved_chunks?: DebugChunk[]
  mmr_selected_chunks?: DebugChunk[]
  reranked_chunks?: DebugChunk[]
  final_context_chunks?: DebugChunk[]
  [key: string]: any
}

export interface ChatQueryRequest {
  knowledge_base_id?: string | null
  query: string
  conversation_id?: string | null
  top_k_retrieve?: number
  top_k_rerank?: number
  enable_auto_route?: boolean
  debug?: boolean
  debug_chunks?: boolean
}

export interface ChatQueryResponse {
  answer: string
  answer_source?: string
  conversation_id: string
  knowledge_base_id: string
  knowledge_base_name?: string | null
  auto_routed?: boolean
  citations: Citation[]
  debug?: ChatDebug
}

export interface ChatStreamStartEvent {
  conversation_id: string
  knowledge_base_id: string
  knowledge_base_name?: string | null
  auto_routed?: boolean
}

interface StreamHandlers {
  onStart?: (payload: ChatStreamStartEvent) => void
  onToken?: (delta: string) => void
  onEnd?: (payload: ChatQueryResponse) => void
}

const parseSseBlock = (block: string) => {
  // 后端按 event/data 发送 SSE；data 可能跨多行，所以先合并再解析 JSON。
  const lines = block.split(/\r?\n/)
  let event = 'message'
  const dataLines: string[] = []

  for (const line of lines) {
    if (line.startsWith('event:')) {
      event = line.slice(6).trim()
      continue
    }
    if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trim())
    }
  }

  const rawData = dataLines.join('\n')
  return {
    event,
    data: rawData ? JSON.parse(rawData) : null,
  }
}

const extractSseBoundary = (buffer: string) => {
  // 兼容 Linux 和 Windows 换行，避免 Docker/本机环境换行差异导致流式解析卡住。
  const unixBoundary = buffer.indexOf('\n\n')
  const windowsBoundary = buffer.indexOf('\r\n\r\n')

  if (unixBoundary === -1) {
    return windowsBoundary === -1 ? null : { index: windowsBoundary, length: 4 }
  }

  if (windowsBoundary === -1 || unixBoundary < windowsBoundary) {
    return { index: unixBoundary, length: 2 }
  }

  return { index: windowsBoundary, length: 4 }
}

const processSseBlock = (block: string, handlers: StreamHandlers) => {
  // start/token/end 三类事件分别更新会话、追加 token、写入最终结果。
  if (!block) return

  const parsed = parseSseBlock(block)
  if (parsed.event === 'start' && parsed.data) {
    handlers.onStart?.(parsed.data as ChatStreamStartEvent)
  } else if (parsed.event === 'token' && parsed.data) {
    handlers.onToken?.(String((parsed.data as Record<string, unknown>).delta || ''))
  } else if (parsed.event === 'end' && parsed.data) {
    handlers.onEnd?.(parsed.data as ChatQueryResponse)
  } else if (parsed.event === 'error' && parsed.data) {
    throw parsed.data
  }
}

export const chatApi = {
  query: (data: ChatQueryRequest): Promise<ChatQueryResponse> => client.post('/chat/query', data),
  async streamQuery(
    data: ChatQueryRequest,
    handlers: StreamHandlers,
    signal?: AbortSignal,
  ): Promise<void> {
    const response = await fetch(`${resolveApiBaseUrl()}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
      signal,
    })

    if (!response.ok) {
      let apiError: ApiError | null = null
      try {
        apiError = (await response.json()) as ApiError
      } catch {
        apiError = null
      }
      throw (
        apiError || {
          code: 'UNKNOWN_ERROR',
          message: `流式请求失败 (${response.status})`,
        }
      )
    }

    if (!response.body) {
      throw {
        code: 'UNKNOWN_ERROR',
        message: '流式响应不可用',
      } satisfies ApiError
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      // fetch stream 可能把一个 SSE block 拆成多段，所以要先放入 buffer 再按边界切块。
      const { value, done } = await reader.read()
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done })

      let boundary = extractSseBoundary(buffer)
      while (boundary) {
        const block = buffer.slice(0, boundary.index).trim()
        buffer = buffer.slice(boundary.index + boundary.length)
        processSseBlock(block, handlers)
        boundary = extractSseBoundary(buffer)
      }

      if (done) {
        processSseBlock(buffer.trim(), handlers)
        break
      }
    }
  },
}
