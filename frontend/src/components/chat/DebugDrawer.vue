<script setup lang="ts">
// 调试抽屉只在开发态入口打开，用于对照后端 debug payload 的各阶段数据。
import { computed, ref, watch } from 'vue'
import { type ChatDebug, type DebugChunk } from '../../api/chat'

const props = defineProps<{
  open: boolean
  debugData: ChatDebug | null
}>()

defineEmits(['close'])

const formatMs = (ms: number | null | undefined) => {
  if (ms === undefined) return '-'
  return `${ms}ms`
}

const formatScore = (value: number | null | undefined) => {
  if (value === undefined || value === null) return '-'
  return value.toFixed(4)
}

const formatIntentMode = (value: string | null | undefined) => {
  const mapping: Record<string, string> = {
    rule: '规则',
    heuristic: '启发式',
    trained: '训练分类器',
    prototype: 'Prototype 回退',
    llm: '云端兜底',
    default: '默认放行',
    local_classifier: '本地分类器'
  }
  return mapping[value || ''] || (value ?? '-')
}

const formatSourceType = (value: string | null | undefined) => {
  const mapping: Record<string, string> = {
    text: '正文',
    table: '表格',
    image_ocr: '图片 OCR',
    image_ocr_table: '表格截图 OCR'
  }
  return mapping[value || ''] || (value ?? '-')
}

const formatBlockIndex = (value: number | null | undefined) => {
  if (value === undefined || value === null || value < 0) return '-'
  return String(value + 1)
}

const getChunkText = (chunk: DebugChunk) => chunk.child_text || chunk.parent_text || '-'

const getTimingValue = (debug: ChatDebug | null | undefined, key: string) => {
  // 新版后端把阶段耗时放到 stage_timings_ms，旧字段仍保留为兼容回退。
  const stageTimings = debug?.stage_timings_ms
  const stageValue = stageTimings?.[key]
  if (stageValue !== undefined) return stageValue
  return debug?.[key]
}

const stageTimingMetrics = computed(() => {
  // 展示顺序按一次问答的真实执行链路排列，方便定位慢在哪一段。
  const debug = props.debugData
  return [
    { label: '历史补全', value: getTimingValue(debug, 'history_rewrite_ms') },
    { label: '查询重写', value: getTimingValue(debug, 'rewrite_ms') },
    { label: '向量检索', value: getTimingValue(debug, 'retrieve_ms') },
    { label: 'Rerank', value: getTimingValue(debug, 'rerank_ms') },
    { label: '上下文构造', value: getTimingValue(debug, 'context_build_ms') },
    { label: 'LLM 首 Token', value: getTimingValue(debug, 'llm_first_token_ms') },
    { label: 'LLM 后续生成', value: getTimingValue(debug, 'llm_after_first_token_ms') },
    { label: 'LLM 总生成', value: getTimingValue(debug, 'generate_ms') },
    { label: '服务端额外开销', value: getTimingValue(debug, 'service_overhead_ms') },
    { label: '总端到端耗时', value: debug?.latency_ms, highlight: true }
  ]
})

const chunkStages = computed(() => {
  // 四个阶段对应后端 retrieved/MMR/rerank/final context，便于排查候选在哪一步丢失。
  const debug = props.debugData
  return [
    {
      key: 'retrieved_chunks',
      title: '初始检索',
      description: '向量召回后的原始候选',
      chunks: debug?.retrieved_chunks ?? []
    },
    {
      key: 'mmr_selected_chunks',
      title: 'MMR 选择',
      description: '去重后的候选集',
      chunks: debug?.mmr_selected_chunks ?? []
    },
    {
      key: 'reranked_chunks',
      title: 'Rerank 排序',
      description: '重排模型输出顺序',
      chunks: debug?.reranked_chunks ?? []
    },
    {
      key: 'final_context_chunks',
      title: '最终上下文',
      description: '实际送入 LLM 的上下文',
      chunks: debug?.final_context_chunks ?? []
    }
  ]
})

const expandedStages = ref<Record<string, boolean>>({})

const syncExpandedStages = () => {
  // 默认展开初始检索和最终上下文，中间阶段按需展开，减少抽屉首屏噪音。
  expandedStages.value = Object.fromEntries(
    chunkStages.value.map((stage, index) => [stage.key, index === 0 || stage.key === 'final_context_chunks'])
  )
}

watch(chunkStages, syncExpandedStages, { immediate: true })

const toggleStage = (key: string) => {
  expandedStages.value[key] = !expandedStages.value[key]
}
</script>

<template>
  <div class="debug-drawer" :class="{ open }">
    <div class="drawer-header">
      <h3>RAG 调试详情</h3>
      <button class="close-btn" @click="$emit('close')">&times;</button>
    </div>
    
    <div class="drawer-content" v-if="debugData">
      <div class="debug-section">
        <h4>意图路由</h4>
        <div class="metrics-grid">
          <div class="metric">
            <span class="m-label">意图标签 · intent_name</span>
            <span class="m-value">{{ debugData.intent_name || '-' }}</span>
          </div>
          <div class="metric">
            <span class="m-label">意图来源 · intent_source</span>
            <span class="m-value">{{ debugData.intent_source || '-' }}</span>
          </div>
          <div class="metric">
            <span class="m-label">分类来源 · intent_classifier_source</span>
            <span class="m-value">{{ debugData.intent_classifier_source || '-' }}</span>
          </div>
          <div class="metric">
            <span class="m-label">分类模式 · intent_classifier_mode</span>
            <span class="m-value">{{ formatIntentMode(debugData.intent_classifier_mode) }}</span>
          </div>
          <div class="metric">
            <span class="m-label">分类得分 · intent_classifier_score</span>
            <span class="m-value">{{ formatScore(debugData.intent_classifier_score) }}</span>
          </div>
          <div class="metric">
            <span class="m-label">分类间隔 · intent_classifier_margin</span>
            <span class="m-value">{{ formatScore(debugData.intent_classifier_margin) }}</span>
          </div>
        </div>
      </div>

      <div class="debug-section">
        <h4>查询重写</h4>
        <div class="debug-card">
          <p class="label">重写后的 Query</p>
          <p class="value">{{ debugData.rewritten_query || '未重写' }}</p>
        </div>
      </div>

      <div class="debug-section">
        <h4>检索性能</h4>
        <div class="metrics-grid">
          <div
            v-for="metric in stageTimingMetrics"
            :key="metric.label"
            class="metric"
            :class="{ highlight: metric.highlight }"
          >
            <span class="m-label">{{ metric.label }}</span>
            <span class="m-value">{{ formatMs(metric.value) }}</span>
          </div>
        </div>
      </div>

      <div class="debug-section">
        <h4>数据统计</h4>
        <div class="metrics-grid">
          <div class="metric">
            <span class="m-label">检索 Chunk 数</span>
            <span class="m-value">{{ debugData.retrieved_count }}</span>
          </div>
          <div class="metric">
            <span class="m-label">重排后 Chunk 数</span>
            <span class="m-value">{{ debugData.reranked_count }}</span>
          </div>
        </div>
      </div>

      <div class="debug-section">
        <h4>模型信息</h4>
        <div class="debug-card">
          <p class="label">LLM 模型</p>
          <p class="value code">{{ debugData.llm_model || '-' }}</p>
          <div class="token-usage">
            <span>Prompt: {{ debugData.llm_prompt_tokens ?? '-' }}</span>
            <span>Completion: {{ debugData.llm_completion_tokens ?? '-' }}</span>
            <span class="total">Total: {{ debugData.llm_total_tokens ?? '-' }}</span>
          </div>
        </div>
      </div>

      <div class="debug-section">
        <h4>Chunk 分阶段明细</h4>
        <div class="chunk-stages">
          <section v-for="stage in chunkStages" :key="stage.key" class="chunk-stage">
            <button class="stage-header" type="button" @click="toggleStage(stage.key)">
              <div>
                <p class="stage-title">{{ stage.title }}</p>
                <p class="stage-description">{{ stage.description }}</p>
              </div>
              <div class="stage-meta">
                <span class="stage-count">{{ stage.chunks.length }} 条</span>
                <span class="stage-toggle">{{ expandedStages[stage.key] ? '收起' : '展开' }}</span>
              </div>
            </button>

            <div v-if="expandedStages[stage.key]" class="stage-body">
              <div v-if="stage.chunks.length === 0" class="chunk-empty">当前阶段无详细 chunk 数据</div>
              <article v-for="chunk in stage.chunks" :key="`${stage.key}-${chunk.chunk_id}-${chunk.parent_id}`" class="chunk-card">
                <div class="chunk-meta-grid">
                  <div class="meta-item">
                    <span class="meta-label">chunk_id</span>
                    <code class="meta-value">{{ chunk.chunk_id }}</code>
                  </div>
                  <div class="meta-item">
                    <span class="meta-label">parent_id</span>
                    <code class="meta-value">{{ chunk.parent_id }}</code>
                  </div>
                  <div class="meta-item">
                    <span class="meta-label">distance</span>
                    <span class="meta-value">{{ formatScore(chunk.distance) }}</span>
                  </div>
                  <div class="meta-item">
                    <span class="meta-label">rerank_score</span>
                    <span class="meta-value">{{ formatScore(chunk.rerank_score) }}</span>
                  </div>
                  <div class="meta-item">
                    <span class="meta-label">source_type</span>
                    <span class="meta-value">{{ formatSourceType(chunk.source_type) }}</span>
                  </div>
                  <div class="meta-item">
                    <span class="meta-label">page</span>
                    <span class="meta-value">{{ chunk.page_number ?? '-' }}</span>
                  </div>
                  <div class="meta-item">
                    <span class="meta-label">block</span>
                    <span class="meta-value">{{ formatBlockIndex(chunk.block_index) }}</span>
                  </div>
                </div>
                <div class="chunk-text-block">
                  <span class="meta-label">text</span>
                  <pre class="chunk-text">{{ getChunkText(chunk) }}</pre>
                </div>
              </article>
            </div>
          </section>
        </div>
      </div>
    </div>

    <div class="drawer-content empty" v-else>
      <p>无调试数据</p>
    </div>
  </div>
</template>

<style scoped>
.debug-drawer {
  position: fixed;
  top: var(--header-height);
  right: -560px;
  width: 560px;
  height: calc(100vh - var(--header-height));
  background-color: #f8fafc;
  border-left: 1px solid var(--border-color);
  box-shadow: -4px 0 12px rgba(0, 0, 0, 0.1);
  transition: right 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  z-index: 110;
  display: flex;
  flex-direction: column;
}

.debug-drawer.open { right: 0; }

.drawer-header {
  padding: 20px 24px;
  background-color: white;
  border-bottom: 1px solid var(--border-color);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.drawer-header h3 { font-size: 1rem; font-weight: 600; color: var(--text-color); }

.close-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: var(--text-muted);
}

.drawer-content {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.debug-section h4 {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  margin-bottom: 12px;
  letter-spacing: 0.05em;
}

.debug-card {
  background-color: white;
  border: 1px solid var(--border-color);
  border-radius: var(--radius);
  padding: 16px;
}

.debug-card .label { font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px; }
.debug-card .value { font-size: 0.9rem; color: var(--text-color); line-height: 1.5; }
.debug-card .value.code { font-family: monospace; background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }

.metrics-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.metric {
  background-color: white;
  border: 1px solid var(--border-color);
  border-radius: var(--radius);
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.metric.highlight { border-color: var(--primary-color); background-color: #eff6ff; }

.m-label { font-size: 0.7rem; color: var(--text-muted); }
.m-value { font-size: 1rem; font-weight: 600; color: var(--text-color); }
.highlight .m-value { color: var(--primary-color); }

.token-usage {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px dashed var(--border-color);
  display: flex;
  gap: 12px;
  font-size: 0.75rem;
  color: var(--text-muted);
  flex-wrap: wrap;
}

.token-usage .total { font-weight: 600; color: var(--text-color); }

.chunk-stages {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.chunk-stage {
  background-color: white;
  border: 1px solid var(--border-color);
  border-radius: var(--radius);
  overflow: hidden;
}

.stage-header {
  width: 100%;
  padding: 14px 16px;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  background-color: white;
  text-align: left;
}

.stage-header:hover {
  background-color: #f8fafc;
}

.stage-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-color);
}

.stage-description {
  font-size: 0.75rem;
  color: var(--text-muted);
  margin-top: 4px;
}

.stage-meta {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
  white-space: nowrap;
}

.stage-count {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--primary-color);
}

.stage-toggle {
  font-size: 0.75rem;
  color: var(--text-muted);
}

.stage-body {
  border-top: 1px solid var(--border-color);
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.chunk-empty {
  border: 1px dashed var(--border-color);
  border-radius: var(--radius);
  padding: 14px;
  font-size: 0.8rem;
  color: var(--text-muted);
  background-color: #f8fafc;
}

.chunk-card {
  border: 1px solid var(--border-color);
  border-radius: var(--radius);
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  background-color: #fcfdff;
}

.chunk-meta-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px 12px;
}

.meta-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.meta-label {
  font-size: 0.7rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.meta-value {
  font-size: 0.82rem;
  color: var(--text-color);
  line-height: 1.4;
  word-break: break-all;
}

code.meta-value {
  background-color: #f1f5f9;
  padding: 2px 6px;
  border-radius: 4px;
}

.chunk-text-block {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.chunk-text {
  margin: 0;
  padding: 12px;
  border-radius: 8px;
  background-color: #f8fafc;
  border: 1px solid var(--border-color);
  color: var(--text-color);
  font-family: inherit;
  font-size: 0.82rem;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}

.empty { align-items: center; justify-content: center; color: var(--text-muted); }
</style>
