<script setup lang="ts">
// 趋势表按知识库比较当前窗口和上一窗口，用来先定位“哪个知识库在变差”。
import type { KnowledgeBaseTaskTrend } from '../../api/tasks'

const props = defineProps<{
  items: KnowledgeBaseTaskTrend[]
  loading?: boolean
}>()

const deltaLabel = (value: number) => {
  if (value > 0) return `+${value}`
  return `${value}`
}

const deltaClass = (value: number, positiveIsBad = true) => {
  // 失败/重试上升是坏事，完成数上升是好事，因此用 positiveIsBad 区分颜色含义。
  if (value === 0) return 'trend-delta--flat'
  if (positiveIsBad) {
    return value > 0 ? 'trend-delta--up' : 'trend-delta--down'
  }
  return value > 0 ? 'trend-delta--down' : 'trend-delta--up'
}

const formatTime = (value: string | null | undefined) => {
  if (!value) return '—'
  return new Date(value).toLocaleString('zh-CN')
}
</script>

<template>
  <section class="trend-panel">
    <div class="trend-panel__head">
      <div>
        <span class="trend-panel__eyebrow">Knowledge Base Trends</span>
        <h3>知识库趋势视图</h3>
      </div>
      <p class="trend-panel__hint">对比“当前监控窗口”和“上一窗口”，先看哪一个知识库在变差。</p>
    </div>

    <div v-if="props.loading" class="trend-skeleton-list">
      <div v-for="index in 4" :key="index" class="trend-skeleton"></div>
    </div>

    <div v-else-if="props.items.length === 0" class="trend-empty">
      当前还没有足够的知识库任务趋势数据。
    </div>

    <div v-else class="trend-table">
      <article v-for="item in props.items" :key="item.knowledge_base_id" class="trend-row">
        <div class="trend-row__main">
          <div class="trend-row__header">
            <strong>{{ item.knowledge_base_name }}</strong>
            <span>{{ formatTime(item.updated_at) }}</span>
          </div>
          <div class="trend-row__meta">
            <span class="trend-chip">积压 {{ item.pending }}</span>
            <span class="trend-chip">运行 {{ item.running }}</span>
            <span class="trend-chip">本窗完成 {{ item.recent_completed }}</span>
          </div>
        </div>

        <div class="trend-row__stats">
          <div class="trend-stat">
            <span class="trend-stat__label">失败趋势</span>
            <strong>{{ item.recent_failed }} / {{ item.previous_failed }}</strong>
            <span class="trend-delta" :class="deltaClass(item.failed_delta)">{{ deltaLabel(item.failed_delta) }}</span>
          </div>
          <div class="trend-stat">
            <span class="trend-stat__label">重试趋势</span>
            <strong>{{ item.recent_retried }} / {{ item.previous_retried }}</strong>
            <span class="trend-delta" :class="deltaClass(item.retried_delta)">{{ deltaLabel(item.retried_delta) }}</span>
          </div>
          <div class="trend-stat">
            <span class="trend-stat__label">完成趋势</span>
            <strong>{{ item.recent_completed }} / {{ item.previous_completed }}</strong>
            <span class="trend-delta" :class="deltaClass(item.completed_delta, false)">{{ deltaLabel(item.completed_delta) }}</span>
          </div>
        </div>
      </article>
    </div>
  </section>
</template>

<style scoped>
.trend-panel {
  padding: 0;
  border-radius: var(--radius-md);
  background: white;
  border: 1px solid var(--border-color);
  overflow: hidden;
}

.trend-panel__head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 18px;
  margin-bottom: 0;
  border-bottom: 1px solid var(--border-color);
}

.trend-panel__eyebrow {
  display: block;
  margin-bottom: 8px;
  color: var(--primary-color);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.trend-panel h3 {
  margin: 0;
  color: #0f172a;
  font-size: 1.2rem;
}

.trend-panel__hint {
  margin: 0;
  color: #64748b;
  font-size: 0.82rem;
}

.trend-table,
.trend-skeleton-list {
  display: flex;
  flex-direction: column;
  gap: 0;
}

.trend-row,
.trend-skeleton,
.trend-empty {
  padding: 14px 18px;
  border-radius: 0;
  border: 0;
  border-bottom: 1px solid var(--border-color);
  background: #ffffff;
}

.trend-row:last-child,
.trend-skeleton:last-child {
  border-bottom: 0;
}

.trend-row {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
  gap: 16px;
}

.trend-row__header,
.trend-row__meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.trend-row__header strong {
  color: #0f172a;
  font-size: 0.95rem;
}

.trend-row__header span {
  color: #64748b;
  font-size: 0.78rem;
}

.trend-row__meta {
  margin-top: 10px;
  justify-content: flex-start;
}

.trend-chip {
  display: inline-flex;
  min-height: 28px;
  padding: 0 10px;
  border-radius: 4px;
  background: #f8fafc;
  color: #475569;
  font-size: 0.76rem;
  font-weight: 600;
}

.trend-row__stats {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.trend-stat {
  padding: 0 0 0 12px;
  border-radius: 0;
  background: transparent;
  border: 0;
  border-left: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.trend-stat__label {
  color: #64748b;
  font-size: 0.74rem;
}

.trend-stat strong {
  color: #0f172a;
  font-size: 0.86rem;
}

.trend-delta {
  font-size: 0.76rem;
  font-weight: 700;
}

.trend-delta--up {
  color: #b91c1c;
}

.trend-delta--down {
  color: #166534;
}

.trend-delta--flat {
  color: #64748b;
}

.trend-empty {
  color: #64748b;
  text-align: center;
}

.trend-skeleton {
  min-height: 128px;
  background: linear-gradient(90deg, #e2e8f0 0%, #f8fafc 50%, #e2e8f0 100%);
  background-size: 200% 100%;
  animation: trend-wave 1.2s ease-in-out infinite;
}

@keyframes trend-wave {
  0% { background-position: 100% 50%; }
  100% { background-position: 0 50%; }
}

@media (max-width: 1080px) {
  .trend-row {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 720px) {
  .trend-panel {
    padding: 0;
  }

  .trend-panel__head {
    flex-direction: column;
    align-items: flex-start;
  }

  .trend-row__stats {
    grid-template-columns: 1fr;
  }
}
</style>
