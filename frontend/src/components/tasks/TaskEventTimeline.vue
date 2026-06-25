<script setup lang="ts">
// 单任务时间线用于下钻排查：任务摘要看结果，事件列表看卡在哪个阶段。
import { computed } from 'vue'
import type { Task, TaskEvent } from '../../api/tasks'

const props = defineProps<{
  task: Task | null
  events: TaskEvent[]
  loading?: boolean
  knowledgeBaseName?: string
}>()

const statusLabel = computed(() => {
  const map: Record<string, string> = {
    pending: '待处理',
    running: '运行中',
    completed: '完成',
    failed: '失败',
  }
  return props.task ? map[props.task.status] || props.task.status : '未选择'
})

const eventLabel = (eventType: string) => {
  const map: Record<string, string> = {
    created: '创建',
    claimed: '领取',
    heartbeat: '心跳',
    started: '开始',
    retrying: '重试',
    completed: '完成',
    failed: '失败',
    notice: '提示',
    document_started: '文档开始',
    document_completed: '文档完成',
    document_failed: '文档失败',
    document_rebuild_started: '文档重建',
  }
  return map[eventType] || eventType
}

const formatTime = (value: string | null | undefined) => {
  if (!value) return '—'
  return new Date(value).toLocaleString('zh-CN')
}

const eventSummary = computed(() => {
  // 从事件 payload 中汇总 worker 和文档信息，减少用户逐条读事件的成本。
  const counts = new Map<string, number>()
  const documentIds = new Set<string>()
  let latestWorker = props.task?.locked_by || ''

  for (const event of props.events) {
    counts.set(event.event_type, (counts.get(event.event_type) || 0) + 1)
    const payload = event.payload || {}
    const workerId = typeof payload.worker_id === 'string' ? payload.worker_id : ''
    const documentId = typeof payload.document_id === 'string' ? payload.document_id : ''

    if (workerId) latestWorker = workerId
    if (documentId) documentIds.add(documentId)
  }

  return {
    latestWorker: latestWorker || '—',
    touchedDocuments: documentIds.size,
    eventCount: props.events.length,
    topTypes: Array.from(counts.entries())
      .sort((left, right) => right[1] - left[1])
      .slice(0, 4),
  }
})
</script>

<template>
  <section class="timeline-panel">
    <div class="timeline-panel__head">
      <div>
        <span class="timeline-panel__eyebrow">Task Timeline</span>
        <h3>任务详情</h3>
      </div>
    </div>

    <div v-if="!props.task" class="timeline-empty">
      从左侧任务列表选择一条记录，查看状态、错误和完整事件时间线。
    </div>

    <template v-else>
      <div class="task-summary">
        <div class="task-summary__row">
          <span class="task-summary__label">任务状态</span>
          <strong>{{ statusLabel }}</strong>
        </div>
        <div class="task-summary__row">
          <span class="task-summary__label">知识库</span>
          <strong>{{ props.knowledgeBaseName || props.task.knowledge_base_id }}</strong>
        </div>
        <div class="task-summary__row">
          <span class="task-summary__label">任务类型</span>
          <strong>{{ props.task.task_type || 'ingest' }}</strong>
        </div>
        <div class="task-summary__row">
          <span class="task-summary__label">最近信息</span>
          <strong>{{ props.task.message || '—' }}</strong>
        </div>
        <div class="task-summary__row">
          <span class="task-summary__label">尝试次数</span>
          <strong>{{ props.task.attempt_count || 0 }}</strong>
        </div>
        <div class="task-summary__row">
          <span class="task-summary__label">最近错误</span>
          <strong>{{ props.task.last_error || '—' }}</strong>
        </div>
        <div class="task-summary__row">
          <span class="task-summary__label">开始时间</span>
          <strong>{{ formatTime(props.task.started_at) }}</strong>
        </div>
        <div class="task-summary__row">
          <span class="task-summary__label">结束时间</span>
          <strong>{{ formatTime(props.task.finished_at) }}</strong>
        </div>
      </div>

      <div class="event-summary">
        <div class="event-summary__card">
          <span class="event-summary__label">事件总数</span>
          <strong>{{ eventSummary.eventCount }}</strong>
        </div>
        <div class="event-summary__card">
          <span class="event-summary__label">触达文档</span>
          <strong>{{ eventSummary.touchedDocuments }}</strong>
        </div>
        <div class="event-summary__card">
          <span class="event-summary__label">最近 Worker</span>
          <strong>{{ eventSummary.latestWorker }}</strong>
        </div>
        <div class="event-summary__card event-summary__card--types">
          <span class="event-summary__label">高频事件</span>
          <div class="event-summary__chips">
            <span v-for="[eventType, count] in eventSummary.topTypes" :key="eventType" class="event-summary__chip">
              {{ eventLabel(eventType) }} × {{ count }}
            </span>
            <span v-if="eventSummary.topTypes.length === 0" class="event-summary__chip">暂无</span>
          </div>
        </div>
      </div>

      <div v-if="props.loading" class="timeline-skeleton-list">
        <div v-for="index in 4" :key="index" class="timeline-skeleton"></div>
      </div>

      <div v-else-if="props.events.length === 0" class="timeline-empty timeline-empty--events">
        当前任务还没有可展示的事件。
      </div>

      <div v-else class="timeline-list">
        <article v-for="event in props.events" :key="event.id" class="timeline-item">
          <div class="timeline-item__dot"></div>
          <div class="timeline-item__content">
            <div class="timeline-item__head">
              <strong>{{ eventLabel(event.event_type) }}</strong>
              <span>{{ formatTime(event.created_at) }}</span>
            </div>
            <p>{{ event.message }}</p>
          </div>
        </article>
      </div>
    </template>
  </section>
</template>

<style scoped>
.timeline-panel {
  padding: 0;
  border-radius: var(--radius-md);
  background: white;
  border: 1px solid var(--border-color);
  min-height: 100%;
  overflow: hidden;
}

.timeline-panel__head {
  padding: 14px 18px;
  margin-bottom: 0;
  border-bottom: 1px solid var(--border-color);
}

.timeline-panel__eyebrow {
  display: block;
  margin-bottom: 8px;
  color: var(--primary-color);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.timeline-panel h3 {
  margin: 0;
  color: #0f172a;
  font-size: 1.2rem;
}

.task-summary {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0;
  margin-bottom: 0;
  border-bottom: 1px solid var(--border-color);
}

.task-summary__row {
  padding: 12px 18px;
  border-radius: 0;
  border: 0;
  border-right: 1px solid var(--border-color);
  border-bottom: 1px solid var(--border-color);
  background: #ffffff;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.task-summary__row:nth-child(2n) {
  border-right: 0;
}

.task-summary__row:nth-last-child(-n + 2) {
  border-bottom: 0;
}

.task-summary__label {
  color: #64748b;
  font-size: 0.76rem;
}

.task-summary__row strong {
  color: #0f172a;
  font-size: 0.88rem;
  line-height: 1.6;
  word-break: break-word;
}

.event-summary {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0;
  margin-bottom: 0;
  border-bottom: 1px solid var(--border-color);
}

.event-summary__card {
  padding: 12px 18px;
  border-radius: 0;
  border: 0;
  border-right: 1px solid #dbeafe;
  background: #eff6ff;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.event-summary__card--types {
  grid-column: 1 / -1;
  border-top: 1px solid #dbeafe;
  border-right: 0;
}

.event-summary__label {
  color: #64748b;
  font-size: 0.76rem;
}

.event-summary__card strong {
  color: #0f172a;
  font-size: 0.92rem;
  line-height: 1.5;
  word-break: break-word;
}

.event-summary__chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.event-summary__chip {
  display: inline-flex;
  min-height: 26px;
  padding: 0 10px;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.9);
  color: #334155;
  font-size: 0.74rem;
  font-weight: 600;
}

.timeline-list {
  display: flex;
  flex-direction: column;
  gap: 0;
  padding: 16px 18px 18px;
}

.timeline-item {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  position: relative;
  padding-bottom: 14px;
}

.timeline-item::before {
  content: '';
  position: absolute;
  left: 5px;
  top: 18px;
  bottom: -2px;
  width: 1px;
  background: #dbeafe;
}

.timeline-item:last-child {
  padding-bottom: 0;
}

.timeline-item:last-child::before {
  display: none;
}

.timeline-item__dot {
  width: 11px;
  height: 11px;
  margin-top: 7px;
  border-radius: 50%;
  flex: none;
  background: var(--primary-color);
  box-shadow: 0 0 0 4px #eff6ff;
  position: relative;
  z-index: 1;
}

.timeline-item__content {
  flex: 1;
  min-width: 0;
  padding: 0 0 14px;
  border-radius: 0;
  border: 0;
  border-bottom: 1px solid var(--border-color);
  background: #ffffff;
}

.timeline-item:last-child .timeline-item__content {
  border-bottom: 0;
  padding-bottom: 0;
}

.timeline-item__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
}

.timeline-item__head strong {
  color: #0f172a;
}

.timeline-item__head span {
  color: #64748b;
  font-size: 0.78rem;
}

.timeline-item__content p {
  margin: 0;
  color: #475569;
  line-height: 1.7;
}

.timeline-empty {
  min-height: 180px;
  padding: 18px;
  border-radius: 0;
  border: 0;
  color: #64748b;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  line-height: 1.8;
}

.timeline-empty--events {
  min-height: 120px;
}

.timeline-skeleton-list {
  display: flex;
  flex-direction: column;
  gap: 0;
  padding: 16px 18px;
}

.timeline-skeleton {
  min-height: 82px;
  border-radius: 0;
  border-bottom: 1px solid var(--border-color);
  background: linear-gradient(90deg, #e2e8f0 0%, #f8fafc 50%, #e2e8f0 100%);
  background-size: 200% 100%;
  animation: timeline-wave 1.2s ease-in-out infinite;
}

@keyframes timeline-wave {
  0% { background-position: 100% 50%; }
  100% { background-position: 0 50%; }
}

@media (max-width: 920px) {
  .task-summary {
    grid-template-columns: 1fr;
  }

  .task-summary__row {
    border-right: 0;
  }

  .task-summary__row:nth-last-child(-n + 2) {
    border-bottom: 1px solid var(--border-color);
  }

  .task-summary__row:last-child {
    border-bottom: 0;
  }

  .event-summary {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .event-summary__card--types {
    grid-column: 1 / -1;
  }
}

@media (max-width: 640px) {
  .timeline-panel {
    padding: 0;
  }

  .event-summary {
    grid-template-columns: 1fr;
  }

  .event-summary__card--types {
    grid-column: span 1;
  }

  .timeline-item__head {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
