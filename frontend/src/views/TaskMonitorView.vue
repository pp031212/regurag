<script setup lang="ts">
// 任务观测页聚合后台 worker 状态、任务列表、告警和事件时间线。
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  taskApi,
  type KnowledgeBaseTaskTrend,
  type Task,
  type TaskAlert,
  type TaskEvent,
  type TaskOverviewResponse,
  type TaskStatus,
} from '../api/tasks'
import { useKBStore } from '../stores/kb'
import KnowledgeBaseTrendTable from '../components/tasks/KnowledgeBaseTrendTable.vue'
import TaskAlertStack from '../components/tasks/TaskAlertStack.vue'
import TaskEventTimeline from '../components/tasks/TaskEventTimeline.vue'
import TaskStatusOverview from '../components/tasks/TaskStatusOverview.vue'

const route = useRoute()
const router = useRouter()
const kbStore = useKBStore()

const selectedKnowledgeBaseId = ref(String(route.query.knowledge_base_id || ''))
const selectedStatus = ref<'all' | TaskStatus>('all')
const selectedTaskId = ref('')

const tasks = ref<Task[]>([])
const overview = ref<TaskOverviewResponse | null>(null)
const alerts = ref<TaskAlert[]>([])
const trends = ref<KnowledgeBaseTaskTrend[]>([])
const events = ref<TaskEvent[]>([])

const loading = ref(false)
const refreshing = ref(false)
const eventLoading = ref(false)
const error = ref('')
const dashboardLoaded = ref(false)

let refreshTimer: number | null = null

const currentKnowledgeBaseFilter = computed(() => selectedKnowledgeBaseId.value || undefined)
const selectedTask = computed(() => tasks.value.find((task) => task.id === selectedTaskId.value) || null)
const isBusy = computed(() => loading.value || refreshing.value)

const filteredKnowledgeBaseName = computed(() => {
  if (!selectedKnowledgeBaseId.value) return '全部知识库'
  return kbStore.knowledgeBases.find((item) => item.id === selectedKnowledgeBaseId.value)?.name || selectedKnowledgeBaseId.value
})

const alertSummary = computed(() => {
  const critical = alerts.value.filter((item) => item.severity === 'critical').length
  const warning = alerts.value.filter((item) => item.severity === 'warning').length
  return { critical, warning, total: alerts.value.length }
})

const selectedTaskKnowledgeBaseName = computed(() => {
  const knowledgeBaseId = selectedTask.value?.knowledge_base_id
  if (!knowledgeBaseId) return ''
  return kbStore.knowledgeBases.find((item) => item.id === knowledgeBaseId)?.name || knowledgeBaseId
})

const statusLabel = (status: TaskStatus) => {
  const map: Record<TaskStatus, string> = {
    pending: '待处理',
    running: '运行中',
    completed: '已完成',
    failed: '已失败',
  }
  return map[status]
}

const formatTime = (value: string | null | undefined) => {
  if (!value) return '—'
  return new Date(value).toLocaleString('zh-CN')
}

const syncRouteQuery = () => {
  // 知识库筛选写入 URL，方便从知识库详情页跳转后刷新仍保留范围。
  const query = {
    ...route.query,
    knowledge_base_id: selectedKnowledgeBaseId.value || undefined,
  }
  router.replace({ query })
}

const fetchEvents = async (taskId: string, options: { showLoading?: boolean } = {}) => {
  // 事件时间线比任务列表更细，用于定位任务卡在预处理、入库还是 worker 重试。
  const showLoading = options.showLoading ?? events.value.length === 0
  if (showLoading) eventLoading.value = true
  try {
    const response = await taskApi.events(taskId, { limit: 100 })
    events.value = response.items
  } catch (err: any) {
    error.value = err.message || '获取任务事件失败'
  } finally {
    if (showLoading) eventLoading.value = false
  }
}

const fetchDashboard = async (options: { showLoading?: boolean } = {}) => {
  if (loading.value || refreshing.value) return

  // overview/alerts/trends/list 可以并行拉取，减少观测页自动刷新时的等待。
  const showLoading = options.showLoading ?? !dashboardLoaded.value
  if (showLoading) {
    loading.value = true
  } else {
    refreshing.value = true
  }
  error.value = ''
  try {
    const [overviewResponse, alertsResponse, trendsResponse, listResponse] = await Promise.all([
      taskApi.overview({ knowledge_base_id: currentKnowledgeBaseFilter.value }),
      taskApi.alerts({ knowledge_base_id: currentKnowledgeBaseFilter.value }),
      taskApi.trends({ limit: 12 }),
      taskApi.list({
        knowledge_base_id: currentKnowledgeBaseFilter.value,
        status: selectedStatus.value === 'all' ? undefined : selectedStatus.value,
        limit: 50,
      }),
    ])

    overview.value = overviewResponse
    alerts.value = alertsResponse.items
    trends.value = trendsResponse.items
    tasks.value = listResponse.items

    if (selectedTaskId.value && tasks.value.some((task) => task.id === selectedTaskId.value)) {
      // 刷新后仍能找到当前任务，就保持详情面板不跳动，只更新事件。
      await fetchEvents(selectedTaskId.value, { showLoading: false })
      dashboardLoaded.value = true
      return
    }

    const nextTask = tasks.value[0] || null
    const taskChanged = selectedTaskId.value !== (nextTask?.id || '')
    selectedTaskId.value = nextTask?.id || ''
    if (nextTask) {
      await fetchEvents(nextTask.id, { showLoading: showLoading || taskChanged })
    } else {
      events.value = []
    }
    dashboardLoaded.value = true
  } catch (err: any) {
    error.value = err.message || '获取任务观测数据失败'
  } finally {
    if (showLoading) {
      loading.value = false
    } else {
      refreshing.value = false
    }
  }
}

const startAutoRefresh = () => {
  // 观测页打开期间自动刷新；离开页面时在 onBeforeUnmount 中清理定时器。
  if (refreshTimer !== null) return
  refreshTimer = window.setInterval(() => {
    void fetchDashboard({ showLoading: false })
  }, 10000)
}

const stopAutoRefresh = () => {
  if (refreshTimer === null) return
  window.clearInterval(refreshTimer)
  refreshTimer = null
}

const handleTaskSelect = async (taskId: string) => {
  if (selectedTaskId.value === taskId) return
  selectedTaskId.value = taskId
  await fetchEvents(taskId)
}

const openFilteredKnowledgeBase = () => {
  if (!selectedKnowledgeBaseId.value) return
  router.push({ name: 'KnowledgeBaseDetail', params: { id: selectedKnowledgeBaseId.value } })
}

watch(selectedKnowledgeBaseId, async () => {
  syncRouteQuery()
  await fetchDashboard({ showLoading: true })
})

watch(selectedStatus, async () => {
  await fetchDashboard({ showLoading: true })
})

onMounted(async () => {
  if (kbStore.knowledgeBases.length === 0) {
    await kbStore.fetchKBs()
  }

  await fetchDashboard({ showLoading: true })
  startAutoRefresh()
})

onBeforeUnmount(() => {
  stopAutoRefresh()
})
</script>

<template>
  <div class="task-monitor-page">
    <section class="page-head">
      <div>
        <span class="page-head__eyebrow">Task Monitoring</span>
        <h2>任务观测</h2>
        <p>这里看的是后台 worker 的运行状态，不是单纯的任务列表。你可以先判断有没有积压、告警和长时任务，再下钻到单任务时间线。</p>
      </div>
      <div class="page-head__actions">
        <button v-if="selectedKnowledgeBaseId" class="head-btn" @click="openFilteredKnowledgeBase">
          回到当前知识库
        </button>
        <button class="head-btn head-btn--primary" :disabled="isBusy" @click="fetchDashboard({ showLoading: false })">
          {{ isBusy ? '刷新中...' : '立即刷新' }}
        </button>
      </div>
    </section>

    <section class="toolbar">
      <div class="toolbar-summary">
        <span class="toolbar-summary__label">当前范围</span>
        <strong>{{ filteredKnowledgeBaseName }}</strong>
      </div>

      <label class="toolbar-field">
        <span>知识库范围</span>
        <select v-model="selectedKnowledgeBaseId">
          <option value="">全部知识库</option>
          <option v-for="kb in kbStore.knowledgeBases" :key="kb.id" :value="kb.id">{{ kb.name }}</option>
        </select>
      </label>

      <label class="toolbar-field">
        <span>任务状态</span>
        <select v-model="selectedStatus">
          <option value="all">全部状态</option>
          <option value="pending">待处理</option>
          <option value="running">运行中</option>
          <option value="completed">已完成</option>
          <option value="failed">已失败</option>
        </select>
      </label>
    </section>

    <section class="quick-state" :class="{ 'quick-state--critical': alertSummary.critical > 0 }">
      <span>自动刷新 <strong>10s</strong></span>
      <span>当前告警 <strong>{{ alertSummary.total }}</strong></span>
      <span>严重 <strong>{{ alertSummary.critical }}</strong></span>
      <span>提醒 <strong>{{ alertSummary.warning }}</strong></span>
    </section>

    <p v-if="error" class="page-error">{{ error }}</p>

    <TaskStatusOverview :overview="overview" :loading="loading" />

    <TaskAlertStack :alerts="alerts" :loading="loading" />

    <KnowledgeBaseTrendTable :items="trends" :loading="loading" />

    <div class="monitor-grid">
      <section class="task-list-panel">
        <div class="task-list-panel__head">
          <div>
            <span class="task-list-panel__eyebrow">Task Queue</span>
            <h3>任务列表</h3>
          </div>
          <span class="task-list-panel__count">{{ tasks.length }} 条</span>
        </div>

        <div v-if="loading" class="task-list-skeleton">
          <div v-for="index in 6" :key="index" class="task-list-skeleton__row"></div>
        </div>

        <div v-else-if="tasks.length === 0" class="task-list-empty">
          当前筛选条件下没有任务。
        </div>

        <div v-else class="task-list">
          <button
            v-for="task in tasks"
            :key="task.id"
            class="task-row"
            :class="{ 'task-row--active': task.id === selectedTaskId }"
            @click="handleTaskSelect(task.id)"
          >
            <div class="task-row__head">
              <span class="task-row__title">{{ task.task_type || 'ingest' }} · {{ statusLabel(task.status) }}</span>
              <span class="task-row__time">{{ formatTime(task.updated_at) }}</span>
            </div>
            <div class="task-row__meta">
              <span>{{ kbStore.knowledgeBases.find((item) => item.id === task.knowledge_base_id)?.name || task.knowledge_base_id }}</span>
              <span>尝试 {{ task.attempt_count || 0 }}</span>
            </div>
            <p class="task-row__message">{{ task.message || '暂无任务信息' }}</p>
          </button>
        </div>
      </section>

      <TaskEventTimeline
        :task="selectedTask"
        :events="events"
        :loading="eventLoading"
        :knowledge-base-name="selectedTaskKnowledgeBaseName"
      />
    </div>
  </div>
</template>

<style scoped>
.task-monitor-page {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.page-head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 20px;
  padding: 20px 24px;
  border-radius: var(--radius);
  background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 100%);
  border: 1px solid #dbeafe;
}

.page-head__eyebrow {
  display: block;
  margin-bottom: 8px;
  color: var(--primary-color);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.page-head h2 {
  margin: 0 0 8px;
  font-size: 1.45rem;
  color: #0f172a;
}

.page-head p {
  margin: 0;
  color: #475569;
  line-height: 1.7;
  max-width: 760px;
}

.page-head__actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.head-btn {
  min-height: 42px;
  padding: 0 16px;
  border-radius: var(--radius);
  border: 1px solid var(--border-color);
  background: white;
  color: var(--primary-color);
  font-weight: 600;
}

.head-btn--primary {
  border-color: transparent;
  background: var(--primary-color);
  color: white;
}

.toolbar {
  display: flex;
  align-items: flex-end;
  gap: 14px;
  flex-wrap: wrap;
  padding: 16px 18px;
  border-radius: var(--radius-md);
  background: white;
  border: 1px solid var(--border-color);
}

.toolbar-field {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.toolbar-field span {
  color: #64748b;
  font-size: 0.78rem;
}

.toolbar-field select {
  min-width: 220px;
  min-height: 42px;
  padding: 0 14px;
  border-radius: var(--radius-md);
  border: 1px solid var(--border-color);
  background: #ffffff;
  color: #0f172a;
}

.toolbar-summary {
  min-width: 240px;
  padding: 10px 14px;
  border-radius: var(--radius-md);
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  border-left: 4px solid var(--primary-color);
  display: flex;
  flex-direction: column;
  gap: 6px;
  align-self: stretch;
  justify-content: center;
}

.toolbar-summary__label {
  color: #64748b;
  font-size: 0.76rem;
}

.toolbar-summary strong {
  color: #0f172a;
  font-size: 1rem;
  line-height: 1.35;
}

.page-error {
  margin: 0;
  padding: 14px 16px;
  border-radius: var(--radius-md);
  background: #fff1f2;
  border: 1px solid #fecdd3;
  color: #be123c;
}

.quick-state {
  min-height: 42px;
  padding: 0 14px;
  display: flex;
  align-items: center;
  gap: 18px;
  flex-wrap: wrap;
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  border-left: 4px solid var(--primary-color);
  background: #f8fafc;
}

.quick-state--critical {
  border-left-color: #dc2626;
  background: #fff7ed;
}

.quick-state span {
  color: #64748b;
  font-size: 0.82rem;
}

.quick-state strong {
  color: #0f172a;
  font-size: 0.9rem;
}

.monitor-grid {
  display: grid;
  grid-template-columns: minmax(0, 0.95fr) minmax(420px, 1.05fr);
  gap: 18px;
  align-items: stretch;
}

.task-list-panel {
  padding: 0;
  border-radius: var(--radius-md);
  background: white;
  border: 1px solid var(--border-color);
  overflow: hidden;
}

.task-list-panel__head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 12px;
  padding: 16px 18px;
  margin-bottom: 0;
  border-bottom: 1px solid var(--border-color);
}

.task-list-panel__eyebrow {
  display: block;
  margin-bottom: 8px;
  color: var(--primary-color);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.task-list-panel h3 {
  margin: 0;
  color: #0f172a;
  font-size: 1.2rem;
}

.task-list-panel__count {
  color: #64748b;
  font-size: 0.84rem;
}

.task-list {
  display: flex;
  flex-direction: column;
  gap: 0;
}

.task-row {
  width: 100%;
  padding: 14px 18px 14px 16px;
  border: 0;
  border-left: 3px solid transparent;
  border-bottom: 1px solid var(--border-color);
  border-radius: 0;
  background: #ffffff;
  display: flex;
  flex-direction: column;
  gap: 8px;
  text-align: left;
}

.task-row:last-child {
  border-bottom: 0;
}

.task-row:hover {
  background: #f8fafc;
}

.task-row--active {
  border-left-color: var(--primary-color);
  background: #eff6ff;
}

.task-row__head,
.task-row__meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.task-row__title {
  color: #0f172a;
  font-weight: 700;
}

.task-row__time,
.task-row__meta {
  color: #64748b;
  font-size: 0.78rem;
}

.task-row__message {
  margin: 0;
  color: #475569;
  line-height: 1.7;
}

.task-list-empty {
  min-height: 240px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: #64748b;
  border-radius: 0;
  border: 0;
}

.task-list-skeleton {
  display: flex;
  flex-direction: column;
  gap: 0;
}

.task-list-skeleton__row {
  min-height: 96px;
  border-radius: 0;
  border-bottom: 1px solid var(--border-color);
  background: linear-gradient(90deg, #e2e8f0 0%, #f8fafc 50%, #e2e8f0 100%);
  background-size: 200% 100%;
  animation: task-row-wave 1.2s ease-in-out infinite;
}

@keyframes task-row-wave {
  0% { background-position: 100% 50%; }
  100% { background-position: 0 50%; }
}

@media (max-width: 1080px) {
  .monitor-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 720px) {
  .page-head {
    padding: 20px;
    flex-direction: column;
    align-items: flex-start;
  }

  .toolbar {
    padding: 16px;
  }

  .toolbar-field,
  .toolbar-field select {
    width: 100%;
  }

  .toolbar-summary {
    width: 100%;
    margin-left: 0;
  }

  .page-head__actions {
    width: 100%;
  }

  .head-btn {
    width: 100%;
    justify-content: center;
  }

  .quick-state {
    align-items: flex-start;
    flex-direction: column;
    padding: 12px 14px;
    gap: 8px;
  }

  .task-list-panel {
    padding: 0;
  }

  .task-row__head,
  .task-row__meta {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
