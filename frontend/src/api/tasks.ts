import client from './client'

// 任务接口覆盖后台入库/重建的状态、告警、趋势和事件流，用于运维监控页。
export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed'

export interface Task {
  id: string
  knowledge_base_id: string
  task_type?: string
  document_ids: string[]
  status: TaskStatus
  message?: string
  attempt_count?: number
  last_error?: string | null
  started_at?: string | null
  finished_at?: string | null
  locked_at?: string | null
  locked_by?: string | null
  created_at: string
  updated_at: string
}

export interface TaskListResponse {
  items: Task[]
  total: number
}

export interface TaskStatsResponse {
  total: number
  pending: number
  running: number
  completed: number
  failed: number
  retrying: number
  stale_running: number
}

export interface KnowledgeBaseFailureSummary {
  knowledge_base_id: string
  task_count: number
}

export interface TaskOverviewResponse extends TaskStatsResponse {
  active_workers: number
  oldest_pending_age_seconds: number | null
  long_running: number
  recent_failed: number
  recent_retried: number
  knowledge_bases_with_recent_failures: KnowledgeBaseFailureSummary[]
}

export interface TaskAlert {
  code: string
  severity: 'critical' | 'warning'
  message: string
  count: number
  details: Record<string, unknown>
}

export interface TaskAlertListResponse {
  items: TaskAlert[]
  total: number
}

export interface KnowledgeBaseTaskTrend {
  knowledge_base_id: string
  knowledge_base_name: string
  pending: number
  running: number
  recent_failed: number
  previous_failed: number
  failed_delta: number
  recent_retried: number
  previous_retried: number
  retried_delta: number
  recent_completed: number
  previous_completed: number
  completed_delta: number
  updated_at?: string | null
}

export interface KnowledgeBaseTaskTrendListResponse {
  items: KnowledgeBaseTaskTrend[]
  total: number
}

export interface TaskEvent {
  id: string
  task_id: string
  event_type: string
  message: string
  payload?: Record<string, unknown> | null
  created_at: string
}

export interface TaskEventListResponse {
  items: TaskEvent[]
  total: number
}

export const taskApi = {
  get: (id: string): Promise<Task> => client.get(`/tasks/${id}`),
  list: (params?: { knowledge_base_id?: string; status?: TaskStatus; limit?: number }): Promise<TaskListResponse> =>
    client.get('/tasks', { params }),
  // stats 是轻量计数，overview/alerts/trends/events 面向更完整的监控视图。
  stats: (params?: { knowledge_base_id?: string }): Promise<TaskStatsResponse> =>
    client.get('/tasks/stats', { params }),
  overview: (params?: { knowledge_base_id?: string }): Promise<TaskOverviewResponse> =>
    client.get('/tasks/overview', { params }),
  alerts: (params?: { knowledge_base_id?: string }): Promise<TaskAlertListResponse> =>
    client.get('/tasks/alerts', { params }),
  trends: (params?: { limit?: number }): Promise<KnowledgeBaseTaskTrendListResponse> =>
    client.get('/tasks/trends', { params }),
  events: (id: string, params?: { limit?: number }): Promise<TaskEventListResponse> =>
    client.get(`/tasks/${id}/events`, { params }),
}
