<script setup lang="ts">
// 告警组件把后端机器可读的 alert code 翻译成人能执行的排查动作。
import { computed } from 'vue'
import type { TaskAlert } from '../../api/tasks'

const props = defineProps<{
  alerts: TaskAlert[]
  loading?: boolean
}>()

const severityLabel = (severity: TaskAlert['severity']) => {
  return severity === 'critical' ? '严重' : '提醒'
}

const explainAlert = (alert: TaskAlert) => {
  // 文案在前端展开，后端只需要稳定提供 code/details。
  const details = alert.details || {}
  switch (alert.code) {
    case 'PENDING_WITHOUT_ACTIVE_WORKERS':
      return {
        title: '队列里还有任务，但当前没有活跃 worker。',
        action: '优先检查 worker 进程是否退出、日志是否持续报错，或本机是否还在运行旧进程。',
        context: `pending=${details.pending ?? alert.count}，active_workers=${details.active_workers ?? 0}`,
      }
    case 'STALE_RUNNING_TASKS':
      return {
        title: '有任务长时间停留在 running，租约可能已经过期。',
        action: '检查任务是否被卡死，确认是否需要重启 worker 或人工观察任务事件时间线。',
        context: `stale_running=${details.stale_running ?? alert.count}`,
      }
    case 'LONG_RUNNING_TASKS':
      return {
        title: '存在超过长时阈值的任务。',
        action: '先看时间线停在哪个文档或阶段，再判断是 OCR 慢、向量入库慢，还是任务已卡住。',
        context: `threshold=${details.threshold_seconds ?? '—'} 秒`,
      }
    case 'RECENT_FAILURE_SPIKE':
      return {
        title: '最近窗口内失败任务数量明显偏高。',
        action: '先看是否集中在同一知识库或同一类文档，再决定是修数据、修流程，还是调重试策略。',
        context: `recent_failed=${details.recent_failed ?? alert.count}，window=${details.window_hours ?? '—'} 小时`,
      }
    case 'RECENT_RETRY_SPIKE':
      return {
        title: '最近窗口内重试次数偏高，系统在恢复但不够稳。',
        action: '先检查失败是否可自动恢复；如果同类错误反复出现，应该回到根因而不是继续依赖重试。',
        context: `recent_retried=${details.recent_retried ?? alert.count}，window=${details.window_hours ?? '—'} 小时`,
      }
    default:
      return {
        title: alert.message,
        action: '请结合任务列表和事件时间线继续下钻排查。',
        context: '',
      }
  }
}

const explainedAlerts = computed(() =>
  props.alerts.map((alert) => ({
    ...alert,
    explanation: explainAlert(alert),
  })),
)
</script>

<template>
  <section class="alert-panel">
    <div class="alert-panel__head">
      <div>
        <span class="alert-panel__eyebrow">Task Alerts</span>
        <h3>最小告警判定</h3>
      </div>
    </div>

    <div v-if="props.loading" class="alert-list">
      <div v-for="index in 2" :key="index" class="alert-skeleton"></div>
    </div>

    <div v-else-if="props.alerts.length === 0" class="alert-empty">
      当前没有命中告警规则，任务队列处于可接受状态。
    </div>

    <div v-else class="alert-list">
      <article
        v-for="alert in explainedAlerts"
        :key="alert.code"
        class="alert-item"
        :class="`alert-item--${alert.severity}`"
      >
        <div class="alert-item__meta">
          <span class="alert-item__tag">{{ severityLabel(alert.severity) }}</span>
          <strong>{{ alert.code }}</strong>
        </div>
        <p class="alert-item__title">{{ alert.explanation.title }}</p>
        <p class="alert-item__action">{{ alert.explanation.action }}</p>
        <div class="alert-item__footer">
          <span class="alert-item__count">数量：{{ alert.count }}</span>
          <span v-if="alert.explanation.context" class="alert-item__context">{{ alert.explanation.context }}</span>
        </div>
      </article>
    </div>
  </section>
</template>

<style scoped>
.alert-panel {
  padding: 0;
  border-radius: var(--radius-md);
  background: white;
  border: 1px solid var(--border-color);
  overflow: hidden;
}

.alert-panel__head {
  padding: 14px 18px;
  margin-bottom: 0;
  border-bottom: 1px solid var(--border-color);
}

.alert-panel__eyebrow {
  display: block;
  margin-bottom: 8px;
  color: var(--primary-color);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.alert-panel h3 {
  margin: 0;
  color: #0f172a;
  font-size: 1.2rem;
}

.alert-list {
  display: flex;
  flex-direction: column;
  gap: 0;
}

.alert-item,
.alert-skeleton,
.alert-empty {
  border-radius: 0;
  padding: 14px 18px 14px 16px;
  border: 0;
  border-left: 3px solid var(--primary-color);
  border-bottom: 1px solid var(--border-color);
  background: #ffffff;
}

.alert-item:last-child,
.alert-skeleton:last-child {
  border-bottom: 0;
}

.alert-item--critical {
  border-left-color: #dc2626;
  background: #fff7ed;
}

.alert-item--warning {
  border-left-color: #f59e0b;
  background: #fffaf0;
}

.alert-item__meta {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
  color: #0f172a;
}

.alert-item__tag {
  display: inline-flex;
  min-height: 24px;
  padding: 0 8px;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.85);
  color: #991b1b;
  font-size: 0.72rem;
  font-weight: 700;
}

.alert-item__title,
.alert-item__action {
  margin: 0 0 8px;
  color: #475569;
  line-height: 1.7;
}

.alert-item__title {
  color: #0f172a;
  font-weight: 600;
}

.alert-item__footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.alert-item__count,
.alert-item__context {
  color: #64748b;
  font-size: 0.8rem;
}

.alert-empty {
  color: #475569;
  line-height: 1.7;
}

.alert-skeleton {
  min-height: 92px;
  background: linear-gradient(90deg, #e2e8f0 0%, #f8fafc 50%, #e2e8f0 100%);
  background-size: 200% 100%;
  animation: alert-wave 1.2s ease-in-out infinite;
}

@keyframes alert-wave {
  0% { background-position: 100% 50%; }
  100% { background-position: 0 50%; }
}

@media (max-width: 640px) {
  .alert-panel {
    padding: 0;
  }

  .alert-item__footer {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
