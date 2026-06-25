<script setup lang="ts">
// 任务概览卡片只展示关键健康指标，详细排查留给告警和事件时间线。
import type { TaskOverviewResponse } from '../../api/tasks'

const props = defineProps<{
  overview: TaskOverviewResponse | null
  loading?: boolean
}>()

const metricCards = [
  // 顺序按“队列状态 -> 异常信号 -> worker 状态 -> 近期趋势”排列。
  { key: 'pending', label: '待处理', tone: 'pending' },
  { key: 'running', label: '运行中', tone: 'running' },
  { key: 'failed', label: '失败', tone: 'failed' },
  { key: 'retrying', label: '重试过', tone: 'retrying' },
  { key: 'stale_running', label: '僵尸运行', tone: 'stale' },
  { key: 'active_workers', label: '活跃 Worker', tone: 'workers' },
  { key: 'long_running', label: '长时任务', tone: 'long' },
  { key: 'recent_failed', label: '近窗失败', tone: 'failed' },
] as const

const formatAge = (seconds: number | null | undefined) => {
  if (!seconds || seconds <= 0) return '无积压'
  if (seconds < 60) return `${seconds} 秒`
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分钟`
  return `${Math.floor(seconds / 3600)} 小时`
}
</script>

<template>
  <section class="overview-panel">
    <div class="overview-panel__head">
      <div>
        <span class="overview-panel__eyebrow">Task Overview</span>
        <h3>任务聚合概览</h3>
      </div>
      <p class="overview-panel__hint">
        最老待处理任务年龄：<strong>{{ formatAge(props.overview?.oldest_pending_age_seconds) }}</strong>
      </p>
    </div>

    <div v-if="props.loading" class="overview-grid overview-grid--loading">
      <div v-for="index in metricCards.length" :key="index" class="overview-skeleton"></div>
    </div>

    <div v-else-if="props.overview" class="overview-grid">
      <article
        v-for="card in metricCards"
        :key="card.key"
        class="overview-card"
        :class="`overview-card--${card.tone}`"
      >
        <span class="overview-card__label">{{ card.label }}</span>
        <strong class="overview-card__value">{{ props.overview[card.key] }}</strong>
      </article>
    </div>
  </section>
</template>

<style scoped>
.overview-panel {
  padding: 0;
  border-radius: var(--radius-md);
  background: white;
  border: 1px solid var(--border-color);
  overflow: hidden;
}

.overview-panel__head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 18px;
  margin-bottom: 0;
  border-bottom: 1px solid var(--border-color);
}

.overview-panel__eyebrow {
  display: block;
  margin-bottom: 8px;
  color: var(--primary-color);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.overview-panel h3 {
  margin: 0;
  color: #0f172a;
  font-size: 1.2rem;
}

.overview-panel__hint {
  margin: 0;
  color: #475569;
  font-size: 0.84rem;
}

.overview-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0;
}

.overview-card {
  min-height: 86px;
  padding: 14px 18px;
  border: 0;
  border-right: 1px solid var(--border-color);
  border-bottom: 1px solid var(--border-color);
  border-radius: 0;
  background: white;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.overview-card:nth-child(4n) {
  border-right: 0;
}

.overview-card:nth-last-child(-n + 4) {
  border-bottom: 0;
}

.overview-card__label {
  color: #64748b;
  font-size: 0.8rem;
}

.overview-card__value {
  color: #0f172a;
  font-size: clamp(1.5rem, 2vw, 1.9rem);
}

.overview-card--failed,
.overview-card--stale,
.overview-card--long {
  background: #fff7ed;
}

.overview-card--pending,
.overview-card--retrying {
  background: #eff6ff;
}

.overview-card--running,
.overview-card--workers {
  background: #f0fdfa;
}

.overview-grid--loading {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.overview-skeleton {
  min-height: 86px;
  border-radius: 0;
  border-right: 1px solid var(--border-color);
  border-bottom: 1px solid var(--border-color);
  background: linear-gradient(90deg, #e2e8f0 0%, #f8fafc 50%, #e2e8f0 100%);
  background-size: 200% 100%;
  animation: overview-wave 1.2s ease-in-out infinite;
}

@keyframes overview-wave {
  0% { background-position: 100% 50%; }
  100% { background-position: 0 50%; }
}

@media (max-width: 1080px) {
  .overview-grid,
  .overview-grid--loading {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .overview-card:nth-child(4n),
  .overview-skeleton:nth-child(4n) {
    border-right: 1px solid var(--border-color);
  }

  .overview-card:nth-child(2n),
  .overview-skeleton:nth-child(2n) {
    border-right: 0;
  }

  .overview-card:nth-last-child(-n + 4),
  .overview-skeleton:nth-last-child(-n + 4) {
    border-bottom: 1px solid var(--border-color);
  }

  .overview-card:nth-last-child(-n + 2),
  .overview-skeleton:nth-last-child(-n + 2) {
    border-bottom: 0;
  }
}

@media (max-width: 640px) {
  .overview-panel {
    padding: 0;
  }

  .overview-panel__head {
    flex-direction: column;
    align-items: flex-start;
  }

  .overview-grid,
  .overview-grid--loading {
    grid-template-columns: 1fr;
  }

  .overview-card,
  .overview-skeleton {
    border-right: 0;
    border-bottom: 1px solid var(--border-color);
  }

  .overview-card:last-child,
  .overview-skeleton:last-child {
    border-bottom: 0;
  }
}
</style>
