<script setup lang="ts">
import { computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ChevronLeftRound } from '@vicons/material'
import DocumentsView from './DocumentsView.vue'
import { useKBStore } from '../stores/kb'

const route = useRoute()
const router = useRouter()
const kbStore = useKBStore()

const routeKbId = computed(() => String(route.params.id || ''))

const currentKB = computed(() =>
  kbStore.knowledgeBases.find((item) => item.id === routeKbId.value) || null,
)

const domainLabel = computed(() => {
  return kbStore.getDomainLabel(currentKB.value?.domain || kbStore.defaultDomain)
})

const statusLabel = computed(() => {
  const mapping: Record<string, string> = {
    empty: '待导入',
    indexing: '索引中',
    ready: '就绪',
    failed: '失败',
  }
  return mapping[currentKB.value?.status || 'empty'] || currentKB.value?.status || '未知'
})

const syncCurrentKB = async () => {
  if (kbStore.knowledgeBases.length === 0) {
    await kbStore.fetchKBs()
  }

  const matchedKB = kbStore.knowledgeBases.find((item) => item.id === routeKbId.value) || null
  if (!matchedKB) {
    router.replace({ name: 'KnowledgeBases' })
    return
  }

  kbStore.setCurrentKB(matchedKB)
}

onMounted(syncCurrentKB)
watch(routeKbId, syncCurrentKB)
</script>

<template>
  <div class="kb-detail-page">
    <section class="detail-header">
      <div class="detail-header__actions">
        <button class="back-btn" @click="router.push({ name: 'KnowledgeBases' })">
          <ChevronLeftRound aria-hidden="true" class="back-btn__icon" />
          <span>返回知识库列表</span>
        </button>
        <button class="monitor-btn" @click="router.push({ name: 'TaskMonitor', query: { knowledge_base_id: routeKbId } })">
          查看任务观测
        </button>
      </div>
    </section>

    <div class="resource-panel">
      <div v-if="currentKB" class="resource-summary">
        <div class="resource-summary__header">
          <div>
            <span class="resource-summary__eyebrow">当前知识库</span>
            <h2>{{ currentKB.name }}</h2>
          </div>
          <div class="resource-summary__meta">
            <span class="pill pill--domain">{{ domainLabel }}</span>
            <span class="pill" :class="`pill--${currentKB.status}`">{{ statusLabel }}</span>
          </div>
        </div>
        <p class="resource-summary__desc">{{ currentKB.description || '当前知识库还没有补充描述。' }}</p>
        <dl class="resource-summary__facts">
          <div>
            <dt>业务主题</dt>
            <dd>{{ currentKB.subject }}</dd>
          </div>
          <div>
            <dt>最近更新</dt>
            <dd>{{ new Date(currentKB.updated_at).toLocaleString('zh-CN') }}</dd>
          </div>
          <div>
            <dt>知识库 ID</dt>
            <dd>{{ currentKB.id }}</dd>
          </div>
          <div>
            <dt>当前状态</dt>
            <dd :class="['fact-status', `fact-status--${currentKB.status}`]">{{ statusLabel }}</dd>
          </div>
        </dl>
      </div>
    </div>

    <DocumentsView embedded />
  </div>
</template>

<style scoped>
.kb-detail-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.detail-header {
  display: flex;
  justify-content: flex-start;
}

.detail-header__actions {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.back-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  height: 40px;
  padding: 0 16px;
  border-radius: var(--radius);
  border: 1px solid #cbd5e1;
  background: white;
  color: #0f172a;
  font-weight: 600;
}

.back-btn:hover {
  border-color: #93c5fd;
  color: #1d4ed8;
  background: #f8fbff;
}

.back-btn__icon {
  width: 20px;
  height: 20px;
  flex: none;
}

.monitor-btn {
  display: inline-flex;
  align-items: center;
  min-height: 40px;
  padding: 0 16px;
  border-radius: var(--radius);
  border: 1px solid var(--border-color);
  background: white;
  color: var(--primary-color);
  font-weight: 700;
}

.monitor-btn:hover {
  background: #eff6ff;
  border-color: #bfdbfe;
}

.resource-panel {
  padding: 20px 24px;
  border-radius: var(--radius);
  background: var(--surface-color);
  border: 1px solid var(--border-color);
}

.resource-summary {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.resource-summary__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.resource-summary__eyebrow {
  display: block;
  margin-bottom: 8px;
  color: var(--primary-color);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.resource-summary__header h2 {
  margin: 0;
  font-size: 1.45rem;
  color: #0f172a;
}

.resource-summary__meta {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.pill {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  padding: 0 10px;
  border-radius: 999px;
  background: #f1f5f9;
  color: #475569;
  font-size: 0.76rem;
  font-weight: 700;
}

.pill--domain {
  background: #eef2ff;
  color: #4338ca;
}

.pill--ready {
  background: #dcfce7;
  color: #166534;
}

.pill--indexing {
  background: #fef9c3;
  color: #854d0e;
}

.pill--failed {
  background: #fee2e2;
  color: #991b1b;
}

.resource-summary__desc {
  margin: 0;
  color: #475569;
  line-height: 1.8;
}

.resource-summary__facts {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin: 0;
}

.resource-summary__facts div {
  padding: 12px 14px;
  border-radius: var(--radius);
  background: #f8fafc;
  border: 1px solid var(--border-color);
}

.resource-summary__facts dt {
  margin-bottom: 6px;
  color: #64748b;
  font-size: 0.78rem;
}

.resource-summary__facts dd {
  margin: 0;
  color: #0f172a;
  font-weight: 600;
  line-height: 1.6;
  word-break: break-all;
}

.resource-summary__facts dd.fact-status--ready {
  color: #166534;
}

.resource-summary__facts dd.fact-status--indexing {
  color: #854d0e;
}

.resource-summary__facts dd.fact-status--failed {
  color: #991b1b;
}

@media (max-width: 720px) {
  .resource-summary__header {
    flex-direction: column;
  }

  .resource-summary__facts {
    grid-template-columns: 1fr;
  }
}
</style>
