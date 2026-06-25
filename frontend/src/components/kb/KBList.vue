<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useKBStore } from '../../stores/kb'
import { useTaskStore } from '../../stores/task'
import ConfirmDialog from '../common/ConfirmDialog.vue'

const kbStore = useKBStore()
const taskStore = useTaskStore()
const router = useRouter()
const dialogState = ref<{
  title: string
  message: string
  confirmText: string
  tone: 'default' | 'danger'
  showCancel: boolean
  action: (() => Promise<void> | void) | null
} | null>(null)
const dialogLoading = ref(false)
const hasKnowledgeBases = computed(() => kbStore.knowledgeBases.length > 0)

onMounted(() => {
  kbStore.fetchKBs()
})

const getStatusLabel = (status: string) => {
  const map: Record<string, string> = {
    empty: '待导入',
    indexing: '索引中',
    ready: '就绪',
    failed: '失败'
  }
  return map[status] || status
}

const openDetail = (id: string) => {
  const target = kbStore.knowledgeBases.find((item) => item.id === id)
  if (target) {
    kbStore.setCurrentKB(target)
  }
  router.push({ name: 'KnowledgeBaseDetail', params: { id } })
}

const handleDelete = async (e: Event, id: string) => {
  e.stopPropagation()
  const target = kbStore.knowledgeBases.find((item) => item.id === id)
  if (target?.is_default) {
    dialogState.value = {
      title: '无法删除默认知识库',
      message: '默认知识库当前受保护，不能在前端直接删除。如需清理，请先调整后端配置或处理历史残留数据。',
      confirmText: '知道了',
      tone: 'default',
      showCancel: false,
      action: null,
    }
    return
  }
  dialogState.value = {
    title: '删除知识库',
    message: '这会同时清空该知识库的关联文档、聊天记录和向量索引，且无法恢复。',
    confirmText: '确认删除',
    tone: 'danger',
    showCancel: true,
    action: async () => {
      await kbStore.deleteKB(id)
    },
  }
}

const handleRebuild = async (e: Event, id: string) => {
  e.stopPropagation()
  dialogState.value = {
    title: '重建索引',
    message: '系统会清空当前知识库的向量数据并重新扫描文档。任务启动后可在任务状态中继续跟踪。',
    confirmText: '开始重建',
    tone: 'default',
    showCancel: true,
    action: async () => {
      const task = await kbStore.rebuildKB(id)
      taskStore.addTask(task)
      dialogState.value = {
        title: '任务已启动',
        message: '重建索引任务已经进入后台执行，你可以稍后在任务状态中查看进度。',
        confirmText: '知道了',
        tone: 'default',
        showCancel: false,
        action: null,
      }
    },
  }
}

const dialogOpen = computed(() => dialogState.value !== null)

const closeDialog = () => {
  if (dialogLoading.value) return
  dialogState.value = null
}

const handleDialogConfirm = async () => {
  if (!dialogState.value?.action) {
    closeDialog()
    return
  }

  dialogLoading.value = true
  try {
    await dialogState.value.action()
    if (dialogState.value?.showCancel) {
      dialogState.value = null
    }
  } catch (err: any) {
    dialogState.value = {
      title: '操作失败',
      message: err.message || '请求失败，请稍后重试。',
      confirmText: '知道了',
      tone: 'default',
      showCancel: false,
      action: null,
    }
  } finally {
    dialogLoading.value = false
  }
}
</script>

<template>
  <div class="kb-list-container">
    <div v-if="kbStore.loading" class="loading">加载中...</div>
    <div v-else-if="kbStore.error" class="error">{{ kbStore.error }}</div>
    <div v-else-if="!hasKnowledgeBases" class="empty-state">
      <div class="empty-state__card">
        <span class="empty-state__eyebrow">Knowledge Resources</span>
        <h3>先创建一个知识库</h3>
        <p>上传制度、手册、流程或 FAQ 后，聊天页才能基于资料回答并展示引用来源。</p>
        <button class="empty-state__action" type="button" @click="$emit('create')">创建知识库</button>
      </div>
    </div>
    <div v-else class="grid">
      <div 
        v-for="kb in kbStore.knowledgeBases" 
        :key="kb.id" 
        class="kb-card"
      >
        <div class="card-header">
          <h3 class="kb-name">{{ kb.name }}</h3>
          <div class="header-right">
            <span class="type-badge domain">{{ kbStore.getDomainLabel(kb.domain) }}</span>
            <span v-if="kb.is_default" class="type-badge default">默认库</span>
            <span class="status-badge" :class="kb.status">{{ getStatusLabel(kb.status) }}</span>
          </div>
        </div>
        <p class="kb-desc">{{ kb.description || '暂无描述' }}</p>
        <div class="card-footer">
          <span class="subject">{{ kb.subject }}</span>
          <div class="card-actions">
            <button class="text-btn" @click="openDetail(kb.id)">详情</button>
            <button class="icon-btn" @click="handleRebuild($event, kb.id)" title="重建索引">
              <svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
            </button>
            <button class="icon-btn delete" :disabled="kb.is_default" @click="handleDelete($event, kb.id)" title="删除知识库">
              <svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
            </button>
          </div>
        </div>
      </div>
      
      <div class="kb-card add-card" @click="$emit('create')">
        <div class="add-icon">+</div>
        <p>创建新知识库</p>
      </div>
    </div>
    <ConfirmDialog
      :open="dialogOpen"
      :title="dialogState?.title || ''"
      :message="dialogState?.message || ''"
      :confirm-text="dialogState?.confirmText || '确定'"
      :tone="dialogState?.tone || 'default'"
      :show-cancel="dialogState?.showCancel ?? true"
      :loading="dialogLoading"
      @close="closeDialog"
      @confirm="handleDialogConfirm"
    />
  </div>
</template>

<style scoped>
.kb-list-container { width: 100%; }

.empty-state {
  min-height: 280px;
  display: grid;
  place-items: center;
}

.empty-state__card {
  width: min(520px, 100%);
  padding: 28px;
  border-radius: var(--radius);
  border: 1px solid var(--border-color);
  background: white;
  text-align: center;
}

.empty-state__eyebrow {
  display: block;
  margin-bottom: 10px;
  color: var(--primary-color);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.empty-state__card h3 {
  margin: 0 0 10px;
  color: var(--text-color);
  font-size: 1.25rem;
}

.empty-state__card p {
  margin: 0;
  color: var(--text-muted);
  line-height: 1.7;
}

.empty-state__action {
  min-height: 40px;
  margin-top: 18px;
  padding: 0 16px;
  border-radius: var(--radius);
  background: var(--primary-color);
  color: white;
  font-weight: 600;
}

.empty-state__action:hover {
  background: var(--primary-hover);
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
}

.kb-card {
  background: var(--card-bg);
  border: 1px solid var(--border-color);
  border-radius: var(--radius);
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  position: relative;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.kb-name {
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--text-color);
}

.type-badge,
.status-badge {
  font-size: 0.75rem;
  padding: 2px 8px;
  border-radius: 999px;
}

.type-badge.default {
  background-color: #dbeafe;
  color: #1d4ed8;
}

.type-badge.domain {
  background-color: #eef2ff;
  color: #4338ca;
}

.status-badge {
  background-color: #f1f5f9;
  color: #64748b;
}

.status-badge.ready { background-color: #dcfce7; color: #166534; }
.status-badge.indexing { background-color: #fef9c3; color: #854d0e; }
.status-badge.failed { background-color: #fee2e2; color: #991b1b; }

.kb-desc {
  font-size: 0.875rem;
  color: var(--text-muted);
  flex: 1;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.75rem;
  color: var(--text-muted);
  border-top: 1px solid var(--border-color);
  padding-top: 12px;
  gap: 12px;
}

.card-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.text-btn {
  height: 32px;
  padding: 0 12px;
  border-radius: var(--radius);
  border: 1px solid #bfdbfe;
  background: #eff6ff;
  color: #1d4ed8;
  font-size: 0.82rem;
  font-weight: 600;
}

.text-btn:hover {
  background: #dbeafe;
}

.icon-btn {
  background: none;
  border: 1px solid var(--border-color);
  padding: 4px;
  border-radius: 8px;
  color: var(--text-muted);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}

.icon-btn svg { width: 16px; height: 16px; }

.icon-btn:hover {
  border-color: var(--primary-color);
  color: var(--primary-color);
  background-color: #eff6ff;
}

.icon-btn.delete:hover {
  border-color: #ef4444;
  color: #ef4444;
  background-color: #fee2e2;
}

.icon-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.add-card {
  border: 1px dashed var(--border-color);
  background: transparent;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  cursor: pointer;
  transition:
    transform 0.18s ease,
    box-shadow 0.18s ease,
    border-color 0.18s ease,
    background-color 0.18s ease,
    color 0.18s ease;
}

.add-card:hover {
  border-color: var(--primary-color);
  color: var(--primary-color);
  background-color: #eff6ff;
  transform: translateY(-2px);
  box-shadow: none;
}

.add-icon {
  font-size: 2rem;
  font-weight: 300;
  margin-bottom: 8px;
}
</style>
