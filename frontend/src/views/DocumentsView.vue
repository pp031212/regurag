<script setup lang="ts">
// 文档页负责当前知识库的上传、导入、重建和删除操作；真正的入库在后台 worker 中执行。
import { computed, ref, onMounted, watch } from 'vue'
import { useKBStore } from '../stores/kb'
import { useTaskStore } from '../stores/task'
import { docApi, type Document } from '../api/documents'
import { taskApi } from '../api/tasks'
import DocumentUpload from '../components/kb/DocumentUpload.vue'
import ConfirmDialog from '../components/common/ConfirmDialog.vue'

const props = withDefaults(
  defineProps<{
    embedded?: boolean
  }>(),
  {
    embedded: false,
  },
)

const kbStore = useKBStore()
const taskStore = useTaskStore()

const documents = ref<Document[]>([])
const loading = ref(false)
const ingestLoading = ref(false)
const rebuildingDocumentId = ref<string | null>(null)
const deletingDocumentId = ref<string | null>(null)
const noticeDialog = ref<{
  title: string
  message: string
} | null>(null)
const deleteDialog = ref<Document | null>(null)
const workerWarning = ref('')

const currentKBActiveTasks = computed(() =>
  // 同一知识库同一时间只允许一个导入/重建任务，避免重复写索引。
  Array.from(taskStore.activeTasks.values()).filter(
    (task) =>
      task.knowledge_base_id === kbStore.currentKB?.id &&
      (task.status === 'pending' || task.status === 'running'),
  ),
)
const currentKBFailedTasks = computed(() =>
  Array.from(taskStore.activeTasks.values())
    .filter(
      (task) =>
        task.knowledge_base_id === kbStore.currentKB?.id &&
        task.status === 'failed',
    )
    .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()),
)
const ingestableDocuments = computed(() =>
  documents.value.filter((doc) => doc.status === 'uploaded' || doc.status === 'failed'),
)
const ingestBlocking = computed(() => ingestLoading.value || currentKBActiveTasks.value.length > 0)
const activeIngestTask = computed(() => currentKBActiveTasks.value[0] || null)
const latestFailedTask = computed(() => currentKBFailedTasks.value[0] || null)
const currentKBTaskFingerprint = computed(() =>
  // 用任务 id/status/updated_at 组成指纹，任务变化时触发文档列表和 worker 告警刷新。
  Array.from(taskStore.activeTasks.values())
    .filter((task) => task.knowledge_base_id === kbStore.currentKB?.id)
    .map((task) => `${task.id}:${task.status}:${task.updated_at}`)
    .sort()
    .join('|'),
)
const ingestOverlayText = computed(() => {
  if (ingestLoading.value) return '正在启动导入任务...'
  if (activeIngestTask.value?.message?.includes('整页 OCR 降级处理')) {
    return '当前文档进入了整页 OCR 降级处理，系统仍在尝试提取可用内容。'
  }
  if (activeIngestTask.value?.status === 'pending') return '导入任务已排队，正在准备处理文档...'
  return '文档正在导入中，请稍候...'
})
const ingestOverlayHint = computed(() => {
  const message = activeIngestTask.value?.message?.trim() || ''
  if (!message.includes('整页 OCR 降级处理')) return ''
  return '建议优先使用文本型 PDF、DOCX、XLSX 等更常规文件；扫描件、矢量字形或版式复杂 PDF 会显著增加处理时间。'
})
const failureBannerMessage = computed(() => {
  const message = latestFailedTask.value?.message?.trim()
  if (!message) return ''
  return humanizeFailureMessage(message)
})

const fetchDocuments = async () => {
  if (!kbStore.currentKB) return
  loading.value = true
  try {
    const res = await docApi.listByKB(kbStore.currentKB.id)
    documents.value = res.items
  } catch (err) {
    console.error('Fetch docs failed:', err)
  } finally {
    loading.value = false
  }
}

const syncCurrentKBActiveTasks = async () => {
  // 先从后端恢复 pending/running 任务，避免刷新页面后前端误以为没有任务在跑。
  if (!kbStore.currentKB) return
  try {
    await taskStore.hydrateActiveTasks({ knowledgeBaseId: kbStore.currentKB.id })
  } catch (err) {
    console.error('Hydrate active tasks failed:', err)
  }
}

const refreshWorkerWarning = async () => {
  // pending 任务存在但没有 active worker 时，单纯刷新页面不会推进导入，需要显式提示。
  if (!kbStore.currentKB) {
    workerWarning.value = ''
    return
  }

  try {
    const response = await taskApi.alerts({ knowledge_base_id: kbStore.currentKB.id })
    const missingWorkerAlert = response.items.find((item) => item.code === 'PENDING_WITHOUT_ACTIVE_WORKERS')
    workerWarning.value = missingWorkerAlert
      ? '检测到当前知识库存在排队中的导入任务，但后台任务 worker 没有运行。仅刷新浏览器不会继续处理，请启动 worker 或到任务观测页确认。'
      : ''
  } catch (err) {
    console.error('Fetch task alerts failed:', err)
    workerWarning.value = ''
  }
}

const refreshCurrentKBState = async () => {
  if (!kbStore.currentKB) return
  await syncCurrentKBActiveTasks()
  await Promise.all([fetchDocuments(), refreshWorkerWarning()])
}

const triggerIngest = async () => {
  if (!kbStore.currentKB) return

  // 提交新任务前再次同步后端任务状态，避免多个浏览器标签页重复创建导入任务。
  await syncCurrentKBActiveTasks()
  if (currentKBActiveTasks.value.length > 0) {
    noticeDialog.value = {
      title: '已有任务在进行',
      message: workerWarning.value || '当前知识库已经有排队中或运行中的导入任务，请等待当前任务结束后再继续提交。',
    }
    return
  }

  if (ingestableDocuments.value.length === 0) {
    noticeDialog.value = {
      title: '无需导入',
      message: '当前没有待导入或导入失败的文档，无需重复执行导入。',
    }
    return
  }
  ingestLoading.value = true
  try {
    const task = await docApi.ingest(
      kbStore.currentKB.id,
      ingestableDocuments.value.map((doc) => doc.id),
    )
    taskStore.addTask(task)
    noticeDialog.value = {
      title: '任务已启动',
      message: `已将 ${ingestableDocuments.value.length} 个待处理文档加入导入队列，请关注文档状态更新。`,
    }
  } catch (err: any) {
    noticeDialog.value = {
      title: '启动失败',
      message: err.message || '启动失败，请稍后重试。',
    }
  } finally {
    ingestLoading.value = false
  }
}

const triggerDocumentAction = async (document: Document) => {
  if (!kbStore.currentKB) return

  // 单文档重建也会写同一个知识库索引，所以仍然受当前 KB active task 限制。
  await syncCurrentKBActiveTasks()
  if (currentKBActiveTasks.value.length > 0) {
    noticeDialog.value = {
      title: '已有任务在进行',
      message: workerWarning.value || '当前知识库已经有排队中或运行中的导入任务，请等待当前任务结束后再继续提交。',
    }
    return
  }

  rebuildingDocumentId.value = document.id
  try {
    const isReadyDocument = document.status === 'ready'
    const task = isReadyDocument
      ? await docApi.rebuild(document.id)
      : await docApi.ingest(kbStore.currentKB.id, [document.id])

    taskStore.addTask(task)
    noticeDialog.value = {
      title: isReadyDocument ? '重建任务已启动' : '导入任务已启动',
      message: isReadyDocument
        ? '该文档已进入后台重建索引流程，处理完成后会自动刷新状态。'
        : '该文档已进入后台导入流程，处理完成后会自动刷新状态。',
    }
  } catch (err: any) {
    noticeDialog.value = {
      title: document.status === 'ready' ? '重建启动失败' : '导入启动失败',
      message: err.message || '启动失败，请稍后重试。',
    }
  } finally {
    rebuildingDocumentId.value = null
  }
}

const requestDocumentDelete = (document: Document) => {
  deleteDialog.value = document
}

const confirmDocumentDelete = async () => {
  if (!deleteDialog.value) return

  const document = deleteDialog.value
  deletingDocumentId.value = document.id
  try {
    await docApi.delete(document.id)
    deleteDialog.value = null
    await fetchDocuments()
    noticeDialog.value = {
      title: '删除成功',
      message: `文档“${document.filename}”已从当前知识库移除。`,
    }
  } catch (err: any) {
    noticeDialog.value = {
      title: '删除失败',
      message: err.message || '删除失败，请稍后重试。',
    }
  } finally {
    deletingDocumentId.value = null
  }
}

const getStatusLabel = (status: string) => {
  const map: Record<string, string> = {
    'uploaded': '待导入',
    'indexing': '导入中',
    'ready': '可问答',
    'failed': '导入失败'
  }
  return map[status] || status
}

const getRebuildActionLabel = (status: Document['status']) => {
  if (status === 'uploaded') return '导入'
  if (status === 'failed') return '重试导入'
  return '重建索引'
}

const humanizeFailureMessage = (message: string) => {
  if (message.includes('未提取到可用文本、表格或 OCR 内容')) {
    return '未识别到可用文本、表格或 OCR 内容。当前文件可能是纯图片、画质过低，或内容本身不可解析。'
  }
  if (message.includes('total chunks: 0')) {
    return '文档没有成功产出可入库内容，因此本次导入失败。请检查文件是否为空，或内容是否无法被解析。'
  }
  if (message.includes('unsupported document type')) {
    return '文件类型当前不受支持，请改用页面提示的可上传格式。'
  }
  return message
}

const getDocumentFailureReason = (doc: Document) => {
  // 文档表只存状态，具体失败原因从最近的失败任务里补充成人类可读文案。
  if (doc.status !== 'failed') return ''

  const matchedTask = currentKBFailedTasks.value.find((task) => task.document_ids.includes(doc.id))
  if (matchedTask?.message) {
    return humanizeFailureMessage(matchedTask.message)
  }

  return '本次导入未成功完成，请检查文件内容是否可解析后重试。'
}

onMounted(() => {
  // 嵌入详情页时可能已经有 currentKB；独立进入时需要先拉知识库列表。
  void (async () => {
    if (!kbStore.currentKB) {
      await kbStore.fetchKBs()
    }
    await refreshCurrentKBState()
  })()
})

watch(() => kbStore.currentKB?.id, async () => {
  await refreshCurrentKBState()
})

watch(currentKBTaskFingerprint, async () => {
  // 后台任务状态变化后重新拉文档，拿到 ready/failed/indexing 的最新状态。
  await Promise.all([fetchDocuments(), refreshWorkerWarning()])
})
</script>

<template>
  <div class="docs-page">
    <div v-if="!kbStore.currentKB" class="no-kb-state">
      <p>{{ props.embedded ? '请先在上方知识库列表中选择一个知识库，再管理对应文档。' : '请先在知识库管理中选择一个知识库。' }}</p>
      <router-link v-if="!props.embedded" to="/knowledge-bases" class="btn">去选择</router-link>
    </div>
    
    <template v-else>
      <section class="section">
        <div class="section-header">
          <h2>上传文档</h2>
          <p class="subtitle">这些文件属于当前知识库资源，导入完成后会参与问答检索。</p>
        </div>
        <DocumentUpload :kb-id="kbStore.currentKB.id" :disabled="ingestBlocking" @uploaded="fetchDocuments" />
      </section>

      <section class="section">
        <div class="section-header">
          <div class="title-row">
            <h2>文档列表 ({{ documents.length }})</h2>
            <div class="actions">
              <button 
                class="btn primary" 
                :disabled="ingestableDocuments.length === 0 || ingestBlocking"
                @click="triggerIngest"
              >
                {{ ingestBlocking ? '导入进行中...' : '导入待处理文档' }}
              </button>
              <button class="btn secondary" @click="fetchDocuments" :disabled="loading || ingestBlocking">
                刷新列表
              </button>
            </div>
          </div>
          <p class="subtitle">仅会导入“待导入 / 导入失败”的文档；已可问答的文档不会在这里被重复处理。</p>
          <div class="status-help">
            <span>待导入：已上传，但还没进入检索库</span>
            <span>导入中：正在处理文档</span>
            <span>可问答：已可参与检索和问答</span>
            <span>导入失败：处理过程中出错</span>
          </div>
        </div>
        
        <div class="table-container">
          <div v-if="workerWarning" class="worker-warning-banner">
            <strong>后台 worker 未运行：</strong>
            <span>{{ workerWarning }}</span>
          </div>
          <div v-if="failureBannerMessage" class="failure-banner">
            <strong>最近一次导入失败原因：</strong>
            <span>{{ failureBannerMessage }}</span>
          </div>
          <table v-if="documents.length > 0" class="docs-table">
            <thead>
              <tr>
                <th>文件名</th>
                <th>类型</th>
                <th>状态</th>
                <th>失败原因</th>
                <th>上传时间</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="doc in documents" :key="doc.id">
                <td class="filename">{{ doc.filename }}</td>
                <td>{{ doc.content_type }}</td>
                <td>
                  <span class="status-pill" :class="doc.status">
                    {{ getStatusLabel(doc.status) }}
                  </span>
                </td>
                <td class="failure-reason">
                  <span v-if="doc.status === 'failed'">
                    {{ getDocumentFailureReason(doc) }}
                  </span>
                  <span v-else class="failure-reason__placeholder">-</span>
                </td>
                <td class="date">{{ new Date(doc.created_at).toLocaleString() }}</td>
                <td class="actions-cell">
                  <div class="row-actions">
                    <button
                      class="btn secondary btn-small"
                      :disabled="ingestBlocking || rebuildingDocumentId === doc.id || deletingDocumentId === doc.id"
                      @click="triggerDocumentAction(doc)"
                    >
                      {{ rebuildingDocumentId === doc.id ? '启动中...' : getRebuildActionLabel(doc.status) }}
                    </button>
                    <button
                      class="btn danger-outline btn-small"
                      :disabled="ingestBlocking || deletingDocumentId === doc.id || rebuildingDocumentId === doc.id"
                      @click="requestDocumentDelete(doc)"
                    >
                      {{ deletingDocumentId === doc.id ? '删除中...' : '删除' }}
                    </button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
          <div v-else-if="!loading" class="empty-docs">
            <p>当前知识库还没有文档资源，请从上方开始上传。</p>
          </div>
          <div v-if="loading" class="loading-overlay">加载中...</div>
        </div>
      </section>
      <div v-if="ingestBlocking" class="ingest-overlay">
        <div class="ingest-overlay__card">
          <div class="ingest-overlay__spinner">
            <span></span>
            <span></span>
            <span></span>
          </div>
          <h3>导入进行中</h3>
          <p>{{ ingestOverlayText }}</p>
          <p v-if="ingestOverlayHint" class="ingest-overlay__hint">{{ ingestOverlayHint }}</p>
          <span v-if="activeIngestTask" class="ingest-overlay__meta">
            当前状态：{{ activeIngestTask.status === 'pending' ? '排队中' : '处理中' }}
          </span>
        </div>
      </div>
    </template>
    <ConfirmDialog
      :open="noticeDialog !== null"
      :title="noticeDialog?.title || ''"
      :message="noticeDialog?.message || ''"
      confirm-text="知道了"
      :show-cancel="false"
      @close="noticeDialog = null"
      @confirm="noticeDialog = null"
    />
    <ConfirmDialog
      :open="deleteDialog !== null"
      title="删除文档"
      :message="deleteDialog ? `确定要删除文档“${deleteDialog.filename}”吗？该操作会同时清理该文档的索引和产物，无法撤销。` : ''"
      confirm-text="确认删除"
      cancel-text="取消"
      tone="danger"
      :loading="deletingDocumentId !== null"
      @close="deleteDialog = null"
      @confirm="confirmDocumentDelete"
    />
  </div>
</template>

<style scoped>
.docs-page {
  display: flex;
  flex-direction: column;
  gap: 24px;
  position: relative;
}

.section {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 20px 24px;
  border: 1px solid var(--border-color);
  border-radius: var(--radius);
  background: white;
}

.section-header h2 {
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: 4px;
}

.subtitle {
  font-size: 0.9rem;
  color: var(--text-muted);
}

.status-help {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 12px;
  font-size: 0.75rem;
  color: var(--text-muted);
  line-height: 1.5;
}

.title-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.actions {
  display: flex;
  gap: 12px;
}

.actions-cell {
  width: 220px;
}

.row-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.btn {
  padding: 8px 16px;
  border-radius: var(--radius);
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s;
  border: 1px solid var(--border-color);
  background: white;
  text-decoration: none;
  color: var(--text-color);
}

.btn.primary {
  background-color: var(--primary-color);
  color: white;
  border: none;
}

.btn.primary:hover { background-color: var(--primary-hover); }
.btn.primary:disabled { background-color: #94a3b8; cursor: not-allowed; }

.btn.danger-outline {
  color: #b91c1c;
  background: #fff5f5;
  border-color: #fecaca;
}

.btn.danger-outline:hover:not(:disabled) {
  background: #fee2e2;
}

.table-container {
  background-color: white;
  border: 1px solid var(--border-color);
  border-radius: var(--radius);
  overflow: hidden;
  position: relative;
  min-height: 200px;
}

.btn-small {
  padding: 6px 12px;
  font-size: 0.82rem;
}

.failure-banner {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 14px 18px;
  border-bottom: 1px solid #fecaca;
  background: #fff1f2;
  color: #9f1239;
  font-size: 0.85rem;
  line-height: 1.6;
}

.worker-warning-banner {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 14px 18px;
  border-bottom: 1px solid #fcd34d;
  background: #fffbeb;
  color: #92400e;
  font-size: 0.85rem;
  line-height: 1.6;
}

.docs-table {
  width: 100%;
  border-collapse: collapse;
  text-align: left;
}

.docs-table th, .docs-table td {
  padding: 14px 16px;
  border-bottom: 1px solid var(--border-color);
  vertical-align: middle;
}

.docs-table th {
  background-color: #f8fafc;
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
}

.filename {
  font-weight: 500;
  color: var(--text-color);
}

.failure-reason {
  max-width: 360px;
  color: #991b1b;
  font-size: 0.82rem;
  line-height: 1.6;
}

.failure-reason__placeholder {
  color: #94a3b8;
}

.status-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 72px;
  font-size: 0.75rem;
  padding: 2px 10px;
  border-radius: 999px;
  background-color: #f1f5f9;
  color: #64748b;
  white-space: nowrap;
  word-break: keep-all;
}

.status-pill.uploaded { background-color: #e2e8f0; color: #475569; }
.status-pill.ready { background-color: #dcfce7; color: #166534; }
.status-pill.indexing { background-color: #fef9c3; color: #854d0e; }
.status-pill.failed { background-color: #fee2e2; color: #991b1b; }

.date {
  color: var(--text-muted);
  font-size: 0.85rem;
}

.empty-docs {
  padding: 60px;
  text-align: center;
  color: var(--text-muted);
}

.no-kb-state {
  text-align: center;
  padding: 100px;
  background: white;
  border-radius: var(--radius);
  border: 1px solid var(--border-color);
}

.no-kb-state p {
  margin-bottom: 20px;
  color: var(--text-muted);
}

.loading-overlay {
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(255, 255, 255, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
}

.ingest-overlay {
  position: fixed;
  inset: var(--header-height) 0 0 var(--sidebar-width);
  z-index: var(--z-modal);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background:
    linear-gradient(180deg, rgba(248, 250, 252, 0.72), rgba(241, 245, 249, 0.9));
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
}

.ingest-overlay__card {
  width: min(420px, 100%);
  padding: 28px 24px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  border: 1px solid rgba(191, 219, 254, 0.9);
  border-radius: var(--radius);
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 0 24px 60px rgba(15, 23, 42, 0.12);
  text-align: center;
}

.ingest-overlay__card h3 {
  font-size: 1.1rem;
  font-weight: 700;
  color: #0f172a;
}

.ingest-overlay__card p {
  color: var(--text-muted);
  line-height: 1.7;
}

.ingest-overlay__meta {
  font-size: 0.8rem;
  color: #475569;
}

.ingest-overlay__hint {
  padding: 10px 12px;
  border-radius: 14px;
  background: #fff7ed;
  color: #9a3412;
  font-size: 0.82rem;
  line-height: 1.7;
}

.ingest-overlay__spinner {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  height: 22px;
}

.ingest-overlay__spinner span {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: linear-gradient(180deg, #60a5fa, #2563eb);
  animation: ingest-wave 1s ease-in-out infinite;
}

.ingest-overlay__spinner span:nth-child(2) {
  animation-delay: 0.12s;
}

.ingest-overlay__spinner span:nth-child(3) {
  animation-delay: 0.24s;
}

@keyframes ingest-wave {
  0%, 100% {
    transform: translateY(0);
    opacity: 0.5;
  }

  50% {
    transform: translateY(-6px);
    opacity: 1;
  }
}

@media (max-width: 960px) {
  .ingest-overlay {
    inset: var(--header-height) 0 0 0;
  }

  .failure-banner {
    flex-direction: column;
  }
}
</style>
