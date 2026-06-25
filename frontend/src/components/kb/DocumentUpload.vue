<script setup lang="ts">
import { ref } from 'vue'
import { docApi } from '../../api/documents'

const props = defineProps<{
  kbId: string
  disabled?: boolean
}>()

const emit = defineEmits(['uploaded'])

const isDragging = ref(false)
const uploading = ref(false)
const error = ref<string | null>(null)
const MAX_FILE_SIZE_BYTES_BY_EXTENSION: Record<string, number> = {
  '.txt': 10 * 1024 * 1024,
  '.md': 10 * 1024 * 1024,
  '.pdf': 30 * 1024 * 1024,
  '.docx': 20 * 1024 * 1024,
  '.xlsx': 20 * 1024 * 1024,
  '.png': 10 * 1024 * 1024,
  '.jpg': 10 * 1024 * 1024,
  '.jpeg': 10 * 1024 * 1024,
}
const ALLOWED_EXTENSIONS = new Set(Object.keys(MAX_FILE_SIZE_BYTES_BY_EXTENSION))
const uploadGuidanceSections = [
  {
    title: '推荐直接入库',
    items: ['结构清晰的 .txt / .md / .docx / .xlsx。'],
  },
  {
    title: '支持上传但建议先整理',
    items: ['文本型 PDF、含少量截图的 DOCX。'],
  },
  {
    title: '不建议直接入库',
    items: ['扫描件、长截图、海报式排版、版式复杂 PDF。'],
  },
]
const uploadChecklistItems = [
  '去掉与问答无关的封面、宣传页、页脚说明和重复附件。',
  '把复合规则拆开，尽量让一段文字只描述一个主体、条件或处理结果。',
  '补齐主体、适用情形、时间条件和例外说明，减少模糊代词。',
  '表格、截图或扫描页较多时，先整理成更连续的文本再上传。',
]

const formatSize = (sizeBytes: number) => `${Math.round(sizeBytes / 1024 / 1024)}MB`

const handleFileChange = async (event: Event) => {
  if (props.disabled) return
  const files = (event.target as HTMLInputElement).files
  if (files && files.length > 0) {
    uploadFiles(Array.from(files))
  }
}

const handleDrop = (event: DragEvent) => {
  if (props.disabled) return
  isDragging.value = false
  const files = event.dataTransfer?.files
  if (files && files.length > 0) {
    uploadFiles(Array.from(files))
  }
}

const uploadFiles = async (files: File[]) => {
  uploading.value = true
  error.value = null
  
  try {
    for (const file of files) {
      const normalizedName = file.name.toLowerCase()
      const dotIndex = normalizedName.lastIndexOf('.')
      const extension = dotIndex >= 0 ? normalizedName.slice(dotIndex) : ''
      if (!ALLOWED_EXTENSIONS.has(extension)) {
        error.value = `目前仅支持 .txt / .md / .pdf / .docx / .xlsx / 图片 文件: ${file.name}`
        continue
      }
      const maxSizeBytes = MAX_FILE_SIZE_BYTES_BY_EXTENSION[extension]
      if (file.size > maxSizeBytes) {
        error.value = `${file.name} 超过 ${formatSize(maxSizeBytes)} 限制，请压缩或拆分后再上传。`
        continue
      }
      
      const doc = await docApi.upload(props.kbId, file)
      console.log('File uploaded:', doc)
      emit('uploaded', doc)
    }
  } catch (err: any) {
    if (err?.code === 'DUPLICATE_DOCUMENT') {
      error.value = err.message || '该文件已存在于当前知识库，无需重复上传。'
    } else {
      error.value = err.message || '上传失败'
    }
  } finally {
    uploading.value = false
  }
}
</script>

<template>
  <div 
    class="upload-zone"
    :class="{ dragging: isDragging, uploading, disabled: props.disabled }"
    @dragover.prevent="isDragging = true"
    @dragleave.prevent="isDragging = false"
    @drop.prevent="handleDrop"
  >
    <div class="upload-content">
      <div v-if="uploading" class="loader">
        <div class="spinner"></div>
        <p>正在上传中...</p>
      </div>
      <template v-else>
        <svg class="upload-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
        <div class="text">
          <p class="main-text">{{ props.disabled ? '导入任务进行中，暂时无法上传' : '点击或将文件拖拽到此处上传' }}</p>
          <p class="sub-text">
            {{ props.disabled ? '请等待当前导入任务结束后再继续操作' : '当前支持 .txt / .md / .pdf / .docx / .xlsx / .png / .jpg / .jpeg；文本/图片 10MB，Word/Excel 20MB，PDF 30MB' }}
          </p>
          <div v-if="!props.disabled" class="hint-text">
            <p class="hint-title">上传建议</p>
            <p class="hint-intro">系统支持上传多种格式，但更推荐先把资料整理成适合切块、检索和引用的文本。</p>
            <div class="hint-sections">
              <section v-for="section in uploadGuidanceSections" :key="section.title" class="hint-section">
                <p class="hint-section-title">{{ section.title }}</p>
                <ul class="hint-list">
                  <li v-for="item in section.items" :key="item">{{ item }}</li>
                </ul>
              </section>
            </div>
            <div class="checklist-card">
              <p class="checklist-title">上传前检查清单</p>
              <ul class="hint-list checklist-list">
                <li v-for="item in uploadChecklistItems" :key="item">{{ item }}</li>
              </ul>
            </div>
            <p class="hint-footnote">
              更完整的接入规范已整理到 docs/知识库文档接入规范.md。
            </p>
          </div>
        </div>
        <input type="file" multiple class="file-input" :disabled="props.disabled" @change="handleFileChange" accept=".txt,.md,.pdf,.docx,.xlsx,.png,.jpg,.jpeg,text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,image/png,image/jpeg" />
      </template>
    </div>
    <div v-if="error" class="error-msg">{{ error }}</div>
  </div>
</template>

<style scoped>
.upload-zone {
  border: 1px dashed var(--border-color);
  border-radius: var(--radius);
  background-color: #f8fafc;
  padding: 28px;
  position: relative;
  transition: all 0.2s;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 180px;
}

.upload-zone:hover, .upload-zone.dragging {
  border-color: var(--primary-color);
  background-color: #eff6ff;
}

.upload-zone.uploading {
  cursor: not-allowed;
  pointer-events: none;
}

.upload-zone.disabled {
  cursor: not-allowed;
  opacity: 0.72;
  border-color: #cbd5e1;
  background-color: #f8fafc;
}

.upload-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  text-align: center;
}

.upload-icon {
  width: 48px;
  height: 48px;
  color: #94a3b8;
}

.dragging .upload-icon {
  color: var(--primary-color);
  transform: translateY(-4px);
  transition: transform 0.2s;
}

.main-text {
  font-weight: 500;
  color: var(--text-color);
  margin-bottom: 4px;
}

.sub-text {
  font-size: 0.85rem;
  color: var(--text-muted);
}

.hint-text {
  max-width: 620px;
  font-size: 0.84rem;
  line-height: 1.55;
  color: #475569;
  background: rgba(255, 255, 255, 0.78);
  border: 1px solid #dbeafe;
  border-radius: var(--radius);
  padding: 10px 14px;
  text-align: left;
}

.hint-title {
  margin: 0 0 6px;
  font-weight: 600;
  color: #334155;
}

.hint-intro {
  margin: 0 0 8px;
}

.hint-sections {
  display: grid;
  gap: 8px;
}

.hint-section-title,
.checklist-title {
  margin: 0 0 4px;
  font-weight: 600;
  color: #1e293b;
}

.hint-list {
  margin: 0;
  padding-left: 18px;
}

.hint-list li + li {
  margin-top: 4px;
}

.checklist-card {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid #dbeafe;
}

.hint-footnote {
  margin: 8px 0 0;
  color: #64748b;
}

.file-input {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  opacity: 0;
  cursor: pointer;
}

.error-msg {
  margin-top: 12px;
  color: #ef4444;
  font-size: 0.85rem;
}

/* Spinner */
.spinner {
  width: 32px;
  height: 32px;
  border: 3px solid #e2e8f0;
  border-top-color: var(--primary-color);
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 12px;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
