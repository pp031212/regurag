<script setup lang="ts">
// 引用抽屉展示最终上下文对应的证据，顺序与后端 final_context 保持一致。
import { type Citation } from '../../api/chat'

const formatSourceType = (value: string | null | undefined) => {
  const mapping: Record<string, string> = {
    text: '正文',
    table: '表格',
    image_ocr: '图片 OCR',
    image_ocr_table: '表格截图 OCR'
  }
  return mapping[value || ''] || (value ?? '-')
}

const shouldShowBlockIndex = (value: number | null | undefined) => {
  // block_index 后端以 0 开始，-1/空值表示无法定位到具体块。
  return value !== undefined && value !== null && value >= 0
}

const formatBlockIndex = (value: number | null | undefined) => {
  if (!shouldShowBlockIndex(value)) return ''
  return `块 ${value! + 1}`
}

defineProps<{
  open: boolean
  citations: Citation[]
}>()

defineEmits(['close'])
</script>

<template>
  <div class="citation-drawer-layer" :class="{ open }" @click.self="$emit('close')">
    <div class="citation-drawer" :class="{ open }">
      <div class="drawer-header">
        <h3>引用与证据</h3>
        <button class="close-btn" @click="$emit('close')">&times;</button>
      </div>

      <div class="drawer-content">
        <div v-if="citations.length === 0" class="empty-state">
          <p>暂无引用数据</p>
        </div>
        <div v-else class="citation-list">
          <div v-for="(cit, idx) in citations" :key="idx" class="citation-card">
            <div class="card-meta">
              <span class="source-tag">来源 #{{ idx + 1 }}</span>
              <span class="score-badge">相似度: {{ cit.score.toFixed(4) }}</span>
            </div>
            <div class="meta-pills">
              <span class="meta-pill">{{ formatSourceType(cit.source_type) }}</span>
              <span v-if="cit.page_number" class="meta-pill">第 {{ cit.page_number }} 页</span>
              <span v-if="shouldShowBlockIndex(cit.block_index)" class="meta-pill">{{ formatBlockIndex(cit.block_index) }}</span>
            </div>
            <div class="card-body">
              {{ cit.content }}
            </div>
            <div class="card-footer">
              <span class="doc-id">文档 ID: {{ cit.document_id }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.citation-drawer-layer {
  position: fixed;
  inset: var(--header-height) 0 0 var(--sidebar-width);
  z-index: var(--z-drawer);
  background: rgba(15, 23, 42, 0.08);
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.2s ease;
}

.citation-drawer-layer.open {
  opacity: 1;
  pointer-events: auto;
}

.citation-drawer {
  position: fixed;
  top: var(--header-height);
  right: -400px;
  width: 400px;
  height: calc(100vh - var(--header-height));
  background-color: white;
  border-left: 1px solid var(--border-color);
  box-shadow: -4px 0 12px rgba(0, 0, 0, 0.05);
  transition: right 0.3s ease;
  z-index: calc(var(--z-drawer) + 1);
  display: flex;
  flex-direction: column;
}

.citation-drawer.open {
  right: 0;
}

.drawer-header {
  padding: 20px;
  border-bottom: 1px solid var(--border-color);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.drawer-header h3 {
  font-size: 1rem;
  font-weight: 600;
}

.close-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: var(--text-muted);
}

.drawer-content {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.citation-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.citation-card {
  border: 1px solid var(--border-color);
  border-radius: var(--radius);
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  background-color: var(--bg-color);
}

.meta-pills {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.meta-pill {
  font-size: 0.7rem;
  color: var(--text-muted);
  background-color: #f8fafc;
  border: 1px solid var(--border-color);
  border-radius: 999px;
  padding: 2px 8px;
}

.card-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.source-tag {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--primary-color);
}

.score-badge {
  font-size: 0.7rem;
  color: var(--text-muted);
}

.card-body {
  font-size: 0.875rem;
  line-height: 1.6;
  color: var(--text-color);
  white-space: pre-wrap;
}

.card-footer {
  font-size: 0.7rem;
  color: var(--text-muted);
}

.empty-state {
  text-align: center;
  color: var(--text-muted);
  margin-top: 100px;
}
</style>
