<script setup lang="ts">
// 单条消息组件负责 Markdown 渲染、开发态状态标签、引用/调试入口。
import { type Message } from '../../stores/chat'
import MarkdownIt from 'markdown-it'

const isDev = import.meta.env.DEV
const props = defineProps<{
  msg: Message
}>()

defineEmits(['open-citations', 'open-debug'])

const md = new MarkdownIt()
// 这里渲染的是后端回答和用户输入；如果未来允许任意 HTML，需要再加 sanitize。
const renderMarkdown = (content: string) => md.render(content)
const formatMessageTime = (timestamp: number) =>
  new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

const getAssistantPills = (msg: Message) => {
  // 开发态才展示后端路由/守卫标签，避免正式界面暴露过多内部实现。
  if (!isDev || msg.role !== 'assistant' || !msg.debug) return []

  const pills: Array<{ label: string; tone: 'source' | 'route' | 'guard' }> = []
  if (msg.debug.resolved_knowledge_base_name) {
    pills.push({
      label: `来源：${msg.debug.resolved_knowledge_base_name}`,
      tone: 'source',
    })
  }
  if (msg.debug.auto_routed) {
    pills.push({ label: '自动路由', tone: 'route' })
  }
  if (msg.debug.cross_domain_guard_applied) {
    pills.push({ label: '弱澄清', tone: 'guard' })
  }
  if (msg.debug.answer_guard_applied) {
    pills.push({ label: '保守回答', tone: 'guard' })
  }
  return pills
}
</script>

<template>
  <div class="message-wrapper" :class="msg.role">
    <!-- Avatar Area -->
    <div class="avatar-container">
      <div class="avatar" :class="{ 'bot-avatar': msg.role === 'assistant', 'user-avatar': msg.role === 'user' }">
        <template v-if="msg.role === 'user'">U</template>
        <template v-else>
          <img src="../../assets/logo.svg" alt="ReguRAG 助手" class="logo-img" @error="(e) => (e.target as HTMLImageElement).classList.add('hide')" />
          <span class="fallback-text">RG</span>
        </template>
      </div>
    </div>

    <!-- Content Area -->
    <div class="message-content">
      <div v-if="msg.role === 'assistant' && getAssistantPills(msg).length" class="message-pills">
        <span
          v-for="pill in getAssistantPills(msg)"
          :key="pill.label"
          class="message-pill"
          :class="`message-pill--${pill.tone}`"
        >
          {{ pill.label }}
        </span>
      </div>
      <div
        v-if="msg.streaming"
        class="bubble bubble--streaming"
      >
        {{ msg.content }}
      </div>
      <div v-else class="bubble" v-html="renderMarkdown(msg.content)"></div>
      
      <!-- Actions (Assistant Only) -->
      <div v-if="msg.role === 'assistant'" class="message-actions">
        <button v-if="msg.citations?.length" class="action-btn cite-btn" @click="$emit('open-citations')">
          <svg class="btn-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
          {{ msg.citations.length }} 条引用
        </button>
        <button v-if="isDev && msg.debug" class="action-btn debug-btn" @click="$emit('open-debug')">
          <svg class="btn-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
          调试全链路
        </button>
      </div>
      <div class="message-time">{{ formatMessageTime(msg.timestamp) }}</div>
    </div>
  </div>
</template>

<style scoped>
.message-wrapper {
  display: flex;
  gap: 12px;
  width: min(860px, calc(100% - 48px));
  margin: 0 auto 8px;
}

.message-wrapper.user {
  flex-direction: row-reverse;
}

.avatar-container {
  flex-shrink: 0;
}

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 0.8rem;
  overflow: hidden;
  position: relative;
}

.user-avatar {
  background-color: #334155;
  color: white;
}

.bot-avatar {
  background-color: white;
  border: 1px solid var(--border-color);
}

.logo-img { width: 100%; height: 100%; object-fit: cover; }
.logo-img.hide { display: none; }
.fallback-text { position: absolute; display: none; }
.logo-img.hide + .fallback-text { display: block; }

.message-content {
  display: flex;
  flex-direction: column;
  gap: 6px;
  flex: 1;
  max-width: 100%;
  min-width: 0;
}

.message-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.message-pill {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  padding: 0 8px;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 700;
}

.message-pill--source {
  background: #eef2ff;
  color: #4338ca;
}

.message-pill--route {
  background: #dbeafe;
  color: #1d4ed8;
}

.message-pill--guard {
  background: #fff7ed;
  color: #c2410c;
}

.user .message-content {
  align-items: flex-end;
}

.bubble {
  padding: 12px 16px;
  border-radius: 12px;
  font-size: 0.95rem;
  line-height: 1.6;
  word-break: break-word;
  max-width: min(100%, 720px);
}

.user .bubble {
  background-color: var(--primary-color);
  color: white;
  border-bottom-right-radius: 2px;
}

.assistant .bubble {
  background-color: white;
  border: 1px solid var(--border-color);
  color: var(--text-color);
  border-bottom-left-radius: 2px;
}

.bubble--streaming {
  white-space: pre-wrap;
  font-family: var(--font-sans);
}

/* Markdown refinements inside bubble */
.bubble :deep(p) { margin: 0 0 8px 0; }
.bubble :deep(p:last-child) { margin-bottom: 0; }
.bubble :deep(ul), .bubble :deep(ol) { margin-left: 1.2rem; margin-bottom: 8px; }

.message-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 4px;
}

.message-time {
  font-size: 0.72rem;
  color: var(--text-muted);
  line-height: 1;
}

.action-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  background: white;
  border: 1px solid var(--border-color);
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 0.75rem;
  color: var(--text-muted);
  cursor: pointer;
  transition: all 0.2s;
}

.action-btn:hover {
  border-color: var(--primary-color);
  color: var(--primary-color);
  background-color: #eff6ff;
}

.btn-icon {
  width: 14px;
  height: 14px;
}

@media (max-width: 640px) {
  .message-wrapper {
    width: calc(100% - 24px);
    gap: 10px;
  }

  .avatar {
    width: 32px;
    height: 32px;
  }

  .bubble {
    max-width: 100%;
  }
}
</style>




