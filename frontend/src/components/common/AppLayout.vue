<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import HealthStatus from './HealthStatus.vue'
import ConfirmDialog from './ConfirmDialog.vue'
import { useKBStore } from '../../stores/kb'
import { useChatStore } from '../../stores/chat'

const route = useRoute()
const router = useRouter()
const kbStore = useKBStore()
const chatStore = useChatStore()
const currentPath = computed(() => route.path)
const isChatRoute = computed(() => currentPath.value.startsWith('/chat'))
const isKnowledgeRoute = computed(() => currentPath.value.startsWith('/knowledge-bases'))
const chatSidebarOpen = ref(true)
const clearAllConversationsOpen = ref(false)
const pendingDeleteConversation = ref<{ id: string; title: string } | null>(null)
const hasSidebarConversations = computed(() => chatStore.conversations.length > 0)
const isChatPath = (path: string) => path.startsWith('/chat')

const primaryMenuItems = [
  { name: '智能问答', path: '/chat', icon: 'M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z' },
]

const adminMenuItems = [
  { name: '知识资源', path: '/knowledge-bases', icon: 'M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10' },
  { name: '任务观测', path: '/tasks', icon: 'M9 17v-6m4 6V7m4 10v-4M5 21h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v14a2 2 0 002 2z' }
]

const isMenuItemActive = (path: string) => {
  if (path === '/knowledge-bases') {
    return isKnowledgeRoute.value
  }
  return currentPath.value.startsWith(path)
}

const syncChatSidebar = async (force = false, restoreStoredSelection = false) => {
  if (!force && chatStore.conversations.length > 0) return
  await chatStore.hydrateConversation({ restoreStoredSelection })
}

const handleSidebarConversationSelect = async (conversationId: string) => {
  if (!isChatRoute.value) {
    await router.push('/chat')
  }

  await chatStore.selectConversation(conversationId)
}

const handleSidebarNewChat = async () => {
  if (!isChatRoute.value) {
    await router.push('/chat')
  }

  await chatStore.startNewConversation()
}

const handleSidebarClearAll = async () => {
  if (!hasSidebarConversations.value || chatStore.loading) return
  clearAllConversationsOpen.value = true
}

const handleSidebarDeleteConversation = (conversationId: string, title: string) => {
  if (chatStore.loading) return
  pendingDeleteConversation.value = { id: conversationId, title }
}

const confirmSidebarClearAll = async () => {
  await chatStore.clearAllConversations()
  clearAllConversationsOpen.value = false
}

const confirmSidebarDeleteConversation = async () => {
  const target = pendingDeleteConversation.value
  if (!target) return
  await chatStore.deleteConversation(target.id)
  pendingDeleteConversation.value = null
}

const formatConversationTime = (value: string) =>
  new Date(value).toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })

onMounted(async () => {
  if (kbStore.knowledgeBases.length === 0) {
    await kbStore.fetchKBs()
  }

  await syncChatSidebar(false, true)
})
</script>

<template>
  <div class="app-layout">
    <!-- Sidebar -->
    <aside class="sidebar">
      <div class="logo">
        <img src="../../assets/ReguRagLogo.svg" alt="ReguRAG" class="logo-image-icon" />
        <span class="logo-text">ReguRAG</span>
      </div>
      <div class="sidebar-main">
        <nav class="nav-menu">
          <router-link
            v-for="item in primaryMenuItems"
            :key="item.path"
            :to="item.path"
            class="nav-item"
            :class="{ active: isMenuItemActive(item.path) }"
          >
            <svg class="icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" :d="item.icon" />
            </svg>
            {{ item.name }}
          </router-link>

          <div class="nav-group">
            <span class="nav-group__label">管理</span>
            <router-link
              v-for="item in adminMenuItems"
              :key="item.path"
              :to="item.path"
              class="nav-item nav-item--admin"
              :class="{ active: isMenuItemActive(item.path) }"
            >
              <svg class="icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" :d="item.icon" />
              </svg>
              {{ item.name }}
            </router-link>
          </div>
        </nav>

        <section class="chat-history-panel">
          <div class="chat-history-panel__header">
            <button class="chat-history-panel__toggle" @click="chatSidebarOpen = !chatSidebarOpen">
              <span>会话列表</span>
              <svg class="chat-history-panel__toggle-icon" :class="{ collapsed: !chatSidebarOpen }" viewBox="0 0 20 20" fill="none" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M6 8l4 4 4-4" />
              </svg>
            </button>
            <div class="chat-history-panel__actions">
              <button class="history-action-btn" :disabled="chatStore.loading" @click="handleSidebarNewChat">
                新对话
              </button>
              <button
                class="history-action-btn history-action-btn--danger"
                :disabled="!hasSidebarConversations || chatStore.loading"
                @click="handleSidebarClearAll"
              >
                清空
              </button>
            </div>
          </div>
          <div v-if="chatSidebarOpen" class="chat-history-panel__body">
            <div v-if="!hasSidebarConversations" class="chat-history-panel__empty">
              暂无历史会话
            </div>
            <div v-else class="chat-history-list">
              <div
                v-for="conversation in chatStore.conversations"
                :key="conversation.id"
                class="chat-history-item"
                :class="{ active: conversation.id === chatStore.currentConversationId }"
              >
                <button class="chat-history-item__main" @click="handleSidebarConversationSelect(conversation.id)">
                  <span class="chat-history-item__title">{{ conversation.title }}</span>
                  <span class="chat-history-item__time">{{ formatConversationTime(conversation.updated_at) }}</span>
                </button>
                <button
                  class="chat-history-item__delete"
                  :disabled="chatStore.loading"
                  aria-label="删除会话"
                  title="删除会话"
                  @click.stop="handleSidebarDeleteConversation(conversation.id, conversation.title)"
                >
                  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.6" d="M7.5 4.5h5m-7 2h9m-7.5 0v7m3-7v7m3-7v7M6.5 6.5l.45 7.2a1 1 0 001 .94h3.1a1 1 0 001-.94l.45-7.2m-5.7-2L7.7 3.4A1 1 0 018.67 2.7h2.66a1 1 0 01.97.7l.4 1.1" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </section>
      </div>
      <div class="sidebar-footer">
        <HealthStatus />
        <div class="version">ReguRAG v1.0.0</div>
      </div>
    </aside>

    <!-- Main Content -->
    <main class="main-container">
      <header class="header">
        <div class="header-main">
          <h1 class="page-title">{{ route.meta.title }}</h1>
        </div>
      </header>
      <div class="content">
        <router-view v-slot="{ Component, route: viewRoute }">
          <transition name="fade" mode="out-in">
            <div
              :key="String(viewRoute.name || viewRoute.path)"
              class="view-page"
              :class="{
                'view-page--chat': isChatPath(viewRoute.path),
                'view-page--padded': !isChatPath(viewRoute.path),
              }"
            >
              <component :is="Component" />
            </div>
          </transition>
        </router-view>
      </div>
      <ConfirmDialog
        :open="clearAllConversationsOpen"
        title="清空全部会话"
        message="这会删除全部历史会话和对应消息记录，且无法恢复。"
        confirm-text="全部清空"
        cancel-text="取消"
        tone="danger"
        :loading="chatStore.loading"
        @close="clearAllConversationsOpen = false"
        @confirm="confirmSidebarClearAll"
      />
      <ConfirmDialog
        :open="pendingDeleteConversation !== null"
        title="删除会话"
        :message="`这会删除“${pendingDeleteConversation?.title || ''}”及其对应消息记录，且无法恢复。`"
        confirm-text="删除"
        cancel-text="取消"
        tone="danger"
        :loading="chatStore.loading"
        @close="pendingDeleteConversation = null"
        @confirm="confirmSidebarDeleteConversation"
      />
    </main>
  </div>
</template>

<style scoped>
.app-layout { display: flex; height: 100vh; }
.sidebar {
  width: var(--sidebar-width); background-color: #1e293b; color: white;
  display: flex; flex-direction: column; flex-shrink: 0;
}
.logo {
  height: var(--header-height); display: flex; align-items: center;
  padding: 0 20px; gap: 12px; border-bottom: 1px solid #334155;
}
.logo-image-icon { width: 32px; height: 32px; object-fit: contain; border-radius: 4px; }
.logo-text { font-size: 1.25rem; font-weight: 600; letter-spacing: -0.025em; color: white; }

.sidebar-main {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.nav-menu {
  padding: 24px 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.nav-group {
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px solid #334155;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.nav-group__label {
  padding: 0 12px 6px;
  color: #64748b;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.nav-item {
  display: flex; align-items: center; padding: 10px 12px; border-radius: var(--radius);
  color: #94a3b8; text-decoration: none; transition: all 0.2s; gap: 12px; font-size: 0.95rem;
}
.nav-item--admin {
  font-size: 0.9rem;
}
.nav-item:hover { background-color: #334155; color: white; }
.nav-item.active { background-color: var(--primary-color); color: white; }
.icon { width: 20px; height: 20px; }

.chat-history-panel {
  flex: 1;
  min-height: 0;
  margin: 6px 12px 12px;
  padding-top: 12px;
  border-top: 1px solid #334155;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chat-history-panel__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.chat-history-panel__toggle {
  min-width: 0;
  min-height: 36px;
  padding: 0 8px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: #cbd5e1;
  font-size: 0.84rem;
  font-weight: 600;
  background: transparent;
}

.chat-history-panel__toggle:hover {
  color: white;
}

.chat-history-panel__toggle-icon {
  width: 16px;
  height: 16px;
  transition: transform 0.2s ease;
}

.chat-history-panel__toggle-icon.collapsed {
  transform: rotate(-90deg);
}

.chat-history-panel__body {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chat-history-panel__actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}

.history-action-btn {
  min-height: 30px;
  padding: 0 10px;
  border-radius: 999px;
  background-color: #334155;
  color: #e2e8f0;
  font-size: 0.75rem;
}

.history-action-btn:hover:not(:disabled) {
  background-color: #475569;
  color: white;
}

.history-action-btn--danger {
  background-color: rgba(127, 29, 29, 0.35);
  color: #fecaca;
}

.history-action-btn--danger:hover:not(:disabled) {
  background-color: rgba(153, 27, 27, 0.55);
}

.chat-history-panel__empty {
  padding: 8px 12px;
  color: #94a3b8;
  font-size: 0.76rem;
  line-height: 1.5;
}

.chat-history-list {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.chat-history-item {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 6px;
  border-radius: 12px;
  color: #cbd5e1;
  background-color: transparent;
}

.chat-history-item:hover {
  background-color: #334155;
  color: white;
}

.chat-history-item.active {
  background-color: rgba(37, 99, 235, 0.18);
  color: white;
}

.chat-history-item__main {
  flex: 1;
  min-width: 0;
  padding: 10px 0 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  text-align: left;
  color: inherit;
  background: transparent;
}

.chat-history-item__title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 0.8rem;
  font-weight: 600;
}

.chat-history-item__time {
  font-size: 0.7rem;
  color: #94a3b8;
}

.chat-history-item__delete {
  width: 30px;
  height: 30px;
  margin-right: 8px;
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  color: #94a3b8;
  background: transparent;
  opacity: 0;
  transition: background-color 0.2s ease, color 0.2s ease, opacity 0.2s ease;
}

.chat-history-item:hover .chat-history-item__delete,
.chat-history-item.active .chat-history-item__delete {
  opacity: 1;
}

.chat-history-item__delete:hover:not(:disabled) {
  background-color: rgba(127, 29, 29, 0.45);
  color: #fecaca;
}

.chat-history-item__delete svg {
  width: 16px;
  height: 16px;
}

.sidebar-footer { padding: 16px 24px; border-top: 1px solid #334155; display: flex; flex-direction: column; gap: 4px; }
.version { font-size: 0.7rem; color: #64748b; }

.main-container { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.header {
  height: var(--header-height); background-color: white; border-bottom: 1px solid var(--border-color);
  display: flex; align-items: center; padding: 0 32px; flex-shrink: 0;
}
.header-main {
  width: 100%;
  min-width: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.page-title { font-size: 1.125rem; font-weight: 600; }
.content { flex: 1; overflow-y: auto; padding: 0; }
.view-page { min-height: 100%; }
.view-page--padded { padding: 32px; }
.view-page--chat {
  height: 100%;
  min-height: 0;
  overflow: hidden;
}

@media (max-width: 720px) {
  .header {
    padding: 0 16px;
  }

  .view-page--padded {
    padding: 20px;
  }
}

.fade-enter-active, .fade-leave-active { transition: opacity 0.2s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>
