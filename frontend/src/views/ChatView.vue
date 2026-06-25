<script setup lang="ts">
// 聊天页只负责交互和展示编排：会话状态、滚动定位、引用/调试抽屉都交给 store 或子组件处理。
import { computed, ref, onMounted, nextTick, watch } from 'vue'
import { useKBStore } from '../stores/kb'
import { useChatStore, type Message } from '../stores/chat'
import { useChatScroll } from '../composables/useChatScroll'
import ChatMessage from '../components/chat/ChatMessage.vue'
import CitationDrawer from '../components/chat/CitationDrawer.vue'
import DebugDrawer from '../components/chat/DebugDrawer.vue'
import RunningStatusIcon from '../components/common/RunningStatusIcon.vue'

const isDev = import.meta.env.DEV
const kbStore = useKBStore()
const chatStore = useChatStore()

const inputQuery = ref('')
const chatHistoryRef = ref<HTMLElement | null>(null)
const composerTextareaRef = ref<HTMLTextAreaElement | null>(null)
const { scrollToBottom } = useChatScroll(chatHistoryRef)

const selectedMessage = ref<Message | null>(null)
const drawerOpen = ref(false)
const debugOpen = ref(false)
const activeNavMessageId = ref<string | null>(null)
const navExpanded = ref(false)
const navPanelRef = ref<HTMLElement | null>(null)
const showScrollToBottom = ref(false)

const currentConversation = computed(() =>
  chatStore.currentConversationId
    ? chatStore.conversations.find((conversation) => conversation.id === chatStore.currentConversationId) || null
    : null,
)

const latestAssistantMessage = computed(() =>
  [...chatStore.messages].reverse().find((message) => message.role === 'assistant') || null,
)

const hasStreamingAssistant = computed(() =>
  chatStore.messages.some((message) => message.role === 'assistant' && message.streaming),
)

const latestResolvedKnowledgeBaseName = computed(
  () =>
    latestAssistantMessage.value?.debug?.resolved_knowledge_base_name ||
    latestAssistantMessage.value?.debug?.resolved_knowledge_base_id ||
    kbStore.currentKB?.name ||
    '自动选择',
)

const latestSourceCaption = computed(() => {
  // 这段说明只在开发态状态条里展示，用来帮助判断本轮回答走了哪条后端分支。
  const debug = latestAssistantMessage.value?.debug
  if (!debug) {
    return '当前会话独立存在，知识库只作为本轮问答的资源来源。'
  }
  if (debug.cross_domain_guard_applied) {
    return '这轮消息被识别为跨域复合问题，系统先转为弱澄清，不直接进入单库检索。'
  }
  if (debug.answer_guard_applied) {
    return '这轮消息命中了保守回答守卫，说明限定场景没有被检索证据充分覆盖。'
  }
  if (debug.auto_routed) {
    return '这轮消息已自动切换到更合适的知识库，当前选中知识库只是资源偏好，不再等于会话本身。'
  }
  return '当前仍可手动选择知识来源，但会话本身不会随知识库切换而重置。'
})

const latestStatePills = computed(() => {
  const debug = latestAssistantMessage.value?.debug
  if (!debug) return []

  const pills: Array<{ label: string; tone: 'neutral' | 'route' | 'guard' }> = []
  if (debug.auto_routed) {
    pills.push({ label: '自动路由', tone: 'route' })
  }
  if (debug.cross_domain_guard_applied) {
    pills.push({ label: '弱澄清', tone: 'guard' })
  }
  if (debug.answer_guard_applied) {
    pills.push({ label: '保守回答', tone: 'guard' })
  }
  return pills
})

const conversationSections = computed(() => {
  // 后端消息是线性列表；页面按“一次用户提问 + 一次助手回答”重组，方便侧边导航定位。
  const sections: Array<{
    id: string
    userMessage: Message
    assistantMessage: Message | null
    index: number
    title: string
  }> = []

  let pendingUser: Message | null = null

  for (const message of chatStore.messages) {
    if (message.role === 'user') {
      if (pendingUser) {
        sections.push({
          id: `section-${pendingUser.id}`,
          userMessage: pendingUser,
          assistantMessage: null,
          index: sections.length + 1,
          title: pendingUser.content.replace(/\s+/g, ' ').trim() || `问题 ${sections.length + 1}`,
        })
      }
      pendingUser = message
      continue
    }

    if (pendingUser) {
      sections.push({
        id: `section-${pendingUser.id}`,
        userMessage: pendingUser,
        assistantMessage: message,
        index: sections.length + 1,
        title: pendingUser.content.replace(/\s+/g, ' ').trim() || `问题 ${sections.length + 1}`,
      })
      pendingUser = null
      continue
    }
  }

  if (pendingUser) {
    sections.push({
      id: `section-${pendingUser.id}`,
      userMessage: pendingUser,
      assistantMessage: null,
      index: sections.length + 1,
      title: pendingUser.content.replace(/\s+/g, ' ').trim() || `问题 ${sections.length + 1}`,
    })
  }

  return sections
})

const conversationNavVisibleCount = computed(() => Math.min(conversationSections.value.length, 10))
const conversationNavPanelHeight = computed(() => `${conversationNavVisibleCount.value * 36 + 30}px`)

const updateScrollToBottomVisibility = () => {
  const container = chatHistoryRef.value
  if (!container) {
    showScrollToBottom.value = false
    return
  }

  const remainingDistance = container.scrollHeight - container.scrollTop - container.clientHeight
  showScrollToBottom.value = remainingDistance > 96
}

const updateActiveConversationNav = () => {
  // 根据滚动容器中最接近视口中心的 turn，更新左侧问题导航的高亮项。
  const container = chatHistoryRef.value
  if (!container || conversationSections.value.length === 0) {
    activeNavMessageId.value = null
    return
  }

  const containerRect = container.getBoundingClientRect()
  const viewportCenter = containerRect.top + container.clientHeight / 2
  let activeId = conversationSections.value[0]?.id || null
  let closestDistance = Number.POSITIVE_INFINITY

  for (const section of conversationSections.value) {
    const element = document.getElementById(section.id)
    if (!element) continue
    const rect = element.getBoundingClientRect()
    const sectionCenter = rect.top + rect.height / 2
    const distance = Math.abs(sectionCenter - viewportCenter)
    if (distance < closestDistance) {
      closestDistance = distance
      activeId = section.id
    }
  }

  activeNavMessageId.value = activeId
}

const handleChatHistoryScroll = () => {
  updateActiveConversationNav()
  updateScrollToBottomVisibility()
}

const updateNavScrollbar = () => {
  const panel = navPanelRef.value
  if (!panel) return
}

onMounted(async () => {
  // 页面首次进入时恢复上次会话选择，再同步滚动和输入框高度。
  if (kbStore.knowledgeBases.length === 0) {
    await kbStore.fetchKBs()
  }
  await chatStore.hydrateConversation({ restoreStoredSelection: true })

  await nextTick()
  updateActiveConversationNav()
  updateNavScrollbar()
  syncComposerHeight()
  scrollToBottom()
  updateScrollToBottomVisibility()
})

watch(
  () => chatStore.messages.length,
  async () => {
    // 新消息进入列表后再滚动，避免 DOM 尚未渲染时计算高度不准。
    await nextTick()
    updateActiveConversationNav()
    updateNavScrollbar()
    scrollToBottom()
    updateScrollToBottomVisibility()
  },
  { flush: 'post' }
)

watch(
  () => `${chatStore.messages.at(-1)?.id || ''}:${chatStore.messages.at(-1)?.content.length || 0}`,
  async () => {
    // 流式 token 会持续增长最后一条消息，加载中才自动贴底，避免用户向上阅读时被打断。
    await nextTick()
    if (chatStore.loading) {
      scrollToBottom()
    }
    updateScrollToBottomVisibility()
  },
  { flush: 'post' }
)

watch(navExpanded, async () => {
  await nextTick()
})

watch(inputQuery, async () => {
  await nextTick()
  syncComposerHeight()
})

const syncComposerHeight = () => {
  // 输入框高度随内容增长，但限制最大高度，超过后内部滚动。
  const textarea = composerTextareaRef.value
  if (!textarea) return

  textarea.style.height = '0px'
  const nextHeight = Math.min(textarea.scrollHeight, 200)
  textarea.style.height = `${Math.max(nextHeight, 28)}px`
  textarea.style.overflowY = textarea.scrollHeight > 200 ? 'auto' : 'hidden'
}

const handleSend = async () => {
  if (!inputQuery.value.trim() || !kbStore.currentKB || chatStore.loading) return

  // 先清空输入框并立刻恢复高度，后端流式响应期间输入区不被旧内容撑开。
  const query = inputQuery.value
  inputQuery.value = ''
  await nextTick()
  syncComposerHeight()

  const sendPromise = chatStore.sendMessage(kbStore.currentKB.id, query)
  // 发送后先滚到底部，后续 token 进入时由 watcher 继续贴底。
  scrollToBottom()

  await nextTick()
  scrollToBottom()

  await sendPromise
  scrollToBottom()
}

const openCitations = (msg: Message) => {
  selectedMessage.value = msg
  drawerOpen.value = true
  debugOpen.value = false
}

const openDebug = (msg: Message) => {
  if (!isDev) return
  selectedMessage.value = msg
  debugOpen.value = true
  drawerOpen.value = false
}

const scrollToConversationItem = async (sectionId: string) => {
  // 导航点击使用容器内 offsetTop，而不是 window 滚动，因为聊天区域是独立滚动容器。
  await nextTick()
  const container = chatHistoryRef.value
  const target = document.getElementById(sectionId)
  if (!container || !target) return

  activeNavMessageId.value = sectionId
  container.scrollTo({
    top: Math.max(target.offsetTop - 24, 0),
    behavior: 'smooth',
  })
}

const scrollComposerToBottom = () => {
  const container = chatHistoryRef.value
  if (!container) return

  container.scrollTo({
    top: container.scrollHeight,
    behavior: 'smooth',
  })

  requestAnimationFrame(() => {
    updateScrollToBottomVisibility()
  })
}
</script>

<template>
  <div class="chat-view-container">
    <div class="chat-workspace">
      <section class="chat-main">
        <aside
          v-if="conversationSections.length > 1"
          class="conversation-nav-rail"
          :class="{ 'conversation-nav-rail--expanded': navExpanded }"
          :style="{ height: conversationNavPanelHeight }"
          aria-label="会话导航"
          @mouseenter="navExpanded = true"
          @mouseleave="navExpanded = false"
        >
          <div
            ref="navPanelRef"
            class="conversation-nav-rail__panel"
            :class="{ 'conversation-nav-rail__panel--expanded': navExpanded }"
            :style="{ maxHeight: conversationNavPanelHeight }"
            @scroll="updateNavScrollbar"
          >
            <button
              v-for="section in conversationSections"
              :key="section.id"
              class="conversation-nav-rail__item"
              :class="{ 'conversation-nav-rail__item--active': activeNavMessageId === section.id }"
              :aria-label="`跳转到第 ${section.index} 次提问`"
              @click="scrollToConversationItem(section.id)"
            >
              <span class="conversation-nav-rail__text" :class="{ 'conversation-nav-rail__text--expanded': navExpanded }">
                {{ section.title }}
              </span>
              <span class="conversation-nav-rail__line" :class="{ 'conversation-nav-rail__line--active': activeNavMessageId === section.id }"></span>
            </button>
          </div>
        </aside>

        <div class="chat-history" ref="chatHistoryRef" @scroll="handleChatHistoryScroll">
          <section v-if="isDev" class="conversation-status-bar">
            <div class="conversation-status-bar__title">
              <span class="conversation-status-bar__label">当前会话</span>
              <strong>{{ currentConversation?.title || '新对话' }}</strong>
            </div>
            <div class="conversation-status-bar__meta">
              <span class="status-pill status-pill--neutral">来源：{{ latestResolvedKnowledgeBaseName }}</span>
              <span v-if="latestStatePills.length === 0" class="status-pill status-pill--neutral">直接问答</span>
              <span
                v-for="pill in latestStatePills"
                :key="pill.label"
                class="status-pill"
                :class="`status-pill--${pill.tone}`"
              >
                {{ pill.label }}
              </span>
            </div>
            <p class="conversation-status-bar__hint">{{ latestSourceCaption }}</p>
          </section>

          <div v-if="chatStore.messages.length === 0" class="welcome">
            <div class="welcome-card">
              <span class="welcome-card__eyebrow">ReguRAG</span>
              <h2>你好，我是 ReguRAG 助手</h2>
              <p>你可以直接提问；会话会保留，系统会按需使用“{{ kbStore.currentKB?.subject || '知识资源' }}”或自动切换到更合适的知识库。</p>
            </div>
          </div>

          <section
            v-for="section in conversationSections"
            :id="section.id"
            :key="section.id"
            class="conversation-turn"
          >
            <ChatMessage
              :id="`message-${section.userMessage.id}`"
              :msg="section.userMessage"
              @open-citations="openCitations(section.userMessage)"
              @open-debug="openDebug(section.userMessage)"
            />
            <ChatMessage
              v-if="section.assistantMessage"
              :id="`message-${section.assistantMessage.id}`"
              :msg="section.assistantMessage"
              @open-citations="openCitations(section.assistantMessage)"
              @open-debug="openDebug(section.assistantMessage)"
            />
          </section>

          <div v-if="chatStore.loading && !hasStreamingAssistant" class="loading-message assistant loading">
            <div class="avatar bot-avatar pulse">
              <RunningStatusIcon :visible="true" :speed="200" :size="22" color="rgba(37, 99, 235, 0.92)" />
            </div>
            <div class="message-content">
              <div class="bubble skeleton-bubble">
                <div class="skeleton-line short"></div>
                <div class="skeleton-line medium"></div>
                <div class="skeleton-line long"></div>
              </div>
            </div>
          </div>
        </div>

        <div class="input-area">
          <div class="input-shell">
            <button
              v-if="showScrollToBottom"
              class="scroll-to-bottom-btn"
              type="button"
              aria-label="回到底部"
              @click="scrollComposerToBottom"
            >
              <svg viewBox="0 0 14 14" fill="none" aria-hidden="true">
                <path
                  d="M11.8486 5.5L11.4238 5.92383L8.69727 8.65137C8.44157 8.90706 8.21562 9.13382 8.01172 9.29785C7.79912 9.46883 7.55595 9.61756 7.25 9.66602C7.08435 9.69222 6.91565 9.69222 6.75 9.66602C6.44405 9.61756 6.20088 9.46883 5.98828 9.29785C5.78438 9.13382 5.55843 8.90706 5.30273 8.65137L2.57617 5.92383L2.15137 5.5L3 4.65137L3.42383 5.07617L6.15137 7.80273C6.42595 8.07732 6.59876 8.24849 6.74023 8.3623C6.87291 8.46904 6.92272 8.47813 6.9375 8.48047C6.97895 8.48703 7.02105 8.48703 7.0625 8.48047C7.07728 8.47813 7.12709 8.46904 7.25977 8.3623C7.40124 8.24849 7.57405 8.07732 7.84863 7.80273L10.5762 5.07617L11 4.65137L11.8486 5.5Z"
                  fill="currentColor"
                />
              </svg>
            </button>

            <div class="input-container">
              <textarea
                ref="composerTextareaRef"
                v-model="inputQuery"
                placeholder="输入你的问题..."
                @input="syncComposerHeight"
                @keydown.enter.exact.prevent="handleSend"
                rows="1"
              ></textarea>
              <button class="send-btn" :disabled="!inputQuery.trim() || chatStore.loading" @click="handleSend">
                <svg fill="currentColor" viewBox="0 0 20 20"><path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z"></path></svg>
              </button>
            </div>
          </div>
          <p class="input-tip">按 Enter 发送，Shift + Enter 换行</p>
        </div>
      </section>
    </div>

    <CitationDrawer :open="drawerOpen" :citations="selectedMessage?.citations || []" @close="drawerOpen = false" />
    <DebugDrawer v-if="isDev" :open="debugOpen" :debugData="selectedMessage?.debug || null" @close="debugOpen = false" />
  </div>
</template>

<style scoped>
.chat-view-container {
  height: 100%;
  min-height: 0;
  display: flex;
  flex-direction: column;
  position: relative;
  background:
    radial-gradient(circle at top, rgba(239, 246, 255, 0.88), transparent 36%),
    linear-gradient(180deg, #f8fafc 0%, #f8fafc 100%);
}

.chat-workspace {
  flex: 1;
  display: flex;
  min-height: 0;
}

.chat-main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  position: relative;
  border: none;
  border-radius: 0;
  background: transparent;
  box-shadow: none;
}

.conversation-nav-rail {
  position: absolute;
  top: 50%;
  right: 16px;
  transform: translateY(-50%);
  z-index: 5;
  width: 34px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  overflow: hidden;
  transition: width 0.2s ease;
}

.conversation-nav-rail--expanded {
  width: 240px;
}

.conversation-nav-rail__panel {
  width: 240px;
  padding: 15px 8px 15px 24px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  position: absolute;
  right: 0;
  overflow-x: hidden;
  overflow-y: auto;
  pointer-events: none;
  border-radius: 16px;
  background: transparent;
  box-shadow: none;
  border: 1px solid transparent;
  scrollbar-width: none;
  -ms-overflow-style: none;
  transition:
    background 0.2s ease,
    box-shadow 0.2s ease,
    border-color 0.2s ease,
    opacity 0.2s ease;
}

.conversation-nav-rail__panel--expanded {
  pointer-events: auto;
  background: rgba(255, 255, 255, 0.96);
  box-shadow:
    rgba(0, 0, 0, 0.2) 0 0 1px 0,
    rgba(0, 0, 0, 0.02) 0 0 4px 0,
    rgba(0, 0, 0, 0.08) 0 12px 32px 0;
  border-color: rgba(226, 232, 240, 0.9);
  backdrop-filter: blur(8px);
  scrollbar-width: thin;
  scrollbar-color: rgba(148, 163, 184, 0.6) transparent;
}

.conversation-nav-rail__item {
  width: 100%;
  min-height: 28px;
  padding: 4px 0;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 10px;
  color: #81858c;
  text-align: right;
}

.conversation-nav-rail__item:hover,
.conversation-nav-rail__item:hover .conversation-nav-rail__text {
  color: #0f1115;
}

.conversation-nav-rail__item--active,
.conversation-nav-rail__item--active .conversation-nav-rail__text {
  color: #0f1115;
}

.conversation-nav-rail__text {
  width: 182px;
  margin-right: 8px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  opacity: 0;
  pointer-events: none;
  font-size: 13px;
  line-height: 20px;
  text-align: right;
  transition: opacity 0.1s ease, color 0.2s ease;
}

.conversation-nav-rail__text--expanded {
  opacity: 1;
}

.conversation-nav-rail__line {
  width: 14px;
  height: 3px;
  border-radius: 999px;
  background: #cbd5e1;
  flex: none;
  transition: background-color 0.2s ease, width 0.2s ease;
}

.conversation-nav-rail__panel::-webkit-scrollbar {
  width: 0;
  height: 0;
}

.conversation-nav-rail__panel--expanded::-webkit-scrollbar {
  width: 6px;
}

.conversation-nav-rail__panel--expanded::-webkit-scrollbar-thumb {
  background: rgba(148, 163, 184, 0.6);
  border-radius: 999px;
}

.conversation-nav-rail__item:hover .conversation-nav-rail__line {
  background: #94a3b8;
}

.conversation-nav-rail__line--active {
  width: 18px;
  background: #2563eb;
}

.chat-history {
  flex: 1;
  overflow-y: auto;
  padding: 24px 0 0;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.conversation-turn {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.conversation-status-bar {
  width: min(860px, calc(100% - 48px));
  margin: 0 auto;
  padding: 16px 18px;
  border-radius: 20px;
  border: 1px solid rgba(226, 232, 240, 0.95);
  background: rgba(255, 255, 255, 0.82);
  backdrop-filter: blur(8px);
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.conversation-status-bar__title {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.conversation-status-bar__label {
  color: #64748b;
  font-size: 0.78rem;
  margin-right: 8px;
}

.conversation-status-bar__title strong {
  color: #0f172a;
  font-size: 1rem;
}

.conversation-status-bar__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.conversation-status-bar__hint {
  margin: 0;
  color: #64748b;
  font-size: 0.84rem;
  line-height: 1.7;
}

.status-pill {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  padding: 0 10px;
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 700;
}

.status-pill--neutral {
  background: #e2e8f0;
  color: #475569;
}

.status-pill--route {
  background: #dbeafe;
  color: #1d4ed8;
}

.status-pill--guard {
  background: #fff7ed;
  color: #c2410c;
}

.welcome {
  width: min(860px, calc(100% - 48px));
  margin: auto auto 0;
  padding: 24px 0 48px;
}

.welcome-card {
  padding: 28px 30px;
  border-radius: 28px;
  border: 1px solid rgba(191, 219, 254, 0.8);
  background:
    radial-gradient(circle at top left, rgba(59, 130, 246, 0.12), transparent 30%),
    linear-gradient(140deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 252, 0.95));
  box-shadow: 0 24px 48px rgba(15, 23, 42, 0.06);
}

.welcome-card__eyebrow {
  display: inline-flex;
  margin-bottom: 12px;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(219, 234, 254, 0.9);
  color: #1d4ed8;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.welcome-card h2 {
  margin: 0 0 10px;
  font-size: clamp(1.8rem, 2.8vw, 2.4rem);
  color: #0f172a;
}

.welcome-card p {
  margin: 0;
  color: #475569;
  line-height: 1.8;
}

.loading-message {
  display: flex;
  gap: 12px;
  width: min(860px, calc(100% - 48px));
  margin: 0 auto 8px;
}

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 8px;
}

.bot-avatar {
  background-color: white;
  border: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: center;
}

.message-content {
  flex: 1;
}

.skeleton-bubble {
  width: min(100%, 720px);
  padding: 14px 16px;
  border-radius: 12px;
  border: 1px solid var(--border-color);
  background: white;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.skeleton-line {
  height: 10px;
  border-radius: 999px;
  background: linear-gradient(90deg, #e2e8f0 0%, #f8fafc 50%, #e2e8f0 100%);
  background-size: 200% 100%;
  animation: skeleton-wave 1.2s ease-in-out infinite;
}

.skeleton-line.short { width: 34%; }
.skeleton-line.medium { width: 62%; }
.skeleton-line.long { width: 78%; }

.input-area {
  padding: 18px 24px 24px;
  background: linear-gradient(180deg, rgba(248, 250, 252, 0) 0%, #f8fafc 24%);
}

.input-shell {
  width: min(860px, calc(100% - 48px));
  margin: 0 auto;
  position: relative;
}

.scroll-to-bottom-btn {
  position: absolute;
  right: 40px;
  bottom: 110px;
  width: 34px;
  height: 34px;
  padding: 0;
  border-radius: 999px;
  border: 1px solid rgba(226, 232, 240, 0.96);
  background: rgba(255, 255, 255, 0.96);
  color: #475569;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  box-shadow:
    rgba(0, 0, 0, 0.18) 0 0 1px 0,
    rgba(0, 0, 0, 0.02) 0 0 4px 0,
    rgba(0, 0, 0, 0.08) 0 10px 24px 0;
  backdrop-filter: blur(8px);
  z-index: 4;
}

.scroll-to-bottom-btn:hover {
  color: #0f172a;
  border-color: rgba(203, 213, 225, 1);
}

.scroll-to-bottom-btn svg {
  width: 14px;
  height: 14px;
}

.input-container {
  padding: 10px 12px;
  display: flex;
  align-items: center;
  gap: 12px;
  border-radius: 24px;
  border: 1px solid rgba(203, 213, 225, 0.96);
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 0 18px 38px rgba(15, 23, 42, 0.08);
}

.input-container textarea {
  flex: 1;
  min-width: 0;
  min-height: 24px;
  max-height: 200px;
  resize: none;
  border: none;
  outline: none;
  background: transparent;
  color: #0f172a;
  font-size: 0.96rem;
  line-height: 1.6;
  font-family: inherit;
  padding: 6px 10px;
}

.input-container textarea::placeholder {
  color: #94a3b8;
  font: inherit;
}

.send-btn {
  width: 36px;
  height: 36px;
  padding: 0;
  border-radius: 12px;
  flex: none;
  background: #2563eb;
  color: white;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 10px 22px rgba(37, 99, 235, 0.18);
}

.send-btn:hover:not(:disabled) {
  background: #1d4ed8;
}

.send-btn:disabled {
  background: #94a3b8;
  box-shadow: none;
  cursor: not-allowed;
}

.send-btn svg {
  width: 16px;
  height: 16px;
}

.input-tip {
  width: min(860px, calc(100% - 48px));
  margin: 10px auto 0;
  color: #94a3b8;
  font-size: 0.76rem;
  text-align: center;
}

@keyframes skeleton-wave {
  0% {
    background-position: 100% 50%;
  }

  100% {
    background-position: 0 50%;
  }
}

@media (max-width: 960px) {
  .conversation-nav-rail {
    display: none;
  }
}

@media (max-width: 640px) {
  .chat-history {
    padding-top: 18px;
  }

  .welcome,
  .loading-message,
  .input-container,
  .input-tip,
  .conversation-status-bar {
    width: calc(100% - 24px);
  }

  .welcome-card {
    padding: 22px 20px;
  }

  .input-area {
    padding: 16px 12px 18px;
  }
}
</style>
