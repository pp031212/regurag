import { createRouter, createWebHistory } from 'vue-router'
import ChatView from '../views/ChatView.vue'

const routes = [
  {
    path: '/',
    redirect: '/chat'
  },
  {
    path: '/chat',
    name: 'Chat',
    component: ChatView,
    meta: { title: '智能问答' }
  },
  {
    path: '/knowledge-bases',
    name: 'KnowledgeBases',
    component: () => import('../views/KnowledgeBaseView.vue'),
    meta: { title: '知识资源' }
  },
  {
    path: '/knowledge-bases/:id',
    name: 'KnowledgeBaseDetail',
    component: () => import('../views/KnowledgeBaseDetailView.vue'),
    meta: { title: '知识库详情' }
  },
  {
    path: '/tasks',
    name: 'TaskMonitor',
    component: () => import('../views/TaskMonitorView.vue'),
    meta: { title: '任务观测' }
  },
  {
    path: '/documents',
    redirect: '/knowledge-bases'
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to) => {
  document.title = (to.meta.title as string) || 'ReguRAG'
})

export default router
