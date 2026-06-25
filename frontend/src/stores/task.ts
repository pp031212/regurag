import { defineStore } from 'pinia'
import { taskApi, type Task } from '../api/tasks'

// 轻量任务缓存：文档页只需要跟踪 pending/running 任务，观测页会直接拉完整列表。
export const useTaskStore = defineStore('task', {
  state: () => ({
    activeTasks: new Map<string, Task>(),
    pollingIntervals: new Map<string, any>()
  }),
  
  actions: {
    upsertTask(task: Task) {
      this.activeTasks.set(task.id, task)
    },

    addTask(task: Task) {
      this.upsertTask(task)
      this.startPolling(task.id)
    },

    async hydrateActiveTasks(options?: { knowledgeBaseId?: string }) {
      // 页面刷新或切换知识库后，从后端恢复仍在排队/运行的任务。
      const knowledgeBaseId = options?.knowledgeBaseId
      const [pendingResponse, runningResponse] = await Promise.all([
        taskApi.list({
          knowledge_base_id: knowledgeBaseId,
          status: 'pending',
          limit: 100,
        }),
        taskApi.list({
          knowledge_base_id: knowledgeBaseId,
          status: 'running',
          limit: 100,
        }),
      ])

      const liveTasks = [...pendingResponse.items, ...runningResponse.items]
      const liveTaskIds = new Set(liveTasks.map((task) => task.id))

      for (const task of liveTasks) {
        this.upsertTask(task)
        this.startPolling(task.id)
      }

      for (const [taskId, task] of this.activeTasks.entries()) {
        // 如果后端当前 active 列表里没有这个任务，说明它已经完成/失败或不在当前范围。
        const matchesScope = knowledgeBaseId ? task.knowledge_base_id === knowledgeBaseId : true
        const isActive = task.status === 'pending' || task.status === 'running'
        if (matchesScope && isActive && !liveTaskIds.has(taskId)) {
          this.stopPolling(taskId)
          this.activeTasks.delete(taskId)
        }
      }
    },
    
    async startPolling(taskId: string) {
      // 每个任务只开一个轮询器，任务终态后立即停止。
      if (this.pollingIntervals.has(taskId)) return
      
      const interval = setInterval(async () => {
        try {
          const updatedTask = await taskApi.get(taskId)
          this.upsertTask(updatedTask)
          
          if (updatedTask.status === 'completed' || updatedTask.status === 'failed') {
            this.stopPolling(taskId)
          }
        } catch (err) {
          console.error(`Polling task ${taskId} failed:`, err)
          this.stopPolling(taskId)
        }
      }, 3000) // 每 3 秒轮询一次任务状态
      
      this.pollingIntervals.set(taskId, interval)
    },
    
    stopPolling(taskId: string) {
      const interval = this.pollingIntervals.get(taskId)
      if (interval) {
        clearInterval(interval)
        this.pollingIntervals.delete(taskId)
      }
    }
  }
})
