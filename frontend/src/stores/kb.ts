import { defineStore } from 'pinia'
import { kbApi, type KnowledgeBase, type KnowledgeBaseDomainOption } from '../api/knowledgeBases'

const KB_STORAGE_KEY = 'regurag.current_kb_id'

const readStoredKBId = () => {
  if (typeof window === 'undefined') return null
  return window.localStorage.getItem(KB_STORAGE_KEY)
}

const writeStoredKBId = (kbId: string | null) => {
  if (typeof window === 'undefined') return
  if (kbId) {
    window.localStorage.setItem(KB_STORAGE_KEY, kbId)
    return
  }
  window.localStorage.removeItem(KB_STORAGE_KEY)
}

export const useKBStore = defineStore('kb', {
  state: () => ({
    knowledgeBases: [] as KnowledgeBase[],
    domainOptions: [] as KnowledgeBaseDomainOption[],
    defaultDomain: 'general',
    domainOptionsLoaded: false,
    currentKB: null as KnowledgeBase | null,
    currentKBId: readStoredKBId() as string | null,
    loading: false,
    error: null as string | null
  }),
  
  actions: {
    async fetchDomainOptions({ silent = false }: { silent?: boolean } = {}) {
      try {
        const response = await kbApi.listDomains()
        this.domainOptions = response.items
        this.defaultDomain = response.default_domain || response.items[0]?.value || this.defaultDomain
        this.domainOptionsLoaded = true
        return response
      } catch (err: any) {
        if (!silent) {
          throw err
        }
        return null
      }
    },

    getDomainLabel(domain: string | null | undefined) {
      const value = domain?.trim() || ''
      if (!value) {
        return this.domainOptions.find((item) => item.value === this.defaultDomain)?.label || this.defaultDomain
      }
      return this.domainOptions.find((item) => item.value === value)?.label || value
    },

    async fetchKBs() {
      this.loading = true
      try {
        const response = await kbApi.list()
        this.knowledgeBases = response.items
        await this.fetchDomainOptions({ silent: true })
        const preferredId = this.currentKBId || this.currentKB?.id || null
        const matchedKB = preferredId
          ? this.knowledgeBases.find((kb) => kb.id === preferredId) || null
          : null
        this.currentKB = matchedKB || this.knowledgeBases[0] || null
        this.currentKBId = this.currentKB?.id || null
        writeStoredKBId(this.currentKBId)
      } catch (err: any) {
        this.error = err.message || '获取知识库失败'
      } finally {
        this.loading = false
      }
    },
    
    setCurrentKB(kb: KnowledgeBase) {
      this.currentKB = kb
      this.currentKBId = kb.id
      writeStoredKBId(kb.id)
    },
    
    async createKB(data: Partial<KnowledgeBase>) {
      try {
        const newKB = await kbApi.create(data)
        this.knowledgeBases.push(newKB)
        this.setCurrentKB(newKB)
        return newKB
      } catch (err: any) {
        throw err
      }
    },

    async deleteKB(id: string) {
      try {
        await kbApi.delete(id)
        this.knowledgeBases = this.knowledgeBases.filter(kb => kb.id !== id)
        if (this.currentKB?.id === id) {
          this.currentKB = this.knowledgeBases[0] || null
          this.currentKBId = this.currentKB?.id || null
          writeStoredKBId(this.currentKBId)
        }
      } catch (err: any) {
        throw err
      }
    },

    async rebuildKB(id: string) {
      try {
        return await kbApi.rebuild(id)
      } catch (err: any) {
        throw err
      }
    }
  }
})
