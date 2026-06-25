import client from './client'

// 知识库接口维护元数据和全库重建入口，文档级操作放在 documents.ts。
export interface KnowledgeBase {
  id: string
  name: string
  description: string
  subject: string
  domain: 'general' | 'training_management' | 'labor_law' | string
  status: 'empty' | 'indexing' | 'ready' | 'failed'
  is_default?: boolean
  created_at: string
  updated_at: string
}

export interface KBListResponse {
  items: KnowledgeBase[]
  total: number
}

export interface KnowledgeBaseDomainOption {
  value: string
  label: string
  description: string
}

export interface KnowledgeBaseDomainOptionsResponse {
  items: KnowledgeBaseDomainOption[]
  default_domain: string
}

export const kbApi = {
  list: (): Promise<KBListResponse> => client.get('/knowledge-bases'),
  listDomains: (): Promise<KnowledgeBaseDomainOptionsResponse> => client.get('/knowledge-bases/domains'),
  get: (id: string): Promise<KnowledgeBase> => client.get(`/knowledge-bases/${id}`),
  create: (data: Partial<KnowledgeBase>): Promise<KnowledgeBase> => client.post('/knowledge-bases', data),
  delete: (id: string): Promise<{ id: string, deleted: boolean }> => client.delete(`/knowledge-bases/${id}`),
  // 全库重建仍然可以指定文档子集，方便前端复用同一个任务入口。
  rebuild: (id: string, document_ids: string[] = []): Promise<any> => client.post(`/knowledge-bases/${id}/rebuild`, { document_ids })
}
