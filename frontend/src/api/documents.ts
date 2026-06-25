import client from './client'
import type { Task } from './tasks'

// 文档接口只负责上传、列表、删除和触发入库任务；任务执行状态走 taskApi 查询。
export interface Document {
  id: string
  knowledge_base_id: string
  filename: string
  content_type: string
  file_size: number
  status: 'uploaded' | 'indexing' | 'ready' | 'failed'
  created_at: string
  updated_at: string
}

export interface DocumentListResponse {
  items: Document[]
  total: number
}

export const docApi = {
  upload: (kbId: string, file: File): Promise<Document> => {
    // 上传使用 multipart，知识库 ID 和文件一起提交给后端。
    const formData = new FormData()
    formData.append('knowledge_base_id', kbId)
    formData.append('file', file)
    return client.post('/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
  },
  listByKB: (kbId: string): Promise<DocumentListResponse> => client.get(`/knowledge-bases/${kbId}/documents`),
  // ingest/rebuild 都返回后台任务，页面需要继续轮询任务状态。
  ingest: (kbId: string, document_ids: string[] = []): Promise<Task> => client.post(`/knowledge-bases/${kbId}/ingest`, { document_ids }),
  rebuild: (documentId: string): Promise<Task> => client.post(`/documents/${documentId}/rebuild`),
  delete: (documentId: string): Promise<{ id: string; deleted: boolean }> => client.delete(`/documents/${documentId}`)
}
