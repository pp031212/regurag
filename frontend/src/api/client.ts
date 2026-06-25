import axios from 'axios'
import type { AxiosError, AxiosResponse } from 'axios'

// API 基础地址优先读显式环境变量，便于 Docker、本机和服务器部署共用同一套前端代码。
export const resolveApiBaseUrl = () => {
  const envBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim()
  if (envBaseUrl) {
    return envBaseUrl.replace(/\/$/, '')
  }

  if (typeof window === 'undefined') {
    return '/api/v1'
  }

  const protocol = import.meta.env.VITE_API_PROTOCOL?.trim() || window.location.protocol
  const hostname = import.meta.env.VITE_API_HOST?.trim() || window.location.hostname
  const port = import.meta.env.VITE_API_PORT?.trim() || '8000'

  return `${protocol}//${hostname}:${port}/api/v1`
}

const client = axios.create({
  baseURL: resolveApiBaseUrl(),
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json'
  }
})

export interface ApiError {
  code: string
  message: string
  details?: Record<string, any>
}

client.interceptors.response.use(
  (response: AxiosResponse) => response.data,
  (error: AxiosError<ApiError>) => {
    // 后端已经按统一错误结构返回时，直接把业务错误抛给页面层处理。
    const apiError = error.response?.data
    if (apiError) {
      console.error(`[API Error] ${apiError.code}: ${apiError.message}`, apiError.details)
      return Promise.reject(apiError)
    }

    // 网络中断、跨域或超时没有后端响应，前端补一个统一兜底错误。
    const fallbackError: ApiError = {
      code: 'UNKNOWN_ERROR',
      message: error.message || '网络请求失败'
    }
    return Promise.reject(fallbackError)
  }
)

export default client
