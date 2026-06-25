# ReguRAG 前端

ReguRAG 前端是基于 Vue 3、TypeScript、Pinia 和 Vite 的知识库问答界面。当前页面围绕“智能问答 + 知识资源 + 任务观测”组织，用于完成文档上传、导入任务观测、知识库切换、聊天问答、引用查看和调试信息查看。

## 技术栈

- Vue 3
- TypeScript
- Pinia
- Vue Router
- Vite
- Axios
- Markdown-it

## 本地开发

```bash
npm install
npm run dev
```

默认开发服务由 Vite 启动。前端 API 客户端默认访问 `/api`，本地开发时通过 Vite proxy 转发到后端。

## 生产构建

```bash
npm run build
```

构建命令会先运行 `vue-tsc -b` 做类型检查，再由 Vite 输出静态产物到 `dist/`。

## Docker 运行

项目根目录的 `docker-compose.yml` 已包含 `frontend` 服务。默认启动整套本机 CPU / MySQL 开发线：

```bash
docker compose up --build
```

前端容器默认通过 `http://localhost:8080` 暴露。

## 当前页面

- `智能问答`：聊天主界面，支持会话切换、新会话、引用抽屉、调试抽屉和详细 chunk 阶段信息。
- `知识资源`：知识库列表、知识库详情、文档上传、文档导入、单文档导入/重试/重建索引。
- `任务观测`：任务状态概览、任务事件时间线、告警与知识库趋势。

## 关键目录

- `src/api/`：后端 API 客户端。
- `src/stores/`：Pinia 状态管理。
- `src/views/`：页面级视图。
- `src/components/chat/`：聊天消息、引用抽屉、调试抽屉。
- `src/components/kb/`：知识库和文档上传组件。
- `src/components/tasks/`：任务观测组件。

## 运行前提

前端需要可访问的后端 API。默认 Docker Compose 会同时启动 `backend-api`；本地单独启动前端时，请先确认后端服务可用。
