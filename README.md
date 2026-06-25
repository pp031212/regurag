# ReguRAG

ReguRAG 是一个面向中文法规、制度和业务文档的 RAG 知识库问答系统。项目包含 FastAPI 后端、Vue 3 前端、文档入库流水线、任务队列、检索调试、引用展示和 Docker Compose 本地编排，适合用来演示从资料上传到可追溯问答的完整工程闭环。

## 功能概览

- 多知识库管理：创建、查询、删除知识库，按知识库隔离文档与会话。
- 文档接入：支持文档上传、单文档重建、知识库重建索引和后台入库任务。
- 结构化预处理：覆盖 PDF 正文、表格、图片 OCR 等资料处理场景。
- RAG 问答：包含 query rewrite、dense/sparse 检索、MMR、rerank、上下文收敛和最终回答生成。
- 引用与调试：回答返回来源引用，并提供检索、重排、上下文构造等阶段信息。
- 会话能力：保存对话、消息和上下文，支持多轮追问时的问题补全。
- 任务观测：持久化 ingest 任务、任务事件、重试状态、告警和知识库维度趋势。
- 可替换基础设施：默认使用 MySQL + Redis + Chroma，也保留 Milvus 和 PostgreSQL canary 配置。
- 前端工作台：提供智能问答、知识资源、文档上传、知识库详情和任务监控页面。

## 技术栈

后端：

- Python 3.11
- FastAPI
- SQLAlchemy
- MySQL / PostgreSQL driver
- Redis
- ChromaDB / Milvus
- OpenAI-compatible chat API
- Transformers / Torch
- PDF、DOCX、XLSX 解析与 OCR 相关组件

前端：

- Vue 3
- TypeScript
- Pinia
- Vue Router
- Vite
- Axios
- Markdown-it

工程化：

- Docker Compose
- uv
- pytest
- Nginx

## 目录结构

```text
.
├── backend/                 # FastAPI 后端、RAG 主链路、任务队列、数据库模型与测试
│   ├── app/
│   ├── config/
│   ├── data/models/
│   ├── scripts/
│   ├── sql/
│   └── tests/
├── frontend/                # Vue 3 前端工作台
│   ├── src/api/
│   ├── src/components/
│   ├── src/stores/
│   └── src/views/
├── docs/                    # 当前有效的设计说明、阅读地图和流程图
├── docker-compose.yml       # 默认本地开发编排
├── docker-compose.gpu.yml   # GPU 后端覆盖配置
└── docker-compose.*canary.yml
```

仓库不包含本地密钥、虚拟环境、`node_modules`、构建产物、上传文件、向量库数据和评测输出。这些内容由 `.gitignore` 排除。

## 快速启动

先准备根目录环境变量：

```bash
cp .env.example .env
```

将 `.env` 中的模型配置替换为可用的 OpenAI 兼容接口：

```env
OPENAI_API_KEY=replace-me
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini

REWRITE_API_KEY=replace-me
REWRITE_BASE_URL=https://api.openai.com/v1
REWRITE_MODEL=gpt-4.1-mini
```

启动完整开发环境：

```bash
docker compose up --build
```

默认访问地址：

- 前端：`http://localhost:8080`
- 后端：`http://localhost:8000`
- 后端 OpenAPI：`http://localhost:8000/docs`
- Milvus：`http://localhost:19530`

首次构建会安装 Python、Torch、Chroma、OCR、前端依赖等组件，耗时会比较长。

## 本地开发

后端：

```bash
cd backend
cp .env.example .env
uv sync
uv run uvicorn app.main:app --reload
```

单独启动 worker：

```bash
cd backend
uv run python scripts/run_task_worker.py
```

前端：

```bash
cd frontend
npm install
npm run dev
```

Vite 开发服务默认运行在 `http://localhost:5173`，并将 `/api` 代理到 `http://127.0.0.1:8000`。

## API 模块

后端 API 默认挂载在 `/api/v1`：

- `/health`：健康检查。
- `/chat/query`、`/chat/stream`：普通问答与流式问答。
- `/knowledge-bases`：知识库 CRUD、入库和重建。
- `/documents`：文档上传、删除和重建。
- `/conversations`：会话和消息查询。
- `/tasks`：入库任务列表、详情、统计、事件、告警和趋势。

## 文档

更多实现细节见：

- `backend/README.md`：后端能力、配置、任务队列和评测脚本说明。
- `frontend/README.md`：前端页面、组件和运行说明。
- `docs/README.md`：当前有效文档索引。
- `docs/ReguRAG-主流程图.png`
- `docs/ReguRAG-问答主链路.png`
- `docs/ReguRAG-文档入库主链路.png`

## 适用场景

ReguRAG 更适合作为法规制度类知识库问答、企业内部文档问答、文档入库链路验证、RAG 检索策略调试和全栈项目展示的基础工程。它不是通用 Agent 平台，也没有内置用户认证和权限系统。
