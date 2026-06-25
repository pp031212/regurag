# ReguRAG 后端

ReguRAG 后端是一个基于 FastAPI 的 RAG 服务，当前面向中文规章制度类知识问答场景。  
项目重点覆盖知识库管理、文档导入、检索链路调试、引用返回，以及基于评测结果进行策略调优的最小工程闭环。

## 1. 项目概述

当前后端已经具备以下能力：

- 知识库创建、查询、删除、重建索引
- 文档上传、删除、单文档重建与 ingest 任务创建
- 数据库持久化任务与事件记录 + 独立 worker 执行
- 可替换的任务队列后端边界（当前支持 SQL backend 与 Redis backend）
- worker 执行异常后的最小自动重试与日志可观测性
- 最小任务观测 API（详情 / 列表 / 统计 / 事件流 / 聚合监控 / 告警判定 / 知识库趋势）
- SQL task queue 真实集成测试（当前默认跑 MySQL）
- knowledge base / document / conversation / message_context 的 MySQL 真实集成测试
- Chroma / 文档处理链路真实集成测试
- 完整 RAG 端到端真实集成测试
- RAG 异常链路端到端真实集成测试
- RAG 跨知识库端到端真实集成测试
- worker 失败恢复真实集成测试
- PDF 文档结构化预处理（正文 / 表格 / 图片 OCR）
- 基于 Chroma 的向量检索
- FAQ / 高频问法短路
- 中文制度类切分 profile
- Query Rewrite、MMR、Rerank、最终上下文收敛
- 引用返回与分阶段调试信息输出
- 历史会话持久化与会话上下文参与问题改写
- shadow graph 对照运行与 rollout 验证
- 最小回归测试与批量评测脚本

当前技术栈：

- FastAPI
- SQLAlchemy + MySQL
- ChromaDB
- OpenAI 兼容模型接口
- Hugging Face Embedding / Reranker 模型
- Docker Compose（可选，用于本地统一拉起 MySQL / Redis / API / Worker / Frontend）

补充说明：

- 当前默认元数据库仍是 MySQL
- `TASK_QUEUE_BACKEND` 现在同时接受 `mysql | sql | redis`
- 其中 `sql` 是为后续 PostgreSQL 铺路的中性别名；当前实现仍复用同一套 SQLAlchemy repository
- 当前仓库已补 PostgreSQL driver 依赖位：`psycopg`
- 当前仓库已补 PostgreSQL canary 薄派生镜像：
  - [backend/Dockerfile.postgresql-canary](/D:/System/Desktop/regurag/backend/Dockerfile.postgresql-canary)
  - 作用是在现有 `regurag-backend:local` 之上只补 `psycopg` 和容器内回归所需的 `pytest`
  - 避免每次 PostgreSQL canary 都重新拉整套 `torch` 依赖

## 2. 当前定位

本项目当前定位是一个可运行、可调试、可评测的文本 RAG 后端，而不是通用 Agent 平台。

当前已覆盖：

- 文本知识库问答
- 会话弱耦合下的知识库问答
- 轻量 FAQ / 高频问法模板短路
- 中文制度类文本切分 profile
- 后台 ingest / rebuild 任务创建与数据库持久化排队
- 文档级删除与文档级重建
- PDF -> structured JSON / readable TXT / images 的预处理产物生成
- PDF 结构化结果转译入库
- 检索阶段可观测性
- 会话历史参与问题改写
- shadow compare 离线脚本与开发态对照运行
- `TaskQueueBackend` 抽象与 `SqlTaskQueueBackend` 首版落地
- worker `claimed / completed / retrying / failed` 生命周期日志
- 任务详情已暴露 `attempt_count / last_error / started_at / finished_at / locked_at / locked_by`
- 任务列表、状态统计、事件流、聚合监控与告警接口
- `task_events` 持久化表与任务时间线查询
- `active_workers / oldest_pending_age_seconds / long_running / recent_failed / recent_retried` 聚合视图
- `PENDING_WITHOUT_ACTIVE_WORKERS / STALE_RUNNING_TASKS / LONG_RUNNING_TASKS / RECENT_FAILURE_SPIKE / RECENT_RETRY_SPIKE` 最小告警判定
- 知识库维度的窗口对比趋势接口（当前窗口 vs 上一窗口）
- SQL task queue 的 enqueue / claim / stale reclaim / heartbeat / max-attempts 集成测试（当前默认跑 MySQL）
- knowledge base / document / conversation / message_context 的持久化与级联回收集成测试
- VectorStore 与 RAGPipeline ingest_file 的 Chroma / Markdown / structured JSON 集成测试
- IngestService + RAGService 的真实端到端集成测试
- 无检索命中、answer_guard、cross_domain_guard、knowledge base not ready、conversation not found 的端到端测试
- 自动路由切库后的 conversation / message_context 持久化与连续追问端到端测试
- worker retry recovery、stale lease handoff、partial document failure 的 SQL 集成测试（当前默认跑 MySQL）
- `grounded` / `no_answer` 两类题型的最小评测闭环

当前未覆盖：

- 更完整的 MQ 语义能力（例如 delayed queue / dead-letter queue / 消费组治理）
- 更广泛的多 worker / 多任务失败恢复集成测试
- 用户认证与权限体系

## Docker Compose

项目现在已经补了一版基础容器编排，根目录可直接使用：

```bash
docker compose up --build
```

默认会启动：

- `mysql`
- `redis`
- `milvus-etcd`
- `milvus-minio`
- `milvus-standalone`
- `backend-api`
- `backend-worker`
- `frontend`

当前 Docker 运行方式固定分成三条线：

- 本机默认开发线：使用 `docker-compose.yml`，元数据库走 MySQL，后端镜像走 CPU 版 PyTorch，适合日常开发和本机联调
- PostgreSQL canary 线：叠加 [docker-compose.postgresql-canary.yml](/D:/System/Desktop/regurag/docker-compose.postgresql-canary.yml)，用于验证未来元数据库从 MySQL 切到 PostgreSQL 的可行性；默认 compose 下看到 `regurag-postgresql` orphan 提示时不要直接用 `--remove-orphans` 清掉
- 服务器 GPU 线：叠加 [docker-compose.gpu.yml](/D:/System/Desktop/regurag/docker-compose.gpu.yml)，后端使用 [backend/Dockerfile.gpu](/D:/System/Desktop/regurag/backend/Dockerfile.gpu) 和 CUDA 版 PyTorch，适合正式 GPU 服务器部署

常用入口：

```bash
# 本机默认 CPU / MySQL
docker compose up --build

# PostgreSQL canary
docker compose -f docker-compose.yml -f docker-compose.postgresql-canary.yml up -d postgresql backend-api backend-worker

# GPU 服务器
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d backend-api backend-worker
```

如果你想复跑 Redis 队列的真实 retry smoke，可以在 compose 已启动后执行：

```bash
cd backend
uv run python scripts/run_docker_task_queue_retry_smoke.py
```

这条脚本会：

- 暂停常驻 `backend-worker`
- 创建临时知识库、上传样例文档并创建 ingest 任务
- 用一次性 worker 在首轮认领时注入一个受控异常，验证任务回到 `pending`
- 再用第二次 worker 认领并完成任务
- 最后恢复常驻 `backend-worker`

如果你想复跑最小多 worker / 多任务并发 smoke，可以执行：

```bash
cd backend
uv run python scripts/run_docker_task_queue_concurrency_smoke.py
```

这条脚本会：

- 暂停常驻 `backend-worker`
- 创建两条独立 ingest 任务
- 先让其中一条在首轮认领时进入受控 retry
- 再用两个一次性 worker 分别认领“重试任务”和“陪跑任务”
- 验证两条任务都能在真实 Docker 队列里完成，且彼此不互相拖住
- 最后恢复常驻 `backend-worker`

如果你想复跑更复杂的“`retry + stale reclaim` 混合恢复” smoke，可以执行：

```bash
cd backend
uv run python scripts/run_docker_task_queue_mixed_recovery_smoke.py
```

这条脚本会：

- 暂停常驻 `backend-worker`
- 在 `backend-api` 容器内直接创建两条独立 ingest 任务，避免依赖宿主机 `localhost` 上传链路
- 先让其中一条任务在首轮认领时进入受控 retry
- 再把另一条任务标成 stale running，交给新 worker reclaim
- 启动两个一次性 worker 并发推进“重试任务”和“stale reclaim 任务”
- 最后恢复常驻 `backend-worker`

当前这条脚本已补了 fail-fast 保护：

- 会输出分阶段进度日志
- 默认总超时为 `720s`
- 单阶段等待会受总超时预算约束，不会再无限等到十几二十分钟

当前说明：

- `redis` 已接入任务队列分发层，当前 compose 配置下 `backend-api / backend-worker` 默认使用 Redis backend
- 主链 `vector store` 当前已支持显式切换：
  - `VECTOR_STORE_BACKEND=chroma|milvus`
  - compose 默认仍保持 `chroma`
  - compose 内如切到 `milvus`，当前会使用 `VECTOR_STORE_MILVUS_URI=http://milvus-standalone:19530`
- MySQL 继续作为任务、事件、统计与观测的真实状态存储；Redis 当前只负责 pending task 分发
- 已在真实 Docker 容器里验证两条链路：
  - 正常 ingest 可通过 `backend-api -> Redis pending list -> backend-worker -> MySQL task/event` 端到端完成
  - stale running task 可在 Redis pending list 中被新 worker 以 `stale_reclaimed=true` 重新认领并完成
- retry smoke 现已具备固定脚本入口，可在本地 Docker 中重复验证“首轮 worker 异常 -> 任务回到 pending -> 二次认领成功完成”
- concurrency smoke 现也具备固定脚本入口，可在本地 Docker 中重复验证“一个任务 retry 时，另一个任务仍可由独立 worker 正常推进并完成”
- mixed recovery smoke 现已具备固定脚本入口和 fail-fast 保护，可用于复跑“retry + stale reclaim”混合场景，但当前仍在继续收口真实稳定性
- 当前 worker / ingest / vector store 已补更细的阶段日志与 `stage` 任务事件；如果 mixed recovery 再次卡住，可直接从任务事件和容器日志判断停在：
  - `prepare_ingest_target`
  - `pipeline_ingest_started / completed`
  - `chroma_add_documents_*`
- 向量库和上传目录会通过 volume 持久化
- 后端镜像首构较慢：当前依赖会拉取 `torch / chromadb / OCR` 相关大体积包，首次构建可能需要较长时间
- `backend-api` 与 `backend-worker` 当前复用同一份后端镜像，避免重复构建同一套重依赖

默认访问地址：

- 前端：`http://localhost:8080`
- 后端：`http://localhost:8000`
- Milvus gRPC：`http://localhost:19530`
- Milvus health / metrics：`http://localhost:9091`
- MySQL：默认不暴露到宿主机，仅在 compose 内部通过 `mysql:3306` 访问
- Redis：默认不暴露到宿主机，仅在 compose 内部通过 `redis:6379` 访问

如果需要实际跑通，请先把 `docker-compose.yml` 里的：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `REWRITE_API_KEY`
- `REWRITE_BASE_URL`
- `REWRITE_MODEL`

改成你自己的配置。

## 3. 系统架构

高层流程如下：

1. 前端或调用方发起聊天请求或管理请求。
2. FastAPI 路由完成参数校验并分发到服务层。
3. `RAGService` 先做知识库状态检查和轻量意图短路。
4. `RAGPipeline` 执行主链路：
   - 可选 query rewrite
   - 向量召回
   - MMR 选择
   - 可选 rerank
   - 最终上下文过滤
   - LLM 生成
5. 返回答案、引用和可选调试信息。

存储职责拆分：

- MySQL：知识库、文档、任务等元数据
- Chroma：向量化 chunk 数据

## 4. 目录结构

```text
backend/
  app/
    api/           路由与依赖注入
    core/          配置与领域异常
    db/            数据库初始化与会话
    models/        SQLAlchemy 模型
    rag/           检索链路核心实现
    repositories/  元数据仓储层
    schemas/       请求与响应模型
    services/      应用服务层
  evals/
    rag_eval_set.jsonl
    results/
  scripts/
    eval_rag.py
    regress_pdf_structuring.py
  sql/
    init_mysql.sql
    init_postgresql.sql
  tests/
```

## 5. 核心模块

- [app/services/rag_service.py](/D:/System/Desktop/regurag/backend/app/services/rag_service.py)  
  负责知识库校验、轻量闲聊短路、RAG 主链调度。

- [app/rag/pipeline.py](/D:/System/Desktop/regurag/backend/app/rag/pipeline.py)  
  负责 rewrite、召回、MMR、rerank、最终上下文拼装和生成调用。

- [app/rag/document_processor.py](/D:/System/Desktop/regurag/backend/app/rag/document_processor.py)  
  负责父子块切分。

- [app/document_processing/pdf/service.py](/D:/System/Desktop/regurag/backend/app/document_processing/pdf/service.py)  
  负责 PDF 结构化预处理主入口，产出 `structured.json`、`readable.txt` 和图片裁剪结果。

- [app/rag/structured_document_processor.py](/D:/System/Desktop/regurag/backend/app/rag/structured_document_processor.py)  
  负责将结构化 PDF 结果转成统一检索块。

- [app/rag/vector_store.py](/D:/System/Desktop/regurag/backend/app/rag/vector_store.py)  
  封装 Chroma 的向量写入与查询。

- [app/rag/llm_client.py](/D:/System/Desktop/regurag/backend/app/rag/llm_client.py)  
  封装生成模型调用，并清洗 `<think>...</think>` 这类不应暴露给用户的内部推理内容。

## 6. 环境要求

- Python `>= 3.11`
- MySQL
- 可用的 OpenAI 兼容模型接口
- 足够的本地磁盘空间，用于模型下载与 Chroma 持久化

## 7. 安装方式

当前推荐优先使用根目录 Docker Compose。下面的本地安装方式只适用于不走 Docker 的后端开发或调试。

### 方式一：使用 `uv`

```bash
uv sync
```

### 方式二：使用 conda / pip

```bash
pip install fastapi "uvicorn[standard]" pydantic-settings python-multipart chromadb numpy openai torch transformers sqlalchemy pymysql psycopg[binary] cryptography pytest
```

注意：Docker 默认镜像已改为 CPU 版 PyTorch；服务器 GPU 部署请使用根目录 `docker-compose.gpu.yml`，不要把本地 pip 安装方式当作容器构建口径。

## 8. 环境配置

后端通过 `.env` 读取配置。

常用配置项如下：

```env
APP_ENV=development
APP_HOST=127.0.0.1
APP_PORT=8000
API_V1_PREFIX=/api/v1

DATABASE_URL=mysql+pymysql://root:123456@127.0.0.1:3306/regurag?charset=utf8mb4
TASK_QUEUE_BACKEND=mysql

OPENAI_API_KEY=replace-me
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini
OPENAI_TIMEOUT_SECONDS=45
OPENAI_MAX_TOKENS=700

REWRITE_API_KEY=replace-me
REWRITE_BASE_URL=https://api.openai.com/v1
REWRITE_MODEL=gpt-4.1-mini

EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5
RERANKER_MODEL_NAME=BAAI/bge-reranker-base
VECTOR_STORE_BACKEND=chroma
VECTOR_STORE_MILVUS_URI=http://127.0.0.1:19530
VECTOR_STORE_MILVUS_TOKEN=

CHROMA_PATH=./chroma_db
CHROMA_COLLECTION_NAME=regurag_docs
UPLOADS_DIR=./data/uploads

KNOWLEDGE_BASE_SUBJECT=和鸣教育管理制度
BOOTSTRAP_DEFAULT_KNOWLEDGE_BASE=false
DEFAULT_KNOWLEDGE_BASE_ID=kb_001
DEFAULT_KNOWLEDGE_BASE_NAME=和鸣教育制度库
DEFAULT_KNOWLEDGE_BASE_DESCRIPTION=系统启动时自动导入的默认知识库
SOURCE_DOCUMENT_PATH=../和鸣教育管理制度精简chunk版.md

CHILD_CHUNK_SIZE=50
TOP_K_MMR=8

TASK_WORKER_POLL_INTERVAL_SECONDS=2.0
TASK_WORKER_LEASE_SECONDS=1800
TASK_WORKER_MAX_ATTEMPTS=3
TASK_WORKER_INJECT_FAIL_ON_FIRST_ATTEMPT_DOCUMENT_IDS=
TASK_MONITOR_WINDOW_HOURS=24
TASK_MONITOR_LONG_RUNNING_SECONDS=3600
TASK_MONITOR_RECENT_FAILURE_THRESHOLD=3
TASK_MONITOR_RECENT_RETRY_THRESHOLD=5

CORS_ALLOW_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
CORS_ALLOW_ORIGIN_REGEX=^https?://(localhost|127\.0\.0\.1|10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(:\d+)?$
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=*
CORS_ALLOW_HEADERS=*
```

## 9. 数据库初始化

默认数据库名：

```text
regurag
```

初始化方式：

```sql
source backend/sql/init_mysql.sql;
```

或者手动将 SQL 文件内容导入 MySQL。

如果你要提前为 PostgreSQL 铺路，当前仓库也已提供独立脚手架：

```sql
\i backend/sql/init_postgresql.sql
```

说明：

- 这份脚手架默认假设目标数据库已经手动创建好
- 当前只是“初始化入口预备”，默认运行环境仍保持 MySQL

如果你想检查当前 `DATABASE_URL` 指向的 metadata 库是否可连通、是否已具备核心表，可以执行：

```bash
uv run python scripts/check_metadata_backend.py
uv run python scripts/check_metadata_backend.py --expect-dialect mysql
uv run python scripts/check_metadata_backend.py --expect-dialect postgresql
```

这条脚本会输出：

- 当前 `database_url`
- 归一化后的 `dialect`
- 实际 `driver`
- 已存在表
- 缺失核心表
- `schema_ready`

## 10. 启动服务

### 使用 `uv`

```bash
uv run uvicorn app.main:app --reload
```

### 直接使用当前环境中的 Python 解释器

```bash
python -m uvicorn app.main:app --reload
```

默认本地地址：

```text
http://127.0.0.1:8000/api/v1
```

独立 worker 启动方式：

```bash
uv run python scripts/run_task_worker.py
```

单次消费一个任务后退出：

```bash
uv run python scripts/run_task_worker.py --once
```

## 10.1 PDF 回归用例

已提供可重复执行的 PDF 结构化回归脚本，当前默认内置文本型与表格型两类用例。

执行方式：

```bash
uv run python scripts/regress_pdf_structuring.py --case-id heming_rules_pdf --refresh-output
uv run python scripts/regress_pdf_structuring.py --case-id heming_rules_table_pdf --refresh-output
```

默认会：

- 使用 `backend/evals/pdf_regression_cases.json` 中的回归定义
- 使用 `backend/evals/fixtures/` 中的固定 PDF 样本
- 重新生成该 PDF 的 `structured.json`、`readable.txt` 与 `images/`
- 对页数、关键文本片段、图片数、表格数执行断言
- 输出报告到 `backend/evals/results/pdf_regression_report.json`

当前 fixture 目录已包含：

- `heming_rules.pdf`：文本型 PDF 基准样本
- `heming_rules_table.pdf`：表格型 PDF 基准样本
- `heming_rules_mixed.pdf`：文本 + 表格 + 图片 OCR 混合型 PDF 基准样本

当前已内置回归用例：

- `heming_rules_pdf`
- `heming_rules_table_pdf`
- `heming_rules_mixed_pdf`

回归定义中的路径支持以下前缀：

- `project://` 相对项目根目录
- `backend://` 相对 `backend/` 目录

## 11. API 概览

主要接口如下：

- `GET /api/v1/health`
- `POST /api/v1/chat/query`
- `GET /api/v1/conversations`
- `POST /api/v1/conversations`
- `GET /api/v1/conversations/{conversation_id}/messages`
- `DELETE /api/v1/conversations/{conversation_id}`
- `GET /api/v1/knowledge-bases`
- `GET /api/v1/knowledge-bases/{knowledge_base_id}`
- `POST /api/v1/knowledge-bases`
- `DELETE /api/v1/knowledge-bases/{knowledge_base_id}`
- `POST /api/v1/documents/upload`
- `DELETE /api/v1/documents/{document_id}`
- `POST /api/v1/documents/{document_id}/rebuild`
- `GET /api/v1/knowledge-bases/{knowledge_base_id}/documents`
- `POST /api/v1/knowledge-bases/{knowledge_base_id}/ingest`
- `POST /api/v1/knowledge-bases/{knowledge_base_id}/rebuild`
- `GET /api/v1/tasks`
- `GET /api/v1/tasks/stats`
- `GET /api/v1/tasks/overview`
- `GET /api/v1/tasks/alerts`
- `GET /api/v1/tasks/trends`
- `GET /api/v1/tasks/{task_id}`
- `GET /api/v1/tasks/{task_id}/events`

### 聊天请求示例

```json
{
  "knowledge_base_id": "kb_001",
  "query": "迟到会怎么样？",
  "top_k_retrieve": 15,
  "top_k_rerank": 3,
  "enable_rewrite": false,
  "enable_rerank": true,
  "debug": true,
  "debug_chunks": true
}
```

### 调试字段说明

当 `debug=true` 时，后端可返回：

- 改写后的查询
- 召回数量
- 最终上下文数量
- 各阶段耗时
- LLM 使用信息

当 `debug_chunks=true` 时，还会返回分阶段 chunk 详情：

- `retrieved_chunks`
- `mmr_selected_chunks`
- `reranked_chunks`
- `final_context_chunks`

## 12. 测试

运行回归测试：

```bash
pytest
```

当前测试覆盖：

- API 级回归测试
- 服务层轻量意图短路测试
- 会话上下文、配置 profile、shadow compare 与 graph rollout 相关测试
- `<think>...</think>` 输出清洗测试

当前本地已验证状态：

- 2026-04-11 已执行 `cd backend && uv run pytest tests -q`
- 结果：`195 passed`

当前测试局限：

- 目前仍以 mock 服务依赖为主
- 已补 MySQL task queue、metadata repository、Chroma / 文档处理链路，以及正常/异常/跨知识库 RAG 端到端、基础失败恢复、最小任务观测、任务事件流、聚合监控和知识库趋势聚合集成测试；但更复杂的恢复链路和更多真实资料仍需继续扩充

## 13. 评测

评测集位置：

```text
backend/evals/rag_eval_set.jsonl
backend/evals/rag_eval_split_profile.jsonl
backend/evals/rag_eval_split_profile_strict.jsonl
backend/evals/rag_eval_real_kb_samples.jsonl
```

当前评测集包含两类题型：

- `grounded`
- `no_answer`

运行 baseline：

```bash
uv run python scripts/eval_rag.py --label baseline
uv run python scripts/eval_rag.py --dataset evals/rag_eval_split_profile.jsonl --label split-profile-baseline
uv run python scripts/eval_rag.py --dataset evals/rag_eval_real_kb_samples.jsonl --knowledge-base-id <kb_id> --label real-kb-samples-v1
```

运行策略对比：

```bash
uv run python scripts/eval_rag.py --label rewrite_off --disable-rewrite
uv run python scripts/eval_rag.py --label rerank_off --disable-rerank
```

运行子集评测：

```bash
uv run python scripts/eval_rag.py --dataset evals/rag_eval_split_profile.jsonl --case-id sp002 --case-id sp004 --label split-subset
uv run python scripts/eval_rag.py --dataset evals/rag_eval_split_profile.jsonl --limit 4 --label split-top4
```

对比两次评测结果：

```bash
uv run python scripts/compare_eval_reports.py --baseline evals/results/split-profile-before.json --contender evals/results/split-profile-after.json --label split-profile-diff
```

运行影子链路对照：

```bash
uv run python scripts/compare_shadow_graph.py --label shadow-compare
uv run python scripts/compare_shadow_graph.py --shadow-retrieval-backend langchain_chroma --label shadow-compare-langchain-chroma
uv run python scripts/compare_shadow_graph.py --shadow-retrieval-backend langchain_milvus --label shadow-compare-langchain-milvus
```

运行固定的 shadow retrieval 回归：

```bash
uv run python scripts/run_shadow_retrieval_regression.py --label-prefix shadow-retrieval-regression
```

这个脚本会固定跑 `langchain_chroma` 和 `langchain_milvus` 两个后端，并同时输出：

- 单后端报告：`backend/evals/results/<label-prefix>-<backend>.json`
- 紧凑回归摘要：`backend/evals/results/<label-prefix>-summary.json`

摘要里会固定关注这些口径：

- `graph_error_count`
- `retrieval_drift_case_count`
- `final_answer_hit_parity_rate`
- `final_citation_ids_match_rate`
- `final_context_ids_match_rate`
- `avg_graph_latency_ms`

运行隔离 fixture 的严格 A/B 对照：

```bash
uv run python scripts/compare_split_profiles.py --label split-profile-strict-ab
```

运行真实 PDF 知识库的 before/after A/B：

```bash
uv run python scripts/compare_real_profile_eval.py --label real-profile-ab-v1
```

运行主链 `vector store` 的真实知识库 retrieval-only 对比：

```bash
uv run python scripts/compare_real_vector_store_retrieval.py --label real-vector-store-retrieval-compare
uv run python scripts/compare_real_vector_store_retrieval.py --backends chroma milvus --milvus-uri http://127.0.0.1:19530 --label real-vector-store-retrieval-compare
uv run python scripts/compare_real_vector_store_retrieval.py --label vector-store-smoke --limit 4
```

这个脚本会：

- 为每个 `VECTOR_STORE_BACKEND` 创建独立临时知识库
- 导入真实 PDF fixture
- 跑主链 `_prepare_generation_inputs()`，覆盖：
  - ingest
  - retrieve
  - rerank
  - final context
  - citation
- 输出每个后端的 retrieval-only 报告和对比摘要

当前摘要会固定关注这些口径：

- `retrieval_hit_rate`
- `final_context_hit_rate`
- `citation_hit_rate`
- `avg_retrieve_ms`
- `avg_rerank_ms`
- `avg_context_build_ms`
- `case_drift_count`

运行主链 `vector store` 的真实端到端 eval 对比：

```bash
uv run python scripts/compare_real_vector_store_eval.py --milvus-uri http://127.0.0.1:19530 --label real-vector-store-eval-compare
uv run python scripts/compare_real_vector_store_eval.py --milvus-uri http://127.0.0.1:19530 --label vector-store-eval-smoke --limit 4
uv run python scripts/compare_real_vector_store_eval.py --milvus-uri http://127.0.0.1:19530 --label vector-store-eval-real-kb-v1
```

这个脚本会：

- 为每个 `VECTOR_STORE_BACKEND` 创建独立临时知识库
- 导入真实 PDF fixture
- 跑真实生成阶段的 `run_case()`，覆盖：
  - ingest
  - retrieve
  - rerank
  - final context
  - citation
  - answer
- 输出每个后端的端到端评测报告和对比摘要

当前摘要会固定关注这些口径：

- `retrieval_hit_rate`
- `final_context_hit_rate`
- `citation_hit_rate`
- `answer_hit_rate`
- `avg_latency_ms`
- `case_diffs`

运行主链 `vector store` 的固定回归入口：

```bash
uv run python scripts/run_vector_store_regression.py --label-prefix vector-store-regression
uv run python scripts/run_vector_store_regression.py --skip-live-eval --limit 4 --shadow-limit 4 --label-prefix vector-store-regression-smoke
```

这个脚本会：

- 固定跑 `chroma` 和 `milvus` 两个主链后端
- 生成 retrieval-only 对比报告
- 在有可用模型凭证时继续生成真实端到端 eval 对比报告
- 再分别在两个主链后端下跑一轮 shadow smoke
- 最后输出一个压缩摘要，集中给出：
  - retrieval parity
  - live eval parity
  - shadow health
  - `ready_for_rollout`

补充说明：

- 现在支持 `--stage retrieval|live_eval|shadow`，可以只跑单个阶段
- 现在支持 `--reuse-existing`；即使不显式传这个参数，脚本在“某阶段未被本次请求执行”时也会尝试从同一 `label-prefix` 复用已有阶段结果
- `live_eval` 现在额外支持 `--live-eval-batch-size`
  - 默认按 4 条 case 一批执行
  - 每完成一批就会把 partial report 落到 `backend/evals/results/<label-prefix>-eval-<backend>.json`
  - 如果 full `live_eval` 中途超时，再次执行同一 `label-prefix` 时配合 `--reuse-existing`，会直接跳过已完成的 case，只补剩余 case
- 这意味着可以按同一前缀拆成多次执行：

```bash
uv run python scripts/run_vector_store_regression.py --stage retrieval --label-prefix vector-store-regression-staged
uv run python scripts/run_vector_store_regression.py --stage shadow --label-prefix vector-store-regression-staged
uv run python scripts/run_vector_store_regression.py --stage live_eval --label-prefix vector-store-regression-staged
```

- 第 2、3 次执行会自动加载前面阶段已经落盘的结果，不会重新从头跑整条回归
- 如果 full `live_eval` 比较慢，可以拆成“首次不带 `--reuse-existing` 起跑，后续带 `--reuse-existing` 续跑”的方式：

```bash
uv run python scripts/run_vector_store_regression.py --stage live_eval --label-prefix vector-store-regression-live --live-eval-batch-size 4
uv run python scripts/run_vector_store_regression.py --stage live_eval --label-prefix vector-store-regression-live --live-eval-batch-size 4 --reuse-existing
```

- 这个脚本当前新增了 `--cleanup-timeout-seconds`，默认会对每个临时知识库的删除做 best-effort 超时收口
- 如果报告和 summary 已经写出，但删除临时知识库很慢，脚本会打印 cleanup timeout 提示，而不是一直卡在尾部

当前回归摘要会固定检查这些 rollout gates：

- `retrieval_drift_free`
- `retrieval_hit_parity_ok`
- `final_context_parity_ok`
- `citation_parity_ok`
- `live_eval_available`
- `live_eval_retrieval_parity_ok`
- `live_eval_final_context_parity_ok`
- `live_eval_citation_parity_ok`
- `live_eval_answer_parity_ok`
- `shadow_graph_error_free`
- `shadow_retrieval_drift_free`
- `shadow_citation_alignment_ok`
- `shadow_context_alignment_ok`
- `shadow_answer_hit_parity_ok`

说明：

- 如果显式传了 `--skip-live-eval`，或者当前环境没有可用模型凭证，summary 里的：
  - `live_eval_available`
  - `live_eval_*`
  会是 `false`
- 这种情况下 `ready_for_rollout` 也会保持 `false`，因为脚本会明确把“未完成 live eval”视为未达灰度准入标准，而不是默认放行

评测结果输出目录：

```text
backend/evals/results/
```

如果你想把 summary 进一步压成“现在能不能进入灰度切换”的单条判断，可以执行：

```bash
uv run python scripts/check_vector_store_rollout_readiness.py --summary evals/results/vector-store-regression-live-v1-summary.json --backend milvus
```

这条脚本会固定检查：

- `retrieval / live_eval / shadow` 三个阶段是否都可用
- `milvus.ready_for_rollout` 是否为 `true`
- 是否还有失败的 rollout gates
- 当前 live eval 相对 `chroma` 的时延差值

主链 `milvus` 的灰度切换步骤和回滚步骤已整理在：

- [docs/主链Milvus灰度切换方案.md](/D:/System/Desktop/regurag/docs/主链Milvus灰度切换方案.md)

### 流式性能基线

如果你想观察流式问答的首 token 和总耗时，可以在后端服务启动后运行：

```bash
uv run python scripts/benchmark_chat_stream.py --knowledge-base-id kb_001 --label stream-benchmark-v1
```

默认会复用 `backend/evals/rag_eval_overview_cases.jsonl` 里的题目，只读取其中的 `id / question / category` 字段。

如果你想复跑当前固定的 `5` 题流式基线，可以直接使用 `backend/evals/stream_benchmark_cases_v1.jsonl`：

```bash
uv run python scripts/benchmark_chat_stream.py --inprocess --dataset evals/stream_benchmark_cases_v1.jsonl --label stream-benchmark-v2 --repeats 2 --warmup-runs 1
```

常见用法：

```bash
uv run python scripts/benchmark_chat_stream.py --knowledge-base-id kb_001 --repeats 5 --warmup-runs 2 --label stream-benchmark-v2
uv run python scripts/benchmark_chat_stream.py --dataset evals/rag_eval_real_kb_samples.jsonl --knowledge-base-id kb_001 --limit 4 --label stream-real-kb-top4
uv run python scripts/benchmark_chat_stream.py --knowledge-base-id kb_001 --case-id ovc001 --case-id ovc004 --label stream-subset
uv run python scripts/benchmark_chat_stream.py --inprocess --knowledge-base-id kb_001 --label stream-inprocess
```

脚本会输出并落盘以下指标：

- 客户端观测的 `first_token_ms`
- 客户端观测的 `total_ms`
- 服务端 debug 返回的 `llm_first_token_ms`
- 服务端 debug 返回的 `latency_ms`
- 服务端分阶段耗时 `server_stage_timings_ms`，包括 `rewrite / retrieve / rerank / context_build / generate / llm_after_first_token / service_overhead`
- 每题多次重复后的 `avg / p50 / p95 / max`

结果文件输出到：

```text
backend/evals/results/<label>.json
```

如果你想比较两次流式基线结果：

```bash
uv run python scripts/compare_stream_benchmark.py --baseline evals/results/stream-benchmark-v1.json --contender evals/results/stream-benchmark-v1-warm.json --label stream-benchmark-diff
```

### 当前评测结论

当前 `18` 条样本上：

- `baseline_v7`：`18/18 ok`
- `retrieval_hit_rate = 1.0`
- `final_context_hit_rate = 1.0`
- `citation_hit_rate = 1.0`
- `answer_hit_rate = 1.0`

当前这批规章制度问答样本上的阶段性结论：

- 保留 `rerank`
- `rewrite` 不建议默认开启
- 中文切分 profile 的收益建议优先在 `rag_eval_split_profile.jsonl` 这组 `plain text / OCR text` 样本上观察

当前 split profile 定向评测：

- `split-profile-after-v2`：`8/8 ok`
- `retrieval_hit_rate = 1.0`
- `final_context_hit_rate = 1.0`
- `citation_hit_rate = 1.0`
- `answer_hit_rate = 1.0`

当前 split profile 严格 A/B 对照：

- 基线：root config only；对照：`rules_cn` profile
- 默认隔离参数：`child_chunk_size = 18`，`parent_chunk_size = 32`
- `retrieval_hit / final_context_hit / citation_hit / answer_hit` 均提升 `0.5`
- 当前改善样本：`sab002`（plain text 特殊事项请假条款）、`sab004`（OCR 复合行为条款）
- 这说明中文子句切分对 plain text 的复合制度条款和 OCR 复合行为条款都更容易保住完整上下文

当前已补真实知识库样本评测集：

- 数据集：`backend/evals/rag_eval_real_kb_samples.jsonl`
- 预期知识库内容：已导入 `heming_rules.pdf`、`heming_rules_table.pdf`、`heming_rules_mixed.pdf`
- 覆盖范围：`plain_pdf / table_pdf / mixed_ocr / control`
- 当前样本数：`10`
- 重点题型：病假条件、特殊事项请假、旷课边界、宿舍条款、OCR 课堂纪律、表格短条目、累计扣分后果

当前真实知识库样本评测：

- `real-kb-samples-v1`：`10/10 ok`
- `retrieval_hit_rate = 1.0`
- `final_context_hit_rate = 1.0`
- `citation_hit_rate = 1.0`
- `answer_hit_rate = 1.0`
- 分类结果：
  - `plain_pdf`：`4/4 ok`
  - `table_pdf`：`2/2 ok`
  - `mixed_ocr`：`3/3 ok`
  - `control`：`1/1 ok`

当前真实 PDF 知识库 before/after A/B：

- 脚本：`backend/scripts/compare_real_profile_eval.py`
- 基线：root config only；对照：`rules_cn` profile
- 数据源：`heming_rules.pdf`、`heming_rules_table.pdf`、`heming_rules_mixed.pdf`
- 结果：当前 `10` 条真实样本上，两套 profile 的 `retrieval / final_context / citation / answer hit rate` 均为 `1.0`
- 延迟结果：`rules_cn` 平均延迟约快 `462ms`
- 当前判断：真实样本集已证明两套 profile 在真实 PDF 上都稳定可用，但这批题目还不够难，暂时不足以继续拉开策略收益差异

注意：

这只是当前数据集下的策略结论，不应直接当成所有 RAG 场景的通用规则。

## 14. 运行注意事项

- 文件上传依赖 `python-multipart`
- 元数据使用 MySQL 持久化
- 向量数据使用 Chroma 持久化
- ingest / rebuild 当前通过数据库任务记录 + 独立 worker 执行；仅启动 Web 服务不会自动消费任务
- 当前任务队列后端已支持 `sql / mysql / redis` 三档；compose 默认使用 Redis backend
- `sql` 是推荐给未来 PostgreSQL 迁移使用的中性命名；`mysql` 当前继续保留为兼容别名
- worker 当前已支持“异常失败后回到 pending 再重试”的最小策略，最大尝试次数由配置控制
- `TASK_WORKER_INJECT_FAIL_ON_FIRST_ATTEMPT_DOCUMENT_IDS` 默认关闭，仅用于 Docker/local smoke 回归；命中后只会在该文档的首轮认领注入一次 worker 级异常
- 当前 Docker 实测里，首次 ingest 可能会被容器内 Embedding/Reranker 模型冷启动明显拉长；首个任务长时间停在 `started` 后不一定是 Redis 分发异常
- Docker Compose 现在额外挂了共享 `backend_model_cache` 卷，并给 `backend-api / backend-worker` 设置了 `HF_HOME=/opt/model-cache/huggingface`
- Compose 现在也显式设置了 `PYTHONPATH=/app/backend`，容器内直接跑脚本时会优先使用挂载后的当前源码，而不是镜像里旧的 `site-packages`
- 后端镜像的 Python 依赖环境现在固定在 `/opt/regurag-venv`，不再放在 `/app/backend/.venv`；因此 Docker 模式下宿主机 `backend/.venv` 只是本地开发残留，可以在镜像重建验证通过后删除
- 默认 [Dockerfile](/D:/System/Desktop/regurag/backend/Dockerfile) 面向本机 CPU 开发，会跳过锁文件里的 CUDA 版 `torch / triton / nvidia-*`，再从 PyTorch CPU index 安装 `torch==2.10.0+cpu`
- 服务器 GPU 部署使用 [Dockerfile.gpu](/D:/System/Desktop/regurag/backend/Dockerfile.gpu) 与根目录 [docker-compose.gpu.yml](/D:/System/Desktop/regurag/docker-compose.gpu.yml)，保留锁文件里的 CUDA 版 torch 依赖集
- 主链 `vector store` 当前默认仍是 `chroma`；如需验证主链 Milvus，可显式设置：

```bash
VECTOR_STORE_BACKEND=milvus
VECTOR_STORE_MILVUS_URI=http://127.0.0.1:19530
```

- 当前 Milvus 主链接入仍保持现有主链检索策略不变：
  - dense 检索走 `MilvusVectorStore.search()`
  - sparse 检索继续复用本地 sidecar sparse index / list scan
- 如果当前环境没有可用的 `OPENAI_API_KEY / REWRITE_API_KEY`，仍可先跑 `compare_real_vector_store_retrieval.py`
  来验证主链 `MilvusVectorStore` 在 retrieval-only 口径上的真实知识库对齐；真实生成阶段和 `compare_shadow_graph.py` 仍需要可用模型凭证
- 当前根目录 `docker-compose.yml` 里的模型凭证示例仍是占位值；如需跑真实生成阶段：
  - 本机 `uv run ...` 会读取 `backend/.env`
  - compose 容器内命令则需要你先把 compose 里的模型配置改成可用值
- `backend/evals/rag_eval_real_kb_samples.jsonl` 里的 `rk016` 当前已补 no-answer 等价表述组，允许：
  - `无相关条款可供参照`
  - `未包含相关条款`
  - `未包含相关规定`
  这类语义等价措辞
- `backend/evals/rag_eval_real_kb_samples.jsonl` 里的 `rk011` 当前也已补 grounded answer 等价表述组，允许：
  - `进行相应的扣分处理`
  - `进行相应扣分处理`
  这类和“按事假规则相应扣分”语义等价的回答
- 如需提前填充共享模型缓存，可在 compose 环境里执行：

```bash
docker compose exec -T backend-api python scripts/warm_model_cache.py
docker compose exec -T backend-api python scripts/warm_model_cache.py --only reranker
```

- 当前 retry 语义只覆盖 worker 级执行异常；普通 ingest 失败会直接落成 `failed`，不会自动按任务内容重试
- 首次运行时，Embedding / Reranker 模型可能触发下载
- Hugging Face 未登录时可能出现速率提示，不一定代表功能异常
- 影子链路检索后端当前支持 `legacy`、`langchain_chroma` 和 `langchain_milvus` 三档，可通过 `SHADOW_RETRIEVAL_BACKEND` 切换；默认仍为 `legacy`
- `langchain_milvus` 可通过 `SHADOW_MILVUS_URI` 指向本地 `.db` 文件或远端 Milvus URI；当前 Windows 环境不支持 `Milvus Lite` 本地文件模式，需改用远端/容器内 Milvus endpoint
- 当前 `langchain_milvus` 影子检索已改成 `pymilvus.MilvusClient` 直连负责建表、写入、搜索与自愈重建，不再依赖 `langchain_milvus` 的 ORM bootstrap 路径
- 当前 Docker Compose 已补 `Milvus` 服务并可正常启动；最新 `5` 条 smoke 对照 `graph_error_count = 0`，检索 / citation / final_context 已稳定对齐，剩余差异主要在答案措辞而不是检索漂移
- 如果你要对照“不同主链后端下的 shadow compare”：

```bash
$env:VECTOR_STORE_BACKEND='chroma'; $env:VECTOR_STORE_MILVUS_URI='http://127.0.0.1:19530'; uv run python scripts/compare_shadow_graph.py --label vector-store-shadow-smoke-chroma --limit 4
$env:VECTOR_STORE_BACKEND='milvus'; $env:VECTOR_STORE_MILVUS_URI='http://127.0.0.1:19530'; uv run python scripts/compare_shadow_graph.py --label vector-store-shadow-smoke-milvus --limit 4
```

- 当前这组 smoke 的结论是：
  - 两个主链后端下 `graph_error_count` 都为 `0`
  - `retrieval_drift_case_count` 都为 `0`
  - 差异主要仍在答案措辞，不在检索漂移
- 如果你要在 compose 里做主链 `milvus` 小范围灰度，不需要改默认 compose 文件，当前仓库已提供：
  - [docker-compose.milvus-canary.yml](/D:/System/Desktop/regurag/docker-compose.milvus-canary.yml)
  - 用法：

```bash
$env:COMPOSE_FILE='docker-compose.yml;docker-compose.milvus-canary.yml'
docker compose up -d backend-api backend-worker
```

- 回滚到默认 `chroma`：

```bash
$env:COMPOSE_FILE='docker-compose.yml'
docker compose up -d backend-api backend-worker
```

- 当前这套 canary 已实际补跑通过：
  - retrieval-only compose canary
  - Docker ingest retry smoke
  - Docker concurrency smoke
  - Docker mixed recovery smoke
- 如果你想把这套 canary 观察收成一次固定检查，而不是手工逐条执行，可以直接跑：

```bash
python backend/scripts/run_milvus_canary_checks.py
```

- 当前会输出：
  - [backend/evals/results/milvus-canary-checks-summary.json](/D:/System/Desktop/regurag/backend/evals/results/milvus-canary-checks-summary.json)
- 当前 summary 会固定汇总：
  - readiness check
  - 容器内 `VECTOR_STORE_BACKEND / VECTOR_STORE_MILVUS_URI`
  - retrieval-only canary
  - retry smoke
  - concurrency smoke
  - mixed recovery smoke
- 截至 `2026-04-18` 最新一次统一检查结果为：
  - `canary_ok = true`
- 如果你要为 PostgreSQL 起一套受控 compose canary，当前仓库已提供：
  - [docker-compose.postgresql-canary.yml](/D:/System/Desktop/regurag/docker-compose.postgresql-canary.yml)
  - 用法：

```bash
$env:COMPOSE_FILE='docker-compose.yml;docker-compose.postgresql-canary.yml'
docker compose up -d postgresql backend-api backend-worker
```

- 当前这套 override 的目标是：
  - 保持默认主环境不变
  - 仅把 `backend-api / backend-worker` 的 `DATABASE_URL` 切到 PostgreSQL
  - 同时把 `TASK_QUEUE_BACKEND` 切到 `sql`
- 说明：
  - 这一步当前属于“受控 canary 骨架 + 最小联通验证”，不是默认迁移
  - 当前 `backend-api` 会使用 `backend/Dockerfile.postgresql-canary` 构建 `regurag-backend-postgresql-canary:local`
  - `backend-worker` 会复用同一张 canary 镜像
  - 这样可以只补 `psycopg`，不必重新构建完整后端重依赖镜像
- 当前后端 Docker 构建还做了两处缓存优化：
- `backend-api` 的 build context 已收敛到 `./backend`，不再把整个仓库根目录打包进后端镜像上下文
- 默认 `backend/Dockerfile` 先只复制 `pyproject.toml / uv.lock / README.md`，并用 `uv sync --no-install-project` 预装第三方依赖；普通业务代码改动不再轻易打掉 `transformers / pymilvus` 这类依赖缓存层
- 默认本机 CPU 镜像不再拉取 CUDA 版 `torch / triton / nvidia-*` 依赖；如需服务器 GPU 镜像，使用：

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build backend-api
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d backend-api backend-worker
```

- 本机 CPU 镜像验证 Python / torch 位置：

```bash
docker compose exec backend-api python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```
- PostgreSQL canary 已不再只是最小联通验证。截至 `2026-04-19`，当前已实际验证：
  - metadata check 通过
  - retry smoke 通过
  - concurrency smoke 通过
  - mixed recovery smoke 通过
  - SQL task queue 集成回归通过
- 如果你想复跑 PostgreSQL canary 固定检查，可以直接执行：

```bash
python backend/scripts/run_postgresql_canary_checks.py
```

- 当前会输出：
  - [backend/evals/results/postgresql-canary-checks-summary.json](/D:/System/Desktop/regurag/backend/evals/results/postgresql-canary-checks-summary.json)
- 当前 summary 会固定汇总：
  - 容器内 `DATABASE_URL / TASK_QUEUE_BACKEND / has_psycopg`
  - `check_metadata_backend.py --expect-dialect postgresql`
  - retry smoke
  - concurrency smoke
  - mixed recovery smoke
- 最新一次真实检查结果为：
  - `canary_ok = true`
- 当前 one-off smoke worker 已补 `--idle-timeout-seconds` 短轮询，避免偶发出现“任务已创建但 worker 首次空 claim 后立即退出”的抖动
- 如果你想把 PostgreSQL 下的 SQL task queue 真实集成回归也固定成一次容器内检查，可以直接跑：

```bash
python backend/scripts/run_postgresql_task_queue_integration_checks.py
```

- 当前会在 `backend-api` 容器内执行：
  - `tests/test_task_queue_sql_integration.py`
  - `tests/test_task_worker_sql_recovery_integration.py`
- 说明：
  - 这两组测试当前已经按通用 SQL task queue / repository 语义命名
  - 在 PostgreSQL canary 容器里运行时，实际验证的是 PostgreSQL metadata + `TASK_QUEUE_BACKEND=sql`
- 当前会输出：
  - [backend/evals/results/postgresql-task-queue-integration-summary.json](/D:/System/Desktop/regurag/backend/evals/results/postgresql-task-queue-integration-summary.json)
- 截至 `2026-04-19` 最新一次真实检查结果为：
  - `canary_ok = true`
  - `11 passed in 0.91s`
- 如果你想把 `Milvus` 和 `PostgreSQL` 两条 canary 线统一成一条入口，可以直接跑：

```bash
python backend/scripts/run_canary_overview_checks.py
```

- 当前会顺序执行并汇总：
  - `python backend/scripts/run_milvus_canary_checks.py`
  - `python backend/scripts/run_postgresql_canary_checks.py`
  - `python backend/scripts/run_postgresql_task_queue_integration_checks.py`
- 当前会输出：
  - [backend/evals/results/canary-overview-summary.json](/D:/System/Desktop/regurag/backend/evals/results/canary-overview-summary.json)
- 当前 summary 会固定给出：
  - `overall_ok`
  - `milvus_canary_ok`
  - `postgresql_canary_ok`
  - `postgresql_task_queue_integration_ok`
  - 每条检查对应的 `failed_stage`
  - 每条检查各自的 `summary_path`
- 当前还支持：
  - `--reuse-existing`
  - `--child-timeout-seconds`
- 截至 `2026-04-19` 最新一次总览结果为：
  - `overall_ok = true`
  - `milvus_canary_ok = true`
  - `postgresql_canary_ok = true`
  - `postgresql_task_queue_integration_ok = true`

## 15. 已知限制

- 任务执行机制已脱离 Web 进程内执行；当前 Redis backend 只负责任务分发，真实任务状态与监控统计仍以 MySQL 为准
- 当前尚未接入更完整的 MQ 语义能力，例如 delayed queue、dead-letter queue、独立消费组治理等
- 当前真实 Docker 联调已验证正常 ingest 与 stale reclaim；但 retry 仍主要由测试覆盖，尚未补成一条容易从业务接口直接触发的容器级回归链路
- 当前已补 `backend/scripts/run_docker_task_queue_retry_smoke.py`，可重复触发一次受控 worker 级异常来验证 Redis retry；它依赖本地 compose 正常运行，并会短暂停掉常驻 worker
- 当前已补 `backend/scripts/run_docker_task_queue_concurrency_smoke.py`，可重复验证最小多 worker / 多任务并发恢复路径；它同样会短暂停掉常驻 worker
- 当前已补 `backend/scripts/run_docker_task_queue_mixed_recovery_smoke.py`，用于验证 `retry + stale reclaim` 混合恢复；它默认有 `720s` 总超时和阶段进度输出，超时后会自动走清理并恢复常驻 worker
- 最新在共享模型缓存预热后，`run_docker_task_queue_mixed_recovery_smoke.py --total-timeout-seconds 180` 已在约 `119s` 内完成，通过了 `retry + stale reclaim` 混合恢复验证
- 当前还额外补了 worker / ingest / vector store 的细粒度日志，方便定位 mixed recovery 场景里是模型初始化、文件预处理，还是 Chroma 写入路径卡住
- 当前已具备数据库持久化的任务事件流、基础统计、最小告警判定与知识库趋势聚合，并已接入一版前端任务观测页；当前前端已可展示告警解释、事件聚合摘要和知识库趋势，但尚未接入独立监控平台或更完整的告警推送
- 当前评测集规模仍偏小，且主要聚焦制度型规则问答
- `rewrite` 虽可开关，但尚未沉淀为按题型自动启用的默认策略
- 当前主要耗时仍集中在生成阶段

## 16. 后续计划

短期优先项：

- 继续扩充评测集
- 将 `rewrite / rerank` 的默认策略进一步工程化
- 增加上传校验与重复检测的进一步收口
- 继续细化 worker 监控口径，并打磨前端任务观测页
- 继续补 Redis backend 在 Docker / 多 worker 下更复杂的 retry / reclaim 组合回归，以及更明确的异常分类
- 扩完整恢复测试到更复杂的 worker / 多任务场景
- 视需要补独立告警推送或管理面

中期优先项：

- PDF / OCR 文档解析链路
- 更细粒度的引用定位
- 更稳的任务执行机制
- 更丰富的前端调试体验
