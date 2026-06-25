# 主链 Milvus 灰度切换方案

这份文档对应当前 ReguRAG 的现实状态：

- 主链默认后端仍是 `chroma`
- `MilvusVectorStore` 已接入主链
- 固定回归摘要已能给出 `ready_for_rollout`
- 当前“灰度切换”指的是按环境切换，不是按请求比例分流

也就是说，这里的灰度做法是：

- 先在受控环境里把 `VECTOR_STORE_BACKEND` 从 `chroma` 改成 `milvus`
- 观察一段时间
- 没问题再考虑扩大范围

## 1. 切换前先看什么

先确认最新 fixed regression 结果已经达标。

推荐命令：

```bash
cd backend
uv run python scripts/check_vector_store_rollout_readiness.py --summary evals/results/vector-store-regression-live-v1-summary.json --backend milvus
```

当前这条命令会检查：

- `retrieval / live_eval / shadow` 三个阶段是否都可用
- `milvus.ready_for_rollout` 是否为 `true`
- 哪些 rollout gates 失败了
- 当前 live eval 的时延差值

如果你还想给时延加一个人工上限，可以额外传：

```bash
uv run python scripts/check_vector_store_rollout_readiness.py --summary evals/results/vector-store-regression-live-v1-summary.json --backend milvus --max-live-eval-latency-delta-ms 12000
```

只有脚本退出码为 `0` 时，才建议进入下一步。

## 2. 小范围切换怎么做

当前建议只在一套受控环境里切，不要直接把默认值全局改掉。

### 方案 A：compose 内单环境切换

1. 保留代码默认值不变，只改运行环境：
   - `VECTOR_STORE_BACKEND=milvus`
   - `VECTOR_STORE_MILVUS_URI=http://milvus-standalone:19530`
2. 只重启真正依赖主链 vector store 的服务：
   - `backend-api`
   - `backend-worker`
3. 不需要动：
   - MySQL
   - Redis
   - Milvus 本身以外的其他基础服务

示意做法：

```bash
$env:COMPOSE_FILE='docker-compose.yml;docker-compose.milvus-canary.yml'
docker compose up -d milvus-etcd milvus-minio milvus-standalone
docker compose up -d backend-api backend-worker
```

这里已经提供了现成的 canary override：

- [docker-compose.milvus-canary.yml](/D:/System/Desktop/regurag/docker-compose.milvus-canary.yml)

它只会覆盖：

- `backend-api`
- `backend-worker`

不会改默认的 [docker-compose.yml](/D:/System/Desktop/regurag/docker-compose.yml)。

### 方案 B：本机进程切换

如果不是走 compose，而是本机直接起后端：

```bash
$env:VECTOR_STORE_BACKEND='milvus'
$env:VECTOR_STORE_MILVUS_URI='http://127.0.0.1:19530'
uv run python -m app.main
```

worker 进程也要使用同样的环境变量。

## 3. 切换后固定检查项

切完以后，不要先看“感觉”，先看这 3 类检查。

### 3.1 基础健康

- 服务能正常启动
- 没有新的启动时报错
- 基础问答请求可返回
- ingest 任务能正常创建并完成

### 3.2 最小回归

先跑最小回归，不要一上来就跑全量。

```bash
cd backend
uv run python scripts/run_vector_store_regression.py --stage retrieval --limit 4 --label-prefix milvus-canary
uv run python scripts/run_vector_store_regression.py --stage shadow --shadow-limit 4 --label-prefix milvus-canary
```

如果当前环境有可用模型凭证，再补：

```bash
uv run python scripts/run_vector_store_regression.py --stage live_eval --limit 4 --label-prefix milvus-canary --live-eval-batch-size 2
```

### 3.3 结果检查

重点只看这几项：

- `graph_error_count = 0`
- `retrieval_drift_case_count = 0`
- `ready_for_rollout = true`
- 没有新的 case 级 answer 回退
- 时延虽然可能变高，但没有高到不可接受

如果切换后只是答案措辞有轻微波动，而 retrieval / final context / citation 没退化，优先按评测口径复盘，不要立刻判断成主链后端故障。

## 4. 什么时候继续扩大范围

至少同时满足下面这些，再考虑把 `milvus` 从“受控环境”扩大到“更多环境”：

- `check_vector_store_rollout_readiness.py` 持续通过
- 连续几轮 fixed regression 没有新回退
- 真实 ingest / delete / reset / query 都稳定
- worker 场景没有新增异常
- 团队接受当前时延差异

当前项目没有请求级流量分流设施，所以“扩大范围”更适合理解成：

- 从单一测试环境
- 扩到更多内部环境
- 最后才考虑把默认运行后端改成 `milvus`

## 5. 回滚怎么做

如果切换后发现不稳定，直接回到 `chroma`。

### 回滚步骤

1. 把运行环境改回：
   - compose 场景下，直接去掉 canary override
   - 本机进程场景下，直接改回 `VECTOR_STORE_BACKEND=chroma`
2. 保留：
   - `VECTOR_STORE_MILVUS_URI`
   即使保留也没关系，因为 `chroma` 下不会使用它
3. 重启：
   - `backend-api`
   - `backend-worker`
4. 重新跑一轮最小回归，确认系统回到已知稳定状态

### 回滚后建议检查

- 问答是否恢复正常
- ingest 是否恢复正常
- fixed regression 的 retrieval / shadow 是否恢复到已知基线

compose 回滚示意：

```bash
$env:COMPOSE_FILE='docker-compose.yml'
docker compose up -d backend-api backend-worker
```

## 6. 当前建议

基于当前结果，更稳妥的做法是：

- 默认仍保持 `chroma`
- 把 `milvus` 当成“已达灰度准备完成”的候选后端
- 后续先按环境做小范围切换，不要直接全局扶正

## 7. 当前已验证的 canary 动作

截至 `2026-04-18`，这套 compose canary 已实际跑过：

```bash
$env:COMPOSE_FILE='docker-compose.yml;docker-compose.milvus-canary.yml'
docker compose up -d backend-api backend-worker
docker compose exec -T backend-api python scripts/compare_real_vector_store_retrieval.py --backends milvus --milvus-uri http://milvus-standalone:19530 --label vector-store-retrieval-compose-canary-milvus --limit 4
python backend/scripts/run_docker_task_queue_retry_smoke.py
```

实际结果：

- `backend-api / backend-worker` 容器内都确认是 `VECTOR_STORE_BACKEND=milvus`
- retrieval-only canary：`4/4 retrieval / final_context / citation hit`
- Docker ingest retry smoke：
  - `first_attempt_status = pending`
  - `final_status = completed`
  - `retry_event_count = 1`

后续又在同一套 canary 环境里补跑了：

```bash
$env:PYTHONPATH='D:\System\Desktop\regurag\backend'
$env:COMPOSE_FILE='docker-compose.yml;docker-compose.milvus-canary.yml'
python -m scripts.run_docker_task_queue_concurrency_smoke
python -m scripts.run_docker_task_queue_mixed_recovery_smoke --total-timeout-seconds 240
```

补充结果：

- concurrency smoke：
  - `retried_task_final_status = completed`
  - `companion_task_final_status = completed`
  - `retried_task_retry_event_count = 1`
- mixed recovery smoke：
  - `retry_task_final_status = completed`
  - `stale_task_final_status = completed`
  - `retry_event_count = 1`
  - `stale_reclaimed = true`

如果你不想手工分别敲这些命令，当前也已经有统一入口：

```bash
python backend/scripts/run_milvus_canary_checks.py
```

它会顺序执行并汇总：

- readiness check
- 容器环境核对
- retrieval-only canary
- retry smoke
- concurrency smoke
- mixed recovery smoke

输出文件：

- [backend/evals/results/milvus-canary-checks-summary.json](/D:/System/Desktop/regurag/backend/evals/results/milvus-canary-checks-summary.json)

截至 `2026-04-18` 最新一次统一检查结果：

- `canary_ok = true`
