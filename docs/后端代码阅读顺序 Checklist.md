# 后端代码阅读顺序 Checklist

这份文档解决的问题很简单：

> 当你隔一段时间重新回到项目，不知道先看哪里时，按什么顺序读代码最省力。

建议配合这两份一起看：

- [后端主链路阅读地图.md](/D:/System/Desktop/regurag/docs/后端主链路阅读地图.md)
- [后端模块职责一览.md](/D:/System/Desktop/regurag/docs/后端模块职责一览.md)

---

## 一、5 分钟快速恢复上下文

每次重新进入项目，先不要翻实现细节。

按这个顺序：

1. 看 [项目进度.md](/D:/System/Desktop/regurag/项目进度.md)
   只看：当前状态、当前主线判断、最近几次验证结论

2. 看 [后端主链路阅读地图.md](/D:/System/Desktop/regurag/docs/后端主链路阅读地图.md)
   先把主线重新串起来

3. 看 [后端模块职责一览.md](/D:/System/Desktop/regurag/docs/后端模块职责一览.md)
   先分清每个核心模块归哪层

做到这里，你再进代码就不容易迷路。

---

## 二、第一次进代码时的推荐顺序

### Step 1：先看服务总入口

看：

- [rag_service.py](/D:/System/Desktop/regurag/backend/app/services/rag_service.py:684)

只盯 `RAGService.query()`，不要急着看整个文件。

这一步要回答的问题：

- 请求先经过哪些短路分支？
- 什么条件下才会真正进入主 RAG 链路？
- pipeline 返回之后又做了哪些后处理？

如果你这一步没看明白，后面看 `pipeline` 会丢全局感。

---

### Step 2：再看核心问答流水线

看：

- [pipeline.py](/D:/System/Desktop/regurag/backend/app/rag/pipeline.py:438)

只盯 `RAGPipeline.ask()`。

这一步要回答的问题：

- query 是怎么准备的？
- retrieval 是在哪里调用的？
- MMR / rerank / final context 是怎么接上的？
- 返回结果里 debug 和 citations 是在哪里构建的？

只要 `ask()` 看顺了，主链路就已经恢复 70%。

---

### Step 3：看检索层

按顺序看：

- [hybrid_retriever.py](/D:/System/Desktop/regurag/backend/app/rag/retrievers/hybrid_retriever.py:14)
- [policy.py](/D:/System/Desktop/regurag/backend/app/rag/retrievers/policy.py:7)
- [query_policy.py](/D:/System/Desktop/regurag/backend/app/rag/query_policy.py:9)
- [vector_store.py](/D:/System/Desktop/regurag/backend/app/rag/vector_store.py:13)

这一步要回答的问题：

- dense 和 sparse 是怎么合起来的？
- 检索策略参数由谁控制？
- `sqlite_fts / bm25 / scan / none` 是怎么分发的？
- `VectorStore` 负责到哪一层为止？

---

### Step 4：最后再看影子链路

看：

- [graph.py](/D:/System/Desktop/regurag/backend/app/workflows/rag/graph.py:13)
- [nodes.py](/D:/System/Desktop/regurag/backend/app/workflows/rag/nodes.py:25)

这一步不要把它当新系统看，而要把它当成：

> `RAGPipeline.ask()` 的拆分版

要回答的问题：

- 它覆盖了主链路里的哪几段？
- 哪些步骤和主链路共用同一套实现？
- 哪些外围治理仍留在 `RAGService`？

---

## 三、如果你现在只想排一个线上问答问题

不要按完整阅读顺序来，按问题定位顺序来。

### 1. 问题像“根本没走检索”

先看：

- `RAGService.query()`
- `IntentRouter`
- `KnowledgeBaseRouter`
- cross-domain / FAQ / off-topic 短路分支

因为很可能请求在进入 pipeline 前就被截断了。

### 2. 问题像“检索候选不对”

先看：

- `RAGPipeline.ask()` 里的 retrieval 调用
- `HybridRetriever.retrieve()`
- `VectorStore.search()`
- `VectorStore.keyword_search()` / 当前 sparse provider

### 3. 问题像“召回对了，但最终上下文不对”

先看：

- `MMR`
- `preserve_policy_keyword_docs`
- `preserve_query_keyword_docs`
- `build_parent_docs_for_rerank`
- `build_final_context_parents`

### 4. 问题像“上下文对了，但答案怪”

先看：

- `llm.generate(...)`
- answer guard
- no-answer 清理逻辑
- 评测 case 的 answer 判定口径

### 5. 问题像“graph 和主链路不一致”

先看：

- `run_shadow_graph`
- `nodes.py`
- `shadow_compare`
- 是否复用了同一个 `query_prep / retriever / reranker / llm`

---

## 四、如果你这次是要看 ingest / worker，不是问答链路

那就不要从 `RAGService.query()` 开始。

改成这个顺序：

1. [ingest_service.py](/D:/System/Desktop/regurag/backend/app/services/ingest_service.py)
2. [task_worker.py](/D:/System/Desktop/regurag/backend/app/workers/task_worker.py)
3. [backend/README.md](/D:/System/Desktop/regurag/backend/README.md)
4. 相关 MySQL 集成测试

也就是：

- ingest 看 service
- 执行看 worker
- 约束和运行方式看 README
- 真实行为看测试

---

## 五、最容易把自己读晕的 4 个误区

### 误区 1：一上来就读工具函数

像 `pipeline_steps.py` 这种文件，不要一开始就全看。

先有主线，再回头看工具函数；否则会看到很多局部逻辑，但不知道它们在哪一步生效。

### 误区 2：把 shadow graph 当成完整线上链路

不是。

它现在覆盖的是核心 pipeline，不是完整服务编排。

### 误区 3：把 `VectorStore` 当成完整检索层

不是。

它更偏底层索引存取层；真正的 retrieval 编排在 `Retriever`。

### 误区 4：试图一次性把所有模块都记住

没必要。

先记住：

- 入口在哪
- 主线怎么走
- 每层职责是什么

剩下的遇到问题再局部下钻。

---

## 六、每次重新进入项目时，最低限度要回答的 6 个问题

1. 这次请求先经过 `RAGService` 的哪些短路分支？
2. 它什么时候才真正进入 `pipeline.ask()`？
3. retrieval 是通过哪一层做的？
4. 当前 sparse provider 是什么？
5. rerank / final context 是在哪一步完成的？
6. 最终答案是在 pipeline 内完成，还是在 service 层又被 guard 改写了？

如果这 6 个问题能答出来，你基本就恢复上下文了。

---

## 七、推荐的最小阅读组合

如果时间很少，只读这 4 个点：

1. [rag_service.py](/D:/System/Desktop/regurag/backend/app/services/rag_service.py:684)
2. [pipeline.py](/D:/System/Desktop/regurag/backend/app/rag/pipeline.py:438)
3. [hybrid_retriever.py](/D:/System/Desktop/regurag/backend/app/rag/retrievers/hybrid_retriever.py:14)
4. [graph.py](/D:/System/Desktop/regurag/backend/app/workflows/rag/graph.py:13)

这 4 个点看明白，已经足够支撑你继续往下读了。

---

## 八、一句话版本

> 先看 service 入口，再看 pipeline 主线，再看 retriever，最后把 shadow 当成主线拆分版来读。
