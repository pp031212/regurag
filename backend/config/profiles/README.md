# Config Profiles

这组目录用于给同一套后端代码提供“可切换的业务配置包”。

## 选择方式

- 通过环境变量 `CONFIG_PROFILE` 选择当前业务场景。
- 如果某个 profile 目录下存在同名配置文件，则优先使用该文件。
- 如果 profile 下没有对应文件，则回退到 `backend/config/` 根目录下的通用默认配置。

## 设计目的

- 保留 RAG 对具体业务场景的针对性。
- 避免把业务词表、guard 规则、prompt 示例散落在源码里。
- 让后续新增业务场景时，只需新增一个 profile 目录，而不是改多处代码。

## 当前约定

- `backend/config/`：通用默认配置，适合 `general` 或未覆盖场景。
- `backend/config/profiles/rules_cn/`：当前 `培训管理 + 劳动法` 场景包。
- `backend/config/profiles/_template/`：新业务 profile 模板，可直接复制成新目录后修改。

## 常见配置文件说明

- `knowledge_base_routing.json`
  用于自动路由、领域关键词、知识库优先级和布尔匹配规则。
- `knowledge_base_domains.json`
  用于定义前端创建知识库时可选的业务域、默认业务域，以及后端创建知识库时的 domain 校验口径。
- `retrieval_rules.json`
  用于检索后处理，包括政策类触发词、必须保留片段、低价值上下文过滤等。
- `overview_rules.json`
  用于识别“某类问题一般怎么处理/有哪些情形”这类总括型问题，避免把业务主题词继续硬编码在源码里。
- `light_intents.json`
  用于问候、感谢、离题等轻量短路识别。
- `cross_domain_guard_rules.json`
  用于跨域复合问题的拆问提示和领域标签展示。
- `query_aliases.json`
  用于把用户口语化问法映射成更接近知识库原文的检索别名。
- `answer_guard_rules.json`
  用于限定词证据不足时触发保守回答。
- `query_rewriter_prompts.json`
  用于单轮检索改写和多轮历史补全的 prompt 模板与特定规则改写。
- `source_name_rules.json`
  用于把文档文件名规范化成展示用来源名。
