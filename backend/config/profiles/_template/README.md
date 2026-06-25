# _template Profile

这个目录是“新业务 profile 模板”，用于从零创建新的配置包。

## 使用方式

1. 复制整个 `_template` 目录并重命名，例如 `enterprise_policy`。
2. 按新业务场景修改各个 JSON 文件中的关键词、示例、规则和提示词。
3. 在运行环境中设置 `CONFIG_PROFILE=<你的目录名>`。
4. 如果某个配置文件当前不需要覆盖，可以直接删掉，让系统回退到 `backend/config/` 根目录下的通用默认配置。

## 设计原则

- 只放“结构化骨架”和注释说明，不放当前项目特有业务词。
- 文件名必须与根目录配置同名，否则不会被 profile 加载器识别。
- 每个 JSON 都应该在顶层保留 `_comment`，必要时再加 `_profile_comment`。

## 推荐修改顺序

1. `knowledge_base_routing.json`
2. `knowledge_base_domains.json`
3. `light_intents.json`
4. `query_aliases.json`
5. `retrieval_rules.json`
6. `overview_rules.json`
7. `answer_guard_rules.json`
8. `query_rewriter_prompts.json`
9. `cross_domain_guard_rules.json`
10. `source_name_rules.json`
