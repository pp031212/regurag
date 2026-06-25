from app.rag.query_rewriter import QueryRewriter
from app.rag.query_rewriter_config import load_query_rewriter_prompt_config


def test_query_rewriter_prompt_config_loads_from_json_file() -> None:
    config = load_query_rewriter_prompt_config()

    assert "{query}" in config.rewrite_prompt_template
    assert "{history_text}" in config.history_rewrite_prompt_template
    assert "规章制度/扣分制度" in config.subject_hint
    assert any("第四十二条" in rule.rewritten_query for rule in config.rewrite_rules)


def test_build_rewrite_prompt_uses_config_placeholders() -> None:
    rewriter = object.__new__(QueryRewriter)
    rewriter.prompt_config = load_query_rewriter_prompt_config()

    prompt = rewriter._build_rewrite_prompt("迟到怎么处理")

    assert "规章制度/扣分制度" in prompt
    assert "迟到 早退 旷课 请假 打架 玩手机 吸烟 宿舍卫生" in prompt
    assert "用户问题：迟到怎么处理" in prompt


def test_build_history_rewrite_prompt_uses_config_placeholders() -> None:
    rewriter = object.__new__(QueryRewriter)
    rewriter.prompt_config = load_query_rewriter_prompt_config()

    prompt = rewriter._build_history_rewrite_prompt("那如果是第一次呢", "user: 迟到会怎么处理")

    assert "最近对话：" in prompt
    assert "user: 迟到会怎么处理" in prompt
    assert "当前问题：" in prompt
    assert "那如果是第一次呢" in prompt


def test_match_configured_rewrite_returns_labor_law_overview_query() -> None:
    rewriter = object.__new__(QueryRewriter)
    rewriter.prompt_config = load_query_rewriter_prompt_config()

    rewritten = rewriter._match_configured_rewrite("用人单位辞退员工必须要满足什么条件才能辞退？")

    assert rewritten is not None
    assert "第三十九条" in rewritten
    assert "第四十条" in rewritten
    assert "第四十一条" in rewritten
    assert "第四十二条" in rewritten
