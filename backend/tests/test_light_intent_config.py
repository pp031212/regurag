from app.services.light_intent_config import (
    format_light_intent_response,
    load_light_intent_config,
    normalize_query,
    split_light_intent_clauses,
)


def test_light_intent_config_loads_from_json_files() -> None:
    config = load_light_intent_config()

    assert "规章" in config.policy_keywords
    assert "哈{2,}" in config.off_topic_patterns
    assert "今天天气怎么样" in config.off_topic_patterns
    assert any(rule.name == "greeting" for rule in config.intent_rules)
    assert "off_topic" in config.responses


def test_format_light_intent_response_uses_subject_placeholder() -> None:
    rendered = format_light_intent_response("我是 ReguRAG 助手，主要回答{subject}相关问题。", "劳动合同法")

    assert rendered == "我是 ReguRAG 助手，主要回答劳动合同法相关问题。"


def test_normalize_query_removes_spaces_and_punctuation() -> None:
    assert normalize_query("  谢谢你！ ") == "谢谢你"


def test_split_light_intent_clauses_keeps_multi_part_small_talk() -> None:
    assert split_light_intent_clauses("你好，你是？") == ("你好", "你是")
