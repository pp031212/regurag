from pathlib import Path

from app.core.config import BACKEND_ROOT, get_settings
from app.core.config_profiles import resolve_config_path
from app.rag.query_alias_config import load_query_alias_config
from app.rag.query_rewriter_config import load_query_rewriter_prompt_config
from app.rag.retrieval_rules_config import load_retrieval_rules_config
from app.services.answer_guard_config import load_answer_guard_config
from app.services.cross_domain_guard_config import load_cross_domain_guard_config
from app.services.knowledge_base_routing_config import load_knowledge_base_routing_config
from app.services.light_intent_config import load_light_intent_config


def _clear_profile_caches() -> None:
    get_settings.cache_clear()
    load_query_alias_config.cache_clear()
    load_query_rewriter_prompt_config.cache_clear()
    load_knowledge_base_routing_config.cache_clear()
    load_retrieval_rules_config.cache_clear()
    load_answer_guard_config.cache_clear()
    load_cross_domain_guard_config.cache_clear()
    load_light_intent_config.cache_clear()


def test_resolve_config_path_prefers_profile_specific_file() -> None:
    path = resolve_config_path("knowledge_base_routing.json", profile="rules_cn")

    assert path.parts[-3:] == ("profiles", "rules_cn", "knowledge_base_routing.json")


def test_general_profile_loads_generic_configs(monkeypatch) -> None:
    monkeypatch.setenv("CONFIG_PROFILE", "general")
    _clear_profile_caches()

    routing = load_knowledge_base_routing_config()
    aliases = load_query_alias_config()
    rewriter = load_query_rewriter_prompt_config()
    retrieval = load_retrieval_rules_config()
    answer_guard = load_answer_guard_config()
    light_intent = load_light_intent_config()
    guard = load_cross_domain_guard_config()

    assert "general" in routing.domains.keywords
    assert not routing.knowledge_base_rules
    assert aliases.term_aliases == {}
    assert rewriter.subject_hint == "知识库资料/制度文本"
    assert retrieval.low_value_parent_keywords == ()
    assert answer_guard.qualifier_rules == ()
    assert "和鸣" not in light_intent.policy_keywords
    assert guard.domain_labels == {"general": "通用知识"}

    _clear_profile_caches()


def test_profile_template_contains_expected_files() -> None:
    template_dir = BACKEND_ROOT / "config" / "profiles" / "_template"

    expected = {
        "README.md",
        "knowledge_base_routing.json",
        "retrieval_rules.json",
        "light_intents.json",
        "cross_domain_guard_rules.json",
        "query_aliases.json",
        "answer_guard_rules.json",
        "query_rewriter_prompts.json",
        "source_name_rules.json",
    }

    existing = {path.name for path in Path(template_dir).iterdir()}

    assert expected.issubset(existing)
