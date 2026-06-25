from pathlib import Path

import pytest

from app.core.config import BACKEND_ROOT, Settings


def _build_settings_with_clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    for key in (
        "INTENT_LOCAL_CLASSIFIER_MODEL",
        "INTENT_LLM_CLASSIFIER_API_KEY",
        "INTENT_LLM_CLASSIFIER_BASE_URL",
        "INTENT_LLM_CLASSIFIER_MODEL",
        "CHROMA_COLLECTION_NAME",
        "CONFIG_PROFILE",
        "KNOWLEDGE_BASE_SUBJECT",
        "DEFAULT_KNOWLEDGE_BASE_NAME",
        "DEFAULT_KNOWLEDGE_BASE_DOMAIN",
        "SOURCE_DOCUMENT_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    return Settings(_env_file=env_path)


def test_settings_env_file_points_to_backend_dotenv() -> None:
    env_file = Settings.model_config.get("env_file")

    assert env_file == BACKEND_ROOT / ".env"


def test_settings_expose_shadow_compare_rollout_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _build_settings_with_clean_env(tmp_path, monkeypatch)

    assert settings.chat_shadow_compare_enabled is False
    assert settings.chat_shadow_compare_sample_rate == 0.0
    assert settings.intent_local_classifier_enabled is True
    assert settings.intent_local_classifier_model is None
    assert settings.intent_local_classifier_artifact_path == "./data/models/intent_classifier.pt"
    assert settings.resolved_intent_local_classifier_artifact_path == BACKEND_ROOT / "data" / "models" / "intent_classifier.pt"
    assert settings.intent_local_classifier_min_score == 0.55
    assert settings.intent_local_classifier_min_margin == 0.03
    assert settings.intent_llm_classifier_enabled is False
    assert settings.intent_llm_classifier_api_key is None
    assert settings.intent_llm_classifier_base_url is None
    assert settings.intent_llm_classifier_model is None
    assert settings.intent_llm_classifier_timeout_seconds == 15
    assert settings.intent_llm_classifier_max_tokens == 64
    assert settings.task_worker_poll_interval_seconds == 2.0
    assert settings.task_worker_lease_seconds == 1800
    assert settings.task_worker_max_attempts == 3
    assert settings.task_monitor_window_hours == 24
    assert settings.task_monitor_long_running_seconds == 3600
    assert settings.task_monitor_recent_failure_threshold == 3
    assert settings.task_monitor_recent_retry_threshold == 5
    assert settings.retrieval_sparse_provider == "sqlite_fts"
    assert settings.retrieval_dense_top_k is None
    assert settings.retrieval_sparse_top_k is None
    assert settings.retrieval_sparse_min_hits == 2
    assert settings.retrieval_enable_sparse is True


def test_settings_use_generic_default_knowledge_base_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _build_settings_with_clean_env(tmp_path, monkeypatch)

    assert settings.chroma_collection_name == "regurag_docs"
    assert settings.config_profile == "rules_cn"
    assert settings.knowledge_base_subject == "通用知识主题"
    assert settings.default_knowledge_base_name == "默认知识库"
    assert settings.default_knowledge_base_domain == "general"
    assert settings.source_document_path is None
    assert settings.resolved_source_document_path is None
