"""后端运行配置。

Settings 统一从环境变量和 backend/.env 读取配置。路径类配置在属性里解析成绝对路径，
数据库和队列配置在属性里做规范化，避免业务代码到处重复处理字符串细节。
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine.url import make_url

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """应用配置对象，字段 alias 对应 .env / Docker 环境变量名。"""

    app_name: str = Field(default="ReguRAG Backend", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="127.0.0.1", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")

    database_url: str = Field(
        default="mysql+pymysql://user:password@127.0.0.1:3306/regurag?charset=utf8mb4",
        alias="DATABASE_URL",
    )
    chroma_path: str = Field(default="./chroma_db", alias="CHROMA_PATH")
    uploads_dir: str = Field(default="./data/uploads", alias="UPLOADS_DIR")

    cors_allow_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="CORS_ALLOW_ORIGINS",
    )
    cors_allow_origin_regex: str | None = Field(
        default=r"^https?://(localhost|127\.0\.0\.1|10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(:\d+)?$",
        alias="CORS_ALLOW_ORIGIN_REGEX",
    )
    cors_allow_credentials: bool = Field(default=True, alias="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: str = Field(default="*", alias="CORS_ALLOW_METHODS")
    cors_allow_headers: str = Field(default="*", alias="CORS_ALLOW_HEADERS")

    openai_api_key: str = Field(default="replace-me", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    openai_timeout_seconds: int = Field(default=45, alias="OPENAI_TIMEOUT_SECONDS")
    openai_max_tokens: int = Field(default=700, alias="OPENAI_MAX_TOKENS")

    rewrite_api_key: str = Field(default="replace-me", alias="REWRITE_API_KEY")
    rewrite_base_url: str = Field(default="https://api.openai.com/v1", alias="REWRITE_BASE_URL")
    rewrite_model: str = Field(default="gpt-4.1-mini", alias="REWRITE_MODEL")
    intent_local_classifier_enabled: bool = Field(default=True, alias="INTENT_LOCAL_CLASSIFIER_ENABLED")
    intent_local_classifier_model: str | None = Field(default=None, alias="INTENT_LOCAL_CLASSIFIER_MODEL")
    intent_local_classifier_artifact_path: str = Field(
        default="./data/models/intent_classifier.pt",
        alias="INTENT_LOCAL_CLASSIFIER_ARTIFACT_PATH",
    )
    intent_local_classifier_min_score: float = Field(
        default=0.55,
        alias="INTENT_LOCAL_CLASSIFIER_MIN_SCORE",
        ge=0.0,
        le=1.0,
    )
    intent_local_classifier_min_margin: float = Field(
        default=0.03,
        alias="INTENT_LOCAL_CLASSIFIER_MIN_MARGIN",
        ge=0.0,
        le=1.0,
    )
    intent_llm_classifier_enabled: bool = Field(default=False, alias="INTENT_LLM_CLASSIFIER_ENABLED")
    intent_llm_classifier_api_key: str | None = Field(default=None, alias="INTENT_LLM_CLASSIFIER_API_KEY")
    intent_llm_classifier_base_url: str | None = Field(default=None, alias="INTENT_LLM_CLASSIFIER_BASE_URL")
    intent_llm_classifier_model: str | None = Field(default=None, alias="INTENT_LLM_CLASSIFIER_MODEL")
    intent_llm_classifier_timeout_seconds: int = Field(
        default=15,
        alias="INTENT_LLM_CLASSIFIER_TIMEOUT_SECONDS",
        gt=0,
    )
    intent_llm_classifier_max_tokens: int = Field(
        default=64,
        alias="INTENT_LLM_CLASSIFIER_MAX_TOKENS",
        gt=0,
    )

    embedding_model_name: str = Field(default="BAAI/bge-small-zh-v1.5", alias="EMBEDDING_MODEL_NAME")
    reranker_model_name: str = Field(default="BAAI/bge-reranker-base", alias="RERANKER_MODEL_NAME")
    vector_store_backend: str = Field(default="chroma", alias="VECTOR_STORE_BACKEND")
    vector_store_milvus_uri: str = Field(default="http://127.0.0.1:19530", alias="VECTOR_STORE_MILVUS_URI")
    vector_store_milvus_token: str | None = Field(default=None, alias="VECTOR_STORE_MILVUS_TOKEN")
    retrieval_sparse_provider: str = Field(default="sqlite_fts", alias="RETRIEVAL_SPARSE_PROVIDER")
    retrieval_dense_top_k: int | None = Field(default=None, alias="RETRIEVAL_DENSE_TOP_K", ge=1)
    retrieval_sparse_top_k: int | None = Field(default=None, alias="RETRIEVAL_SPARSE_TOP_K", ge=1)
    retrieval_sparse_min_hits: int = Field(default=2, alias="RETRIEVAL_SPARSE_MIN_HITS", ge=1)
    retrieval_enable_sparse: bool = Field(default=True, alias="RETRIEVAL_ENABLE_SPARSE")
    chroma_collection_name: str = Field(default="regurag_docs", alias="CHROMA_COLLECTION_NAME")
    config_profile: str = Field(default="rules_cn", alias="CONFIG_PROFILE")
    knowledge_base_subject: str = Field(default="通用知识主题", alias="KNOWLEDGE_BASE_SUBJECT")
    source_document_path: str | None = Field(default=None, alias="SOURCE_DOCUMENT_PATH")
    bootstrap_default_knowledge_base: bool = Field(default=False, alias="BOOTSTRAP_DEFAULT_KNOWLEDGE_BASE")
    default_knowledge_base_id: str = Field(default="kb_001", alias="DEFAULT_KNOWLEDGE_BASE_ID")
    default_knowledge_base_name: str = Field(default="默认知识库", alias="DEFAULT_KNOWLEDGE_BASE_NAME")
    default_knowledge_base_description: str = Field(default="系统启动时自动导入的默认知识库", alias="DEFAULT_KNOWLEDGE_BASE_DESCRIPTION")
    default_knowledge_base_domain: str = Field(default="general", alias="DEFAULT_KNOWLEDGE_BASE_DOMAIN")
    child_chunk_size: int = Field(default=50, alias="CHILD_CHUNK_SIZE")
    top_k_mmr: int = Field(default=8, alias="TOP_K_MMR")
    chat_auto_route_enabled: bool = Field(default=True, alias="CHAT_AUTO_ROUTE_ENABLED")
    chat_shadow_compare_enabled: bool = Field(default=False, alias="CHAT_SHADOW_COMPARE_ENABLED")
    chat_shadow_compare_sample_rate: float = Field(
        default=0.0,
        alias="CHAT_SHADOW_COMPARE_SAMPLE_RATE",
        ge=0.0,
        le=1.0,
    )
    shadow_retrieval_backend: str = Field(default="legacy", alias="SHADOW_RETRIEVAL_BACKEND")
    shadow_milvus_uri: str = Field(default="./data/milvus_shadow.db", alias="SHADOW_MILVUS_URI")
    shadow_milvus_token: str | None = Field(default=None, alias="SHADOW_MILVUS_TOKEN")
    shadow_milvus_drop_old: bool = Field(default=False, alias="SHADOW_MILVUS_DROP_OLD")
    pipeline_bootstrap_lock_timeout_seconds: int = Field(
        default=300,
        alias="PIPELINE_BOOTSTRAP_LOCK_TIMEOUT_SECONDS",
        gt=0,
    )
    task_queue_backend: str = Field(default="mysql", alias="TASK_QUEUE_BACKEND")
    redis_url: str = Field(default="redis://127.0.0.1:6379/0", alias="REDIS_URL")
    task_queue_redis_pending_list_key: str = Field(
        default="regurag:tasks:pending",
        alias="TASK_QUEUE_REDIS_PENDING_LIST_KEY",
    )
    task_queue_redis_marker_prefix: str = Field(
        default="regurag:tasks:queued",
        alias="TASK_QUEUE_REDIS_MARKER_PREFIX",
    )
    task_worker_poll_interval_seconds: float = Field(
        default=2.0,
        alias="TASK_WORKER_POLL_INTERVAL_SECONDS",
        gt=0.0,
    )
    task_worker_lease_seconds: int = Field(
        default=1800,
        alias="TASK_WORKER_LEASE_SECONDS",
        gt=0,
    )
    task_worker_max_attempts: int = Field(
        default=3,
        alias="TASK_WORKER_MAX_ATTEMPTS",
        gt=0,
    )
    task_worker_inject_fail_on_first_attempt_document_ids: str = Field(
        default="",
        alias="TASK_WORKER_INJECT_FAIL_ON_FIRST_ATTEMPT_DOCUMENT_IDS",
    )
    task_monitor_window_hours: int = Field(
        default=24,
        alias="TASK_MONITOR_WINDOW_HOURS",
        gt=0,
    )
    task_monitor_long_running_seconds: int = Field(
        default=3600,
        alias="TASK_MONITOR_LONG_RUNNING_SECONDS",
        gt=0,
    )
    task_monitor_recent_failure_threshold: int = Field(
        default=3,
        alias="TASK_MONITOR_RECENT_FAILURE_THRESHOLD",
        ge=1,
    )
    task_monitor_recent_retry_threshold: int = Field(
        default=5,
        alias="TASK_MONITOR_RECENT_RETRY_THRESHOLD",
        ge=1,
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def backend_root(self) -> Path:
        return BACKEND_ROOT

    @property
    def resolved_chroma_path(self) -> Path:
        """Chroma、本地稀疏索引和模型锁文件共用的本地数据目录。"""
        return self._resolve_path(self.chroma_path)

    @property
    def resolved_source_document_path(self) -> Path | None:
        return self._resolve_path(self.source_document_path)

    @property
    def resolved_uploads_dir(self) -> Path:
        """上传文件根目录；相对路径按 backend 根目录解析。"""
        return self._resolve_path(self.uploads_dir)

    @property
    def resolved_intent_local_classifier_artifact_path(self) -> Path:
        resolved = self._resolve_path(self.intent_local_classifier_artifact_path)
        if resolved is None:
            raise ValueError("INTENT_LOCAL_CLASSIFIER_ARTIFACT_PATH cannot be empty")
        return resolved

    @property
    def resolved_shadow_milvus_uri(self) -> str:
        uri = self.shadow_milvus_uri.strip()
        if "://" in uri:
            return uri
        resolved = self._resolve_path(uri)
        if resolved is None:
            raise ValueError("SHADOW_MILVUS_URI cannot be empty")
        return str(resolved)

    @property
    def normalized_vector_store_backend(self) -> str:
        return self.vector_store_backend.strip().lower()

    @property
    def resolved_vector_store_milvus_uri(self) -> str:
        uri = self.vector_store_milvus_uri.strip()
        if "://" in uri:
            return uri
        resolved = self._resolve_path(uri)
        if resolved is None:
            raise ValueError("VECTOR_STORE_MILVUS_URI cannot be empty")
        return str(resolved)

    @property
    def normalized_task_queue_backend(self) -> str:
        return self.task_queue_backend.strip().lower()

    @property
    def normalized_database_dialect(self) -> str:
        """提取数据库方言，供 MySQL/PostgreSQL canary 逻辑判断。"""
        drivername = make_url(self.database_url).drivername.strip().lower()
        dialect = drivername.split("+", 1)[0]
        if dialect.startswith("postgres"):
            return "postgresql"
        return dialect

    @property
    def resolved_redis_url(self) -> str:
        return self.redis_url.strip()

    @property
    def parsed_task_worker_inject_fail_on_first_attempt_document_ids(self) -> list[str]:
        return self._parse_csv(self.task_worker_inject_fail_on_first_attempt_document_ids)

    @property
    def parsed_cors_allow_origins(self) -> list[str]:
        return self._parse_csv(self.cors_allow_origins)

    @property
    def parsed_cors_allow_methods(self) -> list[str]:
        return self._parse_csv(self.cors_allow_methods)

    @property
    def parsed_cors_allow_headers(self) -> list[str]:
        return self._parse_csv(self.cors_allow_headers)

    def _resolve_path(self, path_like: str | None) -> Path | None:
        """把 .env 中的相对路径统一解析到 backend 根目录下。"""
        if not path_like:
            return None
        path = Path(path_like)
        if path.is_absolute():
            return path
        return (self.backend_root / path).resolve()

    @staticmethod
    def _parse_csv(value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    """缓存 Settings，避免每次请求重复读取 .env。"""
    return Settings()
