from app.core.config import Settings
from scripts.check_metadata_backend import (
    build_metadata_backend_summary,
    normalize_database_dialect,
)


def test_normalize_database_dialect_collapses_postgres_aliases() -> None:
    assert normalize_database_dialect("postgresql+psycopg") == "postgresql"
    assert normalize_database_dialect("postgres+psycopg2") == "postgresql"
    assert normalize_database_dialect("mysql+pymysql") == "mysql"


def test_build_metadata_backend_summary_reports_missing_tables() -> None:
    summary = build_metadata_backend_summary(
        database_url="postgresql+psycopg://user:password@localhost:5432/regurag",
        existing_tables=["knowledge_bases", "documents", "tasks"],
        required_tables=["knowledge_bases", "documents", "tasks", "task_events"],
        connection_ok=True,
    )

    assert summary.dialect == "postgresql"
    assert summary.driver == "postgresql+psycopg"
    assert summary.database_name == "regurag"
    assert summary.missing_required_tables == ["task_events"]
    assert summary.schema_ready is False


def test_settings_normalized_database_dialect_supports_mysql_and_postgresql() -> None:
    mysql_settings = Settings(
        _env_file=None,
        DATABASE_URL="mysql+pymysql://user:password@127.0.0.1:3306/regurag?charset=utf8mb4",
    )
    postgresql_settings = Settings(
        _env_file=None,
        DATABASE_URL="postgresql+psycopg://user:password@127.0.0.1:5432/regurag",
    )

    assert mysql_settings.normalized_database_dialect == "mysql"
    assert postgresql_settings.normalized_database_dialect == "postgresql"
