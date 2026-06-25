"""检查元数据数据库后端是否可用。

用于确认当前 DATABASE_URL 指向的 MySQL/PostgreSQL 已连通，并且核心表结构已经创建。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine.url import make_url

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings

DEFAULT_REQUIRED_TABLES = (
    "knowledge_bases",
    "documents",
    "tasks",
    "task_events",
    "conversations",
    "messages",
    "message_contexts",
)


@dataclass(slots=True)
class MetadataBackendSummary:
    database_url: str
    dialect: str
    driver: str
    database_name: str | None
    required_tables: list[str]
    existing_tables: list[str]
    missing_required_tables: list[str]
    connection_ok: bool
    schema_ready: bool


def normalize_database_dialect(drivername: str) -> str:
    dialect = drivername.strip().lower().split("+", 1)[0]
    if dialect.startswith("postgres"):
        return "postgresql"
    return dialect


def build_metadata_backend_summary(
    *,
    database_url: str,
    existing_tables: Iterable[str],
    required_tables: Iterable[str],
    connection_ok: bool,
) -> MetadataBackendSummary:
    parsed = make_url(database_url)
    existing = sorted({table for table in existing_tables if table})
    required = [table for table in required_tables if table]
    missing = [table for table in required if table not in existing]
    return MetadataBackendSummary(
        database_url=database_url,
        dialect=normalize_database_dialect(parsed.drivername),
        driver=parsed.drivername,
        database_name=parsed.database,
        required_tables=required,
        existing_tables=existing,
        missing_required_tables=missing,
        connection_ok=connection_ok,
        schema_ready=connection_ok and not missing,
    )


def inspect_metadata_backend(*, database_url: str, required_tables: Iterable[str]) -> MetadataBackendSummary:
    engine = create_engine(database_url, pool_pre_ping=True)
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    inspector = inspect(engine)
    return build_metadata_backend_summary(
        database_url=database_url,
        existing_tables=inspector.get_table_names(),
        required_tables=required_tables,
        connection_ok=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the configured metadata database connection and required table set."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file path for the summary output.",
    )
    parser.add_argument(
        "--expect-dialect",
        default=None,
        help="Optional expected normalized database dialect, for example mysql or postgresql.",
    )
    parser.add_argument(
        "--required-table",
        action="append",
        dest="required_tables",
        default=None,
        help="Additional required table name. Repeat to override the default set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    required_tables = args.required_tables or list(DEFAULT_REQUIRED_TABLES)
    summary = inspect_metadata_backend(
        database_url=settings.database_url,
        required_tables=required_tables,
    )
    payload = asdict(summary)
    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.expect_dialect is not None:
        expected = normalize_database_dialect(args.expect_dialect)
        if summary.dialect != expected:
            return 1
    return 0 if summary.schema_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
