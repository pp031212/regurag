"""SQLite FTS5 稀疏检索索引。

该索引和 Chroma/Milvus dense 向量库并行维护，用于关键词召回补充。它不存 embedding，
只存 child/parent 文本和定位元数据。
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from ...workflows.rag.pipeline_steps import RagDoc


class SQLiteFTSSparseIndex:
    """基于 SQLite FTS5 的本地关键词索引。"""

    def __init__(self, *, db_path: str | None, collection_name: str) -> None:
        self.db_file = self._resolve_db_file(db_path)
        self.collection_name = collection_name
        self._ensure_schema()

    @staticmethod
    def _resolve_db_file(db_path: str | None) -> Path:
        """把向量库目录映射到同级 sparse_index.sqlite3。"""
        if db_path:
            base = Path(db_path)
            parent = base.parent if base.suffix else base
            parent.mkdir(parents=True, exist_ok=True)
            return parent / "sparse_index.sqlite3"

        fallback_dir = Path(tempfile.gettempdir()) / "regurag_sparse_index"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir / "sparse_index.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_file)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            # collection 字段用于同一个 sqlite 文件承载多个知识库 collection。
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS sparse_docs
                USING fts5(
                    collection UNINDEXED,
                    doc_id UNINDEXED,
                    document_id UNINDEXED,
                    parent_id UNINDEXED,
                    source_type UNINDEXED,
                    page_number UNINDEXED,
                    block_index UNINDEXED,
                    child_text,
                    parent_text,
                    all_text,
                    tokenize = 'unicode61'
                )
                """
            )

    @staticmethod
    def _match_expression(query_keywords: list[str]) -> str:
        escaped = [f'"{keyword.replace(chr(34), chr(34) * 2)}"' for keyword in query_keywords if keyword]
        return " OR ".join(escaped)

    def upsert_documents(self, items: list[tuple[dict[str, object], str]]) -> None:
        """同步写入 dense chunk 对应的 sparse 文本。"""
        if not items:
            return

        with self._connect() as connection:
            for item, doc_id in items:
                connection.execute(
                    "DELETE FROM sparse_docs WHERE collection = ? AND doc_id = ?",
                    (self.collection_name, doc_id),
                )
                child_text = str(item["child_text"])
                parent_text = str(item["parent_text"])
                connection.execute(
                    """
                    INSERT INTO sparse_docs (
                        collection,
                        doc_id,
                        document_id,
                        parent_id,
                        source_type,
                        page_number,
                        block_index,
                        child_text,
                        parent_text,
                        all_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.collection_name,
                        doc_id,
                        str(item.get("document_id") or ""),
                        str(item["parent_id"]),
                        str(item.get("source_type") or "text"),
                        int(item.get("page_number", 0) or 0),
                        int(item.get("block_index", -1) or -1),
                        child_text,
                        parent_text,
                        f"{child_text}\n{parent_text}",
                    ),
                )

    def delete_document(self, document_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM sparse_docs WHERE collection = ? AND document_id = ?",
                (self.collection_name, document_id),
            )

    def clear_collection(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM sparse_docs WHERE collection = ?",
                (self.collection_name,),
            )

    def search(
        self,
        query_keywords: list[str],
        *,
        min_hits: int = 2,
        top_k: int = 5,
    ) -> list[RagDoc]:
        """按关键词检索候选 chunk，并用命中数 + bm25 rank 做排序。"""
        if not query_keywords:
            return []

        match_expression = self._match_expression(query_keywords)
        if not match_expression:
            return []

        candidate_limit = max(top_k * 20, 50)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    doc_id,
                    document_id,
                    parent_id,
                    source_type,
                    page_number,
                    block_index,
                    child_text,
                    parent_text,
                    all_text,
                    bm25(sparse_docs) AS rank
                FROM sparse_docs
                WHERE collection = ?
                  AND sparse_docs MATCH ?
                ORDER BY rank ASC
                LIMIT ?
                """,
                (self.collection_name, match_expression, candidate_limit),
            ).fetchall()

        matches: list[tuple[int, float, int, RagDoc]] = []
        for row in rows:
            text = str(row["all_text"] or "")
            hit_count = sum(1 for keyword in query_keywords if keyword and keyword in text)
            if hit_count < min_hits:
                continue

            # sparse 命中的结果没有 embedding，只能参与补充召回和最终上下文候选。
            child_text = str(row["child_text"] or "")
            matches.append(
                (
                    hit_count,
                    float(row["rank"] or 0.0),
                    len(child_text),
                    {
                        "id": str(row["doc_id"]),
                        "child_text": child_text,
                        "parent_text": str(row["parent_text"] or ""),
                        "parent_id": str(row["parent_id"] or ""),
                        "document_id": str(row["document_id"] or ""),
                        "source_type": str(row["source_type"] or "text"),
                        "page_number": int(row["page_number"] or 0),
                        "block_index": int(row["block_index"] or -1),
                        "distance": None,
                        "embedding": None,
                        "keyword_hit_count": hit_count,
                    },
                )
            )

        matches.sort(key=lambda item: (-item[0], item[1], -item[2]))
        return [item[3] for item in matches[:top_k]]
