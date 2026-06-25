"""分块 profile 和稀疏检索 provider 的离线 A/B 评测。

使用固定 fixture 和确定性向量/重排/LLM 替身，评估不同分块策略或稀疏 provider
对检索、最终上下文、引用和答案命中的影响。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from app.evals.eval_report_compare import compare_eval_reports
from app.rag.document_processor import DocumentProcessor
from app.rag.pipeline import RAGPipeline
from app.rag.retrievers import RetrievalPolicy, build_default_hybrid_retriever
from app.rag.structured_document_processor import StructuredDocumentProcessor

TOKEN_GROUPS = (
    ("病假", "病假"),
    ("就诊证明", "就诊证明"),
    ("事假", "事假"),
    ("毕业答辩", "毕业答辩"),
    ("单独报批", "单独报批"),
    ("一次", "一次"),
    ("特殊事项请假", "特殊事项请假"),
    ("上课睡觉", "上课睡觉"),
    ("睡觉", "睡觉"),
    ("宿舍", "宿舍"),
    ("打牌", "打牌"),
    ("打游戏", "打游戏"),
    ("扣5分", "扣5分"),
    ("扣10分", "扣10分"),
    ("退学", "退学"),
    ("学费不予退还", "学费不予退还"),
)


QUERY_KEYWORD_HINTS = {
    "病假": "病假 就诊证明",
    "毕业答辩": "特殊事项请假 单独报批 一次",
    "上课期间睡觉": "上课睡觉 扣5分",
    "宿舍打牌": "宿舍 打牌 打游戏 扣10分",
    "访客证丢失": "访客证 丢失 补办 临时通行码 蓝卡编号",
    "夜归返校": "夜归返校 登记 签名",
    "快递柜违规存放": "快递柜 违规存放 清柜",
    "借用储物柜钥匙": "储物柜钥匙 登记 银卡编码",
    "临时出门条": "临时出门条 填写 黄卡序号",
    "临时借用雨伞": "临时借用雨伞 登记 灰卡编号",
    "备用雨伞违规摆放": "备用雨伞 违规摆放 锁柜",
}


@dataclass(slots=True)
class EvalCase:
    """离线评测样本。"""

    id: str
    question: str
    answer_mode: str
    expected_answer_keywords: list[str]
    expected_context_keywords: list[str]
    category: str
    notes: str = ""


@dataclass(slots=True)
class EvalCaseResult:
    """单个样本的评测结果。"""

    id: str
    question: str
    category: str
    answer_mode: str
    answer: str
    retrieval_hit: bool
    final_context_hit: bool
    citation_hit: bool
    answer_hit: bool
    answer_hit_ratio: float
    matched_answer_keywords: list[str]
    missing_answer_keywords: list[str]
    matched_context_keywords: list[str]
    error_type: str
    debug: dict[str, object]
    citations: list[dict[str, object]]


def normalize_text(text: str) -> str:
    """归一化文本，便于中文关键词匹配。"""
    lowered = text.lower()
    collapsed = re.sub(r"\s+", "", lowered)
    return re.sub(r"[!！?？,，.。~～、:：;；'\"“”‘’（）()\-\[\]{}]+", "", collapsed)


def keyword_matches(text: str, keywords: list[str]) -> tuple[list[str], list[str]]:
    """返回命中和缺失的关键词。"""
    normalized_text = normalize_text(text)
    matched: list[str] = []
    missing: list[str] = []
    for keyword in keywords:
        if normalize_text(keyword) in normalized_text:
            matched.append(keyword)
        else:
            missing.append(keyword)
    return matched, missing


def load_eval_cases(path: Path) -> list[EvalCase]:
    """从 JSONL 读取离线评测样本。"""
    records: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(
                EvalCase(
                    id=str(payload["id"]),
                    question=str(payload["question"]),
                    answer_mode=str(payload.get("answer_mode", "grounded")),
                    expected_answer_keywords=[str(item) for item in payload["expected_answer_keywords"]],
                    expected_context_keywords=[str(item) for item in payload["expected_context_keywords"]],
                    category=str(payload.get("category", "unknown")),
                    notes=str(payload.get("notes", "")),
                )
            )
    if not records:
        raise ValueError(f"No evaluation cases found in {path}")
    return records


def classify_error(
    answer_mode: str,
    retrieval_hit: bool,
    final_context_hit: bool,
    answer_hit: bool,
    citation_hit: bool,
) -> str:
    """根据命中阶段把失败归因到 retrieval/rerank/generation/citation。"""
    if answer_mode == "no_answer":
        return "ok" if answer_hit else "fallback_error"
    if retrieval_hit and final_context_hit and answer_hit and citation_hit:
        return "ok"
    if not retrieval_hit:
        return "retrieval_error"
    if not final_context_hit:
        return "rerank_or_filter_error"
    if not answer_hit:
        return "generation_error"
    if not citation_hit:
        return "citation_error"
    return "needs_review"


def evaluate_answer_hit(case: EvalCase, answer_text: str) -> tuple[bool, float, list[str], list[str]]:
    """判断答案是否覆盖期望关键词。"""
    matched_answer_keywords, missing_answer_keywords = keyword_matches(answer_text, case.expected_answer_keywords)
    if case.answer_mode == "no_answer":
        answer_hit = len(matched_answer_keywords) > 0
        answer_hit_ratio = 1.0 if answer_hit else 0.0
        return answer_hit, answer_hit_ratio, matched_answer_keywords, missing_answer_keywords

    answer_hit_ratio = (
        len(matched_answer_keywords) / len(case.expected_answer_keywords) if case.expected_answer_keywords else 0.0
    )
    answer_hit = answer_hit_ratio == 1.0
    return answer_hit, answer_hit_ratio, matched_answer_keywords, missing_answer_keywords


def join_debug_chunk_texts(debug_payload: dict[str, object], key: str) -> str:
    """拼接 debug 中的 chunk 文本用于关键词检查。"""
    chunks = debug_payload.get(key, [])
    if not isinstance(chunks, list):
        return ""
    texts: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        parent_text = chunk.get("parent_text")
        child_text = chunk.get("child_text")
        if isinstance(parent_text, str):
            texts.append(parent_text)
        if isinstance(child_text, str):
            texts.append(child_text)
    return "\n".join(texts)


class DeterministicCollection:
    """模拟向量库 collection.get，供 hybrid retriever 读取候选元数据。"""

    def __init__(self, store: "DeterministicVectorStore") -> None:
        self.store = store

    def get(self, include: list[str] | None = None) -> dict[str, list[object]]:
        documents = list(self.store.documents)
        return {
            "ids": [str(item.get("chunk_id") or "") for item in documents],
            "documents": [str(item.get("child_text") or "") for item in documents],
            "metadatas": [
                {
                    "document_id": item.get("document_id"),
                    "parent_text": item.get("parent_text"),
                    "parent_id": item.get("parent_id"),
                    "source_type": item.get("source_type"),
                    "page_number": item.get("page_number"),
                    "block_index": item.get("block_index"),
                }
                for item in documents
            ],
        }


class DeterministicSparseIndex:
    """确定性稀疏索引，用关键词命中数量模拟召回。"""

    def __init__(self, store: "DeterministicVectorStore") -> None:
        self.store = store

    @staticmethod
    def _phrase_score(text: str, keywords: list[str]) -> tuple[int, int]:
        positions: list[int] = []
        for keyword in keywords:
            pos = text.find(keyword)
            if pos >= 0:
                positions.append(pos)
        if len(positions) < 2:
            return 0, 10**6
        in_order = int(all(left <= right for left, right in zip(positions, positions[1:])))
        spread = positions[-1] - positions[0]
        return in_order, spread

    def search(self, query_keywords: list[str], *, min_hits: int = 2, top_k: int = 5) -> list[dict[str, object]]:
        normalized_keywords = [normalize_text(keyword) for keyword in query_keywords if normalize_text(keyword)]
        scored: list[tuple[int, int, int, int, dict[str, object]]] = []
        for item in self.store.documents:
            child_text = normalize_text(str(item.get("child_text") or ""))
            parent_text = normalize_text(str(item.get("parent_text") or ""))
            text = f"{child_text}\n{parent_text}"
            hit_count = sum(keyword in text for keyword in normalized_keywords)
            if hit_count < min_hits:
                continue
            in_order, spread = self._phrase_score(text, normalized_keywords)
            candidate = dict(item)
            candidate["keyword_hit_count"] = hit_count
            scored.append((hit_count, in_order, -spread, len(child_text), candidate))
        scored.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3]))
        return [item[4] for item in scored[:top_k]]


class DeterministicVectorStore:
    """内存向量库替身，避免离线 A/B 依赖真实 embedding 模型。"""

    def __init__(self) -> None:
        self.documents: list[dict[str, object]] = []
        self.collection = DeterministicCollection(self)
        self.sparse_index = DeterministicSparseIndex(self)
        self.retrieval_sparse_provider = "sqlite_fts"

    def _get_sparse_index(self) -> DeterministicSparseIndex:
        return self.sparse_index

    @staticmethod
    def _matched_token_names(text: str) -> set[str]:
        normalized = normalize_text(text)
        return {
            name
            for name, token in TOKEN_GROUPS
            if normalize_text(token) in normalized
        }

    @staticmethod
    def _embedding(text: str) -> list[float]:
        matched = DeterministicVectorStore._matched_token_names(text)
        vector = [1.0 if name in matched else 0.0 for name, _ in TOKEN_GROUPS]
        if not any(vector):
            return [0.05 for _ in TOKEN_GROUPS]
        return vector

    def add_documents(self, chunks_data: list[dict[str, object]]) -> None:
        start_index = len(self.documents)
        for index, chunk in enumerate(chunks_data, start=start_index):
            item = dict(chunk)
            item["chunk_id"] = str(item.get("chunk_id") or f"chunk_{index}")
            item["embedding"] = self._embedding(str(item.get("child_text") or ""))
            item["distance"] = None
            self.documents.append(item)

    def search(self, query: str, top_k: int) -> tuple[list[float], list[dict[str, object]]]:
        query_embedding = self._embedding(query)
        query_tokens = self._matched_token_names(query)
        scored: list[tuple[float, dict[str, object]]] = []
        for item in self.documents:
            embedding = list(item.get("embedding") or [])
            child_tokens = self._matched_token_names(str(item.get("child_text") or ""))
            parent_tokens = self._matched_token_names(str(item.get("parent_text") or ""))
            overlap_score = float(len(query_tokens & child_tokens) * 2 + len(query_tokens & parent_tokens))
            score = overlap_score if overlap_score > 0 else sum(
                query_value * embedding_value for query_value, embedding_value in zip(query_embedding, embedding)
            )
            candidate = dict(item)
            candidate["distance"] = round(max(0.0, 1.0 - score / max(1, len(TOKEN_GROUPS))), 4)
            scored.append((score, candidate))
        scored.sort(key=lambda pair: (pair[0], -len(str(pair[1].get("child_text") or ""))), reverse=True)
        return query_embedding, [item for score, item in scored[:top_k] if score > 0]

    def keyword_search(self, query_keywords: list[str], min_hits: int, top_k: int) -> list[dict[str, object]]:
        normalized_keywords = [normalize_text(keyword) for keyword in query_keywords if normalize_text(keyword)]
        scored: list[tuple[int, dict[str, object]]] = []
        for item in self.documents:
            child_text = normalize_text(str(item.get("child_text") or ""))
            parent_text = normalize_text(str(item.get("parent_text") or ""))
            hit_count = sum(keyword in child_text or keyword in parent_text for keyword in normalized_keywords)
            if hit_count < min_hits:
                continue
            candidate = dict(item)
            scored.append((hit_count, candidate))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]


class StaticAliasExpander:
    """测试用 query alias 扩展器，可按需要把问题映射成关键词提示。"""

    def __init__(self, *, use_query_as_keywords: bool = False) -> None:
        self.use_query_as_keywords = use_query_as_keywords

    def expand(self, query: str) -> str:
        if not self.use_query_as_keywords:
            return ""
        for marker, keywords in QUERY_KEYWORD_HINTS.items():
            if marker in query:
                return keywords
        return query


class StaticRewriter:
    """离线评测默认不做 LLM query rewrite。"""

    def rewrite(self, user_query: str) -> str:
        return ""

    def rewrite_with_history(self, user_query: str, history_messages: list[dict[str, str]]) -> str:
        return user_query


class DeterministicReranker:
    """用关键词命中数量做可复现重排。"""

    def rerank(self, query: str, docs: list[dict[str, object]], top_k: int) -> list[dict[str, object]]:
        normalized_query = normalize_text(query)
        reranked: list[dict[str, object]] = []
        for item in docs:
            parent_text = normalize_text(str(item.get("parent_text") or ""))
            score = sum(
                1
                for token, _ in TOKEN_GROUPS
                if normalize_text(token) in normalized_query and normalize_text(token) in parent_text
            )
            candidate = dict(item)
            candidate["rerank_score"] = float(score)
            reranked.append(candidate)
        reranked.sort(key=lambda doc: float(doc.get("rerank_score") or 0.0), reverse=True)
        return reranked[:top_k]


class DeterministicLLM:
    """返回首段上下文作为答案，便于稳定断言。"""

    def generate(
        self,
        query: str,
        context: str,
        standalone_query: str | None = None,
        *,
        answer_style: str = "concise",
    ) -> dict[str, object]:
        if not context:
            answer = "参考资料未明确，无法确定。"
        else:
            leading_context = context.split("\n\n", 1)[0].strip()
            answer = f"根据资料：{leading_context}"
        return {
            "answer": answer,
            "finish_reason": "stop",
            "model": "deterministic-split-profile-model",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }


def build_pipeline(
    *,
    profile: str | None,
    child_chunk_size: int,
    parent_chunk_size: int,
    sparse_provider: str = "sqlite_fts",
    enable_sparse: bool = True,
    use_query_as_keywords: bool = False,
    sparse_top_k: int | None = None,
) -> RAGPipeline:
    """构造离线 A/B 所需的确定性 RAGPipeline。"""
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.settings = SimpleNamespace(
        top_k_mmr=2,
        retrieval_sparse_provider=sparse_provider,
        retrieval_dense_top_k=None,
        retrieval_sparse_top_k=None,
        retrieval_sparse_min_hits=2,
        retrieval_enable_sparse=enable_sparse,
    )
    pipeline.processor = DocumentProcessor(
        child_chunk_size=child_chunk_size,
        parent_chunk_size=parent_chunk_size,
        profile=profile,
    )
    pipeline.structured_processor = StructuredDocumentProcessor(pipeline.processor)
    pipeline.vector_store = DeterministicVectorStore()
    pipeline.vector_store.retrieval_sparse_provider = sparse_provider
    pipeline.retrieval_policy = RetrievalPolicy(enable_sparse=enable_sparse, sparse_top_k=sparse_top_k)
    pipeline.retriever = build_default_hybrid_retriever(
        pipeline.vector_store,
        sparse_provider=sparse_provider,
        policy=pipeline.retrieval_policy,
    )
    pipeline.reranker = DeterministicReranker()
    pipeline.llm = DeterministicLLM()
    pipeline.rewriter = StaticRewriter()
    pipeline.query_alias_expander = StaticAliasExpander(use_query_as_keywords=use_query_as_keywords)
    pipeline.retrieval_rules = SimpleNamespace(
        policy_trigger_words=(),
        policy_trigger_phrases=(),
        behavior_keywords=(),
        must_keep_child_keywords=(),
        low_value_parent_keywords=(),
        rule_signal_keywords=(),
    )
    return pipeline


def ingest_fixture_corpus(pipeline: RAGPipeline, fixture_dir: Path) -> None:
    """导入固定 fixture 文档。"""
    plain_path = fixture_dir / "plain_rules.txt"
    structured_path = fixture_dir / "ocr_rules.json"
    pipeline.ingest_file(plain_path, document_id="plain_rules")
    pipeline.ingest_file(structured_path, document_id="ocr_rules")


def run_case(pipeline: RAGPipeline, case: EvalCase) -> EvalCaseResult:
    """运行单个样本并判断各阶段命中情况。"""
    result = pipeline.ask(
        query=case.question,
        top_k_retrieve=8,
        top_k_rerank=1,
        knowledge_base_id="kb_split_profile_strict",
        source_name_by_document_id={
            "plain_rules": "《split_profile_plain》",
            "ocr_rules": "《split_profile_ocr》",
        },
        enable_rewrite=True,
        enable_rerank=True,
        answer_style="concise",
    )
    debug_payload = dict(result["debug"])
    answer_text = str(result["answer"])
    citations = list(result["citations"])
    citation_text = "\n".join(str(item.get("content", "")) for item in citations if isinstance(item, dict))
    retrieved_text = join_debug_chunk_texts(debug_payload, "retrieved_chunks")
    final_context_text = join_debug_chunk_texts(debug_payload, "final_context_chunks")

    answer_hit, answer_hit_ratio, matched_answer_keywords, missing_answer_keywords = evaluate_answer_hit(case, answer_text)
    matched_retrieval_keywords, _ = keyword_matches(retrieved_text, case.expected_context_keywords)
    matched_final_context_keywords, _ = keyword_matches(final_context_text, case.expected_context_keywords)
    matched_citation_keywords, _ = keyword_matches(citation_text, case.expected_context_keywords)

    retrieval_hit = len(matched_retrieval_keywords) == len(case.expected_context_keywords)
    final_context_hit = len(matched_final_context_keywords) == len(case.expected_context_keywords)
    citation_hit = len(matched_citation_keywords) == len(case.expected_context_keywords)
    if case.answer_mode == "no_answer":
        final_context_chunks = debug_payload.get("final_context_chunks", [])
        retrieval_hit = len(final_context_chunks) == 0
        final_context_hit = len(final_context_chunks) == 0
        citation_hit = len(citations) == 0
    error_type = classify_error(case.answer_mode, retrieval_hit, final_context_hit, answer_hit, citation_hit)

    return EvalCaseResult(
        id=case.id,
        question=case.question,
        category=case.category,
        answer_mode=case.answer_mode,
        answer=answer_text,
        retrieval_hit=retrieval_hit,
        final_context_hit=final_context_hit,
        citation_hit=citation_hit,
        answer_hit=answer_hit,
        answer_hit_ratio=round(answer_hit_ratio, 4),
        matched_answer_keywords=matched_answer_keywords,
        missing_answer_keywords=missing_answer_keywords,
        matched_context_keywords=matched_final_context_keywords,
        error_type=error_type,
        debug=debug_payload,
        citations=citations,
    )


def build_summary(results: list[EvalCaseResult]) -> dict[str, object]:
    """汇总整批样本的命中率、延迟和错误归因。"""
    total = len(results)
    error_buckets: dict[str, int] = {}
    for item in results:
        error_buckets[item.error_type] = error_buckets.get(item.error_type, 0) + 1

    debug_items = [item.debug for item in results]
    return {
        "case_count": total,
        "grounded_case_count": sum(item.answer_mode == "grounded" for item in results),
        "no_answer_case_count": sum(item.answer_mode == "no_answer" for item in results),
        "retrieval_hit_rate": round(sum(item.retrieval_hit for item in results) / total, 4),
        "final_context_hit_rate": round(sum(item.final_context_hit for item in results) / total, 4),
        "citation_hit_rate": round(sum(item.citation_hit for item in results) / total, 4),
        "answer_hit_rate": round(sum(item.answer_hit for item in results) / total, 4),
        "avg_answer_hit_ratio": round(mean(item.answer_hit_ratio for item in results), 4),
        "avg_rewrite_ms": round(mean(int(item.get("rewrite_ms", 0)) for item in debug_items), 2),
        "avg_retrieve_ms": round(mean(int(item.get("retrieve_ms", 0)) for item in debug_items), 2),
        "avg_rerank_ms": round(mean(int(item.get("rerank_ms", 0)) for item in debug_items), 2),
        "avg_generate_ms": round(mean(int(item.get("generate_ms", 0)) for item in debug_items), 2),
        "avg_latency_ms": round(mean(int(item.get("latency_ms", 0)) for item in debug_items), 2),
        "error_breakdown": error_buckets,
    }


def run_split_profile_eval(
    *,
    dataset_path: Path,
    fixture_dir: Path,
    profile: str | None,
    child_chunk_size: int = 18,
    parent_chunk_size: int = 32,
    sparse_provider: str = "sqlite_fts",
    enable_sparse: bool = True,
    use_query_as_keywords: bool = False,
    sparse_top_k: int | None = None,
) -> dict[str, object]:
    """运行单个 profile/provider 组合的离线评测。"""
    cases = load_eval_cases(dataset_path)
    pipeline = build_pipeline(
        profile=profile,
        child_chunk_size=child_chunk_size,
        parent_chunk_size=parent_chunk_size,
        sparse_provider=sparse_provider,
        enable_sparse=enable_sparse,
        use_query_as_keywords=use_query_as_keywords,
        sparse_top_k=sparse_top_k,
    )
    ingest_fixture_corpus(pipeline, fixture_dir)
    results = [run_case(pipeline, case) for case in cases]
    return {
        "dataset": str(dataset_path),
        "config": {
            "profile": profile or "",
            "child_chunk_size": child_chunk_size,
            "parent_chunk_size": parent_chunk_size,
            "sparse_provider": sparse_provider,
            "enable_sparse": enable_sparse,
            "use_query_as_keywords": use_query_as_keywords,
            "sparse_top_k": sparse_top_k,
        },
        "summary": build_summary(results),
        "results": [asdict(item) for item in results],
    }


def run_split_profile_ab_compare(
    *,
    dataset_path: Path,
    fixture_dir: Path,
    baseline_profile: str | None = "",
    contender_profile: str | None = "rules_cn",
    child_chunk_size: int = 18,
    parent_chunk_size: int = 32,
) -> dict[str, object]:
    """比较 baseline profile 和 contender profile。"""
    baseline_report = run_split_profile_eval(
        dataset_path=dataset_path,
        fixture_dir=fixture_dir,
        profile=baseline_profile,
        child_chunk_size=child_chunk_size,
        parent_chunk_size=parent_chunk_size,
    )
    contender_report = run_split_profile_eval(
        dataset_path=dataset_path,
        fixture_dir=fixture_dir,
        profile=contender_profile,
        child_chunk_size=child_chunk_size,
        parent_chunk_size=parent_chunk_size,
    )
    comparison = compare_eval_reports(baseline_report, contender_report)
    return {
        "dataset": str(dataset_path),
        "fixture_dir": str(fixture_dir),
        "baseline": baseline_report,
        "contender": contender_report,
        "comparison": comparison,
    }


def run_sparse_provider_ab_compare(
    *,
    dataset_path: Path,
    fixture_dir: Path,
    profile: str | None = "rules_cn",
    providers: tuple[str, ...] = ("sqlite_fts", "bm25", "scan", "none"),
    child_chunk_size: int = 18,
    parent_chunk_size: int = 32,
) -> dict[str, object]:
    """比较多个稀疏检索 provider。"""
    reports: dict[str, dict[str, object]] = {}
    for provider in providers:
        reports[provider] = run_split_profile_eval(
            dataset_path=dataset_path,
            fixture_dir=fixture_dir,
            profile=profile,
            child_chunk_size=child_chunk_size,
            parent_chunk_size=parent_chunk_size,
            sparse_provider=provider,
            enable_sparse=provider != "none",
            use_query_as_keywords=True,
            sparse_top_k=1,
        )

    baseline_provider = providers[0]
    comparisons: dict[str, dict[str, object]] = {}
    for provider in providers[1:]:
        comparisons[f"{baseline_provider}_vs_{provider}"] = compare_eval_reports(
            reports[baseline_provider],
            reports[provider],
        )

    return {
        "dataset": str(dataset_path),
        "fixture_dir": str(fixture_dir),
        "profile": profile or "",
        "providers": list(providers),
        "reports": reports,
        "comparisons": comparisons,
    }


