"""RAG pipeline 和 shadow graph 共用的纯函数步骤。

这些函数不依赖数据库、API 或 LLM 实例，主要负责候选文档排序、规则保留、
parent 聚合、最终上下文选择、citations/debug 构建。
"""

from __future__ import annotations

import re

from .overview_config import load_overview_query_config

RagDoc = dict[str, object]
_ARTICLE_NUMBER_RE = re.compile(r"第([一二两三四五六七八九十百千0-9]+)条")
_ARTICLE_REFERENCE_TRIGGER_RE = re.compile(r"详见|参见|另见|见第|按照|依照|依据|根据|适用|参照|遵照|按第")
_ARTICLE_REFERENCE_WINDOW_STOP_RE = re.compile(r"[。；;\n]")
_SPLIT_PARENT_ID_RE = re.compile(r"^(.+)_([0-9]+)$")
_ENUMERATION_START_RE = re.compile(
    r"^\s*(?:[（(][一二三四五六七八九十0-9]+[）)]|[一二三四五六七八九十]+、|\d+[\.．、])"
)


def build_debug_chunk(doc: RagDoc) -> RagDoc:
    """把内部候选文档压缩成前端 debug drawer 可展示的字段。"""
    parent_id = str(doc.get("parent_id", "")) or None
    child_text = doc.get("child_text")
    parent_text = str(doc.get("parent_text") or doc.get("text") or "")
    raw_distance = doc.get("distance")
    raw_rerank_score = doc.get("rerank_score")
    raw_page_number = doc.get("page_number")
    raw_block_index = doc.get("block_index")
    return {
        "chunk_id": str(doc.get("id") or parent_id or "unknown"),
        "parent_id": parent_id,
        "child_text": str(child_text) if child_text is not None else None,
        "parent_text": parent_text,
        "distance": float(raw_distance) if raw_distance is not None else None,
        "rerank_score": float(raw_rerank_score) if raw_rerank_score is not None else None,
        "source_name": str(doc.get("source_name") or "") or None,
        "source_type": str(doc.get("source_type") or "text"),
        "page_number": int(raw_page_number) if raw_page_number is not None else None,
        "block_index": int(raw_block_index) if raw_block_index is not None else None,
    }


def build_debug_chunks(docs: list[RagDoc]) -> list[RagDoc]:
    return [build_debug_chunk(doc) for doc in docs]


def prettify_legal_text(text: str) -> str:
    """对法律条文类文本做轻量换行，提升最终上下文和引用的可读性。"""
    normalized = str(text).strip()
    if not normalized:
        return normalized

    normalized = re.sub(r"(中华人民共和国[^\n]{0,40}?法)", r"\1\n", normalized, count=1)
    normalized = re.sub(
        r"(第[一二两三四五六七八九十百千0-9]+章[^\n]{0,40}?)(第[一二两三四五六七八九十百千0-9]+条)",
        r"\1\n\2",
        normalized,
    )
    normalized = re.sub(
        r"(第[一二两三四五六七八九十百千0-9]+条[^\n]{0,80}?：)(（[一二三四五六七八九十]+）)",
        r"\1\n\2",
        normalized,
    )
    normalized = re.sub(r"；(（[一二三四五六七八九十]+）)", r"；\n\1", normalized)
    normalized = re.sub(r"。(劳动者可以立即解除劳动合同，不需事先告知用人单位。)", r"。\n\1", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def format_context_parent(doc: RagDoc) -> str:
    source_name = str(doc.get("source_name") or "").strip()
    text = prettify_legal_text(str(doc["text"]))
    if not source_name:
        return text
    return f"[来源：{source_name}]\n{text}"


def extract_query_keywords(*parts: str) -> list[str]:
    """从 rewrite/alias 等文本里提取去重关键词。"""
    keywords: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for token in re.split(r"[\s,，。！？；：、（）()\"“”'‘’]+", part):
            normalized = token.strip()
            if len(normalized) < 2 or normalized in seen:
                continue
            seen.add(normalized)
            keywords.append(normalized)
    return keywords


def is_overview_query(query: str) -> bool:
    config = load_overview_query_config()
    overview_markers = config.overview_markers
    topic_markers = config.topic_markers
    return any(marker in query for marker in overview_markers) and any(topic in query for topic in topic_markers)


def parse_chinese_numeral(value: str) -> int | None:
    digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100, "千": 1000}
    if not value:
        return None
    if value.isdigit():
        return int(value)

    total = 0
    current = 0
    for char in value:
        if char in digits:
            current = digits[char]
            continue
        unit = units.get(char)
        if unit is None:
            return None
        total += (current or 1) * unit
        current = 0
    return total + current


def extract_article_number(text: str) -> int | None:
    match = re.match(r"^第([一二两三四五六七八九十百千0-9]+)条", text.strip())
    if not match:
        return None
    return parse_chinese_numeral(match.group(1))


def extract_leading_article_number(text: str) -> int | None:
    """提取父块自身条款号，兼容 Markdown 标题前缀后的条款文本。"""
    normalized = str(text).strip()
    article_number = extract_article_number(normalized)
    if article_number is not None:
        return article_number

    match = _ARTICLE_NUMBER_RE.search(normalized[:120])
    if not match:
        return None
    return parse_chinese_numeral(match.group(1))


def extract_referenced_article_numbers(text: str, *, max_window_chars: int = 90) -> list[int]:
    """识别“详见/依照/按照第 xx 条”等一跳条款引用。"""
    referenced_numbers: list[int] = []
    seen: set[int] = set()
    normalized = str(text)
    for trigger_match in _ARTICLE_REFERENCE_TRIGGER_RE.finditer(normalized):
        window = normalized[trigger_match.start() : trigger_match.start() + max_window_chars]
        stop_match = _ARTICLE_REFERENCE_WINDOW_STOP_RE.search(window)
        if stop_match:
            window = window[: stop_match.start()]
        for article_match in _ARTICLE_NUMBER_RE.finditer(window):
            article_number = parse_chinese_numeral(article_match.group(1))
            if article_number is None or article_number in seen:
                continue
            seen.add(article_number)
            referenced_numbers.append(article_number)
    return referenced_numbers


def count_query_keyword_hits(doc: RagDoc, query_keywords: list[str]) -> int:
    if not query_keywords:
        return 0
    text = f"{doc.get('child_text') or ''}\n{doc.get('parent_text') or doc.get('text') or ''}"
    return sum(1 for keyword in query_keywords if keyword in text)


def source_priority(source_type: str | None) -> int:
    priorities = {
        "table": 0,
        "text": 1,
        "image_ocr_table": 2,
        "image_ocr": 3,
    }
    return priorities.get(str(source_type or "text"), 1)


def page_has_table_source(docs: list[RagDoc], page_number: int | None) -> bool:
    if page_number is None:
        return False
    return any(
        int(doc.get("page_number") or 0) == int(page_number) and str(doc.get("source_type") or "") == "table"
        for doc in docs
    )


def retrieval_penalty(doc: RagDoc, docs: list[RagDoc]) -> float:
    """检索阶段的来源惩罚：优先表格和正文，降低 OCR 噪声权重。"""
    source_type = str(doc.get("source_type") or "text")
    penalties = {
        "table": 0.0,
        "text": 0.03,
        "image_ocr_table": 0.12,
        "image_ocr": 0.2,
    }
    penalty = penalties.get(source_type, 0.05)
    page_number = int(doc.get("page_number") or 0)
    if source_type == "image_ocr_table" and page_has_table_source(docs, page_number):
        penalty += 0.08
    if source_type.startswith("image_ocr") and str(doc.get("ocr_quality") or "") == "low":
        penalty += 0.12
    return penalty


def sort_retrieved_docs(docs: list[RagDoc]) -> list[RagDoc]:
    def sort_key(doc: RagDoc) -> tuple[int, float, int]:
        keyword_hit_count = int(doc.get("keyword_hit_count") or 0)
        distance = float(doc.get("distance") or 0.0)
        adjusted_distance = distance + retrieval_penalty(doc, docs)
        return -keyword_hit_count, adjusted_distance, source_priority(str(doc.get("source_type") or "text"))

    return sorted(docs, key=sort_key)


def rerank_penalty(doc: RagDoc, docs: list[RagDoc]) -> float:
    """rerank 后再做一次来源修正，避免 OCR 噪声压过结构化文本。"""
    source_type = str(doc.get("source_type") or "text")
    penalties = {
        "table": 0.0,
        "text": 0.03,
        "image_ocr_table": 0.18,
        "image_ocr": 0.28,
    }
    penalty = penalties.get(source_type, 0.05)
    page_number = int(doc.get("page_number") or 0)
    if source_type == "image_ocr_table" and page_has_table_source(docs, page_number):
        penalty += 0.12
    if source_type.startswith("image_ocr") and str(doc.get("ocr_quality") or "") == "low":
        penalty += 0.18
    return penalty


def sort_reranked_parents(docs: list[RagDoc]) -> list[RagDoc]:
    def sort_key(doc: RagDoc) -> tuple[float, int]:
        rerank_score = float(doc.get("rerank_score") or 0.0)
        adjusted_score = rerank_score - rerank_penalty(doc, docs)
        return -adjusted_score, source_priority(str(doc.get("source_type") or "text"))

    return sorted(docs, key=sort_key)


def dedupe_retrieved_docs(retrieved_docs: list[RagDoc], supplemental_docs: list[RagDoc]) -> list[RagDoc]:
    """合并 dense 和 sparse 候选，并按文本内容去重。"""
    seen_keys: set[str] = set()
    deduped_docs: list[RagDoc] = []
    for doc in [*retrieved_docs, *supplemental_docs]:
        unique_key = f"{doc['child_text']}{doc['parent_text']}"
        if unique_key not in seen_keys:
            seen_keys.add(unique_key)
            deduped_docs.append(doc)
    return deduped_docs


def apply_source_names(docs: list[RagDoc], source_name_by_document_id: dict[str, str] | None) -> list[RagDoc]:
    if not source_name_by_document_id:
        return docs
    for doc in docs:
        document_id = str(doc.get("document_id") or "")
        if document_id and document_id in source_name_by_document_id:
            doc["source_name"] = source_name_by_document_id[document_id]
    return docs


def should_treat_as_policy_question(
    query: str,
    *,
    policy_trigger_words: tuple[str, ...],
    policy_trigger_phrases: tuple[str, ...],
    behavior_keywords: tuple[str, ...],
) -> bool:
    has_behavior_keyword = any(keyword in query for keyword in behavior_keywords)
    return (
        any(word in query for word in policy_trigger_words) or any(phrase in query for phrase in policy_trigger_phrases)
    ) and not has_behavior_keyword


def preserve_policy_keyword_docs(
    selected_docs: list[RagDoc],
    candidate_docs: list[RagDoc],
    *,
    is_policy_question: bool,
    must_keep_child_keywords: tuple[str, ...],
) -> list[RagDoc]:
    """规则类问题保留强关键词命中的候选，防止 MMR 过早丢掉关键条款。"""
    if not is_policy_question:
        return selected_docs

    existing_keys = {f"{doc['child_text']}{doc['parent_text']}" for doc in selected_docs}
    for doc in candidate_docs:
        unique_key = f"{doc['child_text']}{doc['parent_text']}"
        if unique_key in existing_keys:
            continue
        text_to_check = f"{doc['child_text']}\n{doc['parent_text']}"
        if any(keyword in text_to_check for keyword in must_keep_child_keywords):
            selected_docs.append(doc)
            existing_keys.add(unique_key)
    return selected_docs


def preserve_query_keyword_docs(
    selected_docs: list[RagDoc],
    candidate_docs: list[RagDoc],
    query_keywords: list[str],
    *,
    min_hits: int = 2,
    max_extra_docs: int = 2,
) -> list[RagDoc]:
    """按用户 query 关键词命中数补充候选，增强可解释召回。"""
    if not query_keywords or max_extra_docs <= 0:
        return selected_docs

    selected_keys = {f"{doc['child_text']}{doc['parent_text']}" for doc in selected_docs}
    scored_candidates: list[tuple[int, float, int, RagDoc]] = []
    for doc in candidate_docs:
        unique_key = f"{doc['child_text']}{doc['parent_text']}"
        if unique_key in selected_keys:
            continue
        hit_count = count_query_keyword_hits(doc, query_keywords)
        if hit_count < min_hits:
            continue
        distance = float(doc.get("distance") or 0.0)
        scored_candidates.append((hit_count, distance, source_priority(str(doc.get("source_type") or "text")), doc))

    scored_candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, _, _, doc in scored_candidates[:max_extra_docs]:
        unique_key = f"{doc['child_text']}{doc['parent_text']}"
        if unique_key in selected_keys:
            continue
        selected_docs.append(doc)
        selected_keys.add(unique_key)
    return selected_docs


def preserve_overview_candidate_docs(
    selected_docs: list[RagDoc],
    candidate_docs: list[RagDoc],
    *,
    max_extra_docs: int = 4,
) -> list[RagDoc]:
    """概览型问题需要更宽候选面，因此从去重候选里补一些额外 chunk。"""
    if max_extra_docs <= 0:
        return selected_docs

    selected_keys = {f"{doc['child_text']}{doc['parent_text']}" for doc in selected_docs}
    extra_added = 0
    for doc in candidate_docs:
        if extra_added >= max_extra_docs:
            break
        unique_key = f"{doc['child_text']}{doc['parent_text']}"
        if unique_key in selected_keys:
            continue
        selected_docs.append(doc)
        selected_keys.add(unique_key)
        extra_added += 1
    return selected_docs


def build_parent_docs_for_rerank(mmr_selected_docs: list[RagDoc]) -> list[RagDoc]:
    """把 chunk 聚合回 parent 文本，reranker 和 LLM 都以 parent 为粒度。"""
    unique_parents: dict[str, RagDoc] = {}
    for doc in mmr_selected_docs:
        parent_id = str(doc["parent_id"])
        if parent_id not in unique_parents:
            unique_parents[parent_id] = {
                "parent_id": parent_id,
                "text": str(doc["parent_text"]),
                "matched_children": [str(doc["child_text"])],
                "document_id": str(doc.get("document_id") or ""),
                "source_name": str(doc.get("source_name") or ""),
                "source_type": str(doc.get("source_type") or "text"),
                "page_number": int(doc.get("page_number") or 0),
                "block_index": int(doc.get("block_index") or 0),
            }
        else:
            matched_children = unique_parents[parent_id]["matched_children"]
            assert isinstance(matched_children, list)
            matched_children.append(str(doc["child_text"]))
    return list(unique_parents.values())


def collect_overview_article_parents(parent_docs: list[RagDoc], query: str, query_keywords: list[str]) -> list[RagDoc]:
    """概览型条文问题会把命中条款相邻条款一起带上，避免回答断章。"""
    if not is_overview_query(query):
        return []

    article_docs: list[tuple[int, RagDoc]] = []
    for doc in parent_docs:
        article_number = extract_article_number(str(doc.get("text") or ""))
        if article_number is not None:
            article_docs.append((article_number, doc))
    if not article_docs:
        return []

    article_docs.sort(key=lambda item: item[0])
    topical_numbers = {
        article_number for article_number, doc in article_docs if count_query_keyword_hits(doc, query_keywords) >= 1
    }
    if not topical_numbers:
        return []

    selected_numbers = set(topical_numbers)
    expanded = True
    while expanded:
        expanded = False
        for article_number, _ in article_docs:
            if article_number in selected_numbers:
                continue
            if (article_number - 1) in selected_numbers or (article_number + 1) in selected_numbers:
                selected_numbers.add(article_number)
                expanded = True

    selected_docs = [doc for article_number, doc in article_docs if article_number in selected_numbers]
    selected_docs.sort(key=lambda doc: extract_article_number(str(doc.get("text") or "")) or 0)
    return selected_docs[:4]


def collect_overview_parent_candidates(
    parent_docs: list[RagDoc],
    final_context_parents: list[RagDoc],
    query: str,
    *,
    max_docs: int = 3,
) -> list[RagDoc]:
    if not is_overview_query(query) or max_docs <= 0:
        return []

    present_texts = {str(doc.get("text") or "") for doc in final_context_parents}
    selected_docs: list[RagDoc] = []
    for doc in parent_docs:
        text = str(doc.get("text") or "")
        if not text or text in present_texts:
            continue
        selected_docs.append(doc)
        present_texts.add(text)
        if len(selected_docs) >= max_docs:
            break
    return selected_docs


def collect_related_restriction_article_parents(
    parent_docs: list[RagDoc],
    final_context_parents: list[RagDoc],
    query: str,
) -> list[RagDoc]:
    if not is_overview_query(query):
        return []

    present_numbers = {extract_article_number(str(doc.get("text") or "")) for doc in final_context_parents}
    if 40 not in present_numbers and 41 not in present_numbers:
        return []

    related_docs: list[RagDoc] = []
    for doc in parent_docs:
        if extract_article_number(str(doc.get("text") or "")) == 42:
            related_docs.append(doc)
    return related_docs[:1]


def collect_referenced_article_parents(
    parent_docs: list[RagDoc],
    final_context_parents: list[RagDoc],
    *,
    max_docs: int = 2,
) -> list[RagDoc]:
    """补齐最终上下文里显式引用但尚未出现的一跳条款。"""
    if max_docs <= 0 or not parent_docs or not final_context_parents:
        return []

    present_texts = {str(doc.get("text") or "") for doc in final_context_parents}
    present_numbers = {
        article_number
        for doc in final_context_parents
        if (article_number := extract_leading_article_number(str(doc.get("text") or ""))) is not None
    }

    referenced_numbers: list[int] = []
    referenced_seen: set[int] = set()
    for doc in final_context_parents:
        for article_number in extract_referenced_article_numbers(str(doc.get("text") or "")):
            if article_number in present_numbers or article_number in referenced_seen:
                continue
            referenced_seen.add(article_number)
            referenced_numbers.append(article_number)
    if not referenced_numbers:
        return []

    selected_docs: list[RagDoc] = []
    selected_numbers: set[int] = set()
    for expected_number in referenced_numbers:
        for doc in parent_docs:
            text = str(doc.get("text") or "")
            if not text or text in present_texts:
                continue
            article_number = extract_leading_article_number(text)
            if article_number != expected_number or article_number in selected_numbers:
                continue
            selected_docs.append(doc)
            present_texts.add(text)
            selected_numbers.add(article_number)
            break
        if len(selected_docs) >= max_docs:
            break
    return selected_docs


def parse_split_parent_id(parent_id: str) -> tuple[str, int] | None:
    """解析同一长段被拆成多个 parent 后的 ``prefix_0/prefix_1`` 编号。"""
    match = _SPLIT_PARENT_ID_RE.match(str(parent_id).strip())
    if not match:
        return None
    return match.group(1), int(match.group(2))


def collect_adjacent_split_parent_parents(
    parent_docs: list[RagDoc],
    final_context_parents: list[RagDoc],
    *,
    query_keywords: list[str] | None = None,
    max_docs: int = 2,
) -> list[RagDoc]:
    """补齐同一长条文拆分出来的相邻 parent，避免最终上下文只有半段条文。"""
    if max_docs <= 0 or not parent_docs or not final_context_parents:
        return []

    grouped_docs: dict[tuple[str, str, str], dict[int, RagDoc]] = {}
    for doc in parent_docs:
        parent_id = str(doc.get("parent_id") or "")
        parsed = parse_split_parent_id(parent_id)
        if parsed is None:
            continue
        base_parent_id, parent_index = parsed
        group_key = (
            str(doc.get("document_id") or ""),
            str(doc.get("source_type") or "text"),
            base_parent_id,
        )
        grouped_docs.setdefault(group_key, {}).setdefault(parent_index, doc)

    # 只有存在 prefix_0/prefix_1 这种真实拆分组时才扩展，避免把 rule_doc_38 和 rule_doc_39
    # 误认为同一条文的相邻 parent。
    split_groups = {
        key: indexed_docs
        for key, indexed_docs in grouped_docs.items()
        if 0 in indexed_docs and len(indexed_docs) >= 2
    }
    if not split_groups:
        return []

    present_texts = {str(doc.get("text") or "") for doc in final_context_parents}
    present_parent_ids = {str(doc.get("parent_id") or "") for doc in final_context_parents}
    selected_docs: list[RagDoc] = []

    for doc in final_context_parents:
        parent_id = str(doc.get("parent_id") or "")
        parsed = parse_split_parent_id(parent_id)
        if parsed is None:
            continue
        base_parent_id, parent_index = parsed
        group_key = (
            str(doc.get("document_id") or ""),
            str(doc.get("source_type") or "text"),
            base_parent_id,
        )
        indexed_docs = split_groups.get(group_key)
        if not indexed_docs:
            continue

        for adjacent_index in (parent_index - 1, parent_index + 1):
            adjacent_doc = indexed_docs.get(adjacent_index)
            if adjacent_doc is None:
                continue
            adjacent_parent_id = str(adjacent_doc.get("parent_id") or "")
            text = str(adjacent_doc.get("text") or "")
            if not text or text in present_texts or adjacent_parent_id in present_parent_ids:
                continue
            if not should_expand_adjacent_split_parent(
                current_doc=doc,
                adjacent_doc=adjacent_doc,
                current_index=parent_index,
                adjacent_index=adjacent_index,
                query_keywords=query_keywords or [],
            ):
                continue
            selected_docs.append({**adjacent_doc, "context_expand_reason": "adjacent_split_parent"})
            present_texts.add(text)
            present_parent_ids.add(adjacent_parent_id)
            if len(selected_docs) >= max_docs:
                return selected_docs

    return selected_docs


def should_expand_adjacent_split_parent(
    *,
    current_doc: RagDoc,
    adjacent_doc: RagDoc,
    current_index: int,
    adjacent_index: int,
    query_keywords: list[str],
) -> bool:
    """相邻 parent 扩展的触发条件，避免无脑把同组邻居塞进最终上下文。"""
    if query_keywords and count_query_keyword_hits(adjacent_doc, query_keywords) >= 1:
        return True

    current_text = str(current_doc.get("text") or "")
    adjacent_text = str(adjacent_doc.get("text") or "")
    if adjacent_index > current_index:
        return has_forward_split_dependency(current_text, adjacent_text)
    return has_forward_split_dependency(adjacent_text, current_text)


def has_forward_split_dependency(previous_text: str, next_text: str) -> bool:
    """判断前后两个 split parent 是否存在明显承接关系。"""
    previous_tail = previous_text.strip()[-80:]
    next_head = next_text.strip()[:80]
    if not previous_tail or not next_head:
        return False

    if previous_tail.endswith(("：", ":", "；", ";", "，", ",", "、", "（", "(", "“", "《")):
        return True
    if re.search(r"(如下|包括|分为|分成|下列|以下|按照|依照|根据|参照|符合|属于)[^。！？]{0,30}[：:]?$", previous_tail):
        return True
    if _ENUMERATION_START_RE.match(next_head) and re.search(r"(下列|以下|包括|如下|情形|条件|标准|事项)", previous_tail):
        return True
    if re.match(r"^\s*(的|者|时|后|前|以及|并且|或者|但|但是|同时|其中|其|该)", next_head):
        return True
    return False


def build_final_context_parents(
    *,
    parent_docs_for_rerank: list[RagDoc],
    reranked_parents: list[RagDoc],
    context_parent_pool: list[RagDoc] | None = None,
    query: str,
    query_keywords: list[str],
    is_policy_question: bool,
    must_keep_parent_keywords: tuple[str, ...],
    low_value_parent_keywords: tuple[str, ...],
    rule_signal_keywords: tuple[str, ...],
    behavior_keywords: tuple[str, ...],
    top_k_rerank: int,
) -> list[RagDoc]:
    """综合规则保留、关键词命中、overview 扩展和 rerank 顺序，挑最终上下文。"""
    def has_rule_signal(parent_doc: RagDoc) -> bool:
        text = str(parent_doc["text"])
        return any(keyword in text for keyword in rule_signal_keywords)

    def is_global_policy_parent(parent_doc: RagDoc) -> bool:
        text = str(parent_doc["text"])
        return any(keyword in text for keyword in must_keep_parent_keywords)

    def is_low_value_parent(parent_doc: RagDoc) -> bool:
        text = str(parent_doc["text"])
        has_low_value_marker = any(keyword in text for keyword in low_value_parent_keywords)
        if not has_low_value_marker:
            return False
        return not has_rule_signal(parent_doc)

    parent_pool = context_parent_pool or parent_docs_for_rerank
    query_behavior_keywords = [keyword for keyword in behavior_keywords if keyword in query]

    def is_negative_background_parent(parent_doc: RagDoc) -> bool:
        rerank_score = float(parent_doc.get("rerank_score", 0.0))
        if rerank_score >= 0:
            return False
        text = str(parent_doc["text"])
        return not any(keyword in text for keyword in query_behavior_keywords)

    final_context_parents: list[RagDoc] = []
    added_parent_texts: set[str] = set()

    if is_policy_question:
        # 先放入全局规则类 parent，保证制度性问题有总则背景。
        for parent_doc in parent_docs_for_rerank:
            text = str(parent_doc["text"])
            if text in added_parent_texts:
                continue
            if is_low_value_parent(parent_doc) or is_negative_background_parent(parent_doc):
                continue
            if is_global_policy_parent(parent_doc):
                final_context_parents.append(parent_doc)
                added_parent_texts.add(text)

    for parent_doc in parent_docs_for_rerank:
        # 再补用户 query 关键词命中充分的 parent，提高回答针对性。
        text = str(parent_doc["text"])
        if text in added_parent_texts:
            continue
        if is_low_value_parent(parent_doc) or is_negative_background_parent(parent_doc):
            continue
        if count_query_keyword_hits(parent_doc, query_keywords) >= 2:
            final_context_parents.append(parent_doc)
            added_parent_texts.add(text)

    for parent_doc in collect_overview_article_parents(parent_docs_for_rerank, query, query_keywords):
        text = str(parent_doc["text"])
        if text in added_parent_texts:
            continue
        if is_low_value_parent(parent_doc) or is_negative_background_parent(parent_doc):
            continue
        final_context_parents.append(parent_doc)
        added_parent_texts.add(text)

    for parent_doc in collect_related_restriction_article_parents(parent_docs_for_rerank, final_context_parents, query):
        text = str(parent_doc["text"])
        if text in added_parent_texts:
            continue
        if is_low_value_parent(parent_doc) or is_negative_background_parent(parent_doc):
            continue
        final_context_parents.append(parent_doc)
        added_parent_texts.add(text)

    for parent_doc in collect_referenced_article_parents(parent_docs_for_rerank, final_context_parents):
        text = str(parent_doc["text"])
        if text in added_parent_texts:
            continue
        if is_low_value_parent(parent_doc) or is_negative_background_parent(parent_doc):
            continue
        final_context_parents.append(parent_doc)
        added_parent_texts.add(text)

    for parent_doc in collect_adjacent_split_parent_parents(
        parent_pool,
        final_context_parents,
        query_keywords=query_keywords,
    ):
        text = str(parent_doc["text"])
        if text in added_parent_texts:
            continue
        if is_low_value_parent(parent_doc) or is_negative_background_parent(parent_doc):
            continue
        final_context_parents.append(parent_doc)
        added_parent_texts.add(text)

    for parent_doc in collect_overview_parent_candidates(parent_docs_for_rerank, final_context_parents, query):
        text = str(parent_doc["text"])
        if text in added_parent_texts:
            continue
        if is_low_value_parent(parent_doc) or is_negative_background_parent(parent_doc):
            continue
        final_context_parents.append(parent_doc)
        added_parent_texts.add(text)

    for doc in reranked_parents:
        # 最后按 reranker 顺序补齐剩余额度。
        text = str(doc["text"])
        if is_low_value_parent(doc) or is_negative_background_parent(doc):
            continue
        if text not in added_parent_texts:
            final_context_parents.append(doc)
            added_parent_texts.add(text)

    if not final_context_parents and is_overview_query(query):
        fallback_limit = min(len(parent_docs_for_rerank), top_k_rerank + 2)
        final_context_parents = [
            parent_doc
            for parent_doc in parent_docs_for_rerank
            if not is_low_value_parent(parent_doc) and not is_negative_background_parent(parent_doc)
        ][:fallback_limit]

    overview_buffer = 4 if is_overview_query(query) else 2
    return final_context_parents[: top_k_rerank + overview_buffer]


def build_citations(final_context_parents: list[RagDoc], knowledge_base_id: str) -> list[RagDoc]:
    """citations 必须和最终上下文保持一一对应。"""
    citations: list[RagDoc] = []
    for index, parent_doc in enumerate(final_context_parents, start=1):
        citations.append(
            {
                "document_id": knowledge_base_id,
                "chunk_id": f"{parent_doc['parent_id']}-{index}",
                "content": str(parent_doc["text"]),
                "score": float(parent_doc.get("rerank_score", 0.0)),
                "source_name": str(parent_doc.get("source_name") or "") or None,
                "source_type": str(parent_doc.get("source_type") or "text"),
                "page_number": int(parent_doc.get("page_number") or 0),
                "block_index": int(parent_doc.get("block_index") or 0),
            }
        )
    return citations


def build_debug(
    *,
    rewritten_query: str,
    retrieved_count: int,
    reranked_count: int,
    enable_rewrite: bool,
    enable_rerank: bool,
    rewrite_ms: int,
    retrieve_ms: int,
    rerank_ms: int,
    context_build_ms: int = 0,
    generate_ms: int,
    llm_result: RagDoc | None = None,
    retrieved_chunks: list[RagDoc] | None = None,
    mmr_selected_chunks: list[RagDoc] | None = None,
    reranked_chunks: list[RagDoc] | None = None,
    final_context_chunks: list[RagDoc] | None = None,
    used_conversation_history: bool = False,
    history_message_count: int = 0,
    history_rewritten_query: str | None = None,
    history_rewrite_ms: int = 0,
) -> RagDoc:
    """统一构建前端 debug drawer 所需的阶段计时和候选列表。"""
    llm_first_token_ms = llm_result.get("first_token_ms") if llm_result else None
    llm_after_first_token_ms: int | None = None
    if isinstance(llm_first_token_ms, int):
        llm_after_first_token_ms = max(generate_ms - llm_first_token_ms, 0)

    return {
        "rewritten_query": rewritten_query,
        "retrieved_count": retrieved_count,
        "reranked_count": reranked_count,
        "enable_rewrite": enable_rewrite,
        "enable_rerank": enable_rerank,
        "rewrite_ms": rewrite_ms,
        "retrieve_ms": retrieve_ms,
        "rerank_ms": rerank_ms,
        "context_build_ms": context_build_ms,
        "generate_ms": generate_ms,
        "latency_ms": 0,
        "llm_finish_reason": llm_result.get("finish_reason") if llm_result else None,
        "llm_model": llm_result.get("model") if llm_result else None,
        "llm_prompt_tokens": llm_result.get("prompt_tokens") if llm_result else None,
        "llm_completion_tokens": llm_result.get("completion_tokens") if llm_result else None,
        "llm_total_tokens": llm_result.get("total_tokens") if llm_result else None,
        "llm_first_token_ms": llm_first_token_ms,
        "stage_timings_ms": {
            "history_rewrite_ms": history_rewrite_ms,
            "rewrite_ms": rewrite_ms,
            "retrieve_ms": retrieve_ms,
            "rerank_ms": rerank_ms,
            "context_build_ms": context_build_ms,
            "generate_ms": generate_ms,
            "llm_first_token_ms": llm_first_token_ms,
            "llm_after_first_token_ms": llm_after_first_token_ms,
            "service_overhead_ms": 0,
        },
        "retrieved_chunks": retrieved_chunks or [],
        "mmr_selected_chunks": mmr_selected_chunks or [],
        "reranked_chunks": reranked_chunks or [],
        "final_context_chunks": final_context_chunks or [],
        "used_conversation_history": used_conversation_history,
        "history_message_count": history_message_count,
        "history_rewritten_query": history_rewritten_query,
        "history_rewrite_ms": history_rewrite_ms,
    }
