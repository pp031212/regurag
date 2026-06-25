from app.rag.pipeline import RAGPipeline
from app.rag.query_alias_config import QueryAliasExpander, load_query_alias_config
from app.rag.retrieval_rules_config import load_retrieval_rules_config
from app.workflows.rag.pipeline_steps import build_final_context_parents, collect_adjacent_split_parent_parents


def test_sort_retrieved_docs_prefers_table_over_image_ocr_table_on_same_page() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    docs = [
        {
            "id": "img-table",
            "child_text": "ocr table child",
            "parent_text": "ocr table parent",
            "parent_id": "p2",
            "source_type": "image_ocr_table",
            "page_number": 2,
            "distance": 0.1,
        },
        {
            "id": "table",
            "child_text": "table child",
            "parent_text": "table parent",
            "parent_id": "p1",
            "source_type": "table",
            "page_number": 2,
            "distance": 0.11,
        },
    ]

    sorted_docs = pipeline._sort_retrieved_docs(docs)

    assert sorted_docs[0]["source_type"] == "table"
    assert sorted_docs[1]["source_type"] == "image_ocr_table"


def test_sort_retrieved_docs_prefers_high_keyword_hit_candidate() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    docs = [
        {
            "id": "vector-hit",
            "child_text": "劳动者可以立即解除劳动合同。",
            "parent_text": "立即解除劳动合同情形。",
            "parent_id": "p1",
            "source_type": "text",
            "page_number": 1,
            "distance": 0.05,
        },
        {
            "id": "keyword-hit",
            "child_text": "（三）未依法为劳动者缴纳社会保险费的。",
            "parent_text": "第三十八条 用人单位有下列情形之一的，劳动者可以解除劳动合同：（三）未依法为劳动者缴纳社会保险费的；",
            "parent_id": "p2",
            "source_type": "text",
            "page_number": 1,
            "distance": None,
            "keyword_hit_count": 2,
        },
    ]

    sorted_docs = pipeline._sort_retrieved_docs(docs)

    assert sorted_docs[0]["id"] == "keyword-hit"


def test_sort_reranked_parents_prefers_text_over_image_ocr_when_scores_are_close() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    docs = [
        {
            "parent_id": "text-parent",
            "text": "text parent",
            "source_type": "text",
            "page_number": 1,
            "rerank_score": 0.86,
        },
        {
            "parent_id": "image-parent",
            "text": "image parent",
            "source_type": "image_ocr",
            "page_number": 1,
            "rerank_score": 0.9,
        },
    ]

    sorted_docs = pipeline._sort_reranked_parents(docs)

    assert sorted_docs[0]["source_type"] == "text"


def test_sort_retrieved_docs_demotes_low_quality_ocr_when_scores_are_close() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    docs = [
        {
            "id": "normal-ocr",
            "child_text": "迟到扣10分",
            "parent_text": "迟到扣10分",
            "parent_id": "p1",
            "source_type": "image_ocr",
            "page_number": 1,
            "distance": 0.2,
            "ocr_quality": "normal",
        },
        {
            "id": "low-ocr",
            "child_text": "迟到扣10分",
            "parent_text": "迟到扣10分",
            "parent_id": "p2",
            "source_type": "image_ocr",
            "page_number": 2,
            "distance": 0.12,
            "ocr_quality": "low",
        },
    ]

    sorted_docs = pipeline._sort_retrieved_docs(docs)

    assert sorted_docs[0]["id"] == "normal-ocr"


def test_sort_reranked_parents_demotes_low_quality_ocr_when_scores_are_close() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    docs = [
        {
            "parent_id": "normal-ocr",
            "text": "迟到扣10分",
            "source_type": "image_ocr",
            "page_number": 1,
            "rerank_score": 0.74,
            "ocr_quality": "normal",
        },
        {
            "parent_id": "low-ocr",
            "text": "迟到扣10分",
            "source_type": "image_ocr",
            "page_number": 2,
            "rerank_score": 0.86,
            "ocr_quality": "low",
        },
    ]

    sorted_docs = pipeline._sort_reranked_parents(docs)

    assert sorted_docs[0]["parent_id"] == "normal-ocr"


def test_query_alias_expander_adds_teacher_violence_terms() -> None:
    expander = QueryAliasExpander()

    aliases = expander.expand("肘击老师会怎么样")

    assert "学生与老师之间打架斗殴" in aliases
    assert "与老师发生冲突" in aliases
    assert "扣40分" in aliases


def test_query_alias_expander_adds_teacher_verbal_conflict_terms() -> None:
    expander = QueryAliasExpander()

    aliases = expander.expand("顶撞老师会怎么处理")

    assert "与老师发生语言冲突" in aliases
    assert "顶撞老师" in aliases
    assert "扣20分" in aliases


def test_query_alias_expander_adds_configured_labor_law_term_aliases() -> None:
    expander = QueryAliasExpander()

    aliases = expander.expand("不给劳动者购置五险可以直接离职吗")

    assert "社会保险费" in aliases
    assert "解除劳动合同" in aliases


def test_query_alias_expander_adds_dismissal_terms_for_labor_law_queries() -> None:
    expander = QueryAliasExpander()

    aliases = expander.expand("用人单位辞退员工必须要满足什么条件才能辞退")

    assert "解除劳动合同" in aliases
    assert "用人单位解除劳动合同" in aliases
    assert "裁减人员" in aliases
    assert "不得解除劳动合同" in aliases


def test_query_alias_config_loads_from_json_file() -> None:
    config = load_query_alias_config()

    assert "五险" in config.term_aliases
    assert "社会保险费" in config.term_aliases["五险"]
    assert any("扣40分" in rule.aliases for rule in config.conditional_alias_rules)


def test_retrieval_rules_config_loads_from_json_file() -> None:
    config = load_retrieval_rules_config()

    assert "累计" in config.policy_trigger_words
    assert "迟到" in config.behavior_keywords
    assert "无条件退学" in config.must_keep_child_keywords
    assert "以下为和鸣教育管理制度" in config.low_value_parent_keywords
    assert "课堂违纪" in config.overview_query_expansions
    assert "课堂纪律" in config.overview_query_expansions["课堂违纪"]


def test_preserve_query_keyword_docs_adds_high_coverage_candidate() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    selected_docs = [
        {
            "child_text": "可以立即解除劳动合同，不需事先告知。",
            "parent_text": "立即解除劳动合同情形。",
            "parent_id": "p1",
            "source_type": "text",
            "distance": 0.66,
        }
    ]
    candidate_docs = [
        *selected_docs,
        {
            "child_text": "未依法为劳动者缴纳社会保险费的。",
            "parent_text": "第三十八条 用人单位有下列情形之一的，劳动者可以解除劳动合同：（三）未依法为劳动者缴纳社会保险费的；",
            "parent_id": "p2",
            "source_type": "text",
            "distance": 0.63,
        },
    ]

    preserved_docs = pipeline._preserve_query_keyword_docs(
        selected_docs=selected_docs,
        candidate_docs=candidate_docs,
        query_keywords=["社会保险费", "解除劳动合同"],
    )

    assert len(preserved_docs) == 2
    assert any(doc["parent_id"] == "p2" for doc in preserved_docs)


def test_final_context_should_allow_query_keyword_parent_before_generic_fill() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    query_keywords = ["社会保险费", "解除劳动合同"]
    parent_docs_for_rerank = [
        {
            "parent_id": "rule_doc_38_0",
            "text": "第三十八条 用人单位有下列情形之一的，劳动者可以解除劳动合同：（三）未依法为劳动者缴纳社会保险费的；",
            "source_type": "text",
            "page_number": 0,
            "rerank_score": 4.0,
        },
        {
            "parent_id": "rule_doc_38_1",
            "text": "劳动者可以立即解除劳动合同，不需事先告知用人单位。",
            "source_type": "text",
            "page_number": 0,
            "rerank_score": 5.2,
        },
        {
            "parent_id": "rule_doc_36",
            "text": "第三十六条 用人单位与劳动者协商一致，可以解除劳动合同。",
            "source_type": "text",
            "page_number": 0,
            "rerank_score": 4.6,
        },
        {
            "parent_id": "ocr_parent",
            "text": "第二十四条经劳动合同当事人协商一致，劳动合同可以解除。",
            "source_type": "image_ocr",
            "page_number": 5,
            "rerank_score": 4.5,
        },
    ]
    reranked_parents = [
        parent_docs_for_rerank[1],
        parent_docs_for_rerank[2],
        parent_docs_for_rerank[3],
    ]

    final_context_parents: list[dict[str, object]] = []
    added_parent_texts: set[str] = set()

    for parent_doc in parent_docs_for_rerank:
        text = str(parent_doc["text"])
        if text in added_parent_texts:
            continue
        if pipeline._count_query_keyword_hits(parent_doc, query_keywords) >= 2:
            final_context_parents.append(parent_doc)
            added_parent_texts.add(text)

    for doc in reranked_parents:
        text = str(doc["text"])
        if text not in added_parent_texts:
            final_context_parents.append(doc)
            added_parent_texts.add(text)

    final_context_parents = final_context_parents[: 3 + 2]

    assert final_context_parents[0]["parent_id"] == "rule_doc_38_0"
    assert any(doc["parent_id"] == "rule_doc_38_1" for doc in final_context_parents)


def test_collect_overview_article_parents_keeps_adjacent_article_cluster() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    parent_docs = [
        {
            "parent_id": "rule_doc_39",
            "text": "第三十九条 劳动者有下列情形之一的，用人单位可以解除劳动合同。",
        },
        {
            "parent_id": "rule_doc_40",
            "text": "第四十条 有下列情形之一的，用人单位提前三十日以书面形式通知劳动者本人后，可以解除劳动合同。",
        },
        {
            "parent_id": "rule_doc_41",
            "text": "第四十一条 有下列情形之一，需要裁减人员二十人以上的，可以裁减人员。",
        },
        {
            "parent_id": "rule_doc_42",
            "text": "第四十二条 劳动者有下列情形之一的，用人单位不得依照本法第四十条、第四十一条的规定解除劳动合同。",
        },
    ]

    selected = pipeline._collect_overview_article_parents(
        parent_docs=parent_docs,
        query="用人单位辞退员工必须要满足什么条件才能辞退？",
        query_keywords=["解除劳动合同", "辞退"],
    )

    assert [doc["parent_id"] for doc in selected] == ["rule_doc_39", "rule_doc_40", "rule_doc_41", "rule_doc_42"]


def test_collect_related_restriction_article_parents_adds_article_42_when_40_or_41_present() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    parent_docs = [
        {
            "parent_id": "rule_doc_40",
            "text": "第四十条 有下列情形之一的，用人单位可以解除劳动合同。",
        },
        {
            "parent_id": "rule_doc_41",
            "text": "第四十一条 有下列情形之一，可以裁减人员。",
        },
        {
            "parent_id": "rule_doc_42",
            "text": "第四十二条 劳动者有下列情形之一的，用人单位不得依照本法第四十条、第四十一条的规定解除劳动合同。",
        },
    ]
    final_context_parents = [parent_docs[0], parent_docs[1]]

    selected = pipeline._collect_related_restriction_article_parents(
        parent_docs=parent_docs,
        final_context_parents=final_context_parents,
        query="用人单位辞退员工必须要满足什么条件才能辞退？",
    )

    assert [doc["parent_id"] for doc in selected] == ["rule_doc_42"]


def test_collect_adjacent_split_parent_parents_adds_neighbor_from_same_long_section() -> None:
    parent_docs = [
        {
            "parent_id": "rule_doc_38_0",
            "document_id": "labor_law",
            "source_type": "text",
            "text": "第三十八条 用人单位未依法缴纳社会保险费的，劳动者可以解除劳动合同。",
        },
        {
            "parent_id": "rule_doc_38_1",
            "document_id": "labor_law",
            "source_type": "text",
            "text": "劳动者可以立即解除劳动合同，不需事先告知用人单位。",
        },
        {
            "parent_id": "rule_doc_39",
            "document_id": "labor_law",
            "source_type": "text",
            "text": "第三十九条 用人单位可以解除劳动合同的情形。",
        },
    ]

    selected = collect_adjacent_split_parent_parents(
        parent_docs=parent_docs,
        final_context_parents=[parent_docs[0]],
        query_keywords=["解除劳动合同"],
    )

    assert [doc["parent_id"] for doc in selected] == ["rule_doc_38_1"]
    assert selected[0]["context_expand_reason"] == "adjacent_split_parent"


def test_collect_adjacent_split_parent_parents_does_not_merge_separate_articles() -> None:
    parent_docs = [
        {
            "parent_id": "rule_doc_38",
            "document_id": "labor_law",
            "source_type": "text",
            "text": "第三十八条 劳动者可以解除劳动合同。",
        },
        {
            "parent_id": "rule_doc_39",
            "document_id": "labor_law",
            "source_type": "text",
            "text": "第三十九条 用人单位可以解除劳动合同。",
        },
    ]

    selected = collect_adjacent_split_parent_parents(
        parent_docs=parent_docs,
        final_context_parents=[parent_docs[0]],
    )

    assert selected == []


def test_collect_adjacent_split_parent_parents_requires_trigger_signal() -> None:
    parent_docs = [
        {
            "parent_id": "rule_doc_38_0",
            "document_id": "labor_law",
            "source_type": "text",
            "text": "第三十八条 用人单位未依法缴纳社会保险费的，劳动者可以解除劳动合同。",
        },
        {
            "parent_id": "rule_doc_38_1",
            "document_id": "labor_law",
            "source_type": "text",
            "text": "本条其他说明内容。",
        },
    ]

    selected = collect_adjacent_split_parent_parents(
        parent_docs=parent_docs,
        final_context_parents=[parent_docs[0]],
        query_keywords=["社会保险费"],
    )

    assert selected == []


def test_collect_adjacent_split_parent_parents_expands_boundary_dependency() -> None:
    parent_docs = [
        {
            "parent_id": "rule_doc_38_0",
            "document_id": "labor_law",
            "source_type": "text",
            "text": "第三十八条 用人单位有下列情形之一的，劳动者可以解除劳动合同：",
        },
        {
            "parent_id": "rule_doc_38_1",
            "document_id": "labor_law",
            "source_type": "text",
            "text": "（三）未依法为劳动者缴纳社会保险费的。",
        },
    ]

    selected = collect_adjacent_split_parent_parents(
        parent_docs=parent_docs,
        final_context_parents=[parent_docs[0]],
        query_keywords=[],
    )

    assert [doc["parent_id"] for doc in selected] == ["rule_doc_38_1"]


def test_build_final_context_parents_expands_adjacent_parent_from_context_pool() -> None:
    parent_docs_for_rerank = [
        {
            "parent_id": "rule_doc_38_0",
            "document_id": "labor_law",
            "source_type": "text",
            "text": "第三十八条 用人单位未依法缴纳社会保险费的，劳动者可以解除劳动合同。",
            "rerank_score": 5.0,
        }
    ]
    context_parent_pool = [
        *parent_docs_for_rerank,
        {
            "parent_id": "rule_doc_38_1",
            "document_id": "labor_law",
            "source_type": "text",
            "text": "劳动者可以立即解除劳动合同，不需事先告知用人单位。",
            "rerank_score": 0.0,
        },
    ]

    final_context = build_final_context_parents(
        parent_docs_for_rerank=parent_docs_for_rerank,
        reranked_parents=parent_docs_for_rerank,
        context_parent_pool=context_parent_pool,
        query="没交五险可以直接离职吗",
        query_keywords=["社会保险费", "解除劳动合同"],
        is_policy_question=False,
        must_keep_parent_keywords=(),
        low_value_parent_keywords=(),
        rule_signal_keywords=("解除劳动合同",),
        behavior_keywords=(),
        top_k_rerank=3,
    )

    assert [doc["parent_id"] for doc in final_context] == ["rule_doc_38_0", "rule_doc_38_1"]


def test_format_context_parent_prettifies_law_article_items() -> None:
    formatted = RAGPipeline._format_context_parent(
        {
            "source_name": "《中华人民共和国劳动合同法》",
            "text": (
                "中华人民共和国劳动合同法第四章 劳动合同的解除和终止"
                "第三十八条 用人单位有下列情形之一的，劳动者可以解除劳动合同："
                "（一）未按照劳动合同约定提供劳动保护或者劳动条件的；"
                "（二）未及时足额支付劳动报酬的；"
                "（三）未依法为劳动者缴纳社会保险费的。"
                "劳动者可以立即解除劳动合同，不需事先告知用人单位。"
            ),
        }
    )

    assert formatted.startswith("[来源：《中华人民共和国劳动合同法》]\n中华人民共和国劳动合同法")
    assert "\n第四章 劳动合同的解除和终止\n第三十八条" in formatted
    assert "（二）未及时足额支付劳动报酬的；\n（三）未依法为劳动者缴纳社会保险费的。" in formatted
    assert "。\n劳动者可以立即解除劳动合同，不需事先告知用人单位。" in formatted
