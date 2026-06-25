from app.workflows.rag.pipeline_steps import (
    build_citations,
    build_debug,
    build_final_context_parents,
    collect_referenced_article_parents,
    extract_referenced_article_numbers,
    is_overview_query,
    preserve_overview_candidate_docs,
    sort_retrieved_docs,
)
from app.workflows.rag.overview_config import load_overview_query_config


def test_sort_retrieved_docs_prefers_table_over_image_ocr_table_on_same_page() -> None:
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

    sorted_docs = sort_retrieved_docs(docs)

    assert [doc["source_type"] for doc in sorted_docs] == ["table", "image_ocr_table"]


def test_build_final_context_parents_keeps_query_hit_before_generic_fill() -> None:
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

    final_context_parents = build_final_context_parents(
        parent_docs_for_rerank=parent_docs_for_rerank,
        reranked_parents=parent_docs_for_rerank[1:],
        query="不给劳动者购置五险可以直接离职吗",
        query_keywords=query_keywords,
        is_policy_question=False,
        must_keep_parent_keywords=(),
        low_value_parent_keywords=(),
        rule_signal_keywords=(),
        behavior_keywords=("迟到", "旷课"),
        top_k_rerank=3,
    )

    assert final_context_parents[0]["parent_id"] == "rule_doc_38_0"
    assert any(doc["parent_id"] == "rule_doc_38_1" for doc in final_context_parents)


def test_build_citations_preserves_parent_order_and_metadata() -> None:
    citations = build_citations(
        [
            {
                "parent_id": "p1",
                "text": "第一段",
                "rerank_score": 0.9,
                "source_name": "《A》",
                "source_type": "text",
                "page_number": 1,
                "block_index": 2,
            },
            {
                "parent_id": "p2",
                "text": "第二段",
                "rerank_score": 0.7,
                "source_type": "table",
                "page_number": 3,
                "block_index": 4,
            },
        ],
        "kb_001",
    )

    assert citations == [
        {
            "document_id": "kb_001",
            "chunk_id": "p1-1",
            "content": "第一段",
            "score": 0.9,
            "source_name": "《A》",
            "source_type": "text",
            "page_number": 1,
            "block_index": 2,
        },
        {
            "document_id": "kb_001",
            "chunk_id": "p2-2",
            "content": "第二段",
            "score": 0.7,
            "source_name": None,
            "source_type": "table",
            "page_number": 3,
            "block_index": 4,
        },
    ]


def test_build_debug_keeps_history_fields_and_chunk_lists() -> None:
    debug = build_debug(
        rewritten_query="standalone query",
        retrieved_count=4,
        reranked_count=2,
        enable_rewrite=True,
        enable_rerank=True,
        rewrite_ms=10,
        retrieve_ms=20,
        rerank_ms=30,
        context_build_ms=8,
        generate_ms=40,
        llm_result={"finish_reason": "stop", "model": "test-model", "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        retrieved_chunks=[{"chunk_id": "c1"}],
        mmr_selected_chunks=[{"chunk_id": "c2"}],
        reranked_chunks=[{"chunk_id": "c3"}],
        final_context_chunks=[{"chunk_id": "c4"}],
        used_conversation_history=True,
        history_message_count=3,
        history_rewritten_query="history query",
        history_rewrite_ms=12,
    )

    assert debug["rewritten_query"] == "standalone query"
    assert debug["llm_model"] == "test-model"
    assert debug["final_context_chunks"] == [{"chunk_id": "c4"}]
    assert debug["context_build_ms"] == 8
    assert debug["stage_timings_ms"]["context_build_ms"] == 8
    assert debug["used_conversation_history"] is True
    assert debug["history_message_count"] == 3
    assert debug["history_rewritten_query"] == "history query"



def test_is_overview_query_matches_general_summary_questions() -> None:
    assert is_overview_query("请假一般怎么处理？") is True
    assert is_overview_query("课堂违纪一般会怎么处理？") is True
    assert is_overview_query("老板不发工资怎么办？") is True
    assert is_overview_query("上课期间睡觉会怎么处理？") is False


def test_overview_query_config_loads_from_json_file() -> None:
    config = load_overview_query_config()

    assert "一般怎么处理" in config.overview_markers
    assert "请假" in config.topic_markers
    assert "工资" in config.topic_markers


def test_preserve_overview_candidate_docs_keeps_extra_candidates_in_rank_order() -> None:
    selected_docs = [
        {"child_text": "事假扣5分", "parent_text": "事假规则", "parent_id": "p1"},
    ]
    candidate_docs = [
        {"child_text": "事假扣5分", "parent_text": "事假规则", "parent_id": "p1"},
        {"child_text": "病假无证明按事假", "parent_text": "病假规则", "parent_id": "p2"},
        {"child_text": "特殊事项请假需单独报批", "parent_text": "特殊事项请假规则", "parent_id": "p3"},
    ]

    preserved_docs = preserve_overview_candidate_docs(selected_docs, candidate_docs, max_extra_docs=2)

    assert [doc["parent_id"] for doc in preserved_docs] == ["p1", "p2", "p3"]


def test_build_final_context_parents_keeps_more_overview_branches() -> None:
    parent_docs_for_rerank = [
        {"parent_id": "leave_sick", "text": "病假没有就诊证明时，按事假处理。", "source_type": "text", "page_number": 0, "rerank_score": 4.0},
        {"parent_id": "leave_personal", "text": "请事假扣5分/天，半天按半天计算。", "source_type": "text", "page_number": 0, "rerank_score": 4.2},
        {"parent_id": "leave_special", "text": "特殊事项请假需单独报批，每名学员仅有一次机会。", "source_type": "text", "page_number": 0, "rerank_score": 4.1},
    ]

    final_context_parents = build_final_context_parents(
        parent_docs_for_rerank=parent_docs_for_rerank,
        reranked_parents=[parent_docs_for_rerank[1]],
        query="请假一般怎么处理？",
        query_keywords=["请假", "处理标准"],
        is_policy_question=False,
        must_keep_parent_keywords=(),
        low_value_parent_keywords=(),
        rule_signal_keywords=(),
        behavior_keywords=("迟到", "旷课"),
        top_k_rerank=3,
    )

    assert {doc["parent_id"] for doc in final_context_parents} >= {"leave_sick", "leave_personal", "leave_special"}


def test_build_final_context_parents_falls_back_to_parent_docs_for_overview_query() -> None:
    parent_docs = [
        {"text": "病假按事假处理，扣5分/天"},
        {"text": "事假扣5分/天"},
        {"text": "特殊事项请假需单独报批"},
    ]

    final_context = build_final_context_parents(
        parent_docs_for_rerank=parent_docs,
        reranked_parents=[],
        query="请假一般怎么处理？",
        query_keywords=["病假", "事假", "特殊事项请假"],
        is_policy_question=False,
        must_keep_parent_keywords=(),
        low_value_parent_keywords=(),
        rule_signal_keywords=(),
        behavior_keywords=(),
        top_k_rerank=2,
    )

    assert len(final_context) == 3
    assert "病假" in str(final_context[0]["text"])


def test_extract_referenced_article_numbers_detects_one_hop_references() -> None:
    text = "第四十二条 劳动者有下列情形之一的，用人单位不得依照本法第四十条、第四十一条的规定解除劳动合同。"

    assert extract_referenced_article_numbers(text) == [40, 41]


def test_collect_referenced_article_parents_adds_missing_referenced_articles() -> None:
    parent_docs = [
        {"parent_id": "rule_doc_10", "text": "第十条 违纪处理的具体流程详见第十二条。"},
        {"parent_id": "rule_doc_11", "text": "第十一条 普通请假流程由班主任审批。"},
        {"parent_id": "rule_doc_12", "text": "第十二条 违纪处理应先记录事实，再通知本人确认。"},
    ]

    selected = collect_referenced_article_parents(
        parent_docs=parent_docs,
        final_context_parents=[parent_docs[0]],
    )

    assert [doc["parent_id"] for doc in selected] == ["rule_doc_12"]


def test_build_final_context_parents_keeps_referenced_article_before_generic_fill() -> None:
    parent_docs = [
        {
            "parent_id": "rule_doc_10",
            "text": "第十条 迟到处理的具体扣分标准详见第十二条。",
            "source_type": "text",
            "page_number": 0,
            "rerank_score": 4.8,
        },
        {
            "parent_id": "rule_doc_11",
            "text": "第十一条 宿舍卫生检查不合格扣2分。",
            "source_type": "text",
            "page_number": 0,
            "rerank_score": 4.7,
        },
        {
            "parent_id": "rule_doc_12",
            "text": "第十二条 超过二十分钟按旷课，扣10分。",
            "source_type": "text",
            "page_number": 0,
            "rerank_score": 4.1,
        },
    ]

    final_context = build_final_context_parents(
        parent_docs_for_rerank=parent_docs,
        reranked_parents=[parent_docs[0], parent_docs[1]],
        query="迟到超过二十分钟怎么处理？",
        query_keywords=["迟到", "处理"],
        is_policy_question=False,
        must_keep_parent_keywords=(),
        low_value_parent_keywords=(),
        rule_signal_keywords=(),
        behavior_keywords=("迟到", "旷课"),
        top_k_rerank=2,
    )

    assert [doc["parent_id"] for doc in final_context[:3]] == ["rule_doc_10", "rule_doc_12", "rule_doc_11"]

