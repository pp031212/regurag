"""RAG shadow graph 构建。

这个 graph 是 legacy pipeline 的对照实现，节点顺序保持和 RAGPipeline 主链路一致，
用于灰度验证，不直接承担服务层路由、短路或持久化。
"""

from __future__ import annotations

from .nodes import (
    RagWorkflowDependencies,
    generate_answer_node,
    prepare_query_node,
    retrieve_documents_node,
    select_parents_node,
)
from .state import RagWorkflowState


def build_shadow_graph(deps: RagWorkflowDependencies):
    """构建并编译 shadow graph。"""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is not installed in the current environment. "
            "Install it in the regurag environment before building the shadow graph."
        ) from exc

    builder = StateGraph(RagWorkflowState)
    # 每个节点只返回局部 state update，LangGraph 负责按边顺序合并到总 state。
    builder.add_node("prepare_query", lambda state: prepare_query_node(state, deps))
    builder.add_node("retrieve_documents", lambda state: retrieve_documents_node(state, deps))
    builder.add_node("select_parents", lambda state: select_parents_node(state, deps))
    builder.add_node("generate_answer", lambda state: generate_answer_node(state, deps))
    builder.add_edge(START, "prepare_query")
    builder.add_edge("prepare_query", "retrieve_documents")
    builder.add_edge("retrieve_documents", "select_parents")
    builder.add_edge("select_parents", "generate_answer")
    builder.add_edge("generate_answer", END)
    return builder.compile()
