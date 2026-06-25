"""shadow workflow 函数式执行器。"""

from __future__ import annotations

from .nodes import (
    RagWorkflowDependencies,
    generate_answer_node,
    prepare_query_node,
    retrieve_documents_node,
    select_parents_node,
)
from .state import RagWorkflowState


def run_shadow_workflow(
    initial_state: RagWorkflowState,
    deps: RagWorkflowDependencies,
) -> RagWorkflowState:
    """按固定顺序执行 shadow 节点，便于不依赖 LangGraph 时做测试。"""
    state: RagWorkflowState = dict(initial_state)
    for node in (
        prepare_query_node,
        retrieve_documents_node,
        select_parents_node,
        generate_answer_node,
    ):
        state.update(node(state, deps))
    return state
