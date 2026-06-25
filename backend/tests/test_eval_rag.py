import asyncio

from scripts.eval_rag import EvalCase, run_case


class _StubService:
    def __init__(self, *, answer: str = "病假 就诊证明", debug: dict | None = None) -> None:
        self.payloads = []
        self.answer = answer
        self.debug = debug or {
            "retrieved_chunks": [],
            "final_context_chunks": [],
        }

    async def query(self, payload):
        self.payloads.append(payload)
        return {
            "answer": self.answer,
            "citations": [],
            "debug": self.debug,
        }


class _RetryableTimeout(Exception):
    pass


_RetryableTimeout.__name__ = "APITimeoutError"


class _FlakyService(_StubService):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def query(self, payload):
        self.calls += 1
        if self.calls == 1:
            raise _RetryableTimeout("timeout")
        return await super().query(payload)


def test_eval_run_case_disables_auto_route_by_default() -> None:
    service = _StubService()
    case = EvalCase(
        id="case-1",
        question="病假需要满足什么条件？",
        answer_mode="grounded",
        expected_answer_keywords=["病假", "就诊证明"],
        expected_context_keywords=[],
        category="test",
    )

    asyncio.run(
        run_case(
            service=service,
            case=case,
            knowledge_base_id="kb_001",
            top_k_retrieve=5,
            top_k_rerank=3,
            enable_rewrite=True,
            enable_rerank=True,
        )
    )

    assert len(service.payloads) == 1
    assert service.payloads[0].enable_auto_route is False


def test_eval_run_case_can_enable_auto_route_explicitly() -> None:
    service = _StubService()
    case = EvalCase(
        id="case-2",
        question="病假需要满足什么条件？",
        answer_mode="grounded",
        expected_answer_keywords=["病假", "就诊证明"],
        expected_context_keywords=[],
        category="test",
    )

    asyncio.run(
        run_case(
            service=service,
            case=case,
            knowledge_base_id="kb_001",
            top_k_retrieve=5,
            top_k_rerank=3,
            enable_rewrite=True,
            enable_rerank=True,
            enable_auto_route=True,
        )
    )

    assert len(service.payloads) == 1
    assert service.payloads[0].enable_auto_route is True


def test_eval_run_case_retries_retryable_timeout_once() -> None:
    service = _FlakyService()
    case = EvalCase(
        id="case-3",
        question="病假需要满足什么条件？",
        answer_mode="grounded",
        expected_answer_keywords=["病假", "就诊证明"],
        expected_context_keywords=[],
        category="test",
    )

    result = asyncio.run(
        run_case(
            service=service,
            case=case,
            knowledge_base_id="kb_001",
            top_k_retrieve=5,
            top_k_rerank=3,
            enable_rewrite=True,
            enable_rerank=True,
        )
    )

    assert service.calls == 2
    assert result.answer_hit is True


def test_eval_run_case_supports_context_keyword_groups() -> None:
    service = _StubService(
        answer="上课期间睡觉 扣5分",
        debug={
            "retrieved_chunks": [
                {
                    "child_text": "上课期间睡觉，扣5分/次。",
                    "parent_text": "上课期间睡觉，扣5分/次。",
                }
            ],
            "final_context_chunks": [
                {
                    "child_text": "上课期间睡觉，扣5分/次。",
                    "parent_text": "上课期间睡觉，扣5分/次。",
                }
            ],
        },
    )
    case = EvalCase(
        id="case-4",
        question="上课期间睡觉会怎么处理？",
        answer_mode="grounded",
        expected_answer_keywords=["上课期间睡觉", "扣5分"],
        expected_context_keywords=["上课睡觉", "扣5分"],
        expected_context_keyword_groups=[["上课睡觉", "上课期间睡觉"], ["扣5分"]],
        category="test",
    )

    result = asyncio.run(
        run_case(
            service=service,
            case=case,
            knowledge_base_id="kb_001",
            top_k_retrieve=5,
            top_k_rerank=3,
            enable_rewrite=True,
            enable_rerank=True,
        )
    )

    assert result.retrieval_hit is True
    assert result.final_context_hit is True
    assert result.matched_context_keywords == ["上课期间睡觉", "扣5分"]


def test_eval_run_case_supports_answer_keyword_groups() -> None:
    service = _StubService(
        answer="单独请上午、下午或者晚自习，按请假半天计算，并进行相应的扣分处理。",
        debug={
            "retrieved_chunks": [
                {
                    "child_text": "如单独请上午、下午或者晚自习，一律按照请假半天计算，进行相应的扣分。",
                    "parent_text": "如单独请上午、下午或者晚自习，一律按照请假半天计算，进行相应的扣分。",
                }
            ],
            "final_context_chunks": [
                {
                    "child_text": "如单独请上午、下午或者晚自习，一律按照请假半天计算，进行相应的扣分。",
                    "parent_text": "如单独请上午、下午或者晚自习，一律按照请假半天计算，进行相应的扣分。",
                }
            ],
        },
    )
    case = EvalCase(
        id="case-5",
        question="单独请上午、下午或者晚自习怎么算？",
        answer_mode="grounded",
        expected_answer_keywords=["按半天计算", "扣5分"],
        expected_context_keywords=["上午", "下午", "晚自习", "半天"],
        category="test",
        expected_answer_keyword_groups=[
            ["按半天计算", "按请假半天计算"],
            ["扣5分", "按事假规则相应扣分", "进行相应的扣分处理"],
        ],
    )

    result = asyncio.run(
        run_case(
            service=service,
            case=case,
            knowledge_base_id="kb_001",
            top_k_retrieve=5,
            top_k_rerank=3,
            enable_rewrite=True,
            enable_rerank=True,
        )
    )

    assert result.answer_hit is True
    assert result.answer_hit_ratio == 1.0
    assert result.matched_answer_keywords == ["按请假半天计算", "进行相应的扣分处理"]


def test_eval_run_case_supports_no_answer_keyword_groups() -> None:
    service = _StubService(answer="目前无法依据所给资料确定，现有资料无相关条款可供参照，且未包含相关规定。")
    case = EvalCase(
        id="case-6",
        question="校园里可以带宠物进宿舍吗？",
        answer_mode="no_answer",
        expected_answer_keywords=["不确定", "未提供"],
        expected_context_keywords=[],
        category="test",
        expected_answer_keyword_groups=[
            ["不确定", "无法确定", "无法依据所给资料确定"],
            [
                "未提供",
                "没有相关明确规定",
                "资料不足",
                "没有直接条款",
                "未见直接规定",
                "无相关条款可供参照",
                "未包含相关规定",
                "明确规定",
                "相关规定",
            ],
        ],
    )

    result = asyncio.run(
        run_case(
            service=service,
            case=case,
            knowledge_base_id="kb_001",
            top_k_retrieve=5,
            top_k_rerank=3,
            enable_rewrite=True,
            enable_rerank=True,
        )
    )

    assert result.answer_hit is True
    assert result.answer_hit_ratio == 1.0
    assert result.matched_answer_keywords == ["无法依据所给资料确定", "无相关条款可供参照"]
