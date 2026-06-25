from types import SimpleNamespace

from app.rag.llm_client import LLMGenerator


class StubCompletions:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        if not self._responses:
            raise AssertionError("no stub responses left")
        return self._responses.pop(0)


class StubClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=StubCompletions(responses))


def make_response(*, content, model: str = "stub-model", finish_reason: str = "stop"):
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def make_stream_chunk(*, content=None, finish_reason=None, usage=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                delta=SimpleNamespace(content=content),
            )
        ],
        usage=usage,
    )


def test_sanitize_answer_removes_think_block() -> None:
    raw_answer = "<think>internal reasoning</think>\n\n【结论】\n- 可以正常显示"

    cleaned = LLMGenerator._sanitize_answer(raw_answer)

    assert cleaned == "【结论】\n- 可以正常显示"


def test_sanitize_answer_keeps_normal_answer() -> None:
    raw_answer = "【结论】\n- 这是正常回答"

    cleaned = LLMGenerator._sanitize_answer(raw_answer)

    assert cleaned == raw_answer


def test_build_prompt_includes_standalone_query_when_provided() -> None:
    generator = object.__new__(LLMGenerator)
    generator.subject = "劳动法"

    prompt = generator._build_prompt(
        query="该期间不发工资可以吗",
        context="试用期工资相关条款",
        standalone_query="试用期期间不发工资可以吗",
        answer_style="concise",
    )

    assert "用户当前问题：\n该期间不发工资可以吗" in prompt
    assert "结合会话上下文补全后的完整问题：\n试用期期间不发工资可以吗" in prompt


def test_build_prompt_uses_concise_template_by_default() -> None:
    generator = object.__new__(LLMGenerator)
    generator.subject = "和鸣教育管理制度"

    prompt = generator._build_prompt(
        query="迟到会怎么样",
        context="迟到条款",
    )

    assert "结论：" in prompt
    assert "依据：" in prompt
    assert "【推理/判断】" not in prompt


def test_build_prompt_uses_structured_template_when_requested() -> None:
    generator = object.__new__(LLMGenerator)
    generator.subject = "和鸣教育管理制度"

    prompt = generator._build_prompt(
        query="迟到会怎么样",
        context="迟到条款",
        answer_style="structured",
    )

    assert "【直接依据】" in prompt
    assert "【推理/判断】" in prompt
    assert "【结论】" in prompt


def test_extract_message_text_supports_content_parts() -> None:
    content = [
        {"type": "output_text", "text": "第一段"},
        SimpleNamespace(text="第二段"),
    ]

    extracted = LLMGenerator._extract_message_text(content)

    assert extracted == "第一段\n第二段"


def test_generate_falls_back_when_primary_answer_is_empty() -> None:
    generator = object.__new__(LLMGenerator)
    generator.subject = "和鸣教育管理制度"
    generator.model_name = "primary-model"
    generator.max_tokens = 256
    generator.client = StubClient([make_response(content=None, model="primary-model")])
    generator.fallback_model_name = "fallback-model"
    generator.fallback_client = StubClient([make_response(content="结论：\n- 需要补交病假条", model="fallback-model")])

    result = generator.generate(query="病假怎么请？", context="病假条款")

    assert result["answer"] == "结论：\n- 需要补交病假条"
    assert result["model"] == "fallback-model"
    assert result["fallback_used"] is True


def test_build_prompt_mentions_overview_boundary_rule() -> None:
    generator = object.__new__(LLMGenerator)
    generator.subject = "和鸣教育管理制度"

    prompt = generator._build_prompt(
        query="学生日常违规一般会怎么处理？",
        context="若干具体违规条款",
    )

    assert "需要区分具体情形，按对应规则处理" in prompt


def test_extract_delta_text_supports_content_parts() -> None:
    delta = SimpleNamespace(
        content=[
            {"type": "output_text", "text": "第一段"},
            SimpleNamespace(text="第二段"),
        ]
    )

    extracted = LLMGenerator._extract_delta_text(delta)

    assert extracted == "第一段\n第二段"


def test_stream_generate_collects_tokens_and_usage() -> None:
    generator = object.__new__(LLMGenerator)
    generator.subject = "和鸣教育管理制度"
    generator.model_name = "stream-model"
    generator.max_tokens = 256
    generator.client = StubClient(
        [
            [
                make_stream_chunk(content="结论："),
                make_stream_chunk(content="\n- 可以申请"),
                make_stream_chunk(
                    content=None,
                    finish_reason="stop",
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                ),
            ]
        ]
    )

    stream = generator.stream_generate(query="病假怎么请？", context="病假条款")
    chunks: list[str] = []

    while True:
        try:
            chunks.append(next(stream))
        except StopIteration as stop:
            result = stop.value
            break

    assert chunks == ["结论：", "\n- 可以申请"]
    assert result["answer"] == "结论：\n- 可以申请"
    assert result["finish_reason"] == "stop"
    assert result["model"] == "stream-model"
    assert result["total_tokens"] == 15
    assert isinstance(result["first_token_ms"], int)
