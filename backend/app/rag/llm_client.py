import re
from time import perf_counter
from typing import Any, Generator, Literal

from openai import OpenAI


AnswerStyle = Literal["concise", "structured"]


CONCISE_GENERATION_PROMPT_TEMPLATE = """基于参考资料回答问题。

知识库主题：{subject}
如果问题中的机构名称与该主题一致或明显对应，直接按该知识库规则回答，不要额外声明名称未出现。

要求：
1. 只能依据参考资料作答，不要脱离资料发挥。
2. 先直接回答用户问题，再补充 1-3 条最关键依据；不要默认输出冗长三段式分析。
3. 若同时命中具体条款和更上位的总则、汇总规则或通用说明，先回答与问题最直接相关的结论，再简要补充上位规则。
4. 若参考资料中已经存在与问题情形直接对应的具体条款，优先引用该条款，不要退回只总结更泛化的总则，更不要说“不确定”。
5. 若结论含推断，明确写“这是基于相近条款的解释”。
6. 若资料不足，直接说明“参考资料未明确”以及缺的是什么，不要硬编。
7. 输出简洁，不重复，不使用 LaTeX。
8. 如果提供了“结合会话上下文补全后的完整问题”，优先按这个完整问题理解代词、省略和追问，但回答时仍围绕用户当前问题作答。
9. 如果参考资料中明确标注了不同来源，例如不同法律、不同文件，回答时要按来源区分，不要把不同来源的条文编号混写成同一部文件的连续体系。
10. 如果同一条文同时包含一般规则和特殊例外，先回答与用户问题直接对应的一般规则，再补充说明特殊例外；不要因为存在“立即解除”“不需事先告知”等更强表述，就否定同条文前半段已经明确给出的解除权。
11. 对于“日常违规/一般怎么处理/有哪些情况”这类概览型问题，如果资料只给出了若干具体情形而没有统一总则，明确写“需要区分具体情形，按对应规则处理”，不要拼成一条万能规则。

优先按下面格式输出，没有必要的部分可以省略：
结论：
- 直接回答用户问题
依据：
- 最关键的资料依据
补充说明：
- 仅在需要说明例外、资料不足或推断时补充

参考资料：
{context}

用户当前问题：
{query}

结合会话上下文补全后的完整问题：
{standalone_query}
"""


STRUCTURED_GENERATION_PROMPT_TEMPLATE = """基于参考资料回答问题。

知识库主题：{subject}
如果问题中的机构名称与该主题一致或明显对应，直接按该知识库规则回答，不要额外声明名称未出现。

要求：
1. 只能依据参考资料作答，不要脱离资料发挥。
2. 若同时命中具体条款和更上位的总则、汇总规则或通用说明，必须同时说明。
3. 若参考资料中已经存在与问题情形直接对应的具体条款，优先引用该条款，不要退回只总结更泛化的总则，更不要说“不确定”。
4. 若结论含推断，明确写“这是基于相近条款的解释”。
5. 若资料不足，直接说明不确定原因。
6. 输出简洁，不重复，不使用 LaTeX。
7. 如果提供了“结合会话上下文补全后的完整问题”，优先按这个完整问题理解代词、省略和追问，但回答时仍围绕用户当前问题作答。
8. 如果参考资料中明确标注了不同来源，例如不同法律、不同文件，回答时要按来源区分，不要把不同来源的条文编号混写成同一部文件的连续体系。
9. 如果同一条文同时包含一般规则和特殊例外，先回答与用户问题直接对应的一般规则，再补充说明特殊例外；不要因为存在“立即解除”“不需事先告知”等更强表述，就否定同条文前半段已经明确给出的解除权。
10. 对于“日常违规/一般怎么处理/有哪些情况”这类概览型问题，如果资料只给出了若干具体情形而没有统一总则，明确写“需要区分具体情形，按对应规则处理”，不要拼成一条万能规则。

按下面格式输出：
【直接依据】
- 相关条款
【推理/判断】
- 直接适用或参照理解
- 必要时简要说明判断过程
【结论】
- 最终结论

参考资料：
{context}

用户当前问题：
{query}

结合会话上下文补全后的完整问题：
{standalone_query}
"""


class LLMGenerator:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        subject: str,
        timeout_seconds: int,
        max_tokens: int,
        *,
        fallback_api_key: str | None = None,
        fallback_base_url: str | None = None,
        fallback_model_name: str | None = None,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self.model_name = model_name
        self.subject = subject
        self.max_tokens = max_tokens
        self.fallback_client = None
        self.fallback_model_name = fallback_model_name
        if fallback_api_key and fallback_base_url and fallback_model_name:
            self.fallback_client = OpenAI(
                api_key=fallback_api_key,
                base_url=fallback_base_url,
                timeout=timeout_seconds,
            )

    def generate(
        self,
        query: str,
        context: str,
        standalone_query: str | None = None,
        *,
        answer_style: AnswerStyle = "concise",
    ) -> dict[str, object]:
        prompt = self._build_prompt(
            query=query,
            context=context,
            standalone_query=standalone_query,
            answer_style=answer_style,
        )
        messages = [
            {"role": "system", "content": "你是一个严谨的 AI 助手。"},
            {"role": "user", "content": prompt},
        ]
        response = self._create_completion(self.client, self.model_name, messages)
        choice = response.choices[0]
        usage = response.usage
        answer = self._sanitize_answer(self._extract_message_text(choice.message.content))
        fallback_used = False

        if not answer and self.fallback_client is not None and self.fallback_model_name:
            fallback_response = self._create_completion(self.fallback_client, self.fallback_model_name, messages)
            fallback_choice = fallback_response.choices[0]
            fallback_answer = self._sanitize_answer(self._extract_message_text(fallback_choice.message.content))
            if fallback_answer:
                response = fallback_response
                choice = fallback_choice
                usage = fallback_response.usage
                answer = fallback_answer
                fallback_used = True

        return {
            "answer": answer,
            "finish_reason": choice.finish_reason,
            "model": response.model,
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            "fallback_used": fallback_used,
        }

    def stream_generate(
        self,
        query: str,
        context: str,
        standalone_query: str | None = None,
        *,
        answer_style: AnswerStyle = "concise",
    ) -> Generator[str, None, dict[str, object]]:
        prompt = self._build_prompt(
            query=query,
            context=context,
            standalone_query=standalone_query,
            answer_style=answer_style,
        )
        messages = [
            {"role": "system", "content": "你是一个严谨的 AI 助手。"},
            {"role": "user", "content": prompt},
        ]

        started_at = perf_counter()
        stream = self._create_stream_completion(self.client, self.model_name, messages)
        collected_parts: list[str] = []
        finish_reason: str | None = None
        first_token_ms: int | None = None
        usage = None

        for chunk in stream:
            usage = getattr(chunk, "usage", usage)
            if getattr(chunk, "choices", None):
                choice = chunk.choices[0]
                finish_reason = getattr(choice, "finish_reason", finish_reason)
                delta = self._extract_delta_text(getattr(choice, "delta", None))
                if delta:
                    if first_token_ms is None:
                        first_token_ms = int((perf_counter() - started_at) * 1000)
                    collected_parts.append(delta)
                    yield delta

        answer = self._sanitize_answer("".join(collected_parts))
        return {
            "answer": answer,
            "finish_reason": finish_reason,
            "model": self.model_name,
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            "fallback_used": False,
            "first_token_ms": first_token_ms,
        }

    def _create_completion(self, client: OpenAI, model_name: str, messages: list[dict[str, str]]) -> Any:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=self.max_tokens,
        )

    def _create_stream_completion(self, client: OpenAI, model_name: str, messages: list[dict[str, str]]) -> Any:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=self.max_tokens,
            stream=True,
        )

    def _build_prompt(
        self,
        query: str,
        context: str,
        standalone_query: str | None = None,
        *,
        answer_style: AnswerStyle = "concise",
    ) -> str:
        template = STRUCTURED_GENERATION_PROMPT_TEMPLATE if answer_style == "structured" else CONCISE_GENERATION_PROMPT_TEMPLATE
        return template.format(
            context=context,
            query=query,
            standalone_query=standalone_query or query,
            subject=self.subject,
        )

    @staticmethod
    def _sanitize_answer(text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _extract_message_text(content: object) -> str:
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            return "\n".join(part for part in parts if part).strip()
        return str(content)

    @classmethod
    def _extract_delta_text(cls, delta: object) -> str:
        if delta is None:
            return ""
        content = getattr(delta, "content", None)
        return cls._extract_message_text(content)
