import json
import re

from openai import OpenAI

from ..core.config import get_settings


_ALLOWED_LABELS = frozenset({
    "business_query",
    "follow_up_query",
    "off_topic",
    "meaningless_input",
})
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class IntentLLMClassifier:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout_seconds: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        settings = get_settings()
        resolved_api_key = api_key or settings.intent_llm_classifier_api_key or settings.rewrite_api_key
        resolved_base_url = base_url or settings.intent_llm_classifier_base_url or settings.rewrite_base_url
        resolved_model_name = model_name or settings.intent_llm_classifier_model or settings.rewrite_model
        resolved_timeout_seconds = timeout_seconds or settings.intent_llm_classifier_timeout_seconds
        resolved_max_tokens = max_tokens or settings.intent_llm_classifier_max_tokens

        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            timeout=resolved_timeout_seconds,
        )
        self.model_name = resolved_model_name
        self.max_tokens = resolved_max_tokens

    def _build_prompt(self, query: str, history_messages: list[dict[str, str]] | None = None) -> str:
        history_lines = []
        for message in list(history_messages or [])[-4:]:
            role = str(message.get("role") or "").strip()
            content = str(message.get("content") or "").strip()
            if role and content:
                history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines) if history_lines else "无历史"

        return (
            "你是一个聊天入口意图分类器，只能返回 JSON。\n"
            "请在以下标签中四选一：business_query, follow_up_query, off_topic, meaningless_input。\n"
            "判定规则：\n"
            "1. meaningless_input：纯语气词、笑声、感叹、无实际问题的信息。\n"
            "2. follow_up_query：当前问题明显依赖上文才能理解。\n"
            "3. off_topic：与制度、规则、流程、劳动法等业务问答明显无关。\n"
            "4. business_query：其余应进入业务检索的问题。\n"
            "输出格式必须是：{\"label\":\"...\"}\n"
            f"历史消息：\n{history_text}\n"
            f"当前用户输入：{query}"
        )

    @staticmethod
    def _parse_label(content: str) -> str | None:
        stripped = content.strip()
        if stripped in _ALLOWED_LABELS:
            return stripped

        match = _JSON_BLOCK_RE.search(stripped)
        candidate = match.group(0) if match else stripped
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None

        label = str(payload.get("label") or "").strip()
        if label in _ALLOWED_LABELS:
            return label
        return None

    def classify(self, query: str, *, history_messages: list[dict[str, str]] | None = None) -> str | None:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": self._build_prompt(query, history_messages)}],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        content = str(response.choices[0].message.content or "")
        return self._parse_label(content)
