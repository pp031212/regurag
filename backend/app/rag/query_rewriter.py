from openai import OpenAI

from .query_rewriter_config import load_query_rewriter_prompt_config


class QueryRewriter:
    def __init__(self, api_key: str, base_url: str, model_name: str) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.prompt_config = load_query_rewriter_prompt_config()

    def _build_rewrite_prompt(self, user_query: str) -> str:
        return self.prompt_config.rewrite_prompt_template.format(
            subject_hint=self.prompt_config.subject_hint,
            keyword_examples=self.prompt_config.keyword_examples,
            behavior_examples=self.prompt_config.behavior_examples,
            policy_examples=self.prompt_config.policy_examples,
            query=user_query,
        )

    def _build_history_rewrite_prompt(self, user_query: str, history_text: str) -> str:
        return self.prompt_config.history_rewrite_prompt_template.format(
            history_text=history_text,
            user_query=user_query,
        )

    def _match_configured_rewrite(self, user_query: str) -> str | None:
        for rule in self.prompt_config.rewrite_rules:
            if all(keyword in user_query for keyword in rule.match_all):
                return rule.rewritten_query
        return None

    def rewrite(self, user_query: str) -> str:
        configured = self._match_configured_rewrite(user_query)
        if configured:
            return configured

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": self._build_rewrite_prompt(user_query)}],
            temperature=0.1,
            max_tokens=50,
        )
        return (response.choices[0].message.content or "").strip()

    def rewrite_with_history(self, user_query: str, history_messages: list[dict[str, str]]) -> str:
        if not history_messages:
            return user_query

        history_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in history_messages
            if message.get("content")
        ).strip()
        if not history_text:
            return user_query

        prompt = self._build_history_rewrite_prompt(user_query, history_text)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=120,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten or user_query
