import re
from dataclasses import dataclass
from enum import StrEnum

from ..core.config import get_settings
from .faq_shortcut_config import FAQShortcutRule, load_faq_shortcut_config
from .intent_llm_classifier import IntentLLMClassifier
from .intent_local_classifier import IntentLocalClassifier
from .light_intent_config import LightIntentRule, load_light_intent_config, normalize_query, split_light_intent_clauses


class IntentType(StrEnum):
    BUSINESS_QUERY = "business_query"
    FOLLOW_UP_QUERY = "follow_up_query"
    LIGHT_INTENT = "light_intent"
    FAQ_SHORTCUT = "faq_shortcut"
    OFF_TOPIC = "off_topic"
    MEANINGLESS_INPUT = "meaningless_input"


@dataclass(frozen=True)
class IntentDecision:
    intent: IntentType
    source: str
    short_circuit: bool
    finish_reason: str | None = None
    light_intent_rule: LightIntentRule | None = None
    faq_shortcut_rule: FAQShortcutRule | None = None
    classifier_source: str | None = None
    classifier_mode: str | None = None
    classifier_score: float | None = None
    classifier_margin: float | None = None


_MEANINGLESS_SINGLE_CHARACTERS = frozenset(
    {"哈", "呵", "嘿", "嘻", "嗯", "哦", "啊", "哇", "呃", "额", "?", "？", "!", "！", "~"}
)
_MEANINGLESS_REPEAT_RE = re.compile(r"^(哈|呵|嘿|嘻|h|嗯|哦|啊|哇|呃|额){2,}$", re.IGNORECASE)
_WEAK_REFERENCE_TERMS = ("这个", "这个情况", "这种", "那种", "这样", "那样", "上述", "前面", "刚才", "上面")
_FOLLOW_UP_PATTERNS = (
    r"那如果.*",
    r"那要是.*",
    r"那.*呢",
    r"那么.*",
    r"这个.*呢",
    r"这种.*呢",
    r"然后呢",
    r"后来呢",
    r"第一次呢",
    r"第二次呢",
)


def _match_light_intent_rule(normalized: str) -> LightIntentRule | None:
    config = load_light_intent_config()
    for rule in config.intent_rules:
        if any(re.fullmatch(pattern, normalized) for pattern in rule.patterns):
            return rule
    return None


def _match_light_intent(query: str) -> LightIntentRule | None:
    normalized = normalize_query(query)
    if not normalized:
        return None

    rule = _match_light_intent_rule(normalized)
    if rule is not None:
        return rule

    clauses = split_light_intent_clauses(query)
    if len(clauses) <= 1:
        return None

    matched_rules: list[LightIntentRule] = []
    for clause in clauses:
        clause_rule = _match_light_intent_rule(clause)
        if clause_rule is None:
            return None
        matched_rules.append(clause_rule)

    return next((item for item in reversed(matched_rules) if item.name != "greeting"), matched_rules[-1])


def _match_faq_shortcut(query: str) -> FAQShortcutRule | None:
    config = load_faq_shortcut_config()
    normalized = normalize_query(query)
    if not normalized:
        return None

    for rule in config.rules:
        if any(re.fullmatch(pattern, normalized) for pattern in rule.patterns):
            return rule
    return None


def _is_policy_related_query(query: str) -> bool:
    config = load_light_intent_config()
    return any(keyword in query for keyword in config.policy_keywords)


def _matches_off_topic_pattern(query: str) -> bool:
    config = load_light_intent_config()
    normalized = normalize_query(query)
    if not normalized or _is_policy_related_query(query):
        return False
    return any(re.fullmatch(pattern, normalized) for pattern in config.off_topic_patterns)


def _is_meaningless_input(query: str) -> bool:
    stripped = query.strip()
    normalized = normalize_query(query)

    if not stripped or not normalized:
        return True
    if len(normalized) <= 2 and all(character in _MEANINGLESS_SINGLE_CHARACTERS for character in normalized):
        return True
    if len(normalized) <= 4 and len(set(normalized)) == 1 and normalized[0] in _MEANINGLESS_SINGLE_CHARACTERS:
        return True
    if len(normalized) <= 6 and _MEANINGLESS_REPEAT_RE.fullmatch(normalized):
        return True
    return False


def _looks_like_follow_up(query: str, *, has_history: bool) -> bool:
    if not has_history:
        return False

    normalized = normalize_query(query)
    if not normalized or len(normalized) > 18:
        return False
    return any(re.fullmatch(pattern, normalized) for pattern in _FOLLOW_UP_PATTERNS)


class IntentRouter:
    def __init__(
        self,
        *,
        local_classifier: IntentLocalClassifier | None = None,
        local_classifier_enabled: bool | None = None,
        llm_classifier: IntentLLMClassifier | None = None,
        llm_classifier_enabled: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.local_classifier_enabled = (
            settings.intent_local_classifier_enabled
            if local_classifier_enabled is None
            else local_classifier_enabled
        )
        self.llm_classifier_enabled = (
            settings.intent_llm_classifier_enabled
            if llm_classifier_enabled is None
            else llm_classifier_enabled
        )
        self.local_classifier = local_classifier or (IntentLocalClassifier() if self.local_classifier_enabled else None)
        self.llm_classifier = llm_classifier or (IntentLLMClassifier() if self.llm_classifier_enabled else None)

    @staticmethod
    def _should_use_secondary_classifier(query: str, *, has_history: bool) -> bool:
        normalized = normalize_query(query)
        if not normalized:
            return False
        has_weak_reference = any(term in normalized for term in _WEAK_REFERENCE_TERMS)
        if has_weak_reference and (has_history or len(normalized) <= 12):
            return True
        if _is_policy_related_query(query):
            return False
        if len(normalized) <= 6:
            return True
        if has_history and len(normalized) <= 12:
            return True
        return False

    @staticmethod
    def _decision_from_classifier_label(
        label: str,
        *,
        source: str,
        classifier_mode: str,
        classifier_score: float | None = None,
        classifier_margin: float | None = None,
    ) -> IntentDecision:
        if label == IntentType.FOLLOW_UP_QUERY.value:
            return IntentDecision(
                intent=IntentType.FOLLOW_UP_QUERY,
                source=source,
                short_circuit=False,
                classifier_source=source,
                classifier_mode=classifier_mode,
                classifier_score=classifier_score,
                classifier_margin=classifier_margin,
            )
        if label == IntentType.OFF_TOPIC.value:
            return IntentDecision(
                intent=IntentType.OFF_TOPIC,
                source=source,
                short_circuit=True,
                finish_reason="off_topic_short_circuit",
                classifier_source=source,
                classifier_mode=classifier_mode,
                classifier_score=classifier_score,
                classifier_margin=classifier_margin,
            )
        if label == IntentType.MEANINGLESS_INPUT.value:
            return IntentDecision(
                intent=IntentType.MEANINGLESS_INPUT,
                source=source,
                short_circuit=True,
                finish_reason="meaningless_input_short_circuit",
                classifier_source=source,
                classifier_mode=classifier_mode,
                classifier_score=classifier_score,
                classifier_margin=classifier_margin,
            )
        return IntentDecision(
            intent=IntentType.BUSINESS_QUERY,
            source=source,
            short_circuit=False,
            classifier_source=source,
            classifier_mode=classifier_mode,
            classifier_score=classifier_score,
            classifier_margin=classifier_margin,
        )

    def classify(
        self,
        query: str,
        *,
        has_history: bool = False,
        history_messages: list[dict[str, str]] | None = None,
    ) -> IntentDecision:
        if _is_meaningless_input(query):
            return IntentDecision(
                intent=IntentType.MEANINGLESS_INPUT,
                source="heuristic",
                short_circuit=True,
                finish_reason="meaningless_input_short_circuit",
                classifier_source="heuristic",
                classifier_mode="heuristic",
            )

        light_intent_rule = _match_light_intent(query)
        if light_intent_rule is not None:
            return IntentDecision(
                intent=IntentType.LIGHT_INTENT,
                source="light_intent_rule",
                short_circuit=True,
                finish_reason=light_intent_rule.finish_reason,
                light_intent_rule=light_intent_rule,
                classifier_source="light_intent_rule",
                classifier_mode="rule",
            )

        faq_shortcut_rule = _match_faq_shortcut(query)
        if faq_shortcut_rule is not None:
            return IntentDecision(
                intent=IntentType.FAQ_SHORTCUT,
                source="faq_shortcut_rule",
                short_circuit=True,
                finish_reason=faq_shortcut_rule.finish_reason,
                faq_shortcut_rule=faq_shortcut_rule,
                classifier_source="faq_shortcut_rule",
                classifier_mode="rule",
            )

        if _matches_off_topic_pattern(query):
            return IntentDecision(
                intent=IntentType.OFF_TOPIC,
                source="off_topic_rule",
                short_circuit=True,
                finish_reason="off_topic_short_circuit",
                classifier_source="off_topic_rule",
                classifier_mode="rule",
            )

        if _looks_like_follow_up(query, has_history=has_history):
            return IntentDecision(
                intent=IntentType.FOLLOW_UP_QUERY,
                source="follow_up_heuristic",
                short_circuit=False,
                classifier_source="follow_up_heuristic",
                classifier_mode="heuristic",
            )

        if (
            self.local_classifier_enabled
            and self.local_classifier
            and self._should_use_secondary_classifier(query, has_history=has_history)
        ):
            try:
                if hasattr(self.local_classifier, "classify_with_debug"):
                    local_result = self.local_classifier.classify_with_debug(query, history_messages=history_messages)
                    label = local_result.label if local_result is not None else None
                else:
                    local_result = None
                    label = self.local_classifier.classify(
                        query,
                        history_messages=history_messages,
                    )
            except Exception:
                local_result = None
                label = None
            if label:
                return self._decision_from_classifier_label(
                    label,
                    source="intent_local_classifier",
                    classifier_mode=local_result.mode if local_result is not None else "local_classifier",
                    classifier_score=local_result.score if local_result is not None else None,
                    classifier_margin=local_result.margin if local_result is not None else None,
                )

        if (
            self.llm_classifier_enabled
            and self.llm_classifier
            and self._should_use_secondary_classifier(query, has_history=has_history)
        ):
            try:
                label = self.llm_classifier.classify(query, history_messages=history_messages)
            except Exception:
                label = None
            if label:
                return self._decision_from_classifier_label(
                    label,
                    source="intent_llm_classifier",
                    classifier_mode="llm",
                )

        return IntentDecision(
            intent=IntentType.BUSINESS_QUERY,
            source="default",
            short_circuit=False,
            classifier_source="default",
            classifier_mode="default",
        )
