from app.services.intent_router import IntentRouter, IntentType


class StubIntentClassifier:
    def __init__(
        self,
        label: str,
        *,
        mode: str = "trained",
        score: float = 0.91,
        margin: float = 0.34,
    ) -> None:
        self.label = label
        self.mode = mode
        self.score = score
        self.margin = margin
        self.calls: list[tuple[str, list[dict[str, str]] | None]] = []

    def classify(self, query: str, *, history_messages: list[dict[str, str]] | None = None) -> str | None:
        self.calls.append((query, history_messages))
        return self.label

    def classify_with_debug(self, query: str, *, history_messages: list[dict[str, str]] | None = None):
        from app.services.intent_local_classifier import IntentLocalClassificationResult

        self.calls.append((query, history_messages))
        if not self.label:
            return None
        return IntentLocalClassificationResult(
            label=self.label,
            mode=self.mode,
            score=self.score,
            margin=self.margin,
        )


def test_intent_router_classifies_light_intent() -> None:
    router = IntentRouter()

    decision = router.classify("你好呀")

    assert decision.intent == IntentType.LIGHT_INTENT
    assert decision.short_circuit is True
    assert decision.light_intent_rule is not None
    assert decision.light_intent_rule.name == "greeting"


def test_intent_router_classifies_faq_shortcut() -> None:
    router = IntentRouter()

    decision = router.classify("劳动合同法是什么")

    assert decision.intent == IntentType.FAQ_SHORTCUT
    assert decision.short_circuit is True
    assert decision.faq_shortcut_rule is not None
    assert decision.faq_shortcut_rule.name == "labor_contract_law_overview"


def test_intent_router_classifies_off_topic() -> None:
    router = IntentRouter()

    decision = router.classify("今天天气怎么样？")

    assert decision.intent == IntentType.OFF_TOPIC
    assert decision.short_circuit is True
    assert decision.finish_reason == "off_topic_short_circuit"


def test_intent_router_classifies_meaningless_input() -> None:
    router = IntentRouter()

    decision = router.classify("哈哈哈")

    assert decision.intent == IntentType.MEANINGLESS_INPUT
    assert decision.short_circuit is True
    assert decision.finish_reason == "meaningless_input_short_circuit"


def test_intent_router_classifies_interjection_as_meaningless_input() -> None:
    router = IntentRouter()

    decision = router.classify("哇")

    assert decision.intent == IntentType.MEANINGLESS_INPUT
    assert decision.short_circuit is True
    assert decision.finish_reason == "meaningless_input_short_circuit"


def test_intent_router_classifies_follow_up_when_history_exists() -> None:
    router = IntentRouter()

    decision = router.classify("那如果是第一次呢", has_history=True)

    assert decision.intent == IntentType.FOLLOW_UP_QUERY
    assert decision.short_circuit is False


def test_intent_router_classifies_business_query_by_default() -> None:
    router = IntentRouter()

    decision = router.classify("试用期不签劳动合同，这样的公司能去入职吗？")

    assert decision.intent == IntentType.BUSINESS_QUERY
    assert decision.short_circuit is False


def test_intent_router_uses_local_classifier_for_uncertain_short_query() -> None:
    classifier = StubIntentClassifier("follow_up_query")
    router = IntentRouter(
        local_classifier=classifier,
        local_classifier_enabled=True,
        llm_classifier_enabled=False,
    )

    decision = router.classify(
        "这个咋办",
        has_history=True,
        history_messages=[{"role": "user", "content": "试用期最长多久"}],
    )

    assert decision.intent == IntentType.FOLLOW_UP_QUERY
    assert decision.source == "intent_local_classifier"
    assert decision.classifier_source == "intent_local_classifier"
    assert decision.classifier_mode == "trained"
    assert decision.classifier_score == 0.91
    assert decision.classifier_margin == 0.34
    assert classifier.calls == [("这个咋办", [{"role": "user", "content": "试用期最长多久"}])]


def test_intent_router_local_classifier_can_short_circuit_as_off_topic() -> None:
    classifier = StubIntentClassifier("off_topic")
    router = IntentRouter(
        local_classifier=classifier,
        local_classifier_enabled=True,
        llm_classifier_enabled=False,
    )

    decision = router.classify("聊聊这个", has_history=False, history_messages=[])

    assert decision.intent == IntentType.OFF_TOPIC
    assert decision.short_circuit is True
    assert decision.source == "intent_local_classifier"


def test_intent_router_does_not_call_local_classifier_for_clear_long_business_query() -> None:
    classifier = StubIntentClassifier("off_topic")
    router = IntentRouter(
        local_classifier=classifier,
        local_classifier_enabled=True,
        llm_classifier_enabled=False,
    )

    decision = router.classify("试用期不签劳动合同，这样的公司能去入职吗？", has_history=False)

    assert decision.intent == IntentType.BUSINESS_QUERY
    assert decision.source == "default"
    assert classifier.calls == []


def test_intent_router_does_not_call_local_classifier_for_clear_short_policy_query() -> None:
    classifier = StubIntentClassifier("off_topic")
    router = IntentRouter(
        local_classifier=classifier,
        local_classifier_enabled=True,
        llm_classifier_enabled=False,
    )

    decision = router.classify("请假流程说明", has_history=False)

    assert decision.intent == IntentType.BUSINESS_QUERY
    assert decision.source == "default"
    assert classifier.calls == []


def test_intent_router_falls_back_to_llm_classifier_when_local_classifier_returns_none() -> None:
    local_classifier = StubIntentClassifier(label="")
    llm_classifier = StubIntentClassifier("off_topic")
    router = IntentRouter(
        local_classifier=local_classifier,
        local_classifier_enabled=True,
        llm_classifier=llm_classifier,
        llm_classifier_enabled=True,
    )

    def classify_none(query: str, *, history_messages: list[dict[str, str]] | None = None) -> str | None:
        local_classifier.calls.append((query, history_messages))
        return None

    local_classifier.classify = classify_none  # type: ignore[method-assign]
    local_classifier.classify_with_debug = classify_none  # type: ignore[method-assign]

    decision = router.classify("聊聊这个", has_history=False, history_messages=[])

    assert decision.intent == IntentType.OFF_TOPIC
    assert decision.source == "intent_llm_classifier"
    assert local_classifier.calls
    assert llm_classifier.calls == [("聊聊这个", [])]
