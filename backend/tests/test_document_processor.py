from app.rag.document_processor import DocumentProcessor
from app.rag.document_split_config import load_document_split_config


def test_document_split_config_loads_from_json_file() -> None:
    config = load_document_split_config()

    assert config.default_strategy == "generic_paragraph"
    assert any(strategy.name == "law_article" for strategy in config.strategies)
    assert any(strategy.name == "regulation_rule" for strategy in config.strategies)
    assert any(profile.name == "zh_policy_sentence" for profile in config.sentence_profiles)


def test_document_processor_uses_law_article_strategy_for_legal_text() -> None:
    processor = DocumentProcessor(child_chunk_size=500, parent_chunk_size=500)
    text = (
        "第一条 为了保护劳动者合法权益，制定本法。\n"
        "第二条 中华人民共和国境内的企业与劳动者建立劳动关系，适用本法。\n"
        "第三十八条 用人单位未依法为劳动者缴纳社会保险费的，劳动者可以解除劳动合同。"
    )

    chunks = processor.process(text)

    assert len(chunks) == 3
    assert chunks[0]["parent_text"].startswith("第一条")
    assert chunks[1]["parent_text"].startswith("第二条")
    assert chunks[2]["parent_text"].startswith("第三十八条")


def test_document_processor_does_not_split_inline_article_references() -> None:
    processor = DocumentProcessor(child_chunk_size=500, parent_chunk_size=500)
    text = (
        "第三十八条 用人单位有下列情形之一的，劳动者可以解除劳动合同：\n"
        "（一）未及时足额支付劳动报酬的；\n"
        "（二）因本法第二十六条第一款规定的情形致使劳动合同无效的；\n"
        "（三）未依法为劳动者缴纳社会保险费的。\n"
        "第三十九条 劳动者有下列情形之一的，用人单位可以解除劳动合同。"
    )

    chunks = processor.process(text)

    assert len(chunks) == 2
    assert chunks[0]["parent_text"].startswith("第三十八条")
    assert "第二十六条第一款规定" in chunks[0]["parent_text"]
    assert "未依法为劳动者缴纳社会保险费" in chunks[0]["parent_text"]
    assert chunks[1]["parent_text"].startswith("第三十九条")


def test_document_processor_uses_regulation_rule_strategy_for_rule_text() -> None:
    processor = DocumentProcessor(child_chunk_size=500, parent_chunk_size=500)
    text = (
        "一、课堂纪律 迟到扣3分。\n"
        "二、宿舍纪律 熄灯后喧哗扣5分。\n"
        "扣分处理标准如下：累计扣除40分及以上，无条件退学。"
    )

    chunks = processor.process(text)

    assert len(chunks) == 3
    assert chunks[0]["parent_text"].startswith("一、")
    assert chunks[1]["parent_text"].startswith("二、")
    assert chunks[2]["parent_text"].startswith("扣分处理标准如下")


def test_document_processor_falls_back_to_generic_paragraph_strategy() -> None:
    processor = DocumentProcessor(child_chunk_size=500, parent_chunk_size=500)
    text = "这是第一段。\n\n这是第二段。\n\n这是第三段。"

    chunks = processor.process(text)

    assert len(chunks) == 3
    assert chunks[0]["parent_text"] == "这是第一段。"
    assert chunks[1]["parent_text"] == "这是第二段。"
    assert chunks[2]["parent_text"] == "这是第三段。"


def test_document_processor_uses_markdown_headings_for_legal_articles() -> None:
    processor = DocumentProcessor(child_chunk_size=500, parent_chunk_size=500)
    text = (
        "# 中华人民共和国劳动合同法\n\n"
        "## 目录\n"
        "* 第一章 总则\n"
        "* 第四章 劳动合同的解除和终止\n\n"
        "---\n\n"
        "## 第四章 劳动合同的解除和终止\n\n"
        "**第三十八条** 用人单位未依法为劳动者缴纳社会保险费的，劳动者可以解除劳动合同。\n\n"
        "**第三十九条** 劳动者严重违反用人单位的规章制度的，用人单位可以解除劳动合同。\n"
    )

    chunks = processor.process(text, source_format="markdown")

    assert len(chunks) == 2
    assert chunks[0]["parent_text"].startswith("中华人民共和国劳动合同法第四章 劳动合同的解除和终止第三十八条")
    assert "目录" not in chunks[0]["parent_text"]
    assert chunks[1]["parent_text"].startswith("中华人民共和国劳动合同法第四章 劳动合同的解除和终止第三十九条")


def test_document_processor_preserves_markdown_lists_inside_articles() -> None:
    processor = DocumentProcessor(child_chunk_size=500, parent_chunk_size=500)
    text = (
        "# 中华人民共和国劳动合同法\n\n"
        "## 第四章 劳动合同的解除和终止\n\n"
        "**第三十八条** 劳动者可以解除劳动合同：\n"
        "* （一）未及时足额支付劳动报酬的；\n"
        "* （二）未依法为劳动者缴纳社会保险费的。\n"
    )

    chunks = processor.process(text, source_format="markdown")

    assert len(chunks) == 1
    assert "第三十八条 劳动者可以解除劳动合同" in chunks[0]["parent_text"]
    assert "（一）未及时足额支付劳动报酬的；" in chunks[0]["parent_text"]
    assert "（二）未依法为劳动者缴纳社会保险费的。" in chunks[0]["parent_text"]


def test_document_processor_uses_chinese_clause_fallback_for_long_plain_text() -> None:
    processor = DocumentProcessor(child_chunk_size=18, parent_chunk_size=200)
    text = "请假材料要求：病假提交就诊证明、事假提前申请、返校后补交审批单、未经批准不得离校"

    chunks = processor.process(text)

    assert len(chunks) >= 3
    assert all(len(str(chunk["child_text"])) <= 18 for chunk in chunks)
    assert any("病假提交就诊证明" in str(chunk["child_text"]) for chunk in chunks)
    assert any("未经批准不得离校" in str(chunk["child_text"]) for chunk in chunks)


def test_document_processor_uses_weak_boundaries_before_hard_split_for_long_clause() -> None:
    processor = DocumentProcessor(child_chunk_size=24, parent_chunk_size=200, profile="rules_cn")
    text = "迟到处理规则：（一）迟到20分钟以内扣3分/次，（二）迟到超过20分钟扣5分/次，（三）旷课按规定扣20分/次"

    chunks = processor.process(text)
    child_texts = [str(chunk["child_text"]) for chunk in chunks]

    assert len(chunks) >= 3
    assert all(len(child_text) <= 24 for child_text in child_texts)
    assert any("迟到处理规则：" in child_text for child_text in child_texts)
    assert any("（一）迟到20分钟以内" in child_text for child_text in child_texts)
    assert any(child_text.startswith("（二）迟到超过20分钟") for child_text in child_texts)
    assert any(child_text.startswith("（三）旷课按规定") for child_text in child_texts)


def test_document_processor_uses_policy_fallback_for_table_text() -> None:
    processor = DocumentProcessor(child_chunk_size=18, parent_chunk_size=200, profile="rules_cn")
    chunks = processor.build_chunks_from_parent(
        parent_text="行为 | 处理方式：迟到20分钟以内扣3分/次，旷课按规定扣20分/次",
        parent_id="table_parent",
        source_format="table",
    )

    child_texts = [str(chunk["child_text"]) for chunk in chunks]

    assert any(child_text.endswith("处理方式：") for child_text in child_texts)
    assert any("迟到20分钟以内" in child_text for child_text in child_texts)
    assert any("旷课按规定" in child_text for child_text in child_texts)
