import json
from pathlib import Path

from app.document_processing.pdf.ocr import build_ocr_result, normalize_image_ocr_text
from app.document_processing.pdf.image_ocr_postprocess import looks_like_ocr_table, render_image_ocr_table_text
from app.rag.document_processor import DocumentProcessor
from app.rag.pipeline import RAGPipeline
from app.rag.structured_document_processor import StructuredDocumentProcessor


class StubVectorStore:
    def __init__(self) -> None:
        self.added_documents: list[dict[str, object]] = []

    def add_documents(self, chunks_data: list[dict[str, object]]) -> None:
        self.added_documents = chunks_data


def test_structured_document_processor_converts_text_table_and_image_blocks() -> None:
    processor = StructuredDocumentProcessor(DocumentProcessor(child_chunk_size=200))
    structured_data = {
        "pdf_file_id": "demo_pdf",
        "pages": [
            {
                "page_number": 1,
                "text": "这是第一页正文。说明请假流程。",
                "tables": [
                    {
                        "index": 0,
                        "text": "事项 | 要求\n请假 | 需审批",
                        "quality": "normal",
                    }
                ],
                "images": [
                    {
                        "index": 0,
                        "ocr_text": "宿舍检查时间：22:30",
                        "translated_text": "宿舍检查时间：22:30",
                        "translated_source_type": "image_ocr",
                    }
                ],
            }
        ],
    }

    chunks = processor.process(structured_data)

    assert len(chunks) == 3
    assert any("[正文转译][第 1 页]" in str(chunk["parent_text"]) for chunk in chunks)
    assert any("[表格转译][第 1 页][表格 1]" in str(chunk["parent_text"]) for chunk in chunks)
    assert any("[图片OCR转译][第 1 页][图片 1]" in str(chunk["parent_text"]) for chunk in chunks)
    assert any(chunk.get("source_type") == "text" for chunk in chunks)
    assert any(chunk.get("source_type") == "table" for chunk in chunks)
    assert any(chunk.get("source_type") == "image_ocr" for chunk in chunks)
    assert all(chunk.get("document_id") == "demo_pdf" for chunk in chunks)


def test_structured_document_processor_uses_translated_image_text_when_present() -> None:
    processor = StructuredDocumentProcessor(DocumentProcessor(child_chunk_size=200))
    structured_data = {
        "pdf_file_id": "demo_pdf",
        "pages": [
            {
                "page_number": 2,
                "text": "",
                "tables": [],
                "images": [
                    {
                        "index": 0,
                        "ocr_text": "原始 OCR 文本",
                        "translated_text": "[条目 11]\n内容: 迟到、早退（20分钟以内）\n结果: 扣3分/次",
                        "translated_source_type": "image_ocr_table",
                    }
                ],
            }
        ],
    }

    chunks = processor.process(structured_data)

    assert len(chunks) == 1
    assert chunks[0]["source_type"] == "image_ocr_table"
    assert "[条目 11]" in str(chunks[0]["parent_text"])


def test_ingest_file_uses_structured_processor_for_json(tmp_path: Path) -> None:
    structured_path = tmp_path / "structured.json"
    structured_payload = {
        "pdf_file_id": "demo_pdf",
        "pages": [
            {
                "page_number": 2,
                "text": "第二页正文。",
                "tables": [],
                "images": [],
            }
        ],
    }
    structured_path.write_text(json.dumps(structured_payload, ensure_ascii=False), encoding="utf-8")

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.processor = DocumentProcessor(child_chunk_size=200)
    pipeline.structured_processor = StructuredDocumentProcessor(pipeline.processor)
    pipeline.vector_store = StubVectorStore()

    chunk_count = pipeline.ingest_file(structured_path, document_id="doc_structured")

    assert chunk_count == 1
    assert len(pipeline.vector_store.added_documents) == 1
    assert "[正文转译][第 2 页]" in str(pipeline.vector_store.added_documents[0]["parent_text"])
    assert pipeline.vector_store.added_documents[0]["source_type"] == "text"
    assert pipeline.vector_store.added_documents[0]["document_id"] == "doc_structured"


def test_structured_document_processor_uses_chinese_clause_fallback_for_ocr_text() -> None:
    processor = StructuredDocumentProcessor(DocumentProcessor(child_chunk_size=18, parent_chunk_size=200))
    structured_data = {
        "pdf_file_id": "demo_pdf",
        "pages": [
            {
                "page_number": 3,
                "text": "",
                "tables": [],
                "images": [
                    {
                        "index": 0,
                        "ocr_text": "请假材料要求：病假提交就诊证明、事假提前申请、返校后补交审批单、未经批准不得离校",
                        "translated_text": "请假材料要求：病假提交就诊证明、事假提前申请、返校后补交审批单、未经批准不得离校",
                        "translated_source_type": "image_ocr",
                    }
                ],
            }
        ],
    }

    chunks = processor.process(structured_data)

    assert len(chunks) >= 3
    assert all(chunk["source_type"] == "image_ocr" for chunk in chunks)
    assert all(len(str(chunk["child_text"])) <= 18 for chunk in chunks)


def test_structured_document_processor_preserves_ocr_quality_metadata() -> None:
    processor = StructuredDocumentProcessor(DocumentProcessor(child_chunk_size=200))
    structured_data = {
        "pdf_file_id": "demo_pdf",
        "pages": [
            {
                "page_number": 4,
                "text": "",
                "tables": [],
                "images": [
                    {
                        "index": 0,
                        "ocr_text": "扣10分/次",
                        "translated_text": "扣10分/次",
                        "translated_source_type": "image_ocr",
                        "ocr_quality": "low",
                        "ocr_avg_confidence": 0.61,
                        "ocr_min_confidence": 0.42,
                        "ocr_low_confidence_line_count": 2,
                    }
                ],
            }
        ],
    }

    chunks = processor.process(structured_data)

    assert chunks[0]["ocr_quality"] == "low"
    assert chunks[0]["ocr_avg_confidence"] == 0.61
    assert chunks[0]["ocr_min_confidence"] == 0.42
    assert chunks[0]["ocr_low_confidence_line_count"] == 2


def test_normalize_image_ocr_text_merges_fragmented_lines() -> None:
    raw_ocr_text = (
        "请事假（病假无就诊证明的视为事假，毕业答辩等特殊情形单独报\n"
        "批，每名学员学习期间仅有一次特殊事项请假的机会。）\n"
        "扣5分/天\n"
        "如单独请上午、下午或者晚自习，一律按照请假 半天计算，进行相\n"
        "应的扣分。"
    )

    normalized = normalize_image_ocr_text(raw_ocr_text)

    assert "单独报" in normalized
    assert "批，每名学员学习期间仅有一次特殊事项请假的机会。" in normalized
    assert "扣5分/天" in normalized
    assert "一律按照请假半天计算，进行相" in normalized
    assert "应的扣分。" in normalized


def test_normalize_image_ocr_text_fixes_numeric_unit_ocr_confusions() -> None:
    raw_ocr_text = "迟到扣l0分/ 次\n第O条规定扣５分/天"

    normalized = normalize_image_ocr_text(raw_ocr_text)

    assert "扣10分/次" in normalized
    assert "第0条规定扣5分/天" in normalized


def test_build_ocr_result_marks_low_quality_by_confidence() -> None:
    result = build_ocr_result("迟到扣10分/次", [0.91, 0.42, 0.51])

    assert result.quality == "low"
    assert result.avg_confidence is not None
    assert result.min_confidence == 0.42
    assert result.low_confidence_line_count == 2
    assert "very_low_min_confidence" in result.quality_reasons


def test_image_ocr_table_postprocess_renders_records() -> None:
    raw_ocr_text = (
        "课堂纪律\n"
        "11\n"
        "迟到、早退（20分钟以内）\n"
        "扣3分/次\n"
        "12\n"
        "不服从班主任管理，拒绝缴纳手机。\n"
        "扣10分/次"
    )

    assert looks_like_ocr_table(raw_ocr_text) is True
    rendered = render_image_ocr_table_text(raw_ocr_text)

    assert "[条目 11]" in rendered
    assert "内容: 迟到、早退（20分钟以内）" in rendered
    assert "结果: 扣3分/次" in rendered
    assert "[条目 12]" in rendered
    assert "结果: 扣10分/次" in rendered
