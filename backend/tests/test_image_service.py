import json
from pathlib import Path

from app.document_processing.image import ImageOCRService
from app.document_processing.pdf.ocr import OCRResult


def test_image_ocr_service_exports_structured_payload(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake-image")
    output_dir = tmp_path / "artifacts"

    monkeypatch.setattr(
        "app.document_processing.image.service.run_ocr_on_image_result",
        lambda _: OCRResult(
            text="课堂纪律\n1旷课\n扣20分/次",
            avg_confidence=0.88,
            min_confidence=0.76,
            quality="normal",
        ),
    )
    monkeypatch.setattr("app.document_processing.image.service.looks_like_ocr_table", lambda _: True)
    monkeypatch.setattr(
        "app.document_processing.image.service.render_image_ocr_table_text",
        lambda _: "[条目 1]\n内容: 旷课\n结果: 扣20分/次",
    )

    service = ImageOCRService()
    artifacts = service.preprocess(image_path, output_dir)
    payload = json.loads(artifacts.structured_json_path.read_text(encoding="utf-8"))

    assert payload["file_name"] == "sample.png"
    assert payload["pages"][0]["images"][0]["translated_source_type"] == "image_ocr_table"
    assert payload["pages"][0]["images"][0]["translated_text"] == "[条目 1]\n内容: 旷课\n结果: 扣20分/次"
    assert payload["pages"][0]["images"][0]["ocr_quality"] == "normal"
    assert payload["pages"][0]["images"][0]["ocr_avg_confidence"] == 0.88
