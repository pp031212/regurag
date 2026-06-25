"""独立图片 OCR 预处理服务。

图片上传会被包装成和 PDF 类似的 structured.json，这样后续 RAG 入库可以复用
结构化文档处理逻辑。
"""

import json
from dataclasses import dataclass
from pathlib import Path

from ..pdf.image_ocr_postprocess import looks_like_ocr_table, render_image_ocr_table_text
from ..pdf.ocr import run_ocr_on_image_result


@dataclass(slots=True)
class ImageOCRArtifacts:
    """图片 OCR 预处理产物路径。"""

    structured_json_path: Path


class ImageOCRService:
    """把单张图片解析为一页结构化文档。"""

    def preprocess(self, image_path: Path, output_dir: Path) -> ImageOCRArtifacts:
        """执行 OCR 并写出 structured.json。"""
        output_dir.mkdir(parents=True, exist_ok=True)
        structured_json_path = output_dir / "structured.json"

        ocr_result = run_ocr_on_image_result(str(image_path))
        ocr_text = ocr_result.text
        source_type = "image_ocr"
        translated_text = ocr_text
        if ocr_text and looks_like_ocr_table(ocr_text):
            source_type = "image_ocr_table"
            translated_text = render_image_ocr_table_text(ocr_text)

        # 保持 pages/images 结构，避免下游为图片类型写一套特殊入库逻辑。
        payload = {
            "file_name": image_path.name,
            "file_path": str(image_path.resolve()),
            "pdf_file_id": image_path.stem,
            "total_pages": 1,
            "pages": [
                {
                    "page_number": 1,
                    "text": "",
                    "tables": [],
                    "images": [
                        {
                            "index": 0,
                            "image_path": str(image_path.resolve()),
                            "ocr_text": ocr_text,
                            "translated_text": translated_text,
                            "translated_source_type": source_type,
                            "ocr_quality": ocr_result.quality,
                            "ocr_avg_confidence": ocr_result.avg_confidence,
                            "ocr_min_confidence": ocr_result.min_confidence,
                            "ocr_low_confidence_line_count": ocr_result.low_confidence_line_count,
                            "ocr_quality_reasons": list(ocr_result.quality_reasons),
                            "image_note": "standalone image upload",
                            "bbox": [],
                        }
                    ],
                }
            ],
        }
        structured_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return ImageOCRArtifacts(structured_json_path=structured_json_path)
