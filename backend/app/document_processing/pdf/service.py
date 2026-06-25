"""PDF 结构化预处理服务。

PDF 会被转成 structured.json 和 readable.txt。structured.json 保留正文、表格、
图片 OCR、页码和块类型，供 RAG 入库使用；readable.txt 主要用于人工排查。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exporters import export_readable_txt
from .extractors import (
    extract_full_page_ocr_block,
    extract_image_blocks,
    extract_page_text_blocks_excluding_tables,
    extract_tables_from_plumber_page,
)
from .postprocess import merge_cross_page_tables
from .utils import (
    build_pdf_file_id,
    ensure_dir,
    mark_toc_pages_for_skip,
    merge_cross_page_text_blocks,
    remove_repeated_page_noise_blocks,
)


@dataclass(slots=True)
class PDFStructuredArtifacts:
    """PDF 预处理产物路径和解析 payload。"""

    structured_json_path: Path
    readable_txt_path: Path
    image_output_dir: Path
    payload: dict[str, Any]
    used_full_page_ocr_fallback: bool


class PDFStructuringService:
    """把 PDF 解析成 RAG pipeline 可消费的结构化 JSON。"""

    def __init__(self, enable_ocr: bool = True) -> None:
        self.enable_ocr = enable_ocr

    def preprocess(self, pdf_path: Path, output_dir: Path) -> PDFStructuredArtifacts:
        """生成或复用 PDF 结构化产物。"""
        ensure_dir(output_dir)
        image_output_dir = output_dir / "images"
        ensure_dir(image_output_dir)

        structured_json_path = output_dir / "structured.json"
        readable_txt_path = output_dir / "readable.txt"

        if self._artifacts_are_fresh(pdf_path, structured_json_path, readable_txt_path):
            # 预处理结果比源文件新时直接复用，避免重复 OCR 和表格抽取。
            payload = json.loads(structured_json_path.read_text(encoding="utf-8"))
            self._ensure_extractable_payload(payload, pdf_path)
            return PDFStructuredArtifacts(
                structured_json_path=structured_json_path,
                readable_txt_path=readable_txt_path,
                image_output_dir=image_output_dir,
                payload=payload,
                used_full_page_ocr_fallback=self._payload_uses_full_page_ocr_fallback(payload),
            )

        payload = self._extract_pdf_structured(
            pdf_path=pdf_path,
            output_json_path=structured_json_path,
            image_output_dir=image_output_dir,
            enable_ocr=self.enable_ocr,
        )
        self._ensure_extractable_payload(payload, pdf_path)
        export_readable_txt(payload, readable_txt_path)
        return PDFStructuredArtifacts(
            structured_json_path=structured_json_path,
            readable_txt_path=readable_txt_path,
            image_output_dir=image_output_dir,
            payload=payload,
            used_full_page_ocr_fallback=self._payload_uses_full_page_ocr_fallback(payload),
        )

    @staticmethod
    def _artifacts_are_fresh(pdf_path: Path, structured_json_path: Path, readable_txt_path: Path) -> bool:
        if not structured_json_path.exists() or not readable_txt_path.exists():
            return False
        source_mtime = pdf_path.stat().st_mtime
        return structured_json_path.stat().st_mtime >= source_mtime and readable_txt_path.stat().st_mtime >= source_mtime

    @staticmethod
    def _ensure_extractable_payload(payload: dict[str, Any], pdf_path: Path) -> None:
        """确保至少有正文、表格或 OCR 内容，否则入库没有意义。"""
        pages = payload.get("pages")
        if not isinstance(pages, list) or not pages:
            raise RuntimeError(f"PDF 解析失败：未生成页面结构数据（{pdf_path.name}）")

        has_extractable_content = any(
            isinstance(page, dict) and PDFStructuringService._page_has_extractable_content(page)
            for page in pages
        )

        if not has_extractable_content:
            raise RuntimeError(
                f"PDF 解析失败：未提取到可用文本、表格或 OCR 内容（{pdf_path.name}），当前暂不支持该类 PDF。"
            )

    @staticmethod
    def _page_has_extractable_content(page: dict[str, Any]) -> bool:
        page_text = str(page.get("text") or "").strip()
        tables = page.get("tables")
        images = page.get("images")
        if page_text:
            return True
        if isinstance(tables, list) and any(isinstance(table, dict) and str(table.get("text") or "").strip() for table in tables):
            return True
        if isinstance(images, list) and any(
            isinstance(image, dict)
            and (str(image.get("translated_text") or "").strip() or str(image.get("ocr_text") or "").strip())
            for image in images
        ):
            return True
        return False

    @staticmethod
    def _payload_uses_full_page_ocr_fallback(payload: dict[str, Any]) -> bool:
        """检查是否触发整页 OCR 降级，用于前端提示用户处理会更慢。"""
        pages = payload.get("pages")
        if not isinstance(pages, list):
            return False

        for page in pages:
            if not isinstance(page, dict):
                continue
            images = page.get("images")
            if not isinstance(images, list):
                continue
            if any(
                isinstance(image, dict)
                and str(image.get("index") or "") == "full_page_ocr"
                for image in images
            ):
                return True
        return False

    def _extract_pdf_structured(
        self,
        pdf_path: Path,
        output_json_path: Path,
        image_output_dir: Path,
        enable_ocr: bool,
    ) -> dict[str, Any]:
        """执行实际 PDF 抽取：正文、表格、图片 OCR 和整页 OCR 降级。"""
        try:
            import fitz
            import pdfplumber
        except Exception as exc:
            raise RuntimeError("PDF preprocessing dependencies are missing: PyMuPDF/pdfplumber") from exc

        pdf_name = pdf_path.name
        pdf_file_id = build_pdf_file_id(pdf_path)
        fitz_doc = fitz.open(pdf_path)

        try:
            result: dict[str, Any] = {
                "file_name": pdf_name,
                "file_path": str(pdf_path.resolve()),
                "pdf_file_id": pdf_file_id,
                "total_pages": len(fitz_doc),
                "pages": [],
            }

            with pdfplumber.open(pdf_path) as plumber_doc:
                if len(plumber_doc.pages) != len(fitz_doc):
                    raise ValueError("pdfplumber 与 PyMuPDF 读取到的页数不一致")

                for page_index in range(len(fitz_doc)):
                    # PyMuPDF 负责页面文本/图片坐标，pdfplumber 负责表格检测。
                    page_number = page_index + 1
                    fitz_page = fitz_doc.load_page(page_index)
                    plumber_page = plumber_doc.pages[page_index]

                    tables = extract_tables_from_plumber_page(plumber_page)
                    table_bboxes = [tuple(table["bbox"]) for table in tables if table.get("bbox")]
                    page_text, text_blocks = extract_page_text_blocks_excluding_tables(fitz_page, fitz, table_bboxes)
                    images = extract_image_blocks(
                        page=fitz_page,
                        fitz_module=fitz,
                        page_number=page_number,
                        image_output_dir=image_output_dir,
                        pdf_file_id=pdf_file_id,
                        enable_ocr=enable_ocr,
                    )

                    page_payload = {
                        "page_number": page_number,
                        "page_height": float(fitz_page.rect.height),
                        "text": page_text,
                        "text_blocks": text_blocks,
                        "images": images,
                        "tables": tables,
                    }
                    if enable_ocr and not self._page_has_extractable_content(page_payload):
                        # 扫描件或矢量字形 PDF 可能没有可抽取文本，最后降级为整页 OCR。
                        fallback_image = extract_full_page_ocr_block(
                            page=fitz_page,
                            fitz_module=fitz,
                            page_number=page_number,
                            image_output_dir=image_output_dir,
                            pdf_file_id=pdf_file_id,
                            enable_ocr=enable_ocr,
                        )
                        if fallback_image is not None:
                            page_payload["images"].append(fallback_image)

                    result["pages"].append(page_payload)

            result = mark_toc_pages_for_skip(result)
            result = remove_repeated_page_noise_blocks(result)
            result = merge_cross_page_tables(result)
            result = merge_cross_page_text_blocks(result)
            output_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        finally:
            fitz_doc.close()
