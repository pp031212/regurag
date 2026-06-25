"""PDF 页面级抽取函数。

这里只做单页内容提取：正文块、图片块、整页 OCR 降级和表格。跨页合并在
postprocess/utils 中完成。
"""

from pathlib import Path
from typing import Any

from .image_ocr_postprocess import looks_like_ocr_table, render_image_ocr_table_text
from .ocr import run_ocr_on_image_result
from .utils import (
    clean_table_rows,
    is_header_or_footer_block,
    is_noise_text_block,
    normalize_paragraph_text,
    rect_intersects_bbox,
    table_rows_to_text,
)


def extract_page_text_excluding_tables(
    page: Any,
    fitz_module: Any,
    table_bboxes: list[tuple[float, float, float, float]],
) -> str:
    """提取页面正文，并排除表格区域、页眉页脚和页码噪声。"""
    page_text, _ = extract_page_text_blocks_excluding_tables(page, fitz_module, table_bboxes)
    return page_text


def extract_page_text_blocks_excluding_tables(
    page: Any,
    fitz_module: Any,
    table_bboxes: list[tuple[float, float, float, float]],
) -> tuple[str, list[dict[str, Any]]]:
    """提取正文文本和轻量版式信息，供跨页连续性判断使用。"""
    page_dict = page.get_text("dict")
    kept_texts: list[str] = []
    kept_blocks: list[dict[str, Any]] = []
    page_height = page.rect.height

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        bbox = block.get("bbox")
        if not bbox:
            continue

        block_rect = fitz_module.Rect(*bbox)
        if any(rect_intersects_bbox(block_rect, tbbox) for tbbox in table_bboxes):
            # 表格区域单独走 table extractor，避免正文和表格重复入库。
            continue
        if is_header_or_footer_block(block_rect, page_height, top_ratio=0.05, bottom_ratio=0.06):
            continue

        block_text_parts: list[str] = []
        font_sizes: list[float] = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text.strip():
                    block_text_parts.append(text)
                size = span.get("size")
                if isinstance(size, (int, float)):
                    font_sizes.append(float(size))

        block_text = normalize_paragraph_text("".join(block_text_parts).strip())
        if not block_text or is_noise_text_block(block_text):
            continue

        kept_texts.append(block_text)
        x0, y0, x1, y1 = bbox
        kept_blocks.append(
            {
                "text": block_text,
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "page_height": float(page_height),
                "avg_font_size": sum(font_sizes) / len(font_sizes) if font_sizes else None,
            }
        )

    return normalize_paragraph_text("\n".join(kept_texts).strip()), kept_blocks


def extract_image_blocks(
    page: Any,
    fitz_module: Any,
    page_number: int,
    image_output_dir: Path,
    pdf_file_id: str,
    enable_ocr: bool = True,
) -> list[dict[str, Any]]:
    """裁剪页面内图片区域，按需 OCR，并识别是否像表格截图。"""
    image_items: list[dict[str, Any]] = []
    page_dict = page.get_text("dict")

    image_index = 0
    for block in page_dict.get("blocks", []):
        if block.get("type") != 1:
            continue

        bbox = block.get("bbox")
        if not bbox:
            continue

        x0, y0, x1, y1 = bbox
        rect = fitz_module.Rect(x0, y0, x1, y1)
        pix = page.get_pixmap(clip=rect, dpi=150)

        image_filename = f"{pdf_file_id}_page_{page_number}_img_{image_index}.png"
        image_path = image_output_dir / image_filename
        pix.save(image_path)

        ocr_result = run_ocr_on_image_result(str(image_path)) if enable_ocr else None
        ocr_text = ocr_result.text if ocr_result is not None else ""
        translated_text = ""
        translated_source_type = "image_ocr"
        if ocr_text:
            if looks_like_ocr_table(ocr_text):
                translated_source_type = "image_ocr_table"
                translated_text = render_image_ocr_table_text(ocr_text)
            else:
                translated_text = ocr_text
        image_note = (
            "检测到图片区域，并识别出其中的文字"
            if ocr_text
            else "检测到图片区域，但未识别到文字，可能是纯图片或无文字图像"
        )

        image_items.append(
            {
                "index": image_index,
                "bbox": [x0, y0, x1, y1],
                "image_path": str(image_path),
                "ocr_text": ocr_text,
                "translated_text": translated_text,
                "translated_source_type": translated_source_type,
                "image_note": image_note,
                "ocr_quality": ocr_result.quality if ocr_result is not None else "disabled",
                "ocr_avg_confidence": ocr_result.avg_confidence if ocr_result is not None else None,
                "ocr_min_confidence": ocr_result.min_confidence if ocr_result is not None else None,
                "ocr_low_confidence_line_count": (
                    ocr_result.low_confidence_line_count if ocr_result is not None else 0
                ),
                "ocr_quality_reasons": list(ocr_result.quality_reasons) if ocr_result is not None else [],
            }
        )
        image_index += 1

    return image_items


def extract_full_page_ocr_block(
    page: Any,
    fitz_module: Any,
    page_number: int,
    image_output_dir: Path,
    pdf_file_id: str,
    enable_ocr: bool = True,
) -> dict[str, Any] | None:
    """整页没有可用内容时的降级 OCR。"""
    image_filename = f"{pdf_file_id}_page_{page_number}_full_page_ocr.png"
    image_path = image_output_dir / image_filename
    pix = page.get_pixmap(dpi=200)
    pix.save(image_path)

    ocr_result = run_ocr_on_image_result(str(image_path)) if enable_ocr else None
    ocr_text = ocr_result.text if ocr_result is not None else ""
    if not ocr_text:
        return None

    translated_text = ocr_text
    translated_source_type = "image_ocr"
    if looks_like_ocr_table(ocr_text):
        translated_source_type = "image_ocr_table"
        translated_text = render_image_ocr_table_text(ocr_text)

    page_rect = page.rect
    return {
        "index": "full_page_ocr",
        "bbox": [page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1],
        "image_path": str(image_path),
        "ocr_text": ocr_text,
        "translated_text": translated_text,
        "translated_source_type": translated_source_type,
        "image_note": "整页未提取到可用文本，已自动降级为整页 OCR",
        "ocr_quality": ocr_result.quality if ocr_result is not None else "disabled",
        "ocr_avg_confidence": ocr_result.avg_confidence if ocr_result is not None else None,
        "ocr_min_confidence": ocr_result.min_confidence if ocr_result is not None else None,
        "ocr_low_confidence_line_count": ocr_result.low_confidence_line_count if ocr_result is not None else 0,
        "ocr_quality_reasons": list(ocr_result.quality_reasons) if ocr_result is not None else [],
    }


def extract_tables_from_plumber_page(plumber_page: Any) -> list[dict[str, Any]]:
    """用 pdfplumber 提取表格，并输出统一 rows/text/bbox 结构。"""
    tables_output: list[dict[str, Any]] = []

    try:
        found_tables = plumber_page.find_tables()

        for index, table in enumerate(found_tables):
            raw_table = table.extract()
            cleaned_rows = clean_table_rows(raw_table)
            table_text = table_rows_to_text(cleaned_rows)
            tables_output.append(
                {
                    "index": index,
                    "bbox": list(table.bbox) if table.bbox else None,
                    "rows": cleaned_rows,
                    "text": table_text,
                    "quality": "low" if len(cleaned_rows) <= 1 else "normal",
                }
            )

        if not tables_output:
            # find_tables 失败时退回 extract_tables，牺牲 bbox 但尽量保留表格文本。
            raw_tables = plumber_page.extract_tables()
            for index, raw_table in enumerate(raw_tables):
                cleaned_rows = clean_table_rows(raw_table)
                table_text = table_rows_to_text(cleaned_rows)
                tables_output.append(
                    {
                        "index": index,
                        "bbox": None,
                        "rows": cleaned_rows,
                        "text": table_text,
                        "quality": "low" if len(cleaned_rows) <= 1 else "normal",
                    }
                )
    except Exception as exc:
        tables_output.append(
            {
                "index": -1,
                "bbox": None,
                "rows": [],
                "text": "",
                "quality": "error",
                "error": f"表格提取失败: {exc}",
            }
        )

    return tables_output
