"""PDF 结构化结果的人类可读导出。

structured.json 面向程序消费，readable.txt 面向排查解析质量。这里负责把页面正文、
图片 OCR 和表格按页展开，便于对照原始文档检查。
"""

from pathlib import Path
from typing import Any


def build_summary_text(structured_data: dict[str, Any]) -> str:
    """生成整份 PDF 的解析概览。"""
    lines = [
        "[文档解析汇总]",
        f"文件名: {structured_data.get('file_name', '')}",
        f"总页数: {structured_data.get('total_pages', 0)}",
        "",
    ]

    for page in structured_data.get("pages", []):
        page_number = page.get("page_number", "")
        text_len = len(page.get("text", "") or "")
        image_count = len(page.get("images", []))
        table_count = len(page.get("tables", []))
        lines.append(f"第 {page_number} 页 | 正文长度: {text_len} | 图片数: {image_count} | 表格数: {table_count}")

    lines.append("")
    return "\n".join(lines)


def page_to_readable_text(page: dict[str, Any]) -> str:
    """把单页结构化数据展开成排查用文本。"""
    parts: list[str] = []
    page_number = page.get("page_number", "")
    parts.append(f"{'=' * 20} PAGE {page_number} {'=' * 20}")

    parts.append("[正文]")
    text = str(page.get("text", "")).strip()
    parts.append(text if text else "（无正文）")

    images = page.get("images", [])
    if images:
        for image in images:
            translated_text = str(image.get("translated_text") or "").strip()
            translated_source_type = str(image.get("translated_source_type") or "image_ocr")
            parts.extend(
                [
                    "",
                    f"[图片区域 {image.get('index', '')}]",
                    f"bbox: {image.get('bbox', [])}",
                    f"image_note: {image.get('image_note', '无说明')}",
                    f"translated_source_type: {translated_source_type}",
                    "translated_text:",
                    translated_text or "（无转译结果）",
                    "ocr_text:",
                    str(image.get("ocr_text") or "").strip() or "（未识别到文字，可能是纯图片）",
                ]
            )
    else:
        parts.extend(["", "[图片区域]", "（无图片区域）"])

    tables = page.get("tables", [])
    if tables:
        for table in tables:
            parts.append("")
            parts.append(f"[表格 {table.get('index', '')}]")
            if table.get("merged_to_previous"):
                parts.append(f"该表格已合并到上一页（第 {table.get('merged_target_page')} 页）的续表中")
                continue

            parts.append(f"quality: {table.get('quality', 'unknown')}")
            if table.get("bbox"):
                parts.append(f"bbox: {table.get('bbox')}")
            if table.get("is_cross_page_merged"):
                parts.append(f"merged_pages: {table.get('merged_pages', [])}")

            table_text = str(table.get("text") or "").strip()
            parts.append(table_text if table_text else "（空表格或未提取到内容）")
    else:
        parts.extend(["", "[表格]", "（无表格）"])

    parts.append("\n")
    return "\n".join(parts)


def export_readable_txt(structured_data: dict[str, Any], output_txt_path: Path) -> None:
    """写出完整 readable.txt。"""
    lines = [build_summary_text(structured_data)]
    for page in structured_data.get("pages", []):
        lines.append(page_to_readable_text(page))
    output_txt_path.write_text("\n".join(lines), encoding="utf-8")
