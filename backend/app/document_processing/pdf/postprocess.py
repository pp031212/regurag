"""PDF 表格跨页合并后处理。

pdfplumber 经常把跨页续表识别成两个独立表格。这里根据列数、页内位置和表头特征
判断是否需要把下一页第一张表接到上一页最后一张表后面。
"""

from typing import Any

from .utils import get_effective_cell_count, table_rows_to_text


def get_table_column_count(table: dict[str, Any]) -> int:
    """按非空单元格估算表格有效列数，用于比较续表结构。"""
    rows = table.get("rows", [])
    if not rows:
        return 0

    effective_counts = [get_effective_cell_count(row) for row in rows if row]
    return max(effective_counts) if effective_counts else 0


def _is_numeric_like_cell(cell: str) -> bool:
    """判断单元格是否更像数值、日期或时间。"""
    compact = cell.strip().replace(",", "").replace("，", "").replace("%", "")
    compact = compact.replace("¥", "").replace("$", "")
    if not compact:
        return False

    date_patterns = (
        r"\d{4}[-/年]\d{1,2}([-/月]\d{1,2}日?)?",
        r"\d{1,2}[:：]\d{1,2}([:：]\d{1,2})?",
    )
    import re

    if any(re.fullmatch(pattern, compact) for pattern in date_patterns):
        return True
    return bool(re.fullmatch(r"[-+]?\d+(\.\d+)?", compact))


def _is_sentence_like_cell(cell: str) -> bool:
    """长句或带标点的单元格通常是正文，不像表头字段名。"""
    punctuation_markers = "，。；：,.!?！？（）()"
    if len(cell) >= 18:
        return True
    if any(marker in cell for marker in punctuation_markers):
        return True
    return len(cell.split()) >= 3


def _non_empty_cells(row: list[str]) -> list[str]:
    return [cell.strip() for cell in row if cell and cell.strip()]


def is_likely_header_row(row: list[str], next_row: list[str] | None = None) -> bool:
    """判断一行是否像表头，避免把新表误并到上一页续表。"""
    if not row:
        return False

    non_empty_cells = _non_empty_cells(row)
    if not non_empty_cells:
        return False
    if len(non_empty_cells) < 2:
        return False

    lengths = [len(cell) for cell in non_empty_cells]
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    short_cell_ratio = sum(1 for value in lengths if value <= 10) / len(lengths)
    numeric_ratio = sum(1 for cell in non_empty_cells if _is_numeric_like_cell(cell)) / len(non_empty_cells)
    sentence_ratio = sum(1 for cell in non_empty_cells if _is_sentence_like_cell(cell)) / len(non_empty_cells)

    base_header_like = (
        short_cell_ratio >= 0.7
        and avg_len <= 12
        and max_len <= 24
        and numeric_ratio <= 0.4
        and sentence_ratio <= 0.4
    )
    if not base_header_like:
        return False

    next_non_empty_cells = _non_empty_cells(next_row or [])
    if not next_non_empty_cells:
        return True

    next_lengths = [len(cell) for cell in next_non_empty_cells]
    next_avg_len = sum(next_lengths) / len(next_lengths)
    next_numeric_ratio = sum(1 for cell in next_non_empty_cells if _is_numeric_like_cell(cell)) / len(next_non_empty_cells)
    next_sentence_ratio = sum(1 for cell in next_non_empty_cells if _is_sentence_like_cell(cell)) / len(next_non_empty_cells)

    # 表头通常更短、更少数值、更少整句；和下一行一起看能减少误判。
    row_is_more_label_like = sentence_ratio < next_sentence_ratio or numeric_ratio < next_numeric_ratio
    row_is_shorter_than_next = avg_len <= next_avg_len * 0.8 or max_len < max(next_lengths)
    return row_is_more_label_like or row_is_shorter_than_next


def should_merge_with_active_table(
    active_table: dict[str, Any],
    current_table: dict[str, Any],
) -> bool:
    """判断当前页第一张表是否是上一页表格的续表。"""
    active_rows = active_table.get("rows", [])
    current_rows = current_table.get("rows", [])
    if not active_rows or not current_rows:
        return False

    active_col_count = get_table_column_count(active_table)
    current_col_count = get_table_column_count(current_table)
    if active_col_count == 0 or current_col_count == 0:
        return False
    if abs(active_col_count - current_col_count) > 1:
        return False

    current_bbox = current_table.get("bbox")
    if current_bbox:
        _, top, _, _ = current_bbox
        if top > 150:
            # 续表一般从页首附近开始，出现在页面中部时更可能是新表。
            return False

    active_first_row = active_rows[0] if active_rows else []
    current_first_row = current_rows[0] if current_rows else []
    current_second_row = current_rows[1] if len(current_rows) > 1 else []
    if active_first_row and current_first_row and active_first_row == current_first_row:
        return True
    if is_likely_header_row(current_first_row, current_second_row):
        return False
    return True


def merge_cross_page_tables(structured_data: dict[str, Any]) -> dict[str, Any]:
    """把跨页续表合并到上一页表格，并给被合并的表留下标记。"""
    pages = structured_data.get("pages", [])
    if len(pages) < 2:
        return structured_data

    active_table: dict[str, Any] | None = None
    active_page_number: int | None = None

    for page_index, page in enumerate(pages):
        tables = page.get("tables", [])
        if not tables:
            active_table = None
            active_page_number = None
            continue

        first_table = tables[0]
        if first_table.get("quality") == "error":
            active_table = None
            active_page_number = None
            continue

        if page_index == 0:
            valid_tables = [table for table in tables if table.get("quality") != "error"]
            if valid_tables:
                active_table = valid_tables[-1]
                active_page_number = page["page_number"]
            continue

        if active_table is not None and should_merge_with_active_table(active_table, first_table):
            active_rows = active_table.get("rows", [])
            current_rows = first_table.get("rows", [])
            active_first_row = active_rows[0] if active_rows else []
            current_first_row = current_rows[0] if current_rows else []

            # 两页重复同一表头时跳过当前页表头，只追加实际数据行。
            merged_rows = active_rows + current_rows[1:] if active_first_row == current_first_row else active_rows + current_rows
            active_table["rows"] = merged_rows
            active_table["text"] = table_rows_to_text(merged_rows)
            active_table["merged_pages"] = active_table.get(
                "merged_pages",
                [active_page_number] if active_page_number is not None else [],
            )
            if page["page_number"] not in active_table["merged_pages"]:
                active_table["merged_pages"].append(page["page_number"])
            active_table["is_cross_page_merged"] = True

            first_table["merged_to_previous"] = True
            first_table["merged_target_page"] = active_page_number

            remaining_valid_tables = [
                table
                for table in tables[1:]
                if table.get("quality") != "error" and not table.get("merged_to_previous")
            ]
            if remaining_valid_tables:
                active_table = remaining_valid_tables[-1]
                active_page_number = page["page_number"]
            continue

        valid_tables = [
            table
            for table in tables
            if table.get("quality") != "error" and not table.get("merged_to_previous")
        ]
        if valid_tables:
            active_table = valid_tables[-1]
            active_page_number = page["page_number"]
        else:
            active_table = None
            active_page_number = None

    return structured_data
