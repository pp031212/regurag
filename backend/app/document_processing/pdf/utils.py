"""PDF 文本和表格清洗工具。"""

import hashlib
import math
import re
from pathlib import Path
from typing import Any

CROSS_PAGE_CONTINUATION_MIN_SCORE = 2
REPEATED_NOISE_MAX_TEXT_LENGTH = 80
REPEATED_NOISE_MIN_PAGES = 3
REPEATED_NOISE_MIN_PAGE_RATIO = 0.5
REPEATED_NOISE_MAX_X_SPREAD = 50
REPEATED_NOISE_MAX_Y_RATIO_SPREAD = 0.06
TOC_MIN_ITEM_LINES = 3
TOC_MIN_ITEM_RATIO = 0.45
TABLE_HEADER_RECORD_MIN_SCORE = 7
TABLE_HEADER_KEYWORDS = (
    "序号",
    "项目",
    "类别",
    "类型",
    "名称",
    "行为",
    "情形",
    "条件",
    "内容",
    "要求",
    "标准",
    "规则",
    "依据",
    "条款",
    "处理",
    "方式",
    "结果",
    "说明",
    "备注",
    "分值",
    "扣分",
    "处罚",
    "奖励",
    "部门",
    "人员",
    "对象",
    "范围",
    "日期",
    "时间",
    "金额",
    "费用",
    "考核",
    "指标",
    "状态",
    "责任",
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_pdf_file_id(pdf_path: Path) -> str:
    """生成稳定但不易冲突的文件前缀，用于导出页面图片。"""
    pdf_stem = pdf_path.stem
    short_hash = hashlib.md5(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{pdf_stem}_{short_hash}"


def clean_text(text: str) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def normalize_paragraph_text(text: str) -> str:
    """合并 PDF 抽取造成的异常断行，并清理中文字符间多余空格。"""
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    merged_lines: list[str] = []
    for line in lines:
        if not merged_lines:
            merged_lines.append(line)
            continue

        previous = merged_lines[-1]
        if _should_merge_lines(previous, line):
            merged_lines[-1] = f"{previous.rstrip()}{line.lstrip()}"
        else:
            merged_lines.append(line)

    merged_text = "\n".join(merged_lines)
    merged_text = re.sub(r"分/\s*\n\s*次", "分/次", merged_text)
    merged_text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[，。；：、！？）】》])", "", merged_text)
    merged_text = re.sub(r"(?<=[（【《])\s+(?=[\u4e00-\u9fff])", "", merged_text)
    merged_text = re.sub(r"(?<=[，。；：、！？（【《])\s+(?=[\u4e00-\u9fff])", "", merged_text)
    merged_text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", merged_text)
    merged_text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=\d)", "", merged_text)
    merged_text = re.sub(r"(?<=\d)\s+(?=[\u4e00-\u9fff])", "", merged_text)
    merged_text = re.sub(r"[ \t]+", " ", merged_text)
    return merged_text.strip()


def mark_toc_pages_for_skip(structured_data: dict[str, Any]) -> dict[str, Any]:
    """识别明显目录页，并阻止目录导航文本进入正文切块。"""
    pages = structured_data.get("pages", [])
    if not isinstance(pages, list):
        return structured_data

    skipped_pages: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_text = str(page.get("text") or "").strip()
        if not page_text:
            continue
        is_toc, reasons = _is_likely_toc_page(page_text)
        if not is_toc:
            continue

        original_text = page_text
        page["is_toc_page"] = True
        page["skip_reason"] = "toc_page"
        page["toc_detection_reasons"] = reasons
        page["toc_text_preview"] = original_text[:200]
        page["text"] = ""
        page["text_blocks"] = []
        skipped_pages.append(
            {
                "page_number": page.get("page_number"),
                "reason": "toc_page",
                "detection_reasons": reasons,
                "text_preview": original_text[:120],
            }
        )

    if skipped_pages:
        structured_data.setdefault("toc_page_skips", []).extend(skipped_pages)
    return structured_data


def _is_likely_toc_page(text: str) -> tuple[bool, list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False, ["empty_text"]

    head_lines = lines[:3]
    title_hit = any(re.fullmatch(r"(目\s*录|目录)", line) for line in head_lines)
    toc_item_count = sum(1 for line in lines if _is_toc_item_line(line))
    item_ratio = toc_item_count / len(lines)
    reasons: list[str] = []

    if title_hit:
        reasons.append("title_is_toc")
    if toc_item_count >= TOC_MIN_ITEM_LINES:
        reasons.append("many_toc_item_lines")
    if item_ratio >= TOC_MIN_ITEM_RATIO:
        reasons.append("toc_item_ratio_high")

    if title_hit and toc_item_count >= 2:
        return True, reasons
    if toc_item_count >= TOC_MIN_ITEM_LINES and item_ratio >= TOC_MIN_ITEM_RATIO:
        return True, reasons
    return False, reasons or ["toc_signals_insufficient"]


def _is_toc_item_line(line: str) -> bool:
    compact = re.sub(r"\s+", " ", line).strip()
    if len(compact) < 4 or len(compact) > 120:
        return False
    if not re.search(r"\d+\s*$", compact):
        return False
    if re.search(r"[.．·•…]{2,}\s*\d+\s*$", compact):
        return True
    if re.search(r"\s{2,}\d+\s*$", line):
        return True
    return bool(
        re.match(
            r"^(第[一二两三四五六七八九十百千0-9]+[章节]|[一二两三四五六七八九十]+、|\d+[\.．、]|[（(][一二两三四五六七八九十0-9]+[）)]).+\d+\s*$",
            compact,
        )
    )


def remove_repeated_page_noise_blocks(structured_data: dict[str, Any]) -> dict[str, Any]:
    """过滤多页重复且位置稳定的模板噪声，如页眉、页脚、版本号和水印短文本。"""
    pages = structured_data.get("pages", [])
    if not isinstance(pages, list) or len(pages) < REPEATED_NOISE_MIN_PAGES:
        return structured_data

    candidates: dict[str, list[dict[str, Any]]] = {}
    for page_index, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        blocks = page.get("text_blocks")
        if not isinstance(blocks, list):
            continue
        page_number = page.get("page_number", page_index + 1)
        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            text = _normalize_repeated_noise_text(str(block.get("text") or ""))
            if not _is_repeated_noise_candidate(text):
                continue
            bbox = _read_bbox(block)
            page_height = _read_page_height(block) or _read_page_height(page)
            if not bbox or not page_height:
                continue
            candidates.setdefault(text, []).append(
                {
                    "page": page,
                    "page_number": page_number,
                    "block": block,
                    "block_index": block_index,
                    "bbox": bbox,
                    "y_ratio": bbox[1] / page_height,
                }
            )

    min_pages = max(REPEATED_NOISE_MIN_PAGES, math.ceil(len(pages) * REPEATED_NOISE_MIN_PAGE_RATIO))
    repeated_texts: dict[str, list[dict[str, Any]]] = {}
    for text, items in candidates.items():
        page_numbers = {item["page_number"] for item in items}
        if len(page_numbers) < min_pages:
            continue
        if not _has_stable_repeated_layout(items):
            continue
        repeated_texts[text] = items

    if not repeated_texts:
        return structured_data

    removals: list[dict[str, Any]] = []
    for text, items in repeated_texts.items():
        removed_pages: list[object] = []
        for item in items:
            page = item["page"]
            block = item["block"]
            blocks = page.get("text_blocks")
            if not isinstance(blocks, list) or block not in blocks:
                continue
            blocks.remove(block)
            page["text"] = normalize_paragraph_text(
                "\n".join(str(kept.get("text") or "") for kept in blocks if isinstance(kept, dict)).strip()
            )
            page.setdefault("removed_repeated_noise_blocks", []).append(
                {
                    "text": text,
                    "bbox": list(item["bbox"]),
                    "reason": "repeated_stable_layout",
                }
            )
            removed_pages.append(item["page_number"])

        if removed_pages:
            removals.append(
                {
                    "text": text,
                    "pages": removed_pages,
                    "count": len(removed_pages),
                    "reason": "repeated_stable_layout",
                }
            )

    if removals:
        structured_data.setdefault("repeated_text_noise_removals", []).extend(removals)
    return structured_data


def _normalize_repeated_noise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_repeated_noise_candidate(text: str) -> bool:
    if not text:
        return False
    compact = text.strip()
    if len(compact) > REPEATED_NOISE_MAX_TEXT_LENGTH:
        return False
    if "|" in compact:
        return False
    if _starts_new_document_unit(compact):
        return False
    return True


def _has_stable_repeated_layout(items: list[dict[str, Any]]) -> bool:
    if not items:
        return False
    x_values = [item["bbox"][0] for item in items]
    y_ratios = [item["y_ratio"] for item in items]
    return (
        max(x_values) - min(x_values) <= REPEATED_NOISE_MAX_X_SPREAD
        and max(y_ratios) - min(y_ratios) <= REPEATED_NOISE_MAX_Y_RATIO_SPREAD
    )


def _should_merge_lines(previous_line: str, current_line: str) -> bool:
    """判断两行是否像同一句被 PDF 断开。"""
    if not previous_line or not current_line:
        return False

    previous_tail = previous_line.rstrip()
    current_head = current_line.lstrip()

    if not previous_tail or not current_head:
        return False

    terminal_punctuation = "。！？；：.!?;:"
    if previous_tail[-1] in terminal_punctuation:
        return False

    if re.search(r"[\u4e00-\u9fff]$", previous_tail) and re.match(r"^[\u4e00-\u9fff]", current_head):
        return True

    continuation_prefixes = (
        "的", "并", "且", "或", "及", "与", "而", "但", "由", "按", "扣", "处", "给予", "进行",
    )
    if any(current_head.startswith(prefix) for prefix in continuation_prefixes):
        return True

    return False


def merge_cross_page_text_blocks(structured_data: dict[str, Any]) -> dict[str, Any]:
    """把跨页断开的正文尾句合并回上一页。"""
    pages = structured_data.get("pages", [])
    if len(pages) < 2:
        return structured_data

    for index in range(len(pages) - 1):
        current_page = pages[index]
        next_page = pages[index + 1]

        current_text = str(current_page.get("text") or "").strip()
        next_text = str(next_page.get("text") or "").strip()
        if not current_text or not next_text:
            continue

        continuation_text, remaining_text, score, reasons = _split_leading_cross_page_continuation(
            next_text,
            previous_text=current_text,
            current_page=current_page,
            next_page=next_page,
        )
        if not continuation_text:
            continue

        current_lines = [line for line in current_text.splitlines() if line.strip()]
        if not current_lines:
            continue

        last_line = current_lines[-1]
        current_lines[-1] = f"{last_line.rstrip()}{continuation_text.lstrip()}"

        current_page["text"] = "\n".join(current_lines).strip()
        next_page["text"] = remaining_text.strip()
        current_page.setdefault("cross_page_text_merges", []).append(
            {
                "source_page": next_page.get("page_number"),
                "merged_text": continuation_text,
                "score": score,
                "reasons": reasons,
            }
        )

    return structured_data


def _split_leading_cross_page_continuation(
    text: str,
    previous_text: str = "",
    current_page: dict[str, Any] | None = None,
    next_page: dict[str, Any] | None = None,
) -> tuple[str, str, int, list[str]]:
    """识别下一页开头是否是上一页句子或同一制度单元的续写。"""
    stripped = text.lstrip()
    if not stripped:
        return "", text, 0, ["empty_next_text"]

    if _starts_new_document_unit(stripped):
        return "", text, -5, ["next_starts_new_document_unit"]

    previous_tail = _last_non_empty_line(previous_text)
    score, reasons = _score_cross_page_continuation(
        previous_tail=previous_tail,
        next_head=stripped,
        current_page=current_page,
        next_page=next_page,
    )
    if score < CROSS_PAGE_CONTINUATION_MIN_SCORE:
        return "", text, score, reasons

    sentence_match = re.match(r"^(.{0,80}?[。！？；!?;])", stripped)
    if sentence_match:
        continuation = sentence_match.group(1)
        remaining = stripped[len(continuation):].lstrip()
        return continuation, remaining, score, reasons

    if len(stripped) <= 80:
        return stripped, "", score, reasons

    return "", text, score, [*reasons, "continuation_too_long_without_terminal"]


def _score_cross_page_continuation(
    *,
    previous_tail: str,
    next_head: str,
    current_page: dict[str, Any] | None = None,
    next_page: dict[str, Any] | None = None,
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    previous_has_terminal = bool(previous_tail) and previous_tail[-1] in "。！？；.!?;"
    has_strong_continuation = _starts_with_strong_continuation(next_head)
    has_rule_result_signal = _starts_with_rule_result_signal(next_head)

    if previous_has_terminal:
        score -= 1
        reasons.append("previous_has_terminal")
    else:
        score += 1
        reasons.append("previous_missing_terminal")

    if _previous_tail_requires_continuation(previous_tail):
        score += 2
        reasons.append("previous_tail_requires_continuation")

    if has_strong_continuation:
        score += 2
        reasons.append("next_starts_with_continuation_word")
    if has_rule_result_signal:
        score += 3
        reasons.append("next_starts_with_rule_result")
    if (
        not previous_has_terminal
        and _previous_tail_requires_continuation(previous_tail)
        and _starts_with_plain_continuation(next_head)
    ):
        score += 1
        reasons.append("plain_text_can_continue_unfinished_tail")

    layout_score, layout_reasons = _score_cross_page_layout(current_page, next_page)
    score += layout_score
    reasons.extend(layout_reasons)

    return score, reasons


def _last_non_empty_line(text: str) -> str:
    for line in reversed(str(text).splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _starts_new_document_unit(text: str) -> bool:
    head = text.lstrip()
    if not head:
        return False
    new_unit_patterns = (
        r"^第[一二两三四五六七八九十百千0-9]+[章节条]",
        r"^[一二两三四五六七八九十百千0-9]+[章节条]",
        r"^[一二三四五六七八九十]+、",
        r"^\d+[\.．、]",
        r"^（[一二三四五六七八九十0-9]+）",
        r"^[\(（]\d+[\)）]",
        r"^(目录|附则|总则|说明|备注|附件|附录)(?:\s|[:：]|$)",
    )
    return any(re.match(pattern, head) for pattern in new_unit_patterns)


def _starts_with_strong_continuation(text: str) -> bool:
    head = text.lstrip()
    continuation_prefixes = (
        "的", "并", "且", "或", "及", "与", "而", "但", "同时", "其中", "上述", "该", "本项", "本条",
    )
    return any(head.startswith(prefix) for prefix in continuation_prefixes)


def _starts_with_rule_result_signal(text: str) -> bool:
    head = text.lstrip()
    result_prefixes = ("扣", "按", "由", "处", "给予", "进行", "视为", "记", "罚", "取消", "不得", "可以")
    if any(head.startswith(prefix) for prefix in result_prefixes):
        return True
    return bool(re.match(r"^(扣除|扣减)?\d+\s*分", head))


def _starts_with_plain_continuation(text: str) -> bool:
    head = text.lstrip()
    if _starts_with_strong_continuation(head) or _starts_with_rule_result_signal(head):
        return True
    return bool(re.match(r"^[\u4e00-\u9fff]", head))


def _previous_tail_requires_continuation(text: str) -> bool:
    tail = text.rstrip()
    if not tail:
        return False
    if tail[-1] in "，、：（《【(":
        return True
    continuation_suffixes = (
        "按", "由", "为", "与", "和", "及", "或", "并", "但", "而", "扣", "处", "给予", "进行", "视为",
        "不得", "可以", "应当", "应", "未", "不", "被", "将",
    )
    return any(tail.endswith(suffix) for suffix in continuation_suffixes)


def _score_cross_page_layout(
    current_page: dict[str, Any] | None,
    next_page: dict[str, Any] | None,
) -> tuple[int, list[str]]:
    current_block = _last_text_block(current_page)
    next_block = _first_text_block(next_page)
    if not current_block or not next_block:
        return 0, []

    score = 0
    reasons: list[str] = []
    current_bbox = _read_bbox(current_block)
    next_bbox = _read_bbox(next_block)
    if current_bbox and next_bbox:
        current_page_height = _read_page_height(current_block) or _read_page_height(current_page)
        next_page_height = _read_page_height(next_block) or _read_page_height(next_page)
        if current_page_height and current_bbox[3] / current_page_height >= 0.72:
            score += 1
            reasons.append("previous_block_near_page_bottom")
        if next_page_height and next_bbox[1] / next_page_height <= 0.28:
            score += 1
            reasons.append("next_block_near_page_top")
        if abs(current_bbox[0] - next_bbox[0]) <= 24:
            score += 1
            reasons.append("indent_similar")

    current_font_size = _read_float(current_block.get("avg_font_size"))
    next_font_size = _read_float(next_block.get("avg_font_size"))
    if current_font_size is not None and next_font_size is not None and abs(current_font_size - next_font_size) <= 1.5:
        score += 1
        reasons.append("font_size_similar")

    return score, reasons


def _first_text_block(page: dict[str, Any] | None) -> dict[str, Any] | None:
    blocks = page.get("text_blocks") if isinstance(page, dict) else None
    if not isinstance(blocks, list):
        return None
    for block in blocks:
        if isinstance(block, dict) and str(block.get("text") or "").strip():
            return block
    return None


def _last_text_block(page: dict[str, Any] | None) -> dict[str, Any] | None:
    blocks = page.get("text_blocks") if isinstance(page, dict) else None
    if not isinstance(blocks, list):
        return None
    for block in reversed(blocks):
        if isinstance(block, dict) and str(block.get("text") or "").strip():
            return block
    return None


def _read_bbox(block: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = block.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    values = [_read_float(value) for value in bbox]
    if any(value is None for value in values):
        return None
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])


def _read_page_height(source: dict[str, Any] | None) -> float | None:
    if not isinstance(source, dict):
        return None
    return _read_float(source.get("page_height"))


def _read_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def normalize_cell_text(text: Any) -> str:
    if text is None:
        return ""
    normalized = str(text).replace("\r", "\n")
    normalized = re.sub(r"\n+", " ", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return normalized.strip()


def is_row_mostly_empty(row: list[str], empty_ratio_threshold: float = 0.7) -> bool:
    if not row:
        return True
    empty_count = sum(1 for cell in row if not cell.strip())
    return (empty_count / len(row)) >= empty_ratio_threshold


def trim_trailing_empty_cells(row: list[str]) -> list[str]:
    end = len(row)
    while end > 0 and not row[end - 1].strip():
        end -= 1
    return row[:end]


def clean_table_rows(table: list[list[Any]]) -> list[list[str]]:
    """清理 pdfplumber 表格行，去掉空行和尾部空单元格。"""
    cleaned_rows: list[list[str]] = []

    for raw_row in table:
        if raw_row is None:
            continue

        row = [normalize_cell_text(cell) for cell in raw_row]
        row = trim_trailing_empty_cells(row)

        if not row:
            continue
        if all(not cell for cell in row):
            continue
        if is_row_mostly_empty(row):
            continue

        cleaned_rows.append(row)

    return cleaned_rows


def table_rows_to_text(rows: list[list[str]]) -> str:
    """把二维表格转成 RAG 更容易检索的文本，并尽量绑定表头和数据行。"""
    if not rows:
        return ""

    lines: list[str] = []
    for row in rows:
        compact_row = [cell for cell in row if cell.strip()]
        if not compact_row:
            continue
        lines.append(" | ".join(compact_row))

    record_text = table_rows_to_records_text(rows)
    if record_text:
        lines.append(record_text)

    return "\n".join(lines).strip()


def table_rows_to_records_text(rows: list[list[str]]) -> str:
    """将表头后的每行渲染成 key-value 记录，避免单行召回时丢失列语义。"""
    if len(rows) < 2:
        return ""

    header = [cell.strip() for cell in rows[0]]
    if not _is_confident_header_row_for_records(header, rows[1:]):
        return ""

    records: list[str] = []
    for index, row in enumerate(rows[1:], start=1):
        normalized_row = [cell.strip() for cell in row]
        pairs: list[str] = []
        for column_index, header_cell in enumerate(header):
            if not header_cell:
                continue
            value = normalized_row[column_index] if column_index < len(normalized_row) else ""
            if value:
                pairs.append(f"{header_cell}: {value}")
        if pairs:
            records.append(f"[记录 {index}]\n" + "\n".join(pairs))

    return "\n".join(records).strip()


def _is_confident_header_row_for_records(row: list[str], data_rows: list[list[str]]) -> bool:
    score = _score_header_row_for_records(row, data_rows)
    return score >= TABLE_HEADER_RECORD_MIN_SCORE


def _score_header_row_for_records(row: list[str], data_rows: list[list[str]]) -> int:
    non_empty = [cell for cell in row if cell]
    if len(non_empty) < 2:
        return 0

    score = 2
    lengths = [len(cell) for cell in non_empty]
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    short_ratio = sum(1 for length in lengths if length <= 10) / len(lengths)
    sentence_ratio = sum(1 for cell in non_empty if _is_sentence_like_table_cell(cell)) / len(non_empty)
    value_ratio = sum(1 for cell in non_empty if _is_value_like_table_cell(cell)) / len(non_empty)
    keyword_hits = sum(1 for cell in non_empty if _contains_table_header_keyword(cell))

    if short_ratio >= 0.7:
        score += 2
    if avg_len <= 12:
        score += 1
    if max_len <= 24:
        score += 1
    else:
        score -= 3
    if keyword_hits >= 2:
        score += 3
    elif keyword_hits == 1:
        score += 2
    else:
        score -= 2
    if sentence_ratio <= 0.25:
        score += 1
    else:
        score -= 3
    if value_ratio == 0:
        score += 2
    elif value_ratio <= 0.25:
        score += 1
    else:
        score -= 3

    normalized_data_rows = [[cell.strip() for cell in row] for row in data_rows if any(cell.strip() for cell in row)]
    if normalized_data_rows:
        stable_ratio = _table_column_stability_ratio(len(row), normalized_data_rows)
        if stable_ratio >= 0.6:
            score += 2

        data_cells = [cell for data_row in normalized_data_rows for cell in data_row if cell]
        if data_cells:
            data_lengths = [len(cell) for cell in data_cells]
            data_avg_len = sum(data_lengths) / len(data_lengths)
            data_value_ratio = sum(1 for cell in data_cells if _is_value_like_table_cell(cell)) / len(data_cells)
            data_sentence_ratio = sum(1 for cell in data_cells if _is_sentence_like_table_cell(cell)) / len(data_cells)

            if avg_len < data_avg_len:
                score += 1
            if value_ratio < data_value_ratio:
                score += 1
            if sentence_ratio < data_sentence_ratio:
                score += 1

    return score


def _contains_table_header_keyword(cell: str) -> bool:
    return any(keyword in cell for keyword in TABLE_HEADER_KEYWORDS)


def _is_sentence_like_table_cell(cell: str) -> bool:
    punctuation_markers = "，。；：,.!?！？（）()"
    if len(cell) >= 24:
        return True
    if len(cell) >= 16 and any(marker in cell for marker in punctuation_markers):
        return True
    return len(cell.split()) >= 3


def _is_value_like_table_cell(cell: str) -> bool:
    compact = cell.strip().replace(",", "").replace("，", "").replace("%", "")
    compact = compact.replace("¥", "").replace("$", "")
    if not compact:
        return False

    if re.fullmatch(r"[-+]?\d+(\.\d+)?", compact):
        return True
    if re.fullmatch(r"\d{4}[-/年]\d{1,2}([-/月]\d{1,2}日?)?", compact):
        return True
    if re.fullmatch(r"\d{1,2}[:：]\d{1,2}([:：]\d{1,2})?", compact):
        return True
    if re.search(r"\d+\s*(元|万元|%|分/次|分/天|分|次|天|小时|分钟)", compact):
        return True
    return bool(re.search(r"(扣|罚|奖励|赔偿|补贴|报销)\s*\d+", compact))


def _table_column_stability_ratio(header_width: int, data_rows: list[list[str]]) -> float:
    if header_width <= 0 or not data_rows:
        return 0.0

    stable_rows = 0
    for row in data_rows:
        row_width = len(row)
        non_empty_width = len([cell for cell in row if cell])
        if abs(row_width - header_width) <= 1 or abs(non_empty_width - header_width) <= 1:
            stable_rows += 1
    return stable_rows / len(data_rows)


def rect_intersects_bbox(rect: Any, bbox: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    return not (rect.x1 < x0 or rect.x0 > x1 or rect.y1 < y0 or rect.y0 > y1)


def is_header_or_footer_block(
    block_rect: Any,
    page_height: float,
    top_ratio: float = 0.03,
    bottom_ratio: float = 0.08,
) -> bool:
    """用页面纵向位置过滤页眉页脚。"""
    top_threshold = page_height * top_ratio
    bottom_threshold = page_height * (1 - bottom_ratio)
    return block_rect.y1 <= top_threshold or block_rect.y0 >= bottom_threshold


def is_noise_text_block(text: str) -> bool:
    """过滤页码、纯数字等不应进入知识库的噪声块。"""
    if not text:
        return True

    compact = text.strip()
    if re.fullmatch(r"\d+", compact):
        return True
    if re.fullmatch(r"[-—–\s]*\d+[-—–\s]*", compact):
        return True
    if re.fullmatch(r"第\s*\d+\s*页", compact):
        return True
    return False


def get_effective_cell_count(row: list[str]) -> int:
    return sum(1 for cell in row if str(cell).strip())
