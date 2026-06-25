"""图片 OCR 文本后处理。

RapidOCR 对表格截图通常会输出按视觉顺序排列的碎片文本。这里把这类碎片重组为
更接近“条目-内容-结果”的文本，方便后续分块和检索。
"""

import re

from .utils import table_rows_to_records_text


OCR_NUMERIC_CONFUSIONS = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "〇": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9",
        "０": "0",
    }
)
NUMERIC_UNIT_PATTERN = re.compile(r"(?P<number>[0-9Oo〇Il|１２３４５６７８９０]+)\s*(?P<unit>分|分钟|页|条|次|天)")

VALUE_LINE_PATTERNS = (
    r"^扣\d+分/(?:次|天)$",
    r"^如有违反扣\d+分/(?:次|天)$",
    r"^第[一二三四五六七八九十]+次.*$",
    r"^第一次.*$",
    r"^第二次.*$",
    r"^第三次.*$",
)


def normalize_image_ocr_lines(text: str) -> list[str]:
    """清洗 OCR 行文本，重点修复空格和数字单位里的确定性误识别。"""
    if not text:
        return []

    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"[ \t]+", " ", line)
        line = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", line)
        line = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=\d)", "", line)
        line = re.sub(r"(?<=\d)\s+(?=[\u4e00-\u9fff])", "", line)
        line = re.sub(r"分/\s*次", "分/次", line)
        line = re.sub(r"分/\s*天", "分/天", line)
        line = normalize_numeric_unit_confusions(line)
        lines.append(line)
    return lines


def normalize_numeric_unit_confusions(text: str) -> str:
    """只在数字+单位上下文中修复 OCR 常见混淆，避免对正文做语义猜测。"""

    def replace(match: re.Match[str]) -> str:
        number = match.group("number").translate(OCR_NUMERIC_CONFUSIONS)
        return f"{number}{match.group('unit')}"

    return NUMERIC_UNIT_PATTERN.sub(replace, text)


def looks_like_ocr_table(text: str) -> bool:
    """用条目编号、扣分结果和标题关键词粗略识别表格截图。"""
    lines = normalize_image_ocr_lines(text)
    if len(lines) < 5:
        return False

    item_start_count = sum(1 for line in lines if _extract_item_start(line) is not None or re.fullmatch(r"\d{1,2}", line))
    value_line_count = sum(1 for line in lines if _is_value_line(line))
    has_table_header = any(line in {"序号", "扣分", "具体扣分项"} or "课堂纪律" in line for line in lines)
    return (item_start_count >= 2 and value_line_count >= 2) or (has_table_header and item_start_count >= 1 and value_line_count >= 1)


def render_image_ocr_table_text(text: str) -> str:
    """把 OCR 表格碎片渲染成更稳定的结构化纯文本。"""
    lines = normalize_image_ocr_lines(text)
    if not lines:
        return ""

    sections: list[str] = []
    records: list[dict[str, str]] = []
    current_record: dict[str, str] | None = None
    pending_item_id: str | None = None

    for line in lines:
        if pending_item_id and current_record is None and not _is_value_line(line):
            # OCR 可能把“序号”和“内容”拆成两行，这里先暂存序号再补内容。
            item_id, content = pending_item_id, line
            current_record = {"item_id": item_id, "content": content, "value": ""}
            pending_item_id = None
            continue

        if re.fullmatch(r"\d{1,2}", line):
            if current_record is not None:
                records.append(current_record)
                current_record = None
            pending_item_id = line
            continue

        item_start = _extract_item_start(line)
        if item_start is not None:
            if current_record is not None:
                records.append(current_record)
            current_record = {
                "item_id": item_start[0],
                "content": item_start[1],
                "value": "",
            }
            pending_item_id = None
            continue

        if _is_section_title(line) and current_record is None and pending_item_id is None:
            sections.append(line)
            continue

        if _is_value_line(line):
            if current_record is None:
                sections.append(line)
                continue
            # 同一条记录可能有多段处罚结果，追加到 value 字段中。
            current_record["value"] = _append_piece(current_record["value"], line)
            continue

        if current_record is None:
            sections.append(line)
            continue

        if not current_record["value"]:
            current_record["content"] = _append_piece(current_record["content"], line)
        else:
            current_record["value"] = _append_piece(current_record["value"], line)

    if current_record is not None:
        records.append(current_record)
    if pending_item_id is not None:
        sections.append(pending_item_id)

    rendered_parts: list[str] = []
    if sections:
        rendered_parts.append("[图片OCR表格标题]")
        rendered_parts.extend(sections)

    for record in records:
        rendered_parts.extend(
            [
                f"[条目 {record['item_id']}]",
                f"内容: {record['content']}".strip(),
                f"结果: {record['value']}".strip() if record["value"] else "结果:",
            ]
        )

    if records:
        rows = [["条目", "内容", "结果"]]
        rows.extend([record["item_id"], record["content"], record["value"]] for record in records)
        records_text = table_rows_to_records_text(rows)
        if records_text:
            rendered_parts.append(records_text)

    return "\n".join(part for part in rendered_parts if part.strip()).strip()


def _extract_item_start(line: str) -> tuple[str, str] | None:
    match = re.match(r"^(\d{1,2})(.+)$", line)
    if not match:
        return None
    item_id = match.group(1)
    content = match.group(2).strip()
    if not content:
        return None
    if _is_value_line(content):
        return None
    return item_id, content


def _is_value_line(line: str) -> bool:
    compact = line.strip()
    if any(re.fullmatch(pattern, compact) for pattern in VALUE_LINE_PATTERNS):
        return True
    if "扣" in compact and "分" in compact and len(compact) <= 40:
        return True
    if any(keyword in compact for keyword in ("通报批评", "予以开除", "保证书")) and len(compact) <= 60:
        return True
    return False


def _is_section_title(line: str) -> bool:
    compact = line.strip()
    if len(compact) <= 12 and not re.search(r"\d", compact):
        return True
    if compact.startswith(("一", "二", "三", "四", "五", "六", "七", "八", "九", "十")) and len(compact) <= 20:
        return True
    return False


def _append_piece(base: str, piece: str) -> str:
    if not base:
        return piece
    if re.search(r"[。！？；.!?;]$", base):
        return f"{base}{piece}"
    return f"{base} {piece}"
