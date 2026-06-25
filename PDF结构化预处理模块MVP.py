import json
import os
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF：负责 PDF 页面读取、正文提取、图片区域裁剪
import pdfplumber  # 负责表格提取


# =========================================================
# 1. OCR 初始化模块
# 职责：
# - 初始化 OCR 引擎
# - 如果 OCR 依赖不可用，则自动降级为不做 OCR
# =========================================================
try:
    from rapidocr_onnxruntime import RapidOCR
    OCR_ENGINE = RapidOCR()
except Exception:
    OCR_ENGINE = None


# =========================================================
# 2. 通用工具函数模块
# 职责：
# - 目录创建
# - 正文文本清洗
# - 表格单元格清洗
# - 表格行过滤与文本化
# - 图片 OCR 调用
# - 几何区域相交判断
# =========================================================
def ensure_dir(path: str) -> None:
    """
    创建目录。若目录已存在则跳过。
    """
    os.makedirs(path, exist_ok=True)

def build_pdf_file_id(pdf_path: str) -> str:
    """
    基于 PDF 文件路径生成一个稳定的短标识。
    """
    pdf_stem = Path(pdf_path).stem
    short_hash = hashlib.md5(str(Path(pdf_path).resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{pdf_stem}_{short_hash}"

def clean_text(text: str) -> str:
    """
    清洗提取出的正文文本：
    - 去掉空字符串
    - 去掉每行首尾多余空白
    - 保留换行结构，便于人工查看
    """
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def run_ocr_on_image(image_path: str) -> str:
    """
    对裁剪出来的图片区域执行 OCR。

    返回：
    - OCR 成功：返回识别出的文本
    - OCR 不可用 / 识别失败：返回空字符串
    """
    if OCR_ENGINE is None:
        return ""

    try:
        result, _ = OCR_ENGINE(image_path)
        if not result:
            return ""

        texts = [item[1] for item in result if len(item) >= 2]
        return "\n".join(texts).strip()
    except Exception:
        return ""


def normalize_cell_text(text: Any) -> str:
    """
    清洗单元格文本。
    """
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r", "\n")
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def is_row_mostly_empty(row: List[str], empty_ratio_threshold: float = 0.7) -> bool:
    """
    判断一行是否大部分为空。
    """
    if not row:
        return True
    empty_count = sum(1 for cell in row if not cell.strip())
    return (empty_count / len(row)) >= empty_ratio_threshold


def trim_trailing_empty_cells(row: List[str]) -> List[str]:
    """
    裁掉右侧连续空单元格。
    """
    end = len(row)
    while end > 0 and not row[end - 1].strip():
        end -= 1
    return row[:end]


def clean_table_rows(table: List[List[Any]]) -> List[List[str]]:
    """
    对二维表格做基础清洗：
    - 清洗单元格文本
    - 删除右侧连续空列
    - 删除全空行
    - 删除空单元格占比过高的低质量行
    """
    cleaned_rows: List[List[str]] = []

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


def table_rows_to_text(rows: List[List[str]]) -> str:
    """
    将清洗后的表格转成更适合人看的文本。
    """
    if not rows:
        return ""

    lines = []
    for row in rows:
        row = [cell for cell in row if cell.strip()]
        if not row:
            continue
        lines.append(" | ".join(row))

    return "\n".join(lines).strip()


def rect_intersects_bbox(rect: fitz.Rect, bbox: Tuple[float, float, float, float]) -> bool:
    """
    判断 PyMuPDF 的文本块 rect 是否和表格 bbox 相交。
    """
    x0, y0, x1, y1 = bbox
    return not (rect.x1 < x0 or rect.x0 > x1 or rect.y1 < y0 or rect.y0 > y1)

def is_header_or_footer_block(
    block_rect: fitz.Rect,
    page_height: float,
    top_ratio: float = 0.03,
    bottom_ratio: float = 0.08,
) -> bool:
    top_threshold = page_height * top_ratio
    bottom_threshold = page_height * (1 - bottom_ratio)

    if block_rect.y1 <= top_threshold:
        return True

    if block_rect.y0 >= bottom_threshold:
        return True

    return False


def is_noise_text_block(text: str) -> bool:
    """
    判断文本块是否是页码、重复页眉等噪声文本。

    当前规则比较保守：
    - 纯数字页码，如 "1"、"2"、"3"
    - 单独页码形式，如 "- 1 -"、"第 2 页"
    - 可以后续继续扩展
    """
    if not text:
        return True

    text = text.strip()

    # 纯数字页码
    if re.fullmatch(r"\d+", text):
        return True

    # 类似 - 1 -
    if re.fullmatch(r"[-—–\s]*\d+[-—–\s]*", text):
        return True

    # 类似 第 2 页
    if re.fullmatch(r"第\s*\d+\s*页", text):
        return True

    return False
def get_effective_cell_count(row: List[str]) -> int:
    """
    统计一行中非空单元格的数量。
    """
    return sum(1 for cell in row if str(cell).strip())

# =========================================================
# 3. PDF 正文提取模块
# 职责：
# - 按页提取正文文本
# - 基于表格 bbox 尽量排除表格区域
# =========================================================
def extract_page_text_excluding_tables(
    page: fitz.Page,
    table_bboxes: List[Tuple[float, float, float, float]]
) -> str:
    """
    提取页面正文，并尽量排除：
    - 表格区域
    - 页眉页脚区域
    - 页码等噪声文本
    """
    page_dict = page.get_text("dict")
    kept_texts = []
    page_height = page.rect.height

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        bbox = block.get("bbox")
        if not bbox:
            continue

        block_rect = fitz.Rect(*bbox)

        # 1. 如果和任一表格区域相交，则跳过
        if any(rect_intersects_bbox(block_rect, tbbox) for tbbox in table_bboxes):
            continue

        # 2. 如果位于页眉/页脚区域，则先跳过
        if is_header_or_footer_block(block_rect, page_height, top_ratio=0.05, bottom_ratio=0.06):
            continue

        # 3. 提取 block 内文本
        block_text_parts = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text.strip():
                    block_text_parts.append(text)

        block_text = "".join(block_text_parts).strip()
        if not block_text:
            continue

        # 4. 如果文本本身像页码/噪声，也跳过
        if is_noise_text_block(block_text):
            continue

        kept_texts.append(block_text)

    return "\n".join(kept_texts).strip()


# =========================================================
# 4. PDF 图片区域提取模块
# 职责：
# - 找出页面中的图片块
# - 将图片块裁剪保存到本地
# - 对图片块做 OCR（可选）
# =========================================================
def extract_image_blocks(
    page: fitz.Page,
    page_number: int,
    image_output_dir: str,
    pdf_file_id: str,
    enable_ocr: bool = True,
) -> List[Dict[str, Any]]:
    """
    基于页面 block 信息，找出图片区域并导出裁剪图，再做 OCR。

    当前模块职责：
    - 定位 PDF 页面中的图片区域
    - 将图片区域裁剪并保存到本地
    - 对“可能含文字的图片”尝试执行 OCR

    注意：
    - 当前并不做真正的图片语义理解
    - 如果图片是纯图片（如风景图、动物图、插图），OCR 为空是正常现象
    """
    image_items: List[Dict[str, Any]] = []
    page_dict = page.get_text("dict")

    image_index = 0
    for block in page_dict.get("blocks", []):
        if block.get("type") != 1:
            continue

        bbox = block.get("bbox", None)
        if not bbox:
            continue

        x0, y0, x1, y1 = bbox
        rect = fitz.Rect(x0, y0, x1, y1)

        pix = page.get_pixmap(clip=rect, dpi=150)
        image_filename = f"{pdf_file_id}_page_{page_number}_img_{image_index}.png"
        image_path = os.path.join(image_output_dir, image_filename)
        pix.save(image_path)

        ocr_text = run_ocr_on_image(image_path) if enable_ocr else ""

        if ocr_text:
            image_note = "检测到图片区域，并识别出其中的文字"
        else:
            image_note = "检测到图片区域，但未识别到文字，可能是纯图片或无文字图像"

        image_items.append({
            "index": image_index,
            "bbox": [x0, y0, x1, y1],
            "image_path": image_path,
            "ocr_text": ocr_text,
            "image_note": image_note
        })
        image_index += 1

    return image_items


# =========================================================
# 5. PDF 表格提取模块
# 职责：
# - 按页提取表格
# - 优先获取表格 bbox
# - 对原始表格做基础清洗
# - 输出更适合阅读和后续入库的文本
# =========================================================
def extract_tables_from_plumber_page(plumber_page) -> List[Dict[str, Any]]:
    """
    基于已经打开的 pdfplumber page 对象提取表格。

    这样做的好处：
    - 避免每处理一页都重复 open 整份 PDF
    - 性能更稳定
    - 主流程里可以统一管理 pdfplumber 的生命周期
    """
    tables_output: List[Dict[str, Any]] = []

    try:
        # 优先找 table object，这样能拿到 bbox
        found_tables = plumber_page.find_tables()

        for idx, tb in enumerate(found_tables):
            raw_table = tb.extract()
            cleaned_rows = clean_table_rows(raw_table)
            table_text = table_rows_to_text(cleaned_rows)

            tables_output.append({
                "index": idx,
                "bbox": list(tb.bbox) if tb.bbox else None,
                "rows": cleaned_rows,
                "text": table_text,
                "quality": "low" if len(cleaned_rows) <= 1 else "normal"
            })

        # 如果 find_tables 没找到，再降级走 extract_tables
        if not tables_output:
            raw_tables = plumber_page.extract_tables()
            for idx, raw_table in enumerate(raw_tables):
                cleaned_rows = clean_table_rows(raw_table)
                table_text = table_rows_to_text(cleaned_rows)

                tables_output.append({
                    "index": idx,
                    "bbox": None,
                    "rows": cleaned_rows,
                    "text": table_text,
                    "quality": "low" if len(cleaned_rows) <= 1 else "normal"
                })

    except Exception as e:
        tables_output.append({
            "index": -1,
            "bbox": None,
            "rows": [],
            "text": "",
            "quality": "error",
            "error": f"表格提取失败: {e}"
        })

    return tables_output
def get_table_column_count(table: Dict[str, Any]) -> int:
    """
    获取表格的有效列数。

    说明：
    - 不能直接用 len(row)，因为很多 PDF 表头会带大量空单元格
    - 这里改为统计每一行“非空单元格数量”
    - 取所有行中的最大有效列数，作为该表格的列数估计
    """
    rows = table.get("rows", [])
    if not rows:
        return 0

    effective_counts = [get_effective_cell_count(row) for row in rows if row]
    if not effective_counts:
        return 0

    return max(effective_counts)


def is_likely_header_row(row: List[str]) -> bool:
    """
    粗略判断一行是否像表头。

    当前策略比之前更保守：
    1. 优先看是否命中典型表头关键词
    2. 再看是否所有单元格都很短，像字段标签集合
    3. 只要某些单元格已经明显是解释性短句/描述句，就不要轻易判表头
    """
    if not row:
        return False

    non_empty_cells = [cell.strip() for cell in row if cell and cell.strip()]
    if not non_empty_cells:
        return False

    joined = " ".join(non_empty_cells).lower()

    # 1. 典型表头关键词（中英混合，作为强信号）
    header_keywords = [
        "name", "type", "description", "value", "amount", "date", "category",
        "名称", "维度", "类型", "说明", "指标", "优点", "缺点",
        "应用场景", "核心思想", "传统机器学习", "深度学习", "对比维度"
    ]
    keyword_hit = any(keyword in joined for keyword in header_keywords)

    # 2. 统计长度特征
    lengths = [len(cell) for cell in non_empty_cells]
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)

    # 3. 很短的标签型单元格占比
    short_cell_ratio = sum(1 for x in lengths if x <= 10) / len(lengths)

    # 规则 A：关键词命中，直接认为更像表头
    if keyword_hit:
        return True

    # 规则 B：所有单元格都比较短，且整体均值也低，才认为像表头
    # 这样可以避免“决策树 | 通过特征划分不断构造...”这种数据行被误判
    if short_cell_ratio >= 0.8 and avg_len <= 8 and max_len <= 12:
        return True

    return False


def should_merge_with_active_table(
    active_table: Dict[str, Any],
    curr_table: Dict[str, Any],
) -> bool:
    """
    判断当前页表格是否应该并入当前活跃主表。
    """
    active_rows = active_table.get("rows", [])
    curr_rows = curr_table.get("rows", [])

    if not active_rows or not curr_rows:
        return False

    active_col_count = get_table_column_count(active_table)
    curr_col_count = get_table_column_count(curr_table)

    if active_col_count == 0 or curr_col_count == 0:
        return False

    if abs(active_col_count - curr_col_count) > 1:
        return False

    curr_bbox = curr_table.get("bbox")
    if curr_bbox:
        _, top, _, _ = curr_bbox
        if top > 150:
            return False

    active_first_row = active_rows[0] if active_rows else []
    curr_first_row = curr_rows[0] if curr_rows else []

    if active_first_row and curr_first_row and active_first_row == curr_first_row:
        return True

    if is_likely_header_row(curr_first_row):
        return False

    return True


def merge_cross_page_tables(structured_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    对整份 PDF 的相邻页面表格做跨页续表合并。

    升级点：
    - 支持 2 页以上的连续跨页续表
    - 维护一个当前活跃主表 active_table
    """
    pages = structured_data.get("pages", [])
    if len(pages) < 2:
        return structured_data

    active_table = None
    active_page_number = None

    for page_idx, page in enumerate(pages):
        tables = page.get("tables", [])
        print(f"\n[处理第 {page['page_number']} 页] tables={len(tables)}")

        if not tables:
            print("当前页无表格，续表链断开")
            active_table = None
            active_page_number = None
            continue

        first_table = tables[0]

        if first_table.get("quality") == "error":
            print("当前页首表 quality=error，续表链断开")
            active_table = None
            active_page_number = None
            continue

        if page_idx == 0:
            valid_tables = [tb for tb in tables if tb.get("quality") != "error"]
            if valid_tables:
                active_table = valid_tables[-1]
                active_page_number = page["page_number"]
                print(f"初始化 active_table = 第 {active_page_number} 页最后一个表")
            continue

        # 如果当前已有活跃表，则尝试把当前页首表并进去
        if active_table is not None:
            active_rows = active_table.get("rows", [])
            curr_rows = first_table.get("rows", [])

            active_col_count = get_table_column_count(active_table)
            curr_col_count = get_table_column_count(first_table)
            curr_bbox = first_table.get("bbox")
            curr_top = curr_bbox[1] if curr_bbox else None
            curr_first_row = curr_rows[0] if curr_rows else []

            print(f"[续表判断] active_page={active_page_number} -> current_page={page['page_number']}")
            print(f"active_rows={len(active_rows)}, active_cols={active_col_count}")
            print(f"curr_rows={len(curr_rows)}, curr_cols={curr_col_count}")
            print(f"curr_top={curr_top}")
            print(f"curr_first_row={curr_first_row}")
            print(f"curr_first_row像表头吗={is_likely_header_row(curr_first_row)}")

            should_merge = should_merge_with_active_table(active_table, first_table)
            print(f"是否合并={should_merge}")

            if should_merge:
                active_first_row = active_rows[0] if active_rows else []
                curr_first_row = curr_rows[0] if curr_rows else []

                if active_first_row and curr_first_row and active_first_row == curr_first_row:
                    merged_rows = active_rows + curr_rows[1:]
                    print("检测到重复表头，跳过当前页首行再合并")
                else:
                    merged_rows = active_rows + curr_rows

                active_table["rows"] = merged_rows
                active_table["text"] = table_rows_to_text(merged_rows)

                active_table["merged_pages"] = active_table.get(
                    "merged_pages",
                    [active_page_number] if active_page_number is not None else []
                )
                if page["page_number"] not in active_table["merged_pages"]:
                    active_table["merged_pages"].append(page["page_number"])

                active_table["is_cross_page_merged"] = True

                first_table["merged_to_previous"] = True
                first_table["merged_target_page"] = active_page_number

                print(f"已合并：第 {page['page_number']} 页 -> 第 {active_page_number} 页主表")

                # 当前页除首表外，若还有新表，则更新 active_table
                remaining_valid_tables = [
                    tb for tb in tables[1:]
                    if tb.get("quality") != "error" and not tb.get("merged_to_previous")
                ]
                if remaining_valid_tables:
                    active_table = remaining_valid_tables[-1]
                    active_page_number = page["page_number"]
                    print(f"当前页还有新表，active_table 切换为第 {active_page_number} 页最后一个独立表")

                continue

        # 没命中续表，更新 active_table 为当前页最后一个正常表
        valid_tables = [
            tb for tb in tables
            if tb.get("quality") != "error" and not tb.get("merged_to_previous")
        ]
        if valid_tables:
            active_table = valid_tables[-1]
            active_page_number = page["page_number"]
            print(f"未合并，active_table 更新为第 {active_page_number} 页最后一个正常表")
        else:
            active_table = None
            active_page_number = None
            print("当前页无可用表，active_table 清空")

    return structured_data

# =========================================================
# 6. PDF 结构化提取主模块
# 职责：
# - 遍历整份 PDF
# - 按页收集表格、正文、图片
# - 输出统一结构化 JSON
# =========================================================
def extract_pdf_structured(
    pdf_path: str,
    output_json_path: str,
    image_output_dir: str,
    enable_ocr: bool = True,
) -> Dict[str, Any]:
    """
    核心入口：将 PDF 提取为结构化 JSON。

    性能优化点：
    - fitz 整份文档只打开一次
    - pdfplumber 整份文档也只打开一次
    """
    ensure_dir(image_output_dir)

    pdf_name = Path(pdf_path).name
    pdf_file_id = build_pdf_file_id(pdf_path)
    fitz_doc = fitz.open(pdf_path)

    try:
        result: Dict[str, Any] = {
            "file_name": pdf_name,
            "file_path": str(Path(pdf_path).resolve()),
            "pdf_file_id": pdf_file_id,
            "total_pages": len(fitz_doc),
            "pages": []
        }

        with pdfplumber.open(pdf_path) as plumber_doc:
            if len(plumber_doc.pages) != len(fitz_doc):
                raise ValueError("pdfplumber 与 PyMuPDF 读取到的页数不一致")

            for page_idx in range(len(fitz_doc)):
                page_number = page_idx + 1

                # PyMuPDF 页面对象：负责正文、图片
                fitz_page = fitz_doc.load_page(page_idx)

                # pdfplumber 页面对象：负责表格
                plumber_page = plumber_doc.pages[page_idx]

                # 1. 先提表格
                tables = extract_tables_from_plumber_page(plumber_page)

                # 2. 收集表格 bbox，用于正文过滤
                table_bboxes: List[Tuple[float, float, float, float]] = []
                for tb in tables:
                    if tb.get("bbox"):
                        table_bboxes.append(tuple(tb["bbox"]))

                # 3. 再提正文，排除表格区域、页眉页脚等
                page_text = extract_page_text_excluding_tables(fitz_page, table_bboxes)

                # 4. 提取图片区域
                images = extract_image_blocks(
                    page=fitz_page,
                    page_number=page_number,
                    image_output_dir=image_output_dir,
                    pdf_file_id=pdf_file_id,
                    enable_ocr=enable_ocr,
                )

                # 5. 组装单页结果
                page_item = {
                    "page_number": page_number,
                    "text": page_text,
                    "images": images,
                    "tables": tables
                }
                result["pages"].append(page_item)

        # 6. 跨页表格后处理
        result = merge_cross_page_tables(result)

        # 7. 保存 JSON
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    finally:
        fitz_doc.close()


# =========================================================
# 7. 文本导出模块
# 职责：
# - 将结构化 JSON 转成人可读的 txt
# - 便于人工检查解析效果
# =========================================================
def build_summary_text(structured_data: Dict[str, Any]) -> str:
    """
    生成文档解析汇总信息。
    用于 txt 文件顶部，快速查看每页解析情况。
    """
    lines = []
    lines.append("[文档解析汇总]")
    lines.append(f"文件名: {structured_data.get('file_name', '')}")
    lines.append(f"总页数: {structured_data.get('total_pages', 0)}")
    lines.append("")

    for page in structured_data.get("pages", []):
        page_number = page.get("page_number", "")
        text_len = len(page.get("text", "") or "")
        image_count = len(page.get("images", []))
        table_count = len(page.get("tables", []))
        lines.append(
            f"第 {page_number} 页 | 正文长度: {text_len} | 图片数: {image_count} | 表格数: {table_count}"
        )

    lines.append("")
    return "\n".join(lines)


def page_to_readable_text(page: Dict[str, Any]) -> str:
    """
    将单页结构化结果转成适合人工检查的可读文本。
    """
    parts = []
    page_number = page.get("page_number", "")

    parts.append(f"{'=' * 20} PAGE {page_number} {'=' * 20}")

    # ---------- 正文 ----------
    parts.append("[正文]")
    text = page.get("text", "").strip()
    parts.append(text if text else "（无正文）")

    # ---------- 图片区域 ----------
    images = page.get("images", [])
    if images:
        for img in images:
            parts.append("")
            parts.append(f"[图片区域 {img.get('index', '')}]")
            parts.append(f"bbox: {img.get('bbox', [])}")
            parts.append(f"image_note: {img.get('image_note', '无说明')}")
            parts.append("ocr_text:")
            ocr_text = (img.get("ocr_text") or "").strip()
            parts.append(ocr_text if ocr_text else "（未识别到文字，可能是纯图片）")
    else:
        parts.append("")
        parts.append("[图片区域]")
        parts.append("（无图片区域）")

    # ---------- 表格 ----------
    tables = page.get("tables", [])
    if tables:
        has_visible_table = False

        for tb in tables:
            if tb.get("merged_to_previous"):
                parts.append("")
                parts.append(f"[表格 {tb.get('index', '')}]")
                parts.append(f"该表格已合并到上一页（第 {tb.get('merged_target_page')} 页）的续表中")
                continue

            has_visible_table = True
            parts.append("")
            parts.append(f"[表格 {tb.get('index', '')}]")
            parts.append(f"quality: {tb.get('quality', 'unknown')}")

            if tb.get("bbox"):
                parts.append(f"bbox: {tb.get('bbox')}")

            if tb.get("is_cross_page_merged"):
                parts.append(f"merged_pages: {tb.get('merged_pages', [])}")

            table_text = (tb.get("text") or "").strip()
            parts.append(table_text if table_text else "（空表格或未提取到内容）")

        if not has_visible_table and all(tb.get("merged_to_previous") for tb in tables):
            pass
    else:
        parts.append("")
        parts.append("[表格]")
        parts.append("（无表格）")

    parts.append("\n")
    return "\n".join(parts)

def export_readable_txt(structured_data: Dict[str, Any], output_txt_path: str) -> None:
    """
    将整份结构化结果导出成 txt 文件，便于人工检查。
    """
    lines = []
    lines.append(build_summary_text(structured_data))

    for page in structured_data.get("pages", []):
        lines.append(page_to_readable_text(page))

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================================================
# 8. 启动入口模块
# 职责：
# - 指定输入 PDF
# - 指定输出目录
# - 执行结构化提取
# - 导出 JSON 和 TXT
# =========================================================
if __name__ == "__main__":
    # ---------- 输入文件 ----------
    pdf_path = "test.pdf"

    # ---------- 输出目录 ----------
    output_dir = "output"
    output_json_path = os.path.join(output_dir, "test_structured.json")
    output_txt_path = os.path.join(output_dir, "test_readable.txt")
    image_output_dir = os.path.join(output_dir, "images")

    ensure_dir(output_dir)

    # ---------- 执行 PDF 结构化提取 ----------
    data = extract_pdf_structured(
        pdf_path=pdf_path,
        output_json_path=output_json_path,
        image_output_dir=image_output_dir,
        enable_ocr=True,
    )

    # ---------- 导出人工可读 txt ----------
    export_readable_txt(data, output_txt_path)

    # ---------- 输出执行结果 ----------
    print(f"处理完成，共 {data['total_pages']} 页")
    print(f"JSON 已保存到: {output_json_path}")
    print(f"TXT 已保存到: {output_txt_path}")