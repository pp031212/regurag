"""DOCX 文本抽取服务。

把段落、表格和嵌入图片 OCR 统一导出为 extracted.txt，后续入库流程可以像处理
普通文本文件一样处理 DOCX。
"""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import count
from pathlib import Path

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.parts.image import ImagePart
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

from ..pdf.image_ocr_postprocess import looks_like_ocr_table, render_image_ocr_table_text
from ..pdf.ocr import run_ocr_on_image


@dataclass(slots=True)
class DOCXTextExtractionArtifacts:
    """DOCX 预处理产物路径。"""

    extracted_txt_path: Path


class DOCXTextExtractionService:
    """提取 DOCX 中可检索的文本内容。"""

    def preprocess(self, docx_path: Path, output_dir: Path) -> DOCXTextExtractionArtifacts:
        """生成 extracted.txt。"""
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted_txt_path = output_dir / "extracted.txt"
        embedded_image_dir = output_dir / "embedded_images"
        image_sequence = count(1)

        document = Document(docx_path)
        lines = self._extract_container_lines(document, embedded_image_dir, image_sequence)

        extracted_text = "\n\n".join(lines).strip()
        extracted_txt_path.write_text(extracted_text, encoding="utf-8")

        return DOCXTextExtractionArtifacts(extracted_txt_path=extracted_txt_path)

    def _extract_container_lines(
        self,
        container: DocxDocument | _Cell,
        embedded_image_dir: Path,
        image_sequence: Iterator[int],
    ) -> list[str]:
        """递归提取文档正文或表格单元格中的段落、表格和图片。"""
        lines: list[str] = []
        for block in self._iter_block_items(container):
            if isinstance(block, Paragraph):
                paragraph_text = block.text.strip()
                if paragraph_text:
                    lines.append(paragraph_text)
                lines.extend(self._extract_paragraph_image_lines(block, embedded_image_dir, image_sequence))
                continue

            lines.extend(self._extract_table_lines(block, embedded_image_dir, image_sequence))

        return lines

    def _extract_table_lines(
        self,
        table: Table,
        embedded_image_dir: Path,
        image_sequence: Iterator[int],
    ) -> list[str]:
        """把表格行转成管道分隔文本，同时递归处理单元格里的嵌套内容。"""
        lines: list[str] = []
        for row in table.rows:
            row_cells: list[str] = []
            for cell in row.cells:
                cell_lines = self._extract_container_lines(cell, embedded_image_dir, image_sequence)
                cell_text = " ".join(line.strip() for line in cell_lines if line.strip())
                if cell_text:
                    row_cells.append(cell_text)
            if row_cells:
                lines.append(" | ".join(row_cells))
        return lines

    def _extract_paragraph_image_lines(
        self,
        paragraph: Paragraph,
        embedded_image_dir: Path,
        image_sequence: Iterator[int],
    ) -> list[str]:
        """提取段落内嵌图片，OCR 后写回文本流。"""
        lines: list[str] = []
        for relationship_id in paragraph._element.xpath(".//a:blip/@r:embed"):
            part = paragraph.part.related_parts.get(relationship_id)
            if not isinstance(part, ImagePart):
                continue

            image_path = self._write_embedded_image(part, embedded_image_dir, image_sequence)
            ocr_text = run_ocr_on_image(str(image_path))
            if not ocr_text:
                continue

            translated_text = (
                render_image_ocr_table_text(ocr_text)
                if looks_like_ocr_table(ocr_text)
                else ocr_text
            ).strip()
            if translated_text:
                lines.append(translated_text)
        return lines

    def _write_embedded_image(
        self,
        image_part: ImagePart,
        embedded_image_dir: Path,
        image_sequence: Iterator[int],
    ) -> Path:
        """把 DOCX 内嵌图片落盘，供 OCR 引擎读取。"""
        embedded_image_dir.mkdir(parents=True, exist_ok=True)

        image_index = next(image_sequence)
        original_name = getattr(image_part, "filename", "") or f"embedded-image-{image_index}.img"
        suffix = Path(original_name).suffix or ".img"
        image_path = embedded_image_dir / f"embedded-image-{image_index}{suffix}"

        blob = getattr(image_part, "blob", None)
        if blob is None:
            blob = getattr(image_part, "_blob", b"")
        image_path.write_bytes(blob)
        return image_path

    def _iter_block_items(self, parent: DocxDocument | _Cell) -> Iterator[Paragraph | Table]:
        """按文档原始顺序遍历段落和表格。"""
        if isinstance(parent, DocxDocument):
            parent_element = parent.element.body
        else:
            parent_element = parent._tc

        for child in parent_element.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent)
