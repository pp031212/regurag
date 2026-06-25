from typing import Any

from .document_processor import DocumentProcessor


class StructuredDocumentProcessor:
    def __init__(self, text_processor: DocumentProcessor) -> None:
        self.text_processor = text_processor

    @staticmethod
    def is_structured_payload(payload: object) -> bool:
        if not isinstance(payload, dict):
            return False
        pages = payload.get("pages")
        return isinstance(pages, list)

    def process(self, structured_data: dict[str, Any]) -> list[dict[str, object]]:
        if not self.is_structured_payload(structured_data):
            raise ValueError("invalid structured document payload")

        metadata_document_id = str(
            structured_data.get("document_id")
            or structured_data.get("pdf_file_id")
            or structured_data.get("file_name")
            or "structured_document"
        )
        document_id = str(
            structured_data.get("pdf_file_id")
            or structured_data.get("file_name")
            or "structured_document"
        )
        chunks: list[dict[str, object]] = []

        for page in structured_data.get("pages", []):
            page_number = self._parse_page_number(page.get("page_number"))
            chunks.extend(self._build_text_chunks(document_id, metadata_document_id, page_number, page))
            chunks.extend(self._build_table_chunks(document_id, metadata_document_id, page_number, page))
            chunks.extend(self._build_image_chunks(document_id, metadata_document_id, page_number, page))

        for chunk in chunks:
            chunk["document_id"] = metadata_document_id

        return chunks

    @staticmethod
    def _parse_page_number(raw_page_number: object) -> int:
        try:
            return int(raw_page_number)
        except (TypeError, ValueError):
            return 0

    def _build_text_chunks(
        self,
        document_id: str,
        metadata_document_id: str,
        page_number: int,
        page: dict[str, Any],
    ) -> list[dict[str, object]]:
        page_text = str(page.get("text") or "").strip()
        if not page_text:
            return []

        section_text = f"[正文转译][第 {page_number} 页]\n{page_text}"
        return self.text_processor.build_chunks_from_section(
            section_text=section_text,
            parent_id_prefix=f"{document_id}_page_{page_number}_text",
            extra_metadata={
                "document_id": metadata_document_id,
                "source_type": "text",
                "page_number": page_number,
            },
            source_format="structured_text",
        )

    def _build_table_chunks(
        self,
        document_id: str,
        metadata_document_id: str,
        page_number: int,
        page: dict[str, Any],
    ) -> list[dict[str, object]]:
        chunks: list[dict[str, object]] = []
        tables = page.get("tables")
        if not isinstance(tables, list):
            return chunks

        for index, table in enumerate(tables):
            if not isinstance(table, dict):
                continue
            if table.get("merged_to_previous") or table.get("quality") == "error":
                continue

            table_text = str(table.get("text") or "").strip()
            if not table_text:
                continue

            section_text = (
                f"[表格转译][第 {page_number} 页][表格 {index + 1}]\n"
                f"{table_text}"
            )
            chunks.extend(
                self.text_processor.build_chunks_from_section(
                    section_text=section_text,
                    parent_id_prefix=f"{document_id}_page_{page_number}_table_{index}",
                    extra_metadata={
                        "document_id": metadata_document_id,
                        "source_type": "table",
                        "page_number": page_number,
                        "block_index": index,
                    },
                    source_format="table",
                )
            )

        return chunks

    def _build_image_chunks(
        self,
        document_id: str,
        metadata_document_id: str,
        page_number: int,
        page: dict[str, Any],
    ) -> list[dict[str, object]]:
        chunks: list[dict[str, object]] = []
        images = page.get("images")
        if not isinstance(images, list):
            return chunks

        for index, image in enumerate(images):
            if not isinstance(image, dict):
                continue

            translated_text = str(image.get("translated_text") or "").strip()
            ocr_text = str(image.get("ocr_text") or "").strip()
            source_type = str(image.get("translated_source_type") or "image_ocr")
            final_text = translated_text or ocr_text
            if not final_text:
                continue

            section_text = (
                f"[图片OCR转译][第 {page_number} 页][图片 {index + 1}]\n"
                f"{final_text}"
            )
            chunks.extend(
                self.text_processor.build_chunks_from_section(
                    section_text=section_text,
                    parent_id_prefix=f"{document_id}_page_{page_number}_image_{index}",
                    extra_metadata={
                        "document_id": metadata_document_id,
                        "source_type": source_type,
                        "page_number": page_number,
                        "block_index": index,
                        "ocr_quality": str(image.get("ocr_quality") or "unknown"),
                        "ocr_avg_confidence": image.get("ocr_avg_confidence"),
                        "ocr_min_confidence": image.get("ocr_min_confidence"),
                        "ocr_low_confidence_line_count": int(image.get("ocr_low_confidence_line_count") or 0),
                    },
                    source_format=source_type,
                )
            )

        return chunks
