import re

from .document_split_config import DocumentSentenceSplitProfile, DocumentSplitStrategy, load_document_split_config


class DocumentProcessor:
    def __init__(
        self,
        child_chunk_size: int = 50,
        parent_chunk_size: int = 260,
        *,
        profile: str | None = None,
    ) -> None:
        self.child_chunk_size = child_chunk_size
        self.parent_chunk_size = parent_chunk_size
        self.split_config = load_document_split_config(profile=profile)

    @staticmethod
    def _split_with_pattern(text: str, pattern: str) -> list[str]:
        return [sentence.strip() for sentence in re.split(pattern, text) if sentence.strip()]

    def _get_sentence_split_profile(self, source_format: str) -> DocumentSentenceSplitProfile:
        default_profile: DocumentSentenceSplitProfile | None = None
        for profile in self.split_config.sentence_profiles:
            if profile.name == self.split_config.default_sentence_profile:
                default_profile = profile
            if source_format in profile.source_formats:
                return profile
        if default_profile is not None:
            return default_profile
        return DocumentSentenceSplitProfile(
            name="generic_sentence",
            source_formats=(),
            boundary_pattern=r"(?<=[。！？；\n])",
            fallback_boundary_patterns=(),
        )

    @staticmethod
    def _hard_split_text(text: str, target_size: int) -> list[str]:
        stripped = text.strip()
        if not stripped:
            return []
        return [stripped[index : index + target_size].strip() for index in range(0, len(stripped), target_size)]

    def _split_fragment_for_limit(
        self,
        text: str,
        *,
        target_size: int,
        fallback_patterns: tuple[str, ...],
    ) -> list[str]:
        if len(text) <= target_size:
            return [text.strip()]

        for pattern in fallback_patterns:
            parts = self._split_with_pattern(text, pattern)
            if len(parts) <= 1:
                continue
            expanded: list[str] = []
            for part in parts:
                if len(part) <= target_size:
                    expanded.append(part)
                    continue
                expanded.extend(
                    self._split_fragment_for_limit(
                        part,
                        target_size=target_size,
                        fallback_patterns=tuple(item for item in fallback_patterns if item != pattern),
                    )
                )
            if expanded:
                return expanded

        return self._hard_split_text(text, target_size)

    def _split_sentences(self, text: str, *, target_size: int, source_format: str) -> list[str]:
        profile = self._get_sentence_split_profile(source_format)
        sentences = self._split_with_pattern(text, profile.boundary_pattern)
        if not sentences:
            return []

        expanded: list[str] = []
        for sentence in sentences:
            if len(sentence) <= target_size:
                expanded.append(sentence)
                continue
            expanded.extend(
                self._split_fragment_for_limit(
                    sentence,
                    target_size=target_size,
                    fallback_patterns=profile.fallback_boundary_patterns,
                )
            )
        return [sentence.strip() for sentence in expanded if sentence.strip()]

    def _split_parent_chunks(self, parent_text: str, *, source_format: str) -> list[str]:
        sentences = self._split_sentences(
            parent_text,
            target_size=self.parent_chunk_size,
            source_format=source_format,
        )
        if not sentences:
            return []

        parent_chunks: list[str] = []
        current_parent: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            if current_parent and current_length + sentence_length > self.parent_chunk_size:
                parent_chunks.append("".join(current_parent).strip())
                current_parent = []
                current_length = 0

            current_parent.append(sentence)
            current_length += sentence_length

        if current_parent:
            parent_chunks.append("".join(current_parent).strip())

        return parent_chunks

    def build_chunks_from_parent(
        self,
        parent_text: str,
        parent_id: str,
        extra_metadata: dict[str, object] | None = None,
        source_format: str = "plain",
    ) -> list[dict[str, object]]:
        chunks_data: list[dict[str, object]] = []
        sentences = self._split_sentences(
            parent_text,
            target_size=self.child_chunk_size,
            source_format=source_format,
        )
        if not sentences:
            return chunks_data

        current_child: list[str] = []
        current_length = 0
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_child and current_length + sentence_length > self.child_chunk_size:
                chunk: dict[str, object] = {
                    "child_text": "".join(current_child),
                    "parent_text": parent_text,
                    "parent_id": parent_id,
                }
                if extra_metadata:
                    chunk.update(extra_metadata)
                chunks_data.append(chunk)
                current_child = []
                current_length = 0

            current_child.append(sentence)
            current_length += sentence_length

        if current_child:
            chunk = {
                "child_text": "".join(current_child),
                "parent_text": parent_text,
                "parent_id": parent_id,
            }
            if extra_metadata:
                chunk.update(extra_metadata)
            chunks_data.append(chunk)

        return chunks_data

    def build_chunks_from_section(
        self,
        section_text: str,
        parent_id_prefix: str,
        extra_metadata: dict[str, object] | None = None,
        source_format: str = "plain",
    ) -> list[dict[str, object]]:
        chunks_data: list[dict[str, object]] = []
        parent_chunks = self._split_parent_chunks(section_text, source_format=source_format)
        for index, parent_text in enumerate(parent_chunks):
            parent_id = parent_id_prefix if len(parent_chunks) == 1 else f"{parent_id_prefix}_{index}"
            chunks_data.extend(
                self.build_chunks_from_parent(
                    parent_text=parent_text,
                    parent_id=parent_id,
                    extra_metadata=extra_metadata,
                    source_format=source_format,
                )
            )
        return chunks_data

    @staticmethod
    def _match_count(text: str, patterns: tuple[str, ...]) -> int:
        return sum(len(re.findall(pattern, text)) for pattern in patterns)

    def _get_split_strategy(self, text: str) -> DocumentSplitStrategy:
        default_strategy: DocumentSplitStrategy | None = None
        for strategy in self.split_config.strategies:
            if strategy.name == self.split_config.default_strategy:
                default_strategy = strategy
                continue
            if not strategy.detection_patterns:
                continue
            if self._match_count(text, strategy.detection_patterns) >= strategy.min_detection_matches:
                return strategy

        if default_strategy is None:
            raise ValueError("default document split strategy not configured")
        return default_strategy

    def _split_sections(self, text: str) -> list[str]:
        strategy = self._get_split_strategy(text)
        raw_sections = re.split(strategy.boundary_pattern, text)
        sections = [section.strip() for section in raw_sections if section.strip()]
        if sections:
            return sections
        stripped_text = text.strip()
        return [stripped_text] if stripped_text else []

    @staticmethod
    def _strip_markdown_inline(text: str) -> str:
        normalized = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        normalized = re.sub(r"__(.*?)__", r"\1", normalized)
        normalized = re.sub(r"`([^`]+)`", r"\1", normalized)
        return normalized.strip()

    def _process_markdown(self, text: str) -> list[dict[str, object]]:
        lines = text.splitlines()
        document_title = ""
        chapter_heading = ""
        section_heading = ""
        in_toc = False
        current_section_lines: list[str] = []
        sections: list[str] = []

        def flush_current_section() -> None:
            nonlocal current_section_lines
            if not current_section_lines:
                return
            section_text = "\n".join(line for line in current_section_lines if line.strip()).strip()
            if section_text:
                sections.append(section_text)
            current_section_lines = []

        def current_prefix() -> list[str]:
            prefix: list[str] = []
            if document_title:
                prefix.append(document_title)
            if chapter_heading:
                prefix.append(chapter_heading)
            if section_heading:
                prefix.append(section_heading)
            return prefix

        article_pattern = re.compile(r"^\*\*(第[一二两三四五六七八九十百千0-9]+条)\*\*\s*(.*)$")

        for raw_line in lines:
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                if current_section_lines:
                    current_section_lines.append("")
                continue
            if stripped == "---":
                flush_current_section()
                continue

            heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if heading_match:
                flush_current_section()
                level = len(heading_match.group(1))
                heading_text = self._strip_markdown_inline(heading_match.group(2))
                if level == 1:
                    document_title = heading_text
                    chapter_heading = ""
                    section_heading = ""
                    in_toc = False
                elif heading_text == "目录":
                    in_toc = True
                    chapter_heading = ""
                    section_heading = ""
                elif level == 2:
                    in_toc = False
                    chapter_heading = heading_text
                    section_heading = ""
                elif level >= 3:
                    section_heading = heading_text
                continue

            if in_toc:
                continue

            article_match = article_pattern.match(stripped)
            if article_match:
                flush_current_section()
                article_title = article_match.group(1)
                article_body = article_match.group(2)
                current_section_lines = [*current_prefix(), f"{article_title} {article_body}".strip()]
                continue

            normalized_line = self._strip_markdown_inline(stripped)
            if re.match(r"^[*-]\s+", normalized_line):
                normalized_line = re.sub(r"^[*-]\s+", "", normalized_line)
            if current_section_lines:
                current_section_lines.append(normalized_line)
                continue

            current_section_lines = [*current_prefix(), normalized_line]

        flush_current_section()
        chunks_data: list[dict[str, object]] = []
        for index, section_text in enumerate(sections):
            chunks_data.extend(
                self.build_chunks_from_section(
                    section_text=section_text,
                    parent_id_prefix=f"rule_doc_{index}",
                    source_format="markdown",
                )
            )
        return chunks_data

    def process(self, text: str, source_format: str = "plain") -> list[dict[str, object]]:
        if source_format == "markdown":
            return self._process_markdown(text)

        chunks_data: list[dict[str, object]] = []
        parent_sections = self._split_sections(text)

        parent_counter = 0
        for section_text in parent_sections:
            chunks_data.extend(
                self.build_chunks_from_section(
                    section_text=section_text,
                    parent_id_prefix=f"rule_doc_{parent_counter}",
                    source_format=source_format,
                )
            )
            parent_counter += 1

        return chunks_data
