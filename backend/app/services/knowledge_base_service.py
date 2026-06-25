"""知识库和文档管理服务。

这里处理知识库元数据、上传文件、删除文件和索引清理。导入/重建不在这里执行，
而是由 IngestService 创建后台任务后交给 worker。
"""

import hashlib
from io import BytesIO
from pathlib import Path
import shutil
import zipfile

from fastapi import UploadFile

from ..core.config import get_settings
from ..core.exceptions import (
    DefaultKnowledgeBaseProtectedError,
    DocumentFileTooLargeError,
    DuplicateDocumentError,
    DocumentNotFoundError,
    InvalidKnowledgeBaseDomainError,
    KnowledgeBaseNotFoundError,
    UnsupportedDocumentTypeError,
)
from ..repositories.metadata_repository import MetadataRepository, get_metadata_repository
from .knowledge_base_domain_config import load_knowledge_base_domain_config
from .rag_service import (
    RAGPipelineRegistry,
    delete_document_index,
    delete_knowledge_base_index,
    get_default_pipeline_registry,
)


class KnowledgeBaseService:
    """知识库元数据和索引生命周期管理。"""

    def __init__(
        self,
        *,
        repository: MetadataRepository | None = None,
        pipeline_registry: RAGPipelineRegistry | None = None,
    ) -> None:
        self.settings = get_settings()
        self.repository = repository or get_metadata_repository()
        self.pipeline_registry = pipeline_registry or get_default_pipeline_registry()

    def _delete_knowledge_base_index(self, knowledge_base_id: str, subject: str) -> None:
        if self.pipeline_registry is get_default_pipeline_registry():
            delete_knowledge_base_index(knowledge_base_id, subject)
            return
        self.pipeline_registry.delete_knowledge_base_index(knowledge_base_id, subject)

    def _is_default_knowledge_base(self, knowledge_base_id: str) -> bool:
        return (
            self.settings.bootstrap_default_knowledge_base
            and knowledge_base_id == self.settings.default_knowledge_base_id
        )

    def _serialize_knowledge_base(self, record: dict) -> dict:
        return {
            **record,
            "is_default": self._is_default_knowledge_base(str(record["id"])),
        }

    @staticmethod
    def _domain_config():
        return load_knowledge_base_domain_config()

    def _normalize_domain(self, domain: str | None) -> str:
        """校验并归一化知识库业务域，避免前端提交任意字符串。"""
        config = self._domain_config()
        configured_values = {item.value for item in config.options}
        final_domain = (domain or "").strip() or config.default_domain
        if configured_values and final_domain not in configured_values:
            raise InvalidKnowledgeBaseDomainError(
                details={
                    "domain": final_domain,
                    "allowed_domains": sorted(configured_values),
                }
            )
        return final_domain

    def get_domain_options(self) -> dict[str, object]:
        config = self._domain_config()
        return {
            "items": [
                {
                    "value": item.value,
                    "label": item.label,
                    "description": item.description,
                }
                for item in config.options
            ],
            "default_domain": config.default_domain,
        }

    def list_knowledge_bases(self) -> list[dict]:
        return [self._serialize_knowledge_base(item) for item in self.repository.list_knowledge_bases()]

    def get_knowledge_base(self, knowledge_base_id: str) -> dict:
        record = self.repository.get_knowledge_base(knowledge_base_id)
        if record is None:
            raise KnowledgeBaseNotFoundError()
        return self._serialize_knowledge_base(record)

    def create_knowledge_base(self, name: str, description: str, subject: str | None, domain: str | None = None) -> dict:
        final_subject = subject or name
        final_domain = self._normalize_domain(domain)
        record = self.repository.create_knowledge_base(
            name=name,
            description=description,
            subject=final_subject,
            domain=final_domain,
        )
        return self._serialize_knowledge_base(record)

    def delete_knowledge_base(self, knowledge_base_id: str) -> dict:
        """删除知识库、上传目录和对应向量索引。"""
        knowledge_base = self.repository.get_knowledge_base(knowledge_base_id)
        if knowledge_base is None:
            raise KnowledgeBaseNotFoundError()
        if self._is_default_knowledge_base(knowledge_base_id):
            raise DefaultKnowledgeBaseProtectedError()

        documents = self.repository.list_documents(knowledge_base_id=knowledge_base_id)
        upload_dir = self.settings.resolved_uploads_dir / knowledge_base_id
        if upload_dir.exists():
            # 标准上传路径是 uploads/kb_id，整库删除时优先删整个目录。
            shutil.rmtree(upload_dir, ignore_errors=True)
        else:
            # 兼容历史数据：如果文件不在标准目录，就逐个删除记录中的文件路径。
            for document in documents:
                file_path = Path(document["file_path"])
                if file_path.exists() and file_path.is_file():
                    file_path.unlink(missing_ok=True)

        self._delete_knowledge_base_index(knowledge_base_id, knowledge_base["subject"])
        self.repository.delete_knowledge_base(knowledge_base_id)
        return {"id": knowledge_base_id, "deleted": True}


class DocumentService:
    """上传文档文件和管理文档索引。"""

    ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg"}
    MAX_FILE_SIZE_BYTES_BY_EXTENSION = {
        ".txt": 10 * 1024 * 1024,
        ".md": 10 * 1024 * 1024,
        ".pdf": 30 * 1024 * 1024,
        ".docx": 20 * 1024 * 1024,
        ".xlsx": 20 * 1024 * 1024,
        ".png": 10 * 1024 * 1024,
        ".jpg": 10 * 1024 * 1024,
        ".jpeg": 10 * 1024 * 1024,
    }
    UPLOAD_READ_CHUNK_SIZE_BYTES = 1024 * 1024
    CONTENT_TYPE_OVERRIDES = {
        ".md": "text/markdown",
    }
    ALLOWED_CONTENT_TYPES_BY_EXTENSION = {
        ".txt": {"text/plain"},
        ".md": {"text/markdown", "text/plain"},
        ".pdf": {"application/pdf"},
        ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
        ".xlsx": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
        ".png": {"image/png"},
        ".jpg": {"image/jpeg"},
        ".jpeg": {"image/jpeg"},
    }

    def __init__(
        self,
        *,
        repository: MetadataRepository | None = None,
        pipeline_registry: RAGPipelineRegistry | None = None,
    ) -> None:
        self.settings = get_settings()
        self.repository = repository or get_metadata_repository()
        self.pipeline_registry = pipeline_registry or get_default_pipeline_registry()
        self.uploads_dir = self.settings.resolved_uploads_dir
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def _delete_document_index(self, knowledge_base_id: str, subject: str, document_id: str) -> None:
        if self.pipeline_registry is get_default_pipeline_registry():
            delete_document_index(knowledge_base_id, subject, document_id)
            return
        self.pipeline_registry.delete_document_index(knowledge_base_id, subject, document_id)

    @staticmethod
    def _hash_content(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @classmethod
    def _format_size_mb(cls, size_bytes: int) -> str:
        size_mb = size_bytes / 1024 / 1024
        if size_mb.is_integer():
            return f"{int(size_mb)}MB"
        return f"{size_mb:.1f}MB"

    @classmethod
    async def _read_upload_content(cls, upload_file: UploadFile, max_size_bytes: int) -> bytes:
        chunks: list[bytes] = []
        total_size = 0
        while True:
            chunk = await upload_file.read(cls.UPLOAD_READ_CHUNK_SIZE_BYTES)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > max_size_bytes:
                raise DocumentFileTooLargeError(
                    message=(
                        f"文件超过当前类型限制，最大允许 {cls._format_size_mb(max_size_bytes)}。"
                    ),
                    details={
                        "max_size_bytes": max_size_bytes,
                        "max_size": cls._format_size_mb(max_size_bytes),
                        "actual_size_bytes": total_size,
                    },
                )
            chunks.append(chunk)
        return b"".join(chunks)

    @classmethod
    def _validate_declared_content_type(cls, suffix: str, content_type: str | None) -> None:
        normalized_content_type = (content_type or "").split(";", 1)[0].strip().lower()
        if not normalized_content_type or normalized_content_type == "application/octet-stream":
            return
        allowed_types = cls.ALLOWED_CONTENT_TYPES_BY_EXTENSION.get(suffix)
        if allowed_types is None or normalized_content_type in allowed_types:
            return
        raise UnsupportedDocumentTypeError(
            message=f"文件类型与后缀不匹配：{content_type}",
            details={
                "extension": suffix,
                "content_type": normalized_content_type,
                "allowed_content_types": sorted(allowed_types),
            },
        )

    @classmethod
    def _validate_file_signature(cls, suffix: str, content: bytes) -> None:
        if suffix == ".pdf" and not content.startswith(b"%PDF-"):
            cls._raise_invalid_signature(suffix, "PDF 文件头无效")
        if suffix == ".png" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
            cls._raise_invalid_signature(suffix, "PNG 文件头无效")
        if suffix in {".jpg", ".jpeg"} and not content.startswith(b"\xff\xd8\xff"):
            cls._raise_invalid_signature(suffix, "JPEG 文件头无效")
        if suffix == ".docx":
            cls._validate_office_zip_signature(suffix, content, required_prefix="word/")
        if suffix == ".xlsx":
            cls._validate_office_zip_signature(suffix, content, required_prefix="xl/")

    @classmethod
    def _validate_office_zip_signature(cls, suffix: str, content: bytes, *, required_prefix: str) -> None:
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                names = archive.namelist()
        except zipfile.BadZipFile:
            cls._raise_invalid_signature(suffix, "Office 文档不是有效的 zip 容器")
        if "[Content_Types].xml" not in names or not any(name.startswith(required_prefix) for name in names):
            cls._raise_invalid_signature(suffix, "Office 文档内部结构与后缀不匹配")

    @staticmethod
    def _raise_invalid_signature(suffix: str, reason: str) -> None:
        raise UnsupportedDocumentTypeError(
            message=f"文件内容与后缀不匹配：{suffix}",
            details={"extension": suffix, "reason": reason},
        )

    async def upload_document(self, knowledge_base_id: str, upload_file: UploadFile) -> dict:
        """保存原始文件并创建文档元数据，入库任务会后续异步处理。"""
        knowledge_base = self.repository.get_knowledge_base(knowledge_base_id)
        if knowledge_base is None:
            raise KnowledgeBaseNotFoundError()

        suffix = Path(upload_file.filename or "").suffix.lower()
        if suffix not in self.ALLOWED_EXTENSIONS:
            raise UnsupportedDocumentTypeError(
                message=f"unsupported document type: {suffix or 'unknown'}",
                details={"allowed_extensions": sorted(self.ALLOWED_EXTENSIONS)},
            )

        self._validate_declared_content_type(suffix, upload_file.content_type)
        max_size_bytes = self.MAX_FILE_SIZE_BYTES_BY_EXTENSION[suffix]
        document_dir = self.uploads_dir / knowledge_base_id
        document_dir.mkdir(parents=True, exist_ok=True)
        content = await self._read_upload_content(upload_file, max_size_bytes)
        self._validate_file_signature(suffix, content)
        file_size = len(content)
        content_hash = self._hash_content(content)

        # 同名同大小同 hash 视为重复上传，直接阻止，避免重复入库和重复索引。
        duplicate_document = self.repository.find_duplicate_document(
            knowledge_base_id=knowledge_base_id,
            filename=upload_file.filename,
            file_size=file_size,
            content_hash=content_hash,
        )
        if duplicate_document is not None:
            raise DuplicateDocumentError(
                message="该文件已存在于当前知识库，无需重复上传。",
                details={
                    "document_id": duplicate_document["id"],
                    "filename": upload_file.filename,
                    "file_size": file_size,
                },
            )

        file_path = document_dir / upload_file.filename
        file_path.write_bytes(content)

        content_type = (
            self.CONTENT_TYPE_OVERRIDES.get(suffix)
            or upload_file.content_type
            or "application/octet-stream"
        )
        return self.repository.create_document(
            knowledge_base_id=knowledge_base_id,
            filename=upload_file.filename,
            content_type=content_type,
            file_size=file_size,
            content_hash=content_hash,
            file_path=str(file_path),
        )

    def list_documents(self, knowledge_base_id: str) -> list[dict]:
        knowledge_base = self.repository.get_knowledge_base(knowledge_base_id)
        if knowledge_base is None:
            raise KnowledgeBaseNotFoundError()
        return self.repository.list_documents(knowledge_base_id=knowledge_base_id)

    def delete_document(self, document_id: str) -> dict:
        """删除原始文件、预处理产物、向量索引和文档元数据。"""
        document = self.repository.get_document(document_id)
        if document is None:
            raise DocumentNotFoundError()

        knowledge_base = self.repository.get_knowledge_base(str(document["knowledge_base_id"]))
        if knowledge_base is None:
            raise KnowledgeBaseNotFoundError()

        file_path = Path(str(document["file_path"]))
        artifact_dir = file_path.parent / "_artifacts" / document_id
        if artifact_dir.exists():
            # _artifacts 里是 PDF/DOCX/XLSX/OCR 预处理结果，删除文档时一起清理。
            shutil.rmtree(artifact_dir, ignore_errors=True)

        if file_path.exists() and file_path.is_file():
            file_path.unlink(missing_ok=True)

        self._delete_document_index(
            knowledge_base_id=str(document["knowledge_base_id"]),
            subject=str(knowledge_base["subject"]),
            document_id=document_id,
        )
        self.repository.delete_document(document_id)
        return {"id": document_id, "deleted": True}

    def get_document_path(self, document_id: str) -> Path:
        document = self.get_document(document_id)
        return Path(document["file_path"])

    def get_document(self, document_id: str) -> dict:
        document = self.repository.get_document(document_id)
        if document is None:
            raise DocumentNotFoundError()
        return document
