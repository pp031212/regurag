class AppError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        return self.message


class DomainError(AppError):
    status_code = 400
    code = "DOMAIN_ERROR"
    default_message = "domain error"

    def __init__(self, message: str | None = None, details: dict[str, object] | None = None) -> None:
        super().__init__(
            status_code=self.status_code,
            code=self.code,
            message=message or self.default_message,
            details=details,
        )


class KnowledgeBaseNotFoundError(DomainError):
    status_code = 404
    code = "KNOWLEDGE_BASE_NOT_FOUND"
    default_message = "knowledge base not found"


class KnowledgeBaseNotReadyError(DomainError):
    status_code = 409
    code = "KNOWLEDGE_BASE_NOT_READY"
    default_message = "knowledge base is not ready"


class DefaultKnowledgeBaseProtectedError(DomainError):
    status_code = 409
    code = "DEFAULT_KNOWLEDGE_BASE_PROTECTED"
    default_message = "default knowledge base cannot be deleted"


class InvalidKnowledgeBaseDomainError(DomainError):
    status_code = 400
    code = "INVALID_KNOWLEDGE_BASE_DOMAIN"
    default_message = "invalid knowledge base domain"


class DocumentNotFoundError(DomainError):
    status_code = 404
    code = "DOCUMENT_NOT_FOUND"
    default_message = "document not found"


class UnsupportedDocumentTypeError(DomainError):
    status_code = 400
    code = "UNSUPPORTED_DOCUMENT_TYPE"
    default_message = "unsupported document type"


class DocumentFileTooLargeError(DomainError):
    status_code = 413
    code = "DOCUMENT_FILE_TOO_LARGE"
    default_message = "document file is too large"


class DuplicateDocumentError(DomainError):
    status_code = 409
    code = "DUPLICATE_DOCUMENT"
    default_message = "duplicate document already exists in knowledge base"


class DocumentsNotFoundError(DomainError):
    status_code = 404
    code = "DOCUMENTS_NOT_FOUND"
    default_message = "no documents found for knowledge base"


class NoMatchingDocumentsError(DomainError):
    status_code = 404
    code = "DOCUMENTS_NOT_FOUND"
    default_message = "no matching documents found"


class TaskNotFoundError(DomainError):
    status_code = 404
    code = "TASK_NOT_FOUND"
    default_message = "task not found"


class ConversationNotFoundError(DomainError):
    status_code = 404
    code = "CONVERSATION_NOT_FOUND"
    default_message = "conversation not found"
