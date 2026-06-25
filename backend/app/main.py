from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.router import api_router
from .core.config import get_settings
from .core.exceptions import AppError
from .db.session import init_db
from .repositories.metadata_repository import MetadataRepository
from .runtime import AppRuntime
from .schemas.common import ErrorResponse
from .services.rag_service import RAGPipelineRegistry, bootstrap_default_knowledge_base_with_runtime
from .task_queue import get_task_queue_backend


def build_app_runtime() -> AppRuntime:
    repository = MetadataRepository()
    task_queue = get_task_queue_backend(repository=repository)
    pipeline_registry = RAGPipelineRegistry()
    return AppRuntime(
        repository=repository,
        task_queue=task_queue,
        pipeline_registry=pipeline_registry,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    runtime = build_app_runtime()
    app.state.runtime = runtime
    bootstrap_default_knowledge_base_with_runtime(
        repository=runtime.repository,
        pipeline_registry=runtime.pipeline_registry,
    )
    yield


async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    payload = ErrorResponse(code=exc.code, message=exc.message, details=exc.details)
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict):
        payload = ErrorResponse(
            code=str(detail.get("code", "HTTP_ERROR")),
            message=str(detail.get("message", "http error")),
            details=detail.get("details", {}),
        )
    else:
        payload = ErrorResponse(code="HTTP_ERROR", message=str(detail), details={})
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    payload = ErrorResponse(
        code="VALIDATION_ERROR",
        message="request validation failed",
        details={"errors": exc.errors()},
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    payload = ErrorResponse(code="INTERNAL_SERVER_ERROR", message="internal server error", details={})
    return JSONResponse(status_code=500, content=payload.model_dump())



def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        debug=settings.app_debug,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.parsed_cors_allow_origins,
        allow_origin_regex=settings.cors_allow_origin_regex,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.parsed_cors_allow_methods,
        allow_headers=settings.parsed_cors_allow_headers,
    )
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(HTTPException, http_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
