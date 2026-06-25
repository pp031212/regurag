from pydantic import BaseModel, Field


class ListResponse(BaseModel):
    total: int


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)
