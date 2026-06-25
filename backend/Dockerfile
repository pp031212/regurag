# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.8.17 AS uv-binary

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/opt/regurag-venv
ENV VIRTUAL_ENV=/opt/regurag-venv
ENV PATH="/opt/regurag-venv/bin:/root/.local/bin:${PATH}"
ENV UV_HTTP_TIMEOUT=180

ARG TORCH_CPU_SPEC="torch==2.10.0+cpu"
ARG TORCH_CPU_INDEX_URL="https://download.pytorch.org/whl/cpu"

WORKDIR /app/backend

RUN apt-get update -o Acquire::Retries=3 \
    && apt-get install -y --no-install-recommends -o Acquire::Retries=3 build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-binary /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md /app/backend/

# Install third-party dependencies before copying the full source tree so
# routine app changes do not invalidate the heaviest cached layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project \
    --no-install-package torch \
    --no-install-package triton \
    --no-install-package nvidia-cublas-cu12 \
    --no-install-package nvidia-cuda-cupti-cu12 \
    --no-install-package nvidia-cuda-nvrtc-cu12 \
    --no-install-package nvidia-cuda-runtime-cu12 \
    --no-install-package nvidia-cudnn-cu12 \
    --no-install-package nvidia-cufft-cu12 \
    --no-install-package nvidia-cufile-cu12 \
    --no-install-package nvidia-curand-cu12 \
    --no-install-package nvidia-cusolver-cu12 \
    --no-install-package nvidia-cusparse-cu12 \
    --no-install-package nvidia-cusparselt-cu12 \
    --no-install-package nvidia-nccl-cu12 \
    --no-install-package nvidia-nvjitlink-cu12 \
    --no-install-package nvidia-nvshmem-cu12 \
    --no-install-package nvidia-nvtx-cu12 \
    && uv pip install --python "${VIRTUAL_ENV}/bin/python" \
    --index-url "${TORCH_CPU_INDEX_URL}" \
    "${TORCH_CPU_SPEC}"

COPY . /app/backend

RUN mkdir -p /app/backend/data/uploads /app/backend/chroma_db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
