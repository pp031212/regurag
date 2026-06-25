"""测试公共夹具。

这里提供轻量 FastAPI TestClient，屏蔽启动时的数据库初始化和后台运行时构建，
让 API 单测只关注路由行为本身。
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.main as app_main


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """创建不依赖真实外部服务的 FastAPI 测试客户端。"""
    monkeypatch.setattr(app_main, "init_db", lambda: None)
    monkeypatch.setattr(
        app_main,
        "build_app_runtime",
        lambda: SimpleNamespace(repository=None, task_queue=None, pipeline_registry=None),
    )
    monkeypatch.setattr(app_main, "bootstrap_default_knowledge_base_with_runtime", lambda **kwargs: None)
    app = app_main.create_app()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
