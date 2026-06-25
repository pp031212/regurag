# ReguRAG 后端迁移到新设备的 `uv` 配置说明

## 适用范围

本文档只针对后端项目的非 Docker 本地开发环境。

当前项目默认推荐使用根目录 Docker Compose 运行：

- 本机默认开发线：`docker-compose.yml`，MySQL + CPU 版 PyTorch
- PostgreSQL canary 线：`docker-compose.yml + docker-compose.postgresql-canary.yml`
- 服务器 GPU 线：`docker-compose.yml + docker-compose.gpu.yml`

如果你只使用 Docker，不需要在宿主机保留 `backend/.venv`。本文档仅适用于需要在宿主机直接执行 `uv run ...`、IDE 直接接入后端解释器、或不通过容器运行后端的场景。

后端项目根目录是：

```text
<project-root>\\backend
```

真正需要迁移到新设备的是这些文件：

- `backend/pyproject.toml`
- `backend/uv.lock`
- `backend/.python-version`

不需要迁移的是：

- `backend/.venv`

`.venv` 是当前机器本地生成的虚拟环境，新设备上应重新创建。

## 新设备初始化步骤

### 1. 安装 `uv`

先在新设备上安装 `uv`，并确认命令可用：

```powershell
uv --version
```

### 2. 获取项目代码

将整个项目拉到本地，例如：

```powershell
git clone <your-repo-url>
cd regurag
```

如果不是通过 Git，也可以直接拷贝项目目录，但不要拷贝旧机器上的 `.venv`。

### 3. 进入后端目录

```powershell
cd backend
```

### 4. 按项目要求安装并固定 Python

当前项目后端使用 Python `3.11`。

```powershell
uv python install 3.11
uv python pin 3.11
```

如果仓库里已经有 `.python-version`，通常只需要保留即可；重新执行 `pin` 也没问题。

### 5. 根据锁文件创建虚拟环境并安装依赖

```powershell
uv sync
```

执行完成后，`backend/.venv` 会在新设备上自动生成。

## 日常开发命令

以下命令默认都在 `backend` 目录下执行。

### 同步依赖

```powershell
uv sync
```

### 启动后端服务

```powershell
uv run uvicorn app.main:app --reload
```

### 运行全部测试

```powershell
uv run pytest tests
```

### 运行指定测试

```powershell
uv run pytest tests/test_api.py
uv run pytest tests/test_rag_service.py
```

### 运行评测脚本

```powershell
uv run python scripts/eval_rag.py --label baseline
uv run python scripts/compare_shadow_graph.py --limit 1 --label shadow-compare-smoke
uv run python scripts/regress_pdf_structuring.py --case-id heming_rules_pdf --refresh-output
```

## PyCharm 配置

推荐不要在 PyCharm 里使用 `uv` 类型解释器入口，而是直接把项目虚拟环境作为普通 Python 解释器接入。

解释器路径：

```text
<项目路径>\backend\.venv\Scripts\python.exe
```

例如：

```text
<project-root>\\backend\\.venv\\Scripts\\python.exe
```

推荐配置：

- Type: `Python`
- Interpreter: `Existing`
- Path: `<项目路径>\backend\.venv\Scripts\python.exe`
- Working directory: `<项目路径>\backend`

这样做的好处是：

- 不依赖某台机器上的 `uv.exe` 绝对路径
- 换设备时只需要重新执行 `uv sync`
- IDE 与命令行共用同一套 `.venv`

## 校验是否迁移成功

在 `backend` 目录下执行：

```powershell
uv run python -V
uv run pytest tests
```

成功时应满足：

- Python 版本为 `3.11.x`
- 测试可以正常发现并运行

## 常见错误

### 1. `No pyproject.toml found`

说明当前目录不对。应先进入后端目录：

```powershell
cd backend
```

或者显式指定项目：

```powershell
uv sync --project backend
```

### 2. 在仓库根目录执行 `uv run pytest tests` 找不到测试

原因是测试目录在 `backend/tests`，不是仓库根目录的 `tests`。

正确写法：

```powershell
cd backend
uv run pytest tests
```

### 3. PyCharm 提示 `The selected tool is not uv`

说明你进入了 PyCharm 的 `uv` 工具配置入口，却选了 `python.exe`。

对于本项目，推荐不要用这个入口，直接选：

- `Python`
- `Existing interpreter`
- `<项目路径>\backend\.venv\Scripts\python.exe`

## 推荐工作流

以后在新设备上，后端统一按下面这套走：

```powershell
cd <项目路径>\backend
uv sync
uv run pytest tests
uv run uvicorn app.main:app --reload
```

这就是当前项目的标准后端工作流。
