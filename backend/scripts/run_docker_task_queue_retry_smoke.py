"""Docker 环境任务队列重试 smoke 检查。

模拟 worker 执行失败后重新入队，确认后续 worker 能完成任务。
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_FILE = PROJECT_ROOT / "backend" / "evals" / "docker_ingest_sample.txt"
BACKEND_BIND_MOUNT = f"{(PROJECT_ROOT / 'backend').resolve()}:/app/backend"
DEFAULT_RETRY_FINAL_TIMEOUT_SECONDS = 300
DEFAULT_WORKER_IDLE_TIMEOUT_SECONDS = 30


@dataclass(slots=True)
class SmokeResult:
    knowledge_base_id: str
    document_id: str
    task_id: str
    first_attempt_status: str
    final_status: str
    retry_event_count: int


def _json_request(url: str, *, method: str = "GET", payload: dict | None = None) -> dict:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            body = None
            headers = {"Accept": "application/json"}
            if payload is not None:
                body = json.dumps(payload).encode("utf-8")
                headers["Content-Type"] = "application/json"
            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt == 0 and "127.0.0.1:8000" in url:
                _docker_compose("restart", "backend-api")
                _wait_for_api("http://127.0.0.1:8000/api/v1", timeout_seconds=60)
                continue
            raise
    raise RuntimeError(f"request failed for {url}") from last_error


def _multipart_upload(url: str, *, knowledge_base_id: str, file_path: Path) -> dict:
    boundary = f"----ReguRAGBoundary{uuid4().hex}"
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()
    parts = [
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="knowledge_base_id"\r\n\r\n'
            f"{knowledge_base_id}\r\n"
        ).encode("utf-8"),
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8"),
        file_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    body = b"".join(parts)
    headers = {
        "Accept": "application/json",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _docker_compose(*args: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def _get_task(api_base_url: str, task_id: str) -> dict:
    return _json_request(f"{api_base_url}/tasks/{task_id}")


def _get_task_events(api_base_url: str, task_id: str) -> list[dict]:
    payload = _json_request(f"{api_base_url}/tasks/{task_id}/events")
    return list(payload["items"])


def _poll_task_status(
    api_base_url: str,
    task_id: str,
    *,
    acceptable_statuses: set[str],
    timeout_seconds: int,
) -> tuple[dict, list[dict]]:
    deadline = time.time() + timeout_seconds
    last_task: dict | None = None
    last_events: list[dict] = []
    while time.time() < deadline:
        last_task = _get_task(api_base_url, task_id)
        last_events = _get_task_events(api_base_url, task_id)
        if str(last_task["status"]) in acceptable_statuses:
            return last_task, last_events
        time.sleep(2)

    status = None if last_task is None else last_task.get("status")
    raise RuntimeError(f"timed out waiting for task {task_id}, last status={status}")


def _wait_for_api(api_base_url: str, *, timeout_seconds: int = 60) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            _json_request(f"{api_base_url.replace('/api/v1', '')}/api/v1/health")
            return
        except Exception as exc:  # pragma: no cover - defensive polling
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"api not ready at {api_base_url}") from last_error


def _ensure_api_ready(api_base_url: str) -> None:
    try:
        _wait_for_api(api_base_url, timeout_seconds=30)
    except RuntimeError:
        _docker_compose("restart", "backend-api")
        _wait_for_api(api_base_url, timeout_seconds=60)


def run_smoke(
    *,
    api_base_url: str,
    sample_file: Path,
    keep_knowledge_base: bool,
) -> SmokeResult:
    _ensure_api_ready(api_base_url)
    worker_stopped = False
    knowledge_base_id: str | None = None
    try:
        _docker_compose("stop", "backend-worker")
        worker_stopped = True

        kb_payload = {
            "name": f"Docker Retry Smoke {uuid4().hex[:6]}",
            "description": "docker redis retry smoke test",
            "subject": "规章制度测试",
            "domain": "general",
        }
        knowledge_base = _json_request(f"{api_base_url}/knowledge-bases", method="POST", payload=kb_payload)
        knowledge_base_id = str(knowledge_base["id"])

        document = _multipart_upload(
            f"{api_base_url}/documents/upload",
            knowledge_base_id=knowledge_base_id,
            file_path=sample_file,
        )
        document_id = str(document["id"])

        task = _json_request(
            f"{api_base_url}/knowledge-bases/{knowledge_base_id}/ingest",
            method="POST",
            payload={"document_ids": [document_id]},
        )
        task_id = str(task["id"])

        _docker_compose(
            "run",
            "--rm",
            "--no-deps",
            "--volume",
            BACKEND_BIND_MOUNT,
            "-e",
            f"TASK_WORKER_INJECT_FAIL_ON_FIRST_ATTEMPT_DOCUMENT_IDS={document_id}",
            "backend-worker",
            "python",
            "-m",
            "app.workers.task_worker",
            "--once",
            "--idle-timeout-seconds",
            str(DEFAULT_WORKER_IDLE_TIMEOUT_SECONDS),
        )

        first_task, first_events = _poll_task_status(
            api_base_url,
            task_id,
            acceptable_statuses={"pending", "failed", "completed"},
            timeout_seconds=60,
        )
        retry_event_count = sum(1 for item in first_events if item["event_type"] == "retrying")
        if str(first_task["status"]) != "pending" or retry_event_count == 0:
            raise RuntimeError(
                "retry smoke first attempt did not return task to pending with a retrying event"
            )

        _docker_compose(
            "run",
            "--rm",
            "--no-deps",
            "--volume",
            BACKEND_BIND_MOUNT,
            "backend-worker",
            "python",
            "-m",
            "app.workers.task_worker",
            "--once",
            "--idle-timeout-seconds",
            str(DEFAULT_WORKER_IDLE_TIMEOUT_SECONDS),
        )

        final_task, _ = _poll_task_status(
            api_base_url,
            task_id,
            acceptable_statuses={"completed", "failed"},
            timeout_seconds=DEFAULT_RETRY_FINAL_TIMEOUT_SECONDS,
        )
        if str(final_task["status"]) != "completed":
            raise RuntimeError(f"retry smoke final task did not complete: {final_task['status']}")

        return SmokeResult(
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            task_id=task_id,
            first_attempt_status=str(first_task["status"]),
            final_status=str(final_task["status"]),
            retry_event_count=retry_event_count,
        )
    finally:
        if worker_stopped:
            _docker_compose("start", "backend-worker")
        if knowledge_base_id is not None and not keep_knowledge_base:
            try:
                _json_request(f"{api_base_url}/knowledge-bases/{knowledge_base_id}", method="DELETE")
            except (urllib.error.HTTPError, TimeoutError, OSError):
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Dockerized Redis task queue retry smoke test.")
    parser.add_argument(
        "--api-base-url",
        default="http://127.0.0.1:8000/api/v1",
        help="Base API URL for the running Dockerized backend.",
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=DEFAULT_SAMPLE_FILE,
        help="Sample document to upload for the retry smoke task.",
    )
    parser.add_argument(
        "--keep-knowledge-base",
        action="store_true",
        help="Keep the temporary knowledge base instead of deleting it after the run.",
    )
    args = parser.parse_args()

    sample_file = args.sample_file.resolve()
    if not sample_file.exists():
        raise FileNotFoundError(f"sample file not found: {sample_file}")

    result = run_smoke(
        api_base_url=args.api_base_url.rstrip("/"),
        sample_file=sample_file,
        keep_knowledge_base=args.keep_knowledge_base,
    )
    print(
        json.dumps(
            {
                "knowledge_base_id": result.knowledge_base_id,
                "document_id": result.document_id,
                "task_id": result.task_id,
                "first_attempt_status": result.first_attempt_status,
                "final_status": result.final_status,
                "retry_event_count": result.retry_event_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:  # pragma: no cover - CLI guard
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise
