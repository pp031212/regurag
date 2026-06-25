"""Docker 环境任务队列混合恢复 smoke 检查。

覆盖正常任务、失败重试和陈旧 running 任务恢复的组合场景。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import subprocess
import sys
import time
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from scripts.run_docker_task_queue_concurrency_smoke import (
    _extract_claim_worker_ids,
    _run_worker_once,
    _start_worker_once,
)
from scripts.run_docker_task_queue_retry_smoke import (
    DEFAULT_SAMPLE_FILE,
    PROJECT_ROOT,
    _docker_compose,
)


@dataclass(slots=True)
class TaskFixture:
    knowledge_base_id: str
    document_id: str
    task_id: str


@dataclass(slots=True)
class MixedRecoverySmokeResult:
    retry_task_id: str
    stale_task_id: str
    retry_task_claim_worker_ids: list[str]
    stale_task_claim_worker_ids: list[str]
    retry_event_count: int
    stale_reclaimed: bool
    retry_task_final_status: str
    stale_task_final_status: str
    retry_overview: dict[str, object]
    stale_overview: dict[str, object]
    retry_alert_total: int
    stale_alert_total: int


def _log_progress(message: str, *, started_at: float) -> None:
    elapsed = time.time() - started_at
    print(f"[mixed-smoke +{elapsed:6.1f}s] {message}", flush=True)


def _resolve_timeout(*, started_at: float, total_timeout_seconds: int, stage_timeout_seconds: int, stage_name: str) -> int:
    elapsed = time.time() - started_at
    remaining = int(total_timeout_seconds - elapsed)
    if remaining <= 0:
        raise RuntimeError(
            f"mixed recovery smoke exceeded total timeout before {stage_name} "
            f"(total_timeout_seconds={total_timeout_seconds})"
        )
    return max(1, min(stage_timeout_seconds, remaining))


def _create_task_fixture(*, sample_file: Path, name_prefix: str) -> TaskFixture:
    suffix = uuid4().hex[:6]
    encoded_content = base64.b64encode(sample_file.read_bytes()).decode("ascii")
    content_type = mimetypes.guess_type(sample_file.name)[0] or "application/octet-stream"
    bootstrap_script = """
import base64
import hashlib
import json
import sys
from pathlib import Path

from app.core.config import get_settings
from app.repositories.metadata_repository import MetadataRepository
from app.task_queue.factory import get_task_queue_backend

name = sys.argv[1]
description = sys.argv[2]
subject = sys.argv[3]
domain = sys.argv[4]
filename = sys.argv[5]
content_type = sys.argv[6]
encoded_content = sys.argv[7]

file_bytes = base64.b64decode(encoded_content.encode("ascii"))
content_hash = hashlib.sha256(file_bytes).hexdigest()
settings = get_settings()
repository = MetadataRepository()
task_queue = get_task_queue_backend(repository=repository, settings=settings)
knowledge_base = repository.create_knowledge_base(
    name=name,
    description=description,
    subject=subject,
    domain=domain,
)
upload_dir = settings.resolved_uploads_dir / knowledge_base["id"]
upload_dir.mkdir(parents=True, exist_ok=True)
file_path = upload_dir / filename
file_path.write_bytes(file_bytes)
document = repository.create_document(
    knowledge_base_id=knowledge_base["id"],
    filename=filename,
    content_type=content_type,
    file_size=len(file_bytes),
    content_hash=content_hash,
    file_path=str(file_path),
)
task = task_queue.enqueue_task(
    knowledge_base_id=knowledge_base["id"],
    document_ids=[document["id"]],
    task_type="ingest",
    message=f"queued ingest for {filename}",
)
print(
    json.dumps(
        {
            "knowledge_base_id": knowledge_base["id"],
            "document_id": document["id"],
            "task_id": task["id"],
        }
    )
)
""".strip()
    fixture = _docker_compose(
        "exec",
        "-T",
        "backend-api",
        "python",
        "-c",
        bootstrap_script,
        f"{name_prefix} {suffix}",
        f"{name_prefix.lower()} docker smoke",
        "规章制度测试",
        "general",
        sample_file.name,
        content_type,
        encoded_content,
        capture_output=True,
    )
    payload = json.loads(fixture.stdout.strip())
    return TaskFixture(
        knowledge_base_id=str(payload["knowledge_base_id"]),
        document_id=str(payload["document_id"]),
        task_id=str(payload["task_id"]),
    )


def _run_backend_api_json(script: str, *args: str) -> dict | list[dict]:
    result = _docker_compose(
        "exec",
        "-T",
        "backend-api",
        "python",
        "-c",
        script,
        *args,
        capture_output=True,
    )
    return json.loads(result.stdout.strip())


def _wait_for_backend_api_container(*, timeout_seconds: int = 60) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            _docker_compose(
                "exec",
                "-T",
                "backend-api",
                "python",
                "-c",
                "print('ok')",
                capture_output=True,
            )
            return
        except Exception as exc:  # pragma: no cover - defensive polling
            last_error = exc
            time.sleep(2)
    raise RuntimeError("backend-api container not ready") from last_error


def _get_task(task_id: str) -> dict:
    script = """
import json
import sys

from app.repositories.metadata_repository import MetadataRepository

repo = MetadataRepository()
print(json.dumps(repo.get_task(sys.argv[1]), ensure_ascii=False))
""".strip()
    result = _run_backend_api_json(script, task_id)
    if not isinstance(result, dict):
        raise RuntimeError(f"unexpected task payload for {task_id}: {result!r}")
    return result


def _get_task_events(task_id: str, *, limit: int = 100) -> list[dict]:
    script = """
import json
import sys

from app.repositories.metadata_repository import MetadataRepository

repo = MetadataRepository()
print(json.dumps(repo.list_task_events(sys.argv[1], limit=int(sys.argv[2])), ensure_ascii=False))
""".strip()
    result = _run_backend_api_json(script, task_id, str(limit))
    if not isinstance(result, list):
        raise RuntimeError(f"unexpected task events payload for {task_id}: {result!r}")
    return result


def _get_overview(knowledge_base_id: str) -> dict:
    script = """
import json
import sys

from app.services.ingest_service import TaskService

service = TaskService()
print(json.dumps(service.get_task_overview(knowledge_base_id=sys.argv[1]), ensure_ascii=False))
""".strip()
    result = _run_backend_api_json(script, knowledge_base_id)
    if not isinstance(result, dict):
        raise RuntimeError(f"unexpected overview payload for {knowledge_base_id}: {result!r}")
    return result


def _get_alerts(knowledge_base_id: str) -> dict:
    script = """
import json
import sys

from app.services.ingest_service import TaskService

service = TaskService()
print(json.dumps(service.get_task_alerts(knowledge_base_id=sys.argv[1]), ensure_ascii=False))
""".strip()
    result = _run_backend_api_json(script, knowledge_base_id)
    if not isinstance(result, dict):
        raise RuntimeError(f"unexpected alerts payload for {knowledge_base_id}: {result!r}")
    return result


def _delete_knowledge_base(knowledge_base_id: str) -> None:
    script = """
import sys

from app.repositories.metadata_repository import MetadataRepository

repo = MetadataRepository()
repo.delete_knowledge_base(sys.argv[1])
print("ok")
""".strip()
    _docker_compose(
        "exec",
        "-T",
        "backend-api",
        "python",
        "-c",
        script,
        knowledge_base_id,
        capture_output=True,
    )


def _wait_for_task_status(
    task_id: str,
    *,
    acceptable_statuses: set[str],
    timeout_seconds: int,
) -> tuple[dict, list[dict]]:
    deadline = time.time() + timeout_seconds
    last_task: dict | None = None
    last_events: list[dict] = []
    while time.time() < deadline:
        last_task = _get_task(task_id)
        last_events = _get_task_events(task_id)
        if str(last_task["status"]) in acceptable_statuses:
            return last_task, last_events
        time.sleep(2)
    status = None if last_task is None else last_task.get("status")
    raise RuntimeError(f"timed out waiting for task {task_id}, last status={status}")


def _wait_for_retry_pending(task_id: str, *, timeout_seconds: int) -> tuple[dict, list[dict]]:
    deadline = time.time() + timeout_seconds
    last_task: dict | None = None
    last_events: list[dict] = []
    while time.time() < deadline:
        last_task = _get_task(task_id)
        last_events = _get_task_events(task_id)
        retry_event_count = sum(1 for item in last_events if item["event_type"] == "retrying")
        if str(last_task["status"]) == "pending" and retry_event_count >= 1:
            return last_task, last_events
        time.sleep(2)
    status = None if last_task is None else last_task.get("status")
    raise RuntimeError(f"timed out waiting for retried pending task {task_id}, last status={status}")


def _wait_for_claim(task_id: str, *, timeout_seconds: int) -> tuple[dict, list[dict]]:
    deadline = time.time() + timeout_seconds
    last_task: dict | None = None
    last_events: list[dict] = []
    while time.time() < deadline:
        last_task = _get_task(task_id)
        last_events = _get_task_events(task_id)
        if any(item["event_type"] == "claimed" for item in last_events):
            return last_task, last_events
        time.sleep(2)
    status = None if last_task is None else last_task.get("status")
    raise RuntimeError(f"timed out waiting for claimed event on task {task_id}, last status={status}")


def _mark_task_stale(task_id: str) -> None:
    command = (
        "from datetime import UTC, datetime, timedelta; "
        "from app.repositories.metadata_repository import MetadataRepository; "
        "repo = MetadataRepository(); "
        "stale = (datetime.now(UTC) - timedelta(hours=2)).replace(tzinfo=None); "
        f"repo.update_task('{task_id}', status='running', message='mixed recovery stale smoke', "
        "locked_at=stale, locked_by='worker-stale', started_at=stale); "
        f"print(repo.get_task('{task_id}'))"
    )
    _docker_compose(
        "exec",
        "-T",
        "backend-api",
        "python",
        "-c",
        command,
    )

def run_smoke(
    *,
    sample_file: Path,
    keep_knowledge_bases: bool,
    total_timeout_seconds: int,
) -> MixedRecoverySmokeResult:
    started_at = time.time()
    _wait_for_backend_api_container()
    worker_stopped = False
    knowledge_base_ids: list[str] = []
    worker_a: subprocess.Popen[str] | None = None
    worker_b: subprocess.Popen[str] | None = None
    try:
        _log_progress("stopping resident backend-worker", started_at=started_at)
        _docker_compose("stop", "backend-worker")
        worker_stopped = True

        _log_progress("creating retry fixture inside backend-api container", started_at=started_at)
        retry_fixture = _create_task_fixture(
            sample_file=sample_file,
            name_prefix="Docker Mixed Retry",
        )
        _log_progress(
            f"created retry fixture kb={retry_fixture.knowledge_base_id} task={retry_fixture.task_id}",
            started_at=started_at,
        )
        _log_progress("creating stale fixture inside backend-api container", started_at=started_at)
        stale_fixture = _create_task_fixture(
            sample_file=sample_file,
            name_prefix="Docker Mixed Stale",
        )
        _log_progress(
            f"created stale fixture kb={stale_fixture.knowledge_base_id} task={stale_fixture.task_id}",
            started_at=started_at,
        )
        knowledge_base_ids.extend([retry_fixture.knowledge_base_id, stale_fixture.knowledge_base_id])

        _log_progress("running first one-off worker with injected failure", started_at=started_at)
        _run_worker_once(inject_document_id=retry_fixture.document_id)
        _log_progress("waiting for retry task to return to pending", started_at=started_at)
        _, retry_pending_events = _wait_for_retry_pending(
            retry_fixture.task_id,
            timeout_seconds=_resolve_timeout(
                started_at=started_at,
                total_timeout_seconds=total_timeout_seconds,
                stage_timeout_seconds=180,
                stage_name="retry pending transition",
            ),
        )
        retry_event_count = sum(1 for item in retry_pending_events if item["event_type"] == "retrying")
        if retry_event_count < 1:
            raise RuntimeError("retry task did not emit retrying before mixed recovery phase")

        _log_progress("marking stale task for reclaim", started_at=started_at)
        _mark_task_stale(stale_fixture.task_id)

        _log_progress("starting worker A for retry task", started_at=started_at)
        worker_a = _start_worker_once()
        _log_progress("waiting for retry task claim", started_at=started_at)
        _wait_for_claim(
            retry_fixture.task_id,
            timeout_seconds=_resolve_timeout(
                started_at=started_at,
                total_timeout_seconds=total_timeout_seconds,
                stage_timeout_seconds=180,
                stage_name="retry claim",
            ),
        )

        _log_progress("starting worker B for stale reclaim task", started_at=started_at)
        worker_b = _start_worker_once()
        _log_progress("waiting for stale task claim", started_at=started_at)
        _wait_for_claim(
            stale_fixture.task_id,
            timeout_seconds=_resolve_timeout(
                started_at=started_at,
                total_timeout_seconds=total_timeout_seconds,
                stage_timeout_seconds=180,
                stage_name="stale claim",
            ),
        )

        _log_progress("waiting for retry task final status", started_at=started_at)
        retry_task, retry_events = _wait_for_task_status(
            retry_fixture.task_id,
            acceptable_statuses={"completed", "failed"},
            timeout_seconds=_resolve_timeout(
                started_at=started_at,
                total_timeout_seconds=total_timeout_seconds,
                stage_timeout_seconds=360,
                stage_name="retry task completion",
            ),
        )
        _log_progress("waiting for stale task final status", started_at=started_at)
        stale_task, stale_events = _wait_for_task_status(
            stale_fixture.task_id,
            acceptable_statuses={"completed", "failed"},
            timeout_seconds=_resolve_timeout(
                started_at=started_at,
                total_timeout_seconds=total_timeout_seconds,
                stage_timeout_seconds=360,
                stage_name="stale task completion",
            ),
        )

        _log_progress("waiting for one-off workers to exit", started_at=started_at)
        output_a = _wait_process(
            worker_a,
            timeout_seconds=_resolve_timeout(
                started_at=started_at,
                total_timeout_seconds=total_timeout_seconds,
                stage_timeout_seconds=180,
                stage_name="worker A exit",
            ),
        )
        output_b = _wait_process(
            worker_b,
            timeout_seconds=_resolve_timeout(
                started_at=started_at,
                total_timeout_seconds=total_timeout_seconds,
                stage_timeout_seconds=180,
                stage_name="worker B exit",
            ),
        )
        if not output_a and not output_b:
            raise RuntimeError("expected output from at least one mixed recovery worker")

        if str(retry_task["status"]) != "completed":
            raise RuntimeError(f"retry task did not complete successfully: {retry_task['status']}")
        if str(stale_task["status"]) != "completed":
            raise RuntimeError(f"stale task did not complete successfully: {stale_task['status']}")

        stale_reclaimed = any(
            item["event_type"] == "claimed"
            and bool((item.get("payload") or {}).get("stale_reclaimed"))
            for item in stale_events
        )
        if not stale_reclaimed:
            raise RuntimeError("stale task did not emit a stale_reclaimed=true claimed event")

        retry_overview = _get_overview(retry_fixture.knowledge_base_id)
        stale_overview = _get_overview(stale_fixture.knowledge_base_id)
        retry_alerts = _get_alerts(retry_fixture.knowledge_base_id)
        stale_alerts = _get_alerts(stale_fixture.knowledge_base_id)

        for name, overview in (("retry", retry_overview), ("stale", stale_overview)):
            if any(int(overview[key]) != 0 for key in ("pending", "running", "stale_running")):
                raise RuntimeError(f"{name} overview still reports pending/running/stale tasks: {overview}")

        _log_progress("mixed recovery smoke completed", started_at=started_at)
        return MixedRecoverySmokeResult(
            retry_task_id=retry_fixture.task_id,
            stale_task_id=stale_fixture.task_id,
            retry_task_claim_worker_ids=_extract_claim_worker_ids(retry_events),
            stale_task_claim_worker_ids=_extract_claim_worker_ids(stale_events),
            retry_event_count=retry_event_count,
            stale_reclaimed=stale_reclaimed,
            retry_task_final_status=str(retry_task["status"]),
            stale_task_final_status=str(stale_task["status"]),
            retry_overview=retry_overview,
            stale_overview=stale_overview,
            retry_alert_total=int(retry_alerts["total"]),
            stale_alert_total=int(stale_alerts["total"]),
        )
    finally:
        _log_progress("cleaning up mixed recovery smoke resources", started_at=started_at)
        for process in (worker_a, worker_b):
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()
        if worker_stopped:
            _docker_compose("start", "backend-worker")
        if not keep_knowledge_bases:
            for knowledge_base_id in knowledge_base_ids:
                try:
                    _wait_for_backend_api_container()
                    _delete_knowledge_base(knowledge_base_id)
                except (urllib.error.HTTPError, TimeoutError, OSError, RuntimeError):
                    pass


def _wait_process(process: subprocess.Popen[str], *, timeout_seconds: int) -> str:
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        output, _ = process.communicate()
        raise RuntimeError("worker process timed out") from exc
    if process.returncode != 0:
        raise RuntimeError(f"worker process failed with code {process.returncode}:\n{output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Dockerized Redis queue mixed recovery smoke test for retry + stale reclaim."
    )
    parser.add_argument(
        "--api-base-url",
        default="http://127.0.0.1:8000/api/v1",
        help="Base API URL for the running Dockerized backend.",
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=DEFAULT_SAMPLE_FILE,
        help="Sample document to upload for the smoke tasks.",
    )
    parser.add_argument(
        "--keep-knowledge-bases",
        action="store_true",
        help="Keep the temporary knowledge bases instead of deleting them after the run.",
    )
    parser.add_argument(
        "--total-timeout-seconds",
        type=int,
        default=720,
        help="Hard timeout budget for the whole smoke run. Default: 720 seconds.",
    )
    args = parser.parse_args()

    sample_file = args.sample_file.resolve()
    if not sample_file.exists():
        raise FileNotFoundError(f"sample file not found: {sample_file}")

    result = run_smoke(
        sample_file=sample_file,
        keep_knowledge_bases=args.keep_knowledge_bases,
        total_timeout_seconds=args.total_timeout_seconds,
    )
    print(
        json.dumps(
            {
                "retry_task_id": result.retry_task_id,
                "stale_task_id": result.stale_task_id,
                "retry_task_claim_worker_ids": result.retry_task_claim_worker_ids,
                "stale_task_claim_worker_ids": result.stale_task_claim_worker_ids,
                "retry_event_count": result.retry_event_count,
                "stale_reclaimed": result.stale_reclaimed,
                "retry_task_final_status": result.retry_task_final_status,
                "stale_task_final_status": result.stale_task_final_status,
                "retry_overview": result.retry_overview,
                "stale_overview": result.stale_overview,
                "retry_alert_total": result.retry_alert_total,
                "stale_alert_total": result.stale_alert_total,
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
