"""Docker 环境任务队列并发 smoke 检查。

验证多个 worker 并发认领任务时不会重复执行或破坏任务状态。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from scripts.run_docker_task_queue_retry_smoke import (
    DEFAULT_SAMPLE_FILE,
    DEFAULT_WORKER_IDLE_TIMEOUT_SECONDS,
    PROJECT_ROOT,
    _docker_compose,
    _ensure_api_ready,
    _get_task,
    _get_task_events,
    _json_request,
    _multipart_upload,
)

DEFAULT_CONCURRENCY_FINAL_TIMEOUT_SECONDS = 300
DEFAULT_CONCURRENCY_WORKER_EXIT_TIMEOUT_SECONDS = 300


@dataclass(slots=True)
class TaskFixture:
    knowledge_base_id: str
    document_id: str
    task_id: str


@dataclass(slots=True)
class ConcurrencySmokeResult:
    retried_task_id: str
    companion_task_id: str
    retried_task_retry_event_count: int
    retried_task_final_status: str
    companion_task_final_status: str
    retried_task_claim_worker_ids: list[str]
    companion_task_claim_worker_ids: list[str]


def _log_progress(message: str, *, started_at: float) -> None:
    elapsed = time.time() - started_at
    print(f"[concurrency-smoke +{elapsed:6.1f}s] {message}", flush=True)


def _worker_run_command(*, inject_document_id: str | None = None) -> list[str]:
    command = [
        "docker",
        "compose",
        "run",
        "--rm",
        "--no-deps",
        "--volume",
        f"{(PROJECT_ROOT / 'backend').resolve()}:/app/backend",
    ]
    if inject_document_id:
        command.extend(
            [
                "-e",
                f"TASK_WORKER_INJECT_FAIL_ON_FIRST_ATTEMPT_DOCUMENT_IDS={inject_document_id}",
            ]
        )
    command.extend(
        [
            "backend-worker",
            "python",
            "-m",
            "app.workers.task_worker",
            "--once",
            "--idle-timeout-seconds",
            str(DEFAULT_WORKER_IDLE_TIMEOUT_SECONDS),
        ]
    )
    return command


def _run_worker_once(*, inject_document_id: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _worker_run_command(inject_document_id=inject_document_id),
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )


def _start_worker_once(*, inject_document_id: str | None = None) -> subprocess.Popen[str]:
    return subprocess.Popen(
        _worker_run_command(inject_document_id=inject_document_id),
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


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


def _collect_finished_process_output(process: subprocess.Popen[str]) -> str:
    output, _ = process.communicate()
    return output


def _wait_for_task_status(
    api_base_url: str,
    task_id: str,
    *,
    acceptable_statuses: set[str],
    timeout_seconds: int,
    progress_label: str | None = None,
    started_at: float | None = None,
    snapshot_interval_seconds: int = 20,
) -> tuple[dict, list[dict]]:
    deadline = time.time() + timeout_seconds
    last_task: dict | None = None
    last_events: list[dict] = []
    next_snapshot_at = time.time() + snapshot_interval_seconds
    while time.time() < deadline:
        last_task = _get_task(api_base_url, task_id)
        last_events = _get_task_events(api_base_url, task_id)
        if str(last_task["status"]) in acceptable_statuses:
            return last_task, last_events
        if progress_label and started_at is not None and time.time() >= next_snapshot_at:
            _log_progress(
                f"{progress_label} status={last_task['status']} attempt_count={last_task['attempt_count']} locked_by={last_task['locked_by']}",
                started_at=started_at,
            )
            next_snapshot_at = time.time() + snapshot_interval_seconds
        time.sleep(2)
    status = None if last_task is None else last_task.get("status")
    raise RuntimeError(f"timed out waiting for task {task_id}, last status={status}")


def _wait_for_retry_pending(api_base_url: str, task_id: str, *, timeout_seconds: int) -> tuple[dict, list[dict]]:
    deadline = time.time() + timeout_seconds
    last_task: dict | None = None
    last_events: list[dict] = []
    while time.time() < deadline:
        last_task = _get_task(api_base_url, task_id)
        last_events = _get_task_events(api_base_url, task_id)
        retry_event_count = sum(1 for item in last_events if item["event_type"] == "retrying")
        if str(last_task["status"]) == "pending" and retry_event_count >= 1:
            return last_task, last_events
        time.sleep(2)
    status = None if last_task is None else last_task.get("status")
    raise RuntimeError(f"timed out waiting for retried pending task {task_id}, last status={status}")


def _wait_for_tasks_to_finish(
    api_base_url: str,
    *,
    task_ids: list[str],
    watched_processes: dict[str, subprocess.Popen[str]],
    timeout_seconds: int,
    started_at: float,
) -> tuple[dict[str, dict], dict[str, list[dict]], dict[str, str]]:
    deadline = time.time() + timeout_seconds
    process_outputs: dict[str, str] = {}
    while time.time() < deadline:
        tasks = {task_id: _get_task(api_base_url, task_id) for task_id in task_ids}
        events = {task_id: _get_task_events(api_base_url, task_id) for task_id in task_ids}
        unfinished = {
            task_id
            for task_id, task in tasks.items()
            if str(task["status"]) not in {"completed", "failed"}
        }
        if not unfinished:
            return tasks, events, process_outputs

        exited_process_names: list[str] = []
        for process_name, process in watched_processes.items():
            if process_name in process_outputs:
                continue
            if process.poll() is None:
                continue
            output = _collect_finished_process_output(process)
            process_outputs[process_name] = output
            exited_process_names.append(process_name)
            if process.returncode != 0:
                raise RuntimeError(
                    f"{process_name} exited with code {process.returncode} before tasks finished:\n{output}"
                )

        if exited_process_names:
            unfinished_without_claim = [
                task_id
                for task_id in unfinished
                if not any(item["event_type"] == "claimed" for item in events[task_id])
            ]
            if unfinished_without_claim:
                outputs = {
                    process_name: process_outputs[process_name][-2000:]
                    for process_name in exited_process_names
                }
                raise RuntimeError(
                    "concurrency smoke workers exited before pending tasks were claimed: "
                    f"tasks={unfinished_without_claim}, outputs={outputs}"
                )

        for task_id in unfinished:
            task = tasks[task_id]
            _log_progress(
                f"task {task_id} status={task['status']} attempt_count={task['attempt_count']} locked_by={task['locked_by']}",
                started_at=started_at,
            )
        time.sleep(10)

    raise RuntimeError(f"timed out waiting for tasks to finish: {task_ids}")


def _wait_for_claim(api_base_url: str, task_id: str, *, timeout_seconds: int) -> tuple[dict, list[dict]]:
    deadline = time.time() + timeout_seconds
    last_task: dict | None = None
    last_events: list[dict] = []
    while time.time() < deadline:
        last_task = _get_task(api_base_url, task_id)
        last_events = _get_task_events(api_base_url, task_id)
        if any(item["event_type"] == "claimed" for item in last_events):
            return last_task, last_events
        time.sleep(2)
    status = None if last_task is None else last_task.get("status")
    raise RuntimeError(f"timed out waiting for claimed event on task {task_id}, last status={status}")


def _create_task_fixture(api_base_url: str, *, sample_file: Path, name_prefix: str) -> TaskFixture:
    suffix = uuid4().hex[:6]
    kb_payload = {
        "name": f"{name_prefix} {suffix}",
        "description": f"{name_prefix.lower()} docker smoke",
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
    return TaskFixture(
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        task_id=task_id,
    )


def _extract_claim_worker_ids(events: list[dict]) -> list[str]:
    worker_ids: list[str] = []
    for item in events:
        if item["event_type"] != "claimed":
            continue
        payload = item.get("payload") or {}
        worker_id = payload.get("worker_id")
        if isinstance(worker_id, str) and worker_id:
            worker_ids.append(worker_id)
    return worker_ids


def run_smoke(
    *,
    api_base_url: str,
    sample_file: Path,
    keep_knowledge_bases: bool,
) -> ConcurrencySmokeResult:
    started_at = time.time()
    _ensure_api_ready(api_base_url)
    worker_stopped = False
    knowledge_base_ids: list[str] = []
    worker_a: subprocess.Popen[str] | None = None
    worker_b: subprocess.Popen[str] | None = None
    process_outputs: dict[str, str] = {}
    try:
        _log_progress("stopping resident backend-worker", started_at=started_at)
        _docker_compose("stop", "backend-worker")
        worker_stopped = True

        _log_progress("creating retried task fixture", started_at=started_at)
        retried_fixture = _create_task_fixture(
            api_base_url,
            sample_file=sample_file,
            name_prefix="Docker Concurrent Retry",
        )
        _log_progress(
            f"created retried fixture kb={retried_fixture.knowledge_base_id} task={retried_fixture.task_id}",
            started_at=started_at,
        )
        knowledge_base_ids.append(retried_fixture.knowledge_base_id)

        _log_progress("running injected-failure worker for retried task", started_at=started_at)
        _run_worker_once(inject_document_id=retried_fixture.document_id)
        _, retried_events = _wait_for_retry_pending(
            api_base_url,
            retried_fixture.task_id,
            timeout_seconds=60,
        )
        _log_progress("retried task returned to pending after injected failure", started_at=started_at)

        _log_progress("creating companion task fixture", started_at=started_at)
        companion_fixture = _create_task_fixture(
            api_base_url,
            sample_file=sample_file,
            name_prefix="Docker Concurrent Companion",
        )
        _log_progress(
            f"created companion fixture kb={companion_fixture.knowledge_base_id} task={companion_fixture.task_id}",
            started_at=started_at,
        )
        knowledge_base_ids.append(companion_fixture.knowledge_base_id)

        _log_progress("starting worker A", started_at=started_at)
        worker_a = _start_worker_once()
        _log_progress("starting worker B", started_at=started_at)
        worker_b = _start_worker_once()

        _log_progress("waiting for both tasks to finish", started_at=started_at)
        tasks, events, process_outputs = _wait_for_tasks_to_finish(
            api_base_url,
            task_ids=[retried_fixture.task_id, companion_fixture.task_id],
            watched_processes={"worker_a": worker_a, "worker_b": worker_b},
            timeout_seconds=DEFAULT_CONCURRENCY_FINAL_TIMEOUT_SECONDS,
            started_at=started_at,
        )
        companion_task = tasks[companion_fixture.task_id]
        companion_events = events[companion_fixture.task_id]
        retried_task = tasks[retried_fixture.task_id]
        retried_final_events = events[retried_fixture.task_id]
        _log_progress(
            f"companion task reached final status={companion_task['status']}",
            started_at=started_at,
        )
        _log_progress(
            f"retried task reached final status={retried_task['status']}",
            started_at=started_at,
        )

        output_a = process_outputs.get("worker_a")
        if output_a is None:
            output_a = _wait_process(worker_a, timeout_seconds=DEFAULT_CONCURRENCY_WORKER_EXIT_TIMEOUT_SECONDS)
        output_b = process_outputs.get("worker_b")
        if output_b is None:
            output_b = _wait_process(worker_b, timeout_seconds=DEFAULT_CONCURRENCY_WORKER_EXIT_TIMEOUT_SECONDS)
        if not output_a and not output_b:
            raise RuntimeError("expected worker output from at least one concurrent worker")

        if str(companion_task["status"]) != "completed":
            raise RuntimeError(f"companion task did not complete successfully: {companion_task['status']}")
        if str(retried_task["status"]) != "completed":
            raise RuntimeError(f"retried task did not complete successfully: {retried_task['status']}")

        retry_event_count = sum(1 for item in retried_final_events if item["event_type"] == "retrying")
        if retry_event_count < 1:
            raise RuntimeError("retried task did not emit a retrying event")

        retried_claim_worker_ids = _extract_claim_worker_ids(retried_final_events)
        companion_claim_worker_ids = _extract_claim_worker_ids(companion_events)
        if not retried_claim_worker_ids:
            raise RuntimeError("retried task did not emit a claimed event")
        if not companion_claim_worker_ids:
            raise RuntimeError("companion task did not emit a claimed event")
        if len(set(retried_claim_worker_ids + companion_claim_worker_ids)) < 2:
            raise RuntimeError("concurrency smoke expected two distinct worker claims across both tasks")

        _log_progress("concurrency smoke completed", started_at=started_at)
        return ConcurrencySmokeResult(
            retried_task_id=retried_fixture.task_id,
            companion_task_id=companion_fixture.task_id,
            retried_task_retry_event_count=retry_event_count,
            retried_task_final_status=str(retried_task["status"]),
            companion_task_final_status=str(companion_task["status"]),
            retried_task_claim_worker_ids=retried_claim_worker_ids,
            companion_task_claim_worker_ids=companion_claim_worker_ids,
        )
    finally:
        _log_progress("cleaning up concurrency smoke resources", started_at=started_at)
        for process in (worker_a, worker_b):
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()
        if worker_stopped:
            _docker_compose("start", "backend-worker")
        if not keep_knowledge_bases:
            for knowledge_base_id in knowledge_base_ids:
                try:
                    _ensure_api_ready(api_base_url)
                    _json_request(f"{api_base_url}/knowledge-bases/{knowledge_base_id}", method="DELETE")
                except (urllib.error.HTTPError, TimeoutError, OSError, RuntimeError):
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Dockerized multi-worker Redis queue concurrency smoke test.")
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
    args = parser.parse_args()

    sample_file = args.sample_file.resolve()
    if not sample_file.exists():
        raise FileNotFoundError(f"sample file not found: {sample_file}")

    result = run_smoke(
        api_base_url=args.api_base_url.rstrip("/"),
        sample_file=sample_file,
        keep_knowledge_bases=args.keep_knowledge_bases,
    )
    print(
        json.dumps(
            {
                "retried_task_id": result.retried_task_id,
                "companion_task_id": result.companion_task_id,
                "retried_task_retry_event_count": result.retried_task_retry_event_count,
                "retried_task_final_status": result.retried_task_final_status,
                "companion_task_final_status": result.companion_task_final_status,
                "retried_task_claim_worker_ids": result.retried_task_claim_worker_ids,
                "companion_task_claim_worker_ids": result.companion_task_claim_worker_ids,
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
