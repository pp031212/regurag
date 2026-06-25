"""后台入库 worker。

worker 从任务队列里领取 pending 或租约过期的 running 任务，调用 IngestService
执行入库，并负责失败重试和最终失败标记。它可以作为 Docker 后台服务常驻运行，
也可以用 --once 做 smoke test。
"""

import argparse
import logging
import os
import socket
import time
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

from ..core.config import get_settings
from ..services.ingest_service import IngestService
from ..task_queue import TaskQueueBackend

logger = logging.getLogger(__name__)


def build_worker_id() -> str:
    """生成便于日志定位的 worker id。"""
    hostname = socket.gethostname() or "worker"
    return f"{hostname}:{os.getpid()}:{uuid4().hex[:8]}"


class TaskWorker:
    """轮询任务队列并执行入库任务的后台进程。"""

    def __init__(
        self,
        *,
        worker_id: str | None = None,
        ingest_service: IngestService | None = None,
        task_queue: TaskQueueBackend | None = None,
        poll_interval_seconds: float | None = None,
    ) -> None:
        settings = get_settings()
        self.ingest_service = ingest_service or IngestService(task_queue=task_queue)
        self.worker_id = worker_id or build_worker_id()
        self.poll_interval_seconds = poll_interval_seconds or settings.task_worker_poll_interval_seconds
        service_settings = getattr(self.ingest_service, "settings", None) or settings
        self.max_attempts = service_settings.task_worker_max_attempts
        self.inject_fail_on_first_attempt_document_ids = {
            str(item)
            for item in getattr(
                service_settings,
                "parsed_task_worker_inject_fail_on_first_attempt_document_ids",
                [],
            )
            if str(item).strip()
        }

    def run_once(self) -> bool:
        """尝试处理一个任务；没有任务时返回 False。"""
        task = self.ingest_service.claim_next_task(self.worker_id)
        if task is None:
            return False

        started = perf_counter()
        logger.info(
            "task_worker_claimed worker_id=%s task_id=%s task_type=%s attempt=%s",
            self.worker_id,
            task["id"],
            task.get("task_type"),
            task.get("attempt_count"),
        )
        try:
            # 注入失败只用于验证重试/恢复逻辑，正常环境不配置该开关。
            self._maybe_raise_injected_failure(task)
            result = self.ingest_service.run_ingest_task(task["id"])
            logger.info(
                "task_worker_completed worker_id=%s task_id=%s status=%s message=%s elapsed_ms=%s",
                self.worker_id,
                task["id"],
                result.get("status"),
                result.get("message"),
                int((perf_counter() - started) * 1000),
            )
        except Exception as exc:  # pragma: no cover - worker 进程兜底保护
            self._handle_execution_failure(task, exc)
        return True

    def run_forever(self) -> None:
        """常驻模式：无任务时按配置间隔休眠。"""
        logger.info(
            "task_worker_started worker_id=%s poll_interval_seconds=%s max_attempts=%s",
            self.worker_id,
            self.poll_interval_seconds,
            self.max_attempts,
        )
        while True:
            processed = self.run_once()
            if not processed:
                time.sleep(self.poll_interval_seconds)

    def run_until_task_or_timeout(self, *, idle_timeout_seconds: float) -> bool:
        """测试/脚本模式：等待一个任务，超时后退出。"""
        deadline = time.time() + idle_timeout_seconds
        logger.info(
            "task_worker_waiting_for_task worker_id=%s idle_timeout_seconds=%s poll_interval_seconds=%s",
            self.worker_id,
            idle_timeout_seconds,
            self.poll_interval_seconds,
        )
        while time.time() < deadline:
            processed = self.run_once()
            if processed:
                return True
            time.sleep(self.poll_interval_seconds)
        logger.info(
            "task_worker_idle_timeout worker_id=%s idle_timeout_seconds=%s",
            self.worker_id,
            idle_timeout_seconds,
        )
        return False

    def _handle_execution_failure(self, task: dict, exc: Exception) -> None:
        """处理 worker 层异常，并决定是否重新排队。"""
        now = datetime.now(UTC).replace(tzinfo=None)
        attempt_count = int(task.get("attempt_count") or 0)
        if attempt_count < self.max_attempts:
            # 失败但未超过最大尝试次数时，释放锁并放回 pending，等待下一轮认领。
            self.ingest_service.task_queue.create_task_event(
                task["id"],
                event_type="retrying",
                message=f"retry scheduled after worker failure: {exc}",
                payload={
                    "worker_id": self.worker_id,
                    "attempt_count": attempt_count,
                    "max_attempts": self.max_attempts,
                    "error": str(exc),
                },
            )
            self.ingest_service.task_queue.update_task(
                task["id"],
                status="pending",
                message=f"retry scheduled after worker failure: {exc}",
                last_error=str(exc),
                finished_at=None,
                locked_at=None,
                locked_by=None,
            )
            logger.warning(
                "task_worker_retrying worker_id=%s task_id=%s attempt=%s max_attempts=%s error=%s",
                self.worker_id,
                task["id"],
                attempt_count,
                self.max_attempts,
                exc,
            )
            return

        # 达到最大尝试次数后才真正标记 failed，避免短暂错误直接终止任务。
        self.ingest_service.task_queue.create_task_event(
            task["id"],
            event_type="failed",
            message=str(exc),
            payload={
                "worker_id": self.worker_id,
                "attempt_count": attempt_count,
                "max_attempts": self.max_attempts,
                "error": str(exc),
            },
        )
        self.ingest_service.task_queue.update_task(
            task["id"],
            status="failed",
            message=str(exc),
            last_error=str(exc),
            finished_at=now,
            locked_at=None,
            locked_by=None,
        )
        logger.exception(
            "task_worker_failed worker_id=%s task_id=%s attempt=%s max_attempts=%s",
            self.worker_id,
            task["id"],
            attempt_count,
            self.max_attempts,
        )

    def _maybe_raise_injected_failure(self, task: dict) -> None:
        """按配置在第一次尝试时制造失败，用于验证重试链路。"""
        if not self.inject_fail_on_first_attempt_document_ids:
            return

        attempt_count = int(task.get("attempt_count") or 0)
        if attempt_count != 1:
            return

        document_ids = {
            str(item).strip()
            for item in (task.get("document_ids") or [])
            if str(item).strip()
        }
        matched_document_ids = sorted(document_ids & self.inject_fail_on_first_attempt_document_ids)
        if not matched_document_ids:
            return

        message = (
            "injected worker failure on first attempt for documents: "
            + ", ".join(matched_document_ids)
        )
        logger.warning(
            "task_worker_injected_failure worker_id=%s task_id=%s documents=%s",
            self.worker_id,
            task["id"],
            matched_document_ids,
        )
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ReguRAG task worker.")
    parser.add_argument("--once", action="store_true", help="Process at most one task and then exit.")
    parser.add_argument("--worker-id", type=str, default=None, help="Override the generated worker id.")
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=None,
        help="Override the default idle poll interval.",
    )
    parser.add_argument(
        "--idle-timeout-seconds",
        type=float,
        default=None,
        help="When used with --once, keep polling until a task is found or this idle timeout elapses.",
    )
    args = parser.parse_args()

    worker = TaskWorker(
        worker_id=args.worker_id,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    if args.once:
        if args.idle_timeout_seconds is not None and args.idle_timeout_seconds > 0:
            worker.run_until_task_or_timeout(idle_timeout_seconds=args.idle_timeout_seconds)
        else:
            worker.run_once()
        return
    worker.run_forever()


if __name__ == "__main__":
    main()
