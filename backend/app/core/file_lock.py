import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

if os.name == "nt":  # pragma: no cover - platform specific
    import msvcrt
else:  # pragma: no cover - platform specific
    import fcntl


@contextmanager
def advisory_file_lock(
    path: Path,
    *,
    timeout_seconds: float = 120.0,
    poll_interval_seconds: float = 0.1,
) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        _acquire_file_lock(
            handle,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        try:
            yield
        finally:
            _release_file_lock(handle)


def _acquire_file_lock(handle, *, timeout_seconds: float, poll_interval_seconds: float) -> None:
    if os.name == "nt":  # pragma: no cover - Windows dev fallback
        return
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # pragma: no cover - platform specific
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for lock: {handle.name}")
            time.sleep(poll_interval_seconds)


def _release_file_lock(handle) -> None:
    if os.name == "nt":  # pragma: no cover - Windows dev fallback
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # pragma: no cover - platform specific
