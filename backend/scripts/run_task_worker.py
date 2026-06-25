"""启动后台任务 worker。

这是容器和本机手动启动 worker 的最薄入口，实际逻辑在 app.workers.task_worker.main。
"""

from app.workers.task_worker import main


if __name__ == "__main__":
    main()
