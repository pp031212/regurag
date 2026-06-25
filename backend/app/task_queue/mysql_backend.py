from .sql_backend import SqlTaskQueueBackend, get_task_queue_backend


# Compatibility alias kept to avoid breaking existing imports while the task queue
# abstraction is being renamed away from a MySQL-specific label.
MySQLTaskQueueBackend = SqlTaskQueueBackend
