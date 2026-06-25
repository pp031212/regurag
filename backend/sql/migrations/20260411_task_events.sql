CREATE TABLE IF NOT EXISTS `task_events` (
  `id` VARCHAR(64) NOT NULL,
  `task_id` VARCHAR(64) NOT NULL,
  `event_type` VARCHAR(32) NOT NULL,
  `message` TEXT NOT NULL,
  `payload` JSON NULL,
  `created_at` DATETIME NOT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_task_events_task_id` (`task_id`),
  CONSTRAINT `fk_task_events_task_id`
    FOREIGN KEY (`task_id`) REFERENCES `tasks` (`id`)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
