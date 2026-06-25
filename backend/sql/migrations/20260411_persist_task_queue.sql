ALTER TABLE `tasks`
  ADD COLUMN `task_type` VARCHAR(32) NOT NULL DEFAULT 'ingest' AFTER `knowledge_base_id`,
  ADD COLUMN `attempt_count` INT NOT NULL DEFAULT 0 AFTER `message`,
  ADD COLUMN `last_error` TEXT NULL AFTER `attempt_count`,
  ADD COLUMN `started_at` DATETIME NULL AFTER `last_error`,
  ADD COLUMN `finished_at` DATETIME NULL AFTER `started_at`,
  ADD COLUMN `locked_at` DATETIME NULL AFTER `finished_at`,
  ADD COLUMN `locked_by` VARCHAR(128) NULL AFTER `locked_at`;
