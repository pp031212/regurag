SET @schema_name := DATABASE();

CREATE TABLE IF NOT EXISTS `message_contexts` (
  `id` VARCHAR(64) NOT NULL,
  `message_id` VARCHAR(64) NOT NULL,
  `knowledge_base_id` VARCHAR(64) NULL,
  `citations` JSON NULL,
  `debug` JSON NULL,
  `created_at` DATETIME NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_message_contexts_message_id` (`message_id`),
  KEY `idx_message_contexts_message_id` (`message_id`),
  KEY `idx_message_contexts_knowledge_base_id` (`knowledge_base_id`),
  CONSTRAINT `fk_message_contexts_message_id`
    FOREIGN KEY (`message_id`) REFERENCES `messages` (`id`)
    ON DELETE CASCADE,
  CONSTRAINT `fk_message_contexts_knowledge_base_id`
    FOREIGN KEY (`knowledge_base_id`) REFERENCES `knowledge_bases` (`id`)
    ON DELETE SET NULL
);

INSERT INTO `message_contexts` (`id`, `message_id`, `knowledge_base_id`, `citations`, `debug`, `created_at`)
SELECT
  CONCAT('ctx_', SUBSTRING(REPLACE(UUID(), '-', ''), 1, 8)) AS `id`,
  `m`.`id` AS `message_id`,
  `m`.`knowledge_base_id`,
  `m`.`citations`,
  `m`.`debug`,
  `m`.`created_at`
FROM `messages` AS `m`
LEFT JOIN `message_contexts` AS `mc` ON `mc`.`message_id` = `m`.`id`
WHERE `mc`.`message_id` IS NULL
  AND (
    `m`.`knowledge_base_id` IS NOT NULL
    OR `m`.`citations` IS NOT NULL
    OR `m`.`debug` IS NOT NULL
  );

SET @message_fk_name := (
  SELECT CONSTRAINT_NAME
  FROM information_schema.KEY_COLUMN_USAGE
  WHERE TABLE_SCHEMA = @schema_name
    AND TABLE_NAME = 'messages'
    AND COLUMN_NAME = 'knowledge_base_id'
    AND REFERENCED_TABLE_NAME = 'knowledge_bases'
  LIMIT 1
);

SET @sql := IF(
  @message_fk_name IS NOT NULL,
  CONCAT('ALTER TABLE `messages` DROP FOREIGN KEY `', @message_fk_name, '`'),
  'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @has_message_kb_index := (
  SELECT COUNT(*)
  FROM information_schema.STATISTICS
  WHERE TABLE_SCHEMA = @schema_name
    AND TABLE_NAME = 'messages'
    AND INDEX_NAME = 'ix_messages_knowledge_base_id'
);

SET @sql := IF(
  @has_message_kb_index > 0,
  'ALTER TABLE `messages` DROP INDEX `ix_messages_knowledge_base_id`',
  'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @has_message_kb_column := (
  SELECT COUNT(*)
  FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = @schema_name
    AND TABLE_NAME = 'messages'
    AND COLUMN_NAME = 'knowledge_base_id'
);

SET @sql := IF(
  @has_message_kb_column > 0,
  'ALTER TABLE `messages` DROP COLUMN `knowledge_base_id`',
  'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @has_message_citations_column := (
  SELECT COUNT(*)
  FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = @schema_name
    AND TABLE_NAME = 'messages'
    AND COLUMN_NAME = 'citations'
);

SET @sql := IF(
  @has_message_citations_column > 0,
  'ALTER TABLE `messages` DROP COLUMN `citations`',
  'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @has_message_debug_column := (
  SELECT COUNT(*)
  FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = @schema_name
    AND TABLE_NAME = 'messages'
    AND COLUMN_NAME = 'debug'
);

SET @sql := IF(
  @has_message_debug_column > 0,
  'ALTER TABLE `messages` DROP COLUMN `debug`',
  'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
