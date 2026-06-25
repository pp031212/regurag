SET @schema_name := DATABASE();

SET @has_default_kb_column := (
  SELECT COUNT(*)
  FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = @schema_name
    AND TABLE_NAME = 'conversations'
    AND COLUMN_NAME = 'default_knowledge_base_id'
);

SET @sql := IF(
  @has_default_kb_column = 0,
  'ALTER TABLE `conversations` ADD COLUMN `default_knowledge_base_id` VARCHAR(64) NULL AFTER `id`',
  'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

UPDATE `conversations`
SET `default_knowledge_base_id` = `knowledge_base_id`
WHERE `default_knowledge_base_id` IS NULL;

SET @conversation_fk_name := (
  SELECT CONSTRAINT_NAME
  FROM information_schema.KEY_COLUMN_USAGE
  WHERE TABLE_SCHEMA = @schema_name
    AND TABLE_NAME = 'conversations'
    AND COLUMN_NAME = 'knowledge_base_id'
    AND REFERENCED_TABLE_NAME = 'knowledge_bases'
  LIMIT 1
);

SET @sql := IF(
  @conversation_fk_name IS NOT NULL,
  CONCAT('ALTER TABLE `conversations` DROP FOREIGN KEY `', @conversation_fk_name, '`'),
  'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @has_old_conversation_index := (
  SELECT COUNT(*)
  FROM information_schema.STATISTICS
  WHERE TABLE_SCHEMA = @schema_name
    AND TABLE_NAME = 'conversations'
    AND INDEX_NAME = 'ix_conversations_knowledge_base_id'
);

SET @sql := IF(
  @has_old_conversation_index > 0,
  'ALTER TABLE `conversations` DROP INDEX `ix_conversations_knowledge_base_id`',
  'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @has_new_conversation_index := (
  SELECT COUNT(*)
  FROM information_schema.STATISTICS
  WHERE TABLE_SCHEMA = @schema_name
    AND TABLE_NAME = 'conversations'
    AND INDEX_NAME = 'idx_conversations_default_knowledge_base_id'
);

SET @sql := IF(
  @has_new_conversation_index = 0,
  'ALTER TABLE `conversations` ADD INDEX `idx_conversations_default_knowledge_base_id` (`default_knowledge_base_id`)',
  'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @has_new_conversation_fk := (
  SELECT COUNT(*)
  FROM information_schema.KEY_COLUMN_USAGE
  WHERE TABLE_SCHEMA = @schema_name
    AND TABLE_NAME = 'conversations'
    AND COLUMN_NAME = 'default_knowledge_base_id'
    AND REFERENCED_TABLE_NAME = 'knowledge_bases'
);

SET @sql := IF(
  @has_new_conversation_fk = 0,
  'ALTER TABLE `conversations` ADD CONSTRAINT `fk_conversations_default_knowledge_base_id` FOREIGN KEY (`default_knowledge_base_id`) REFERENCES `knowledge_bases` (`id`) ON DELETE SET NULL',
  'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

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

ALTER TABLE `messages`
  MODIFY COLUMN `knowledge_base_id` VARCHAR(64) NULL;

SET @has_new_message_fk := (
  SELECT COUNT(*)
  FROM information_schema.KEY_COLUMN_USAGE
  WHERE TABLE_SCHEMA = @schema_name
    AND TABLE_NAME = 'messages'
    AND COLUMN_NAME = 'knowledge_base_id'
    AND REFERENCED_TABLE_NAME = 'knowledge_bases'
);

SET @sql := IF(
  @has_new_message_fk = 0,
  'ALTER TABLE `messages` ADD CONSTRAINT `fk_messages_knowledge_base_id` FOREIGN KEY (`knowledge_base_id`) REFERENCES `knowledge_bases` (`id`) ON DELETE SET NULL',
  'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @has_old_kb_column := (
  SELECT COUNT(*)
  FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = @schema_name
    AND TABLE_NAME = 'conversations'
    AND COLUMN_NAME = 'knowledge_base_id'
);

SET @sql := IF(
  @has_old_kb_column > 0,
  'ALTER TABLE `conversations` DROP COLUMN `knowledge_base_id`',
  'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
