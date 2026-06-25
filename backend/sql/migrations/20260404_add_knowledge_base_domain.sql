USE `regurag`;

SET @has_domain_column := (
  SELECT COUNT(*)
  FROM information_schema.columns
  WHERE table_schema = DATABASE()
    AND table_name = 'knowledge_bases'
    AND column_name = 'domain'
);

SET @add_domain_column_sql := IF(
  @has_domain_column = 0,
  'ALTER TABLE `knowledge_bases` ADD COLUMN `domain` VARCHAR(64) NOT NULL DEFAULT ''general'' AFTER `subject`',
  'SELECT 1'
);
PREPARE add_domain_column_stmt FROM @add_domain_column_sql;
EXECUTE add_domain_column_stmt;
DEALLOCATE PREPARE add_domain_column_stmt;

UPDATE `knowledge_bases`
SET `domain` = CASE
  WHEN (`subject` LIKE '%劳动%' OR `name` LIKE '%劳动%' OR `name` LIKE '%合同%') THEN 'labor_law'
  WHEN (`subject` LIKE '%和鸣%' OR `subject` LIKE '%规章%' OR `subject` LIKE '%学生%' OR `name` LIKE '%和鸣%' OR `name` LIKE '%规章%') THEN 'training_management'
  ELSE 'general'
END
WHERE `domain` = 'general' OR `domain` IS NULL OR `domain` = '';
