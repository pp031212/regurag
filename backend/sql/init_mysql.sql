CREATE DATABASE IF NOT EXISTS `regurag`
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE `regurag`;

CREATE TABLE IF NOT EXISTS `knowledge_bases` (
  `id` VARCHAR(64) NOT NULL,
  `name` VARCHAR(100) NOT NULL,
  `description` TEXT NOT NULL,
  `subject` VARCHAR(100) NOT NULL,
  `domain` VARCHAR(64) NOT NULL DEFAULT 'general',
  `status` VARCHAR(32) NOT NULL,
  `created_at` DATETIME NOT NULL,
  `updated_at` DATETIME NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `documents` (
  `id` VARCHAR(64) NOT NULL,
  `knowledge_base_id` VARCHAR(64) NOT NULL,
  `filename` VARCHAR(255) NOT NULL,
  `content_type` VARCHAR(100) NOT NULL,
  `file_size` BIGINT NOT NULL DEFAULT 0,
  `content_hash` VARCHAR(64) NOT NULL DEFAULT '',
  `file_path` TEXT NOT NULL,
  `status` VARCHAR(32) NOT NULL,
  `created_at` DATETIME NOT NULL,
  `updated_at` DATETIME NOT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_documents_knowledge_base_id` (`knowledge_base_id`),
  KEY `idx_documents_content_hash` (`content_hash`),
  CONSTRAINT `fk_documents_knowledge_base_id`
    FOREIGN KEY (`knowledge_base_id`) REFERENCES `knowledge_bases` (`id`)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `tasks` (
  `id` VARCHAR(64) NOT NULL,
  `knowledge_base_id` VARCHAR(64) NOT NULL,
  `task_type` VARCHAR(32) NOT NULL DEFAULT 'ingest',
  `document_ids` JSON NOT NULL,
  `status` VARCHAR(32) NOT NULL,
  `message` TEXT NOT NULL,
  `attempt_count` INT NOT NULL DEFAULT 0,
  `last_error` TEXT NULL,
  `started_at` DATETIME NULL,
  `finished_at` DATETIME NULL,
  `locked_at` DATETIME NULL,
  `locked_by` VARCHAR(128) NULL,
  `created_at` DATETIME NOT NULL,
  `updated_at` DATETIME NOT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_tasks_knowledge_base_id` (`knowledge_base_id`),
  CONSTRAINT `fk_tasks_knowledge_base_id`
    FOREIGN KEY (`knowledge_base_id`) REFERENCES `knowledge_bases` (`id`)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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

CREATE TABLE IF NOT EXISTS `conversations` (
  `id` VARCHAR(64) NOT NULL,
  `default_knowledge_base_id` VARCHAR(64) NULL,
  `title` VARCHAR(200) NOT NULL,
  `created_at` DATETIME NOT NULL,
  `updated_at` DATETIME NOT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_conversations_default_knowledge_base_id` (`default_knowledge_base_id`),
  CONSTRAINT `fk_conversations_default_knowledge_base_id`
    FOREIGN KEY (`default_knowledge_base_id`) REFERENCES `knowledge_bases` (`id`)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `messages` (
  `id` VARCHAR(64) NOT NULL,
  `conversation_id` VARCHAR(64) NOT NULL,
  `sequence` INT NOT NULL,
  `role` VARCHAR(32) NOT NULL,
  `content` TEXT NOT NULL,
  `created_at` DATETIME NOT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_messages_conversation_id` (`conversation_id`),
  CONSTRAINT `fk_messages_conversation_id`
    FOREIGN KEY (`conversation_id`) REFERENCES `conversations` (`id`)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
