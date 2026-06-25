-- PostgreSQL scaffold for ReguRAG metadata/task tables.
-- Current expectation:
-- 1. Create the target database yourself.
-- 2. Connect to that database.
-- 3. Run this file.

CREATE TABLE IF NOT EXISTS knowledge_bases (
  id VARCHAR(64) PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  description TEXT NOT NULL,
  subject VARCHAR(100) NOT NULL,
  domain VARCHAR(64) NOT NULL DEFAULT 'general',
  status VARCHAR(32) NOT NULL,
  created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
  updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
  id VARCHAR(64) PRIMARY KEY,
  knowledge_base_id VARCHAR(64) NOT NULL REFERENCES knowledge_bases (id) ON DELETE CASCADE,
  filename VARCHAR(255) NOT NULL,
  content_type VARCHAR(100) NOT NULL,
  file_size BIGINT NOT NULL DEFAULT 0,
  content_hash VARCHAR(64) NOT NULL DEFAULT '',
  file_path TEXT NOT NULL,
  status VARCHAR(32) NOT NULL,
  created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
  updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_knowledge_base_id ON documents (knowledge_base_id);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents (content_hash);

CREATE TABLE IF NOT EXISTS tasks (
  id VARCHAR(64) PRIMARY KEY,
  knowledge_base_id VARCHAR(64) NOT NULL REFERENCES knowledge_bases (id) ON DELETE CASCADE,
  task_type VARCHAR(32) NOT NULL DEFAULT 'ingest',
  document_ids JSONB NOT NULL,
  status VARCHAR(32) NOT NULL,
  message TEXT NOT NULL,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  last_error TEXT NULL,
  started_at TIMESTAMP WITHOUT TIME ZONE NULL,
  finished_at TIMESTAMP WITHOUT TIME ZONE NULL,
  locked_at TIMESTAMP WITHOUT TIME ZONE NULL,
  locked_by VARCHAR(128) NULL,
  created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
  updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_knowledge_base_id ON tasks (knowledge_base_id);

CREATE TABLE IF NOT EXISTS task_events (
  id VARCHAR(64) PRIMARY KEY,
  task_id VARCHAR(64) NOT NULL REFERENCES tasks (id) ON DELETE CASCADE,
  event_type VARCHAR(32) NOT NULL,
  message TEXT NOT NULL,
  payload JSONB NULL,
  created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_events_task_id ON task_events (task_id);

CREATE TABLE IF NOT EXISTS conversations (
  id VARCHAR(64) PRIMARY KEY,
  default_knowledge_base_id VARCHAR(64) NULL REFERENCES knowledge_bases (id) ON DELETE SET NULL,
  title VARCHAR(200) NOT NULL,
  created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
  updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_default_knowledge_base_id ON conversations (default_knowledge_base_id);

CREATE TABLE IF NOT EXISTS messages (
  id VARCHAR(64) PRIMARY KEY,
  conversation_id VARCHAR(64) NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
  sequence INTEGER NOT NULL,
  role VARCHAR(32) NOT NULL,
  content TEXT NOT NULL,
  created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages (conversation_id);

CREATE TABLE IF NOT EXISTS message_contexts (
  id VARCHAR(64) PRIMARY KEY,
  message_id VARCHAR(64) NOT NULL UNIQUE REFERENCES messages (id) ON DELETE CASCADE,
  knowledge_base_id VARCHAR(64) NULL REFERENCES knowledge_bases (id) ON DELETE SET NULL,
  citations JSONB NULL,
  debug JSONB NULL,
  created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_message_contexts_message_id ON message_contexts (message_id);
CREATE INDEX IF NOT EXISTS idx_message_contexts_knowledge_base_id ON message_contexts (knowledge_base_id);
