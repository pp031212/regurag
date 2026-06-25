<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    open: boolean
    title: string
    message: string
    confirmText?: string
    cancelText?: string
    tone?: 'default' | 'danger'
    loading?: boolean
    showCancel?: boolean
  }>(),
  {
    confirmText: '确定',
    cancelText: '取消',
    tone: 'default',
    loading: false,
    showCancel: true,
  },
)

const emit = defineEmits<{
  close: []
  confirm: []
}>()

const handleClose = () => {
  if (props.loading) {
    return
  }
  emit('close')
}

const handleConfirm = () => {
  if (props.loading) {
    return
  }
  emit('confirm')
}
</script>

<template>
  <Teleport to="body">
    <div v-if="open" class="confirm-overlay" @click.self="handleClose">
      <div class="confirm-card">
        <div class="confirm-card__header">
          <div class="confirm-badge" :class="tone">
            <span>{{ tone === 'danger' ? '!' : 'i' }}</span>
          </div>
          <div class="confirm-copy">
            <h3>{{ title }}</h3>
            <p>{{ message }}</p>
          </div>
        </div>

        <div class="confirm-actions">
          <button v-if="showCancel" class="btn secondary" :disabled="loading" @click="handleClose">
            {{ cancelText }}
          </button>
          <button class="btn" :class="tone === 'danger' ? 'danger' : 'primary'" :disabled="loading" @click="handleConfirm">
            {{ loading ? '处理中...' : confirmText }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.confirm-overlay {
  position: fixed;
  inset: 0;
  z-index: var(--z-modal);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background:
    radial-gradient(circle at top, rgba(37, 99, 235, 0.12), transparent 36%),
    rgba(15, 23, 42, 0.4);
  backdrop-filter: blur(6px);
}

.confirm-card {
  width: min(100%, 460px);
  border: 1px solid rgba(226, 232, 240, 0.9);
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18);
  overflow: hidden;
}

.confirm-card__header {
  padding: 24px;
  display: flex;
  gap: 16px;
  align-items: flex-start;
}

.confirm-badge {
  width: 40px;
  height: 40px;
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 14px;
  font-size: 1rem;
  font-weight: 700;
}

.confirm-badge.default {
  background: linear-gradient(180deg, #dbeafe 0%, #bfdbfe 100%);
  color: #1d4ed8;
}

.confirm-badge.danger {
  background: linear-gradient(180deg, #fee2e2 0%, #fecaca 100%);
  color: #dc2626;
}

.confirm-copy h3 {
  margin: 0;
  font-size: 1.05rem;
  font-weight: 700;
  color: #0f172a;
}

.confirm-copy p {
  margin: 8px 0 0;
  font-size: 0.92rem;
  line-height: 1.65;
  color: #475569;
}

.confirm-actions {
  padding: 16px 24px 24px;
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

.btn {
  min-width: 92px;
  padding: 10px 16px;
  border-radius: 12px;
  font-size: 0.92rem;
  font-weight: 600;
  border: 1px solid transparent;
}

.btn.secondary {
  color: #334155;
  background: white;
  border-color: #cbd5e1;
}

.btn.secondary:hover:not(:disabled) {
  background: #f8fafc;
}

.btn.primary {
  color: white;
  background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%);
}

.btn.primary:hover:not(:disabled) {
  filter: brightness(0.98);
}

.btn.danger {
  color: white;
  background: linear-gradient(180deg, #ef4444 0%, #dc2626 100%);
}

.btn.danger:hover:not(:disabled) {
  filter: brightness(0.98);
}
</style>
