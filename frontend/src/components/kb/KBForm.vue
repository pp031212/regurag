<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue'
import { useKBStore } from '../../stores/kb'

const props = defineProps<{
  show: boolean
}>()

const emit = defineEmits(['close', 'success'])
const kbStore = useKBStore()

const loading = ref(false)
const error = ref<string | null>(null)
const overlayPointerDown = ref(false)
const domainOptions = computed(() => {
  if (kbStore.domainOptions.length > 0) {
    return kbStore.domainOptions
  }
  return [
    {
      value: kbStore.defaultDomain,
      label: kbStore.getDomainLabel(kbStore.defaultDomain),
      description: '',
    },
  ]
})

const form = reactive({
  name: '',
  description: '',
  subject: '',
  domain: kbStore.defaultDomain
})

const syncDomainOptions = async (show: boolean) => {
  if (!show) return
  await kbStore.fetchDomainOptions({ silent: true })
  const hasCurrentOption = domainOptions.value.some((option) => option.value === form.domain)
  if (!hasCurrentOption) {
    form.domain = kbStore.defaultDomain
  }
}

watch(
  () => props.show,
  (show) => {
    void syncDomainOptions(show)
  },
  { immediate: true },
)

const handleClose = () => {
  error.value = null
  overlayPointerDown.value = false
  emit('close')
}

const handleOverlayPointerDown = () => {
  overlayPointerDown.value = true
}

const handleOverlayPointerUp = () => {
  if (overlayPointerDown.value) {
    handleClose()
  }
}

const cancelOverlayClose = () => {
  overlayPointerDown.value = false
}

const handleSubmit = async () => {
  if (!form.name || !form.subject) {
    error.value = '请填写名称和主题'
    return
  }

  loading.value = true
  error.value = null
  
  try {
    await kbStore.createKB({
      name: form.name,
      description: form.description,
      subject: form.subject,
      domain: form.domain,
    })
    // Reset form
    form.name = ''
    form.description = ''
    form.subject = ''
    form.domain = kbStore.defaultDomain
    emit('success')
    handleClose()
  } catch (err: any) {
    error.value = err.message || '创建失败'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div
    v-if="show"
    class="modal-overlay"
    @mousedown.self="handleOverlayPointerDown"
    @mouseup.self="handleOverlayPointerUp"
  >
    <div class="modal-container" @mousedown="cancelOverlayClose">
      <div class="modal-header">
        <h3>创建新知识库</h3>
        <button class="close-btn" @click="handleClose">&times;</button>
      </div>
      
      <div class="modal-body">
        <div v-if="error" class="error-alert">{{ error }}</div>
        
        <div class="form-group">
          <label>知识库名称 *</label>
          <input v-model="form.name" type="text" placeholder="例如：财务制度库" :disabled="loading" />
        </div>
        
        <div class="form-group">
          <label>业务主题 *</label>
          <input v-model="form.subject" type="text" placeholder="例如：财务制度" :disabled="loading" />
          <p class="help-text">主题将作为 RAG 检索的重要参考上下文</p>
        </div>

        <div class="form-group">
          <label>业务域</label>
          <select v-model="form.domain" :disabled="loading">
            <option
              v-for="option in domainOptions"
              :key="option.value"
              :value="option.value"
            >
              {{ option.label }}
            </option>
          </select>
          <p class="help-text">业务域会影响自动路由、弱澄清和后续知识组织方式。</p>
        </div>
        
        <div class="form-group">
          <label>描述</label>
          <textarea v-model="form.description" rows="3" placeholder="简要描述该知识库的用途..." :disabled="loading"></textarea>
        </div>
      </div>
      
      <div class="modal-footer">
        <button class="btn secondary" @click="handleClose" :disabled="loading">取消</button>
        <button class="btn primary" @click="handleSubmit" :disabled="loading">
          {{ loading ? '正在创建...' : '立即创建' }}
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  backdrop-filter: blur(2px);
}

.modal-container {
  background-color: white;
  width: 100%;
  max-width: 500px;
  border-radius: var(--radius);
  box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.1);
  overflow: hidden;
  animation: slideUp 0.3s ease-out;
}

@keyframes slideUp {
  from { transform: translateY(20px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

.modal-header {
  padding: 20px 24px;
  border-bottom: 1px solid var(--border-color);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.modal-header h3 {
  font-size: 1.125rem;
  font-weight: 600;
}

.close-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: var(--text-muted);
}

.modal-body {
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.form-group label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-color);
}

input, textarea, select {
  padding: 10px 12px;
  border: 1px solid var(--border-color);
  border-radius: var(--radius);
  font-family: inherit;
  font-size: 0.95rem;
  outline: none;
  transition: border-color 0.2s;
}

input:focus, textarea:focus, select:focus {
  border-color: var(--primary-color);
}

.help-text {
  font-size: 0.75rem;
  color: var(--text-muted);
}

.error-alert {
  padding: 10px 12px;
  background-color: #fee2e2;
  color: #991b1b;
  border-radius: var(--radius);
  font-size: 0.875rem;
}

.modal-footer {
  padding: 16px 24px;
  background-color: #f8fafc;
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

.btn {
  padding: 10px 20px;
  border-radius: var(--radius);
  font-size: 0.95rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn.primary {
  background-color: var(--primary-color);
  color: white;
  border: none;
}

.btn.primary:hover { background-color: var(--primary-hover); }

.btn.secondary {
  background-color: white;
  border: 1px solid var(--border-color);
  color: var(--text-color);
}

.btn.secondary:hover { background-color: #f1f5f9; }
</style>
