<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import axios from 'axios'

const status = ref<'online' | 'offline' | 'checking'>('offline')
let interval: any = null

const checkHealth = async () => {
  try {
    const res = await axios.get('/api/v1/health')
    if (res.data.status === 'ok') {
      status.value = 'online'
    } else {
      status.value = 'offline'
    }
  } catch (err) {
    status.value = 'offline'
  }
}

onMounted(() => {
  checkHealth()
  interval = setInterval(checkHealth, 10000) // Check every 10s
})

onUnmounted(() => {
  if (interval) clearInterval(interval)
})
</script>

<template>
  <div class="health-status">
    <span class="dot" :class="status"></span>
    <span class="text">{{ status === 'online' ? '后端已连接' : status === 'offline' ? '后端未连接' : '检查中...' }}</span>
  </div>
</template>

<style scoped>
.health-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 0;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #94a3b8;
}

.dot.online { background-color: #22c55e; box-shadow: 0 0 8px rgba(34, 197, 94, 0.4); }
.dot.offline { background-color: #ef4444; }

.text {
  font-size: 0.75rem;
  color: #94a3b8;
}
</style>
