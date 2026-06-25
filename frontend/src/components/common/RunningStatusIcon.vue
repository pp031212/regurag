<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import {
  AccessibilityNewOutlined,
  AccessibleForwardOutlined,
  DirectionsBikeOutlined,
  DirectionsRunOutlined,
  PoolOutlined,
  RowingOutlined,
} from '@vicons/material'

interface Props {
  visible?: boolean
  speed?: number
  size?: number
  color?: string
}

const props = withDefaults(defineProps<Props>(), {
  visible: true,
  speed: 220,
  size: 20,
  color: 'rgba(59, 130, 246, 0.92)',
})

const icons = [
  AccessibleForwardOutlined,
  AccessibilityNewOutlined,
  DirectionsBikeOutlined,
  DirectionsRunOutlined,
  PoolOutlined,
  RowingOutlined,
]

const index = ref(0)
let timer: number | null = null

const currentIcon = computed(() => icons[index.value])
const iconStyle = computed(() => ({
  width: `${props.size}px`,
  height: `${props.size}px`,
  color: props.color,
}))

const startTimer = () => {
  if (timer !== null || !props.visible) {
    return
  }

  timer = window.setInterval(() => {
    index.value = (index.value + 1) % icons.length
  }, props.speed)
}

const stopTimer = () => {
  if (timer === null) {
    return
  }

  clearInterval(timer)
  timer = null
}

watch(
  () => [props.speed, props.visible] as const,
  () => {
    stopTimer()
    startTimer()
  },
)

onMounted(() => {
  startTimer()
})

onUnmounted(() => {
  stopTimer()
})
</script>

<template>
  <div v-if="visible" class="running-status-icon" aria-label="思考中" role="img">
    <component :is="currentIcon" :style="iconStyle" class="icon" />
  </div>
</template>

<style scoped>
.running-status-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  user-select: none;
}

.icon {
  display: block;
}
</style>
