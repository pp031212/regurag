import { nextTick, type Ref } from 'vue'

export function useChatScroll(containerRef: Ref<HTMLElement | null>) {
  const scrollToBottom = async () => {
    await nextTick()
    if (containerRef.value) {
      containerRef.value.scrollTop = containerRef.value.scrollHeight
    }
  }

  return {
    scrollToBottom
  }
}
