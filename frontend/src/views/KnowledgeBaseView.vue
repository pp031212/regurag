<script setup lang="ts">
import { ref } from 'vue'
import KBList from '../components/kb/KBList.vue'
import KBForm from '../components/kb/KBForm.vue'

const showForm = ref(false)
</script>

<template>
  <div class="kb-page">
    <section class="section-head">
      <div>
        <span class="section-eyebrow">Knowledge Resources</span>
        <h2>知识库列表</h2>
        <p>这里管理可供聊天使用的知识资源。</p>
      </div>
    </section>

    <section class="list-section">
      <KBList @create="showForm = true" />
    </section>

    <KBForm :show="showForm" @close="showForm = false" @success="() => {}" />
  </div>
</template>

<style scoped>
.kb-page {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.section-head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 20px;
  padding: 20px 24px;
  border-radius: var(--radius);
  background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 100%);
  border: 1px solid #dbeafe;
}

.section-eyebrow {
  display: block;
  margin-bottom: 8px;
  color: var(--primary-color);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.section-head h2 {
  margin: 0 0 8px;
  font-size: 1.45rem;
  color: #0f172a;
}

.section-head p {
  margin: 0;
  color: #475569;
  line-height: 1.7;
}

.list-section {
  min-width: 0;
}

@media (max-width: 720px) {
  .section-head {
    padding: 20px;
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
