<script setup>
import { useLoadingStore } from '@/stores/loading'

const loadingStore = useLoadingStore()
</script>

<template>
  <div v-if="loadingStore.isLoading" class="spinner-overlay">
    <div class="spinner"></div>
  </div>
</template>
