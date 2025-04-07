<script setup>
import { inject, ref, nextTick } from 'vue'
import { useLoadingStore } from '@/stores/loading'

const axios = inject('axios')
const loadingStore = useLoadingStore()

const summarize_list = ref([])
//const selected_date = ref('2022-11-28')
const selected_date = ref('2025-01-09')

const firstArg = ref('')
const secondArg = ref('')
const comparisonModal = ref(null)
const showModal = ref(false)

// 対象日の要約リストを取得
const getSummarizeList = async (mode) => {
  if (!selected_date.value) {
    alert('日付いれいや')
    return
  }

  loadingStore.startLoading()

  try {
    let params = {
      date: selected_date.value,
      mode: mode,
    }

    let summarize_res = await axios.post('/disclosure/summarizelist', params)

    summarize_list.value = summarize_res.data
  } catch (error) {
    alert(`call_error! ★summarize=[${error}]`)
  } finally {
    loadingStore.stopLoading()
  }
}

// 比較ダイアログ表示
const dispCompSummarize = (org, summarize) => {
  firstArg.value = org
  secondArg.value = summarize
  showModal.value = true

  nextTick(() => {
    const modal = new window.bootstrap.Modal(comparisonModal.value, {
      backdrop: true,
    })
    modal.show()
  })
}

const closeModal = () => {
  showModal.value = false
  if (comparisonModal.value) {
    const modalInstance = window.bootstrap.Modal.getInstance(comparisonModal.value)
    if (modalInstance) {
      modalInstance.hide()
    }
  }
}
</script>
<template>
  <div class="container mt-5">
    <!-- 日付入力 -->
    <div class="row mt-3">
      <div class="col">
        <label for="datePicker">日付:</label>
        <input type="date" id="datePicker" class="form-control" v-model="selected_date" />
      </div>
    </div>

    <!-- ボタン -->
    <div class="row mt-3">
      <div class="col d-flex justify-content-between">
        <button class="btn btn-primary" @click="getSummarizeList((mode = 0))">全件表示</button>
      </div>
      <div class="col d-flex justify-content-between">
        <button class="btn btn-danger" @click="getSummarizeList((mode = 1))">決算のみ表示</button>
      </div>
      <div class="col d-flex justify-content-between">
        <button class="btn btn-info" @click="getSummarizeList((mode = 2))">決算外取得表示</button>
      </div>
    </div>

    <!-- 一覧 -->
    <div class="row" v-if="!loadingStore.isLoading && summarize_list.length">
      <div class="col table-scroll-wrapper">
        <table class="table table-scroll">
          <!-- ヘッダー部分 -->
          <thead>
            <tr>
              <th>No.</th>
              <th class="summary">要約</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(list_data, index) in summarize_list" :key="index">
              <td>{{ index + 1 }}</td>
              <td class="summary" @click="dispCompSummarize(list_data.Link, list_data.Summarize)">
                {{ list_data.Summarize }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Bootstrap Modal (比較ダイアログ) -->
    <div
      class="modal fade"
      ref="comparisonModal"
      id="comparisonModal"
      tabindex="-1"
      aria-labelledby="comparisonModalLabel"
      aria-hidden="true"
      v-if="showModal"
    >
      <div class="modal-dialog custom-modal-width">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="comparisonModalLabel">引数の比較</h5>
            <button type="button" class="btn-close" @click="closeModal" aria-label="Close"></button>
          </div>
          <div class="modal-body">
            <div class="row">
              <div class="col-6">
                <p><strong>引数1:</strong></p>
                <p class="text-wrap">{{ firstArg }}</p>
              </div>
              <div class="col-6">
                <p><strong>引数2:</strong></p>
                <p class="text-wrap">{{ secondArg }}</p>
              </div>
            </div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" @click="closeModal">閉じる</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
