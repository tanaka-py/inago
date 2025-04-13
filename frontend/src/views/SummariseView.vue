<script setup>
import { inject, ref, nextTick, onMounted, computed } from 'vue'
import { useLoadingStore } from '@/stores/loading'

const axios = inject('axios')
const loadingStore = useLoadingStore()

const summarize_list = ref([])
const workdata_list = ref([])
//const selected_date = ref('2022-11-28')
const selected_date = ref('')
const confirm_date = ref('2025-01-09')
const is_work_data = ref(false)
const is_model_data = ref(false)
const is_eval_data = ref(false)

const firstArg = ref('')
const secondArg = ref('')
const comparisonModal = ref(null)
const showModal = ref(false)
const dispIndex = ref(0)

// 学習中の要約リストを取得
const getSummarizeList = async () => {
  loadingStore.startLoading()

  try {
    let summarize_res = await axios.post('/disclosure/summarizelist')

    summarize_list.value = summarize_res.data
    workdata_list.value = []
  } catch (error) {
    alert(`call_error! ★summarize=[${error}]`)
  } finally {
    loadingStore.stopLoading()
  }
}

// 株価予想一覧
const getEvalDataList = async () => {
  loadingStore.startLoading()

  try {
    let params = {
      date: confirm_date.value,
    }
    let work_data_res = await axios.post('/disclosure/evallist', params)

    workdata_list.value = work_data_res.data
    summarize_list.value = []

    reLoad()
  } catch (err) {
    alert(`call_error! ★EvalData=[${err}]`)
  } finally {
    loadingStore.stopLoading()
  }
}

// 予想値確認削除用
const deleteEvalData = async () => {
  loadingStore.startLoading()

  try {
    let res = await axios.post('/disclosure/deleteevaldata')

    if (res.status === 200)
    {
      // 状態再描画
      reLoad()
    }
  }
  catch (err)
  {
    alert(`deleteEvalData:${err}`)
  }
  finally {
    loadingStore.stopLoading()
  }
}


// 学習前データ取得
const getWorkDataList = async () => {
  loadingStore.startLoading()

  try {
    let work_data_res = await axios.post('/disclosure/workdatalist')

    workdata_list.value = work_data_res.data
    summarize_list.value = []
  } catch (err) {
    alert(`call_error! ★WorkData=[${err}]`)
  } finally {
    loadingStore.stopLoading()
  }
}

// 比較ダイアログ表示
const dispCompSummarize = (org, summarize, index) => {
  firstArg.value = org
  secondArg.value = summarize
  showModal.value = true
  dispIndex.value = index + 1

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

const reLoad = async () => {
  loadingStore.startLoading()

  try {
    let res = await axios.get('/disclosure/state')

    if (res.status === 200) {
      // 現在学習中の対象日付
      selected_date.value = res.data.target_date
      // 現在学習中の作業データ有無
      is_work_data.value = res.data.is_work_data
      // 学習中のモデルデータあり
      is_model_data.value = res.data.is_model_data
      // 評価用の作業データあり
      is_eval_data.value = res.data.is_eval_data
    }
  }
  catch (err) {
    alert(`reload:${err}`)
  }
  finally
  {
    loadingStore.stopLoading()
  }
}
const escapeHtml = (text) => {
  const map = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;',
    };
  return text.replace(/[&<>"']/g, m => map[m]);
}

// mounted
onMounted(async () => {
  reLoad()
})

// 左側（元の文）：削除された部分を赤く表示
const diff1 = computed(() => getDiffHtmlForOriginal(firstArg.value, secondArg.value))

// 右側（修正後の文）：追加された部分を青く表示
const diff2 = computed(() => getDiffHtmlForModified(firstArg.value, secondArg.value))


// 削除（左側表示用）
const getDiffHtmlForOriginal = (arg1, arg2) => {
  const dmp = new window.diff_match_patch();
  const diffs = dmp.diff_main(arg1, arg2);
  dmp.diff_cleanupSemantic(diffs);

  let result = '';

  diffs.forEach(([op, data]) => {
    const escaped = escapeHtml(data);
    if (op === 0) {
      result += escaped;
    } else if (op === -1) {
      result += `<span style="background:#ffdddd;">${escaped}</span>`;
    } else if (op === 1) {
      result += '&nbsp;'.repeat(data.length); // 空白で位置合わせ
    }
  });

  return result;
};

// 追加（右側表示用）
const getDiffHtmlForModified = (arg1, arg2) => {
  const dmp = new window.diff_match_patch();
  const diffs = dmp.diff_main(arg1, arg2);
  dmp.diff_cleanupSemantic(diffs);

  let result = '';

  diffs.forEach(([op, data]) => {
    const escaped = escapeHtml(data);
    if (op === 0) {
      result += escaped;
    } else if (op === 1) {
      result += `<span style="background:#add8e6;">${escaped}</span>`;
    } else if (op === -1) {
      result += '&nbsp;'.repeat(data.length); // 空白で位置合わせ
    }
  });

  return result;
};

</script>
<template>
  <div v-if="!loadingStore.isLoading" class="container mt-5">
    <div class="row mt-3 g-3">
      <!-- モデルデータ -->
      <div v-if="is_model_data" class="col-lg-6">
        <div class="alert alert-info border rounded-3 shadow-sm py-2 px-3 d-flex align-items-center">
          <i class="bi bi-robot me-2"></i> 学習中モデルデータありだお！
        </div>
      </div>

      <!-- 作業データあり -->
      <div v-if="is_work_data" class="col-lg-6">
        <div class="alert alert-success border rounded-3 shadow-sm py-2 px-3 d-flex align-items-center">
          <i class="bi bi-check-circle-fill me-2"></i>
          現在、<strong class="ms-1">学習前作業データ</strong>で確認中なんだが？ｷﾘｯ
        </div>
      </div>

      <!-- 作業データなし -->
      <div v-else class="col-lg-6">
        <div class="alert alert-danger border rounded-3 shadow-sm py-2 px-3 d-flex align-items-center">
          <i class="bi bi-x-circle-fill me-2"></i>
          現在、<strong class="ms-1">学習前作業データ</strong>が出来ていませんぞ(´；ω；｀)
        </div>
      </div>
    </div>

    <!-- 日付入力 -->
    <div class="row mt-3">
      <div class="col">
        <label for="datePicker">学習中対象日:</label>
        <input type="date" id="datePicker" class="form-control" :value="selected_date" disabled />
      </div>
      <div class="col">
        <label for="datePicker">予測確認対象日:</label>
        <input type="date" id="datePicker" class="form-control" v-model="confirm_date" />
      </div>
    </div>

    <!-- ボタン -->
    <div class="row mt-3">
      <div class="col d-flex justify-content-between">
        <button class="btn btn-primary" @click="getWorkDataList" :disabled="!is_work_data">学習前データ確認</button>
      </div>
      <div class="col d-flex justify-content-between">
        <button class="btn btn-danger" @click="getEvalDataList" :disabled="!is_model_data">予想値確認</button>
      </div>
      <div class="col d-flex justify-content-between">
        <button class="btn btn-warning" @click="deleteEvalData" :disabled="!is_model_data || !is_eval_data">予想値元クリア</button>
      </div>
      <div class="col d-flex justify-content-between">
        <button class="btn btn-info" @click="getSummarizeList">要約比較確認</button>
      </div>
    </div>

    <!-- 学習関連一覧 -->
    <div class="row" v-if="!loadingStore.isLoading && workdata_list.length">
      <div
        class="col table-scroll-wrapper"
        style="max-height: calc(100vh - 100px); overflow-y: auto"
      >
        <div
          v-for="(list_data, index) in workdata_list"
          :key="index"
          class="summary-block p-4 border mb-4 rounded bg-white shadow"
        >
          <!-- NoとDocument Summary（冒頭だけ） -->
          <div class="font-bold mb-2">No. {{ index + 1 }}</div>
          <div class="mb-4">
            <strong>要約：</strong>
            {{ list_data.document_summaries.slice(0, 30) }}...
            <a
              href="#"
              @click.prevent="dispCompSummarize(list_data.Link, list_data.document_summaries)"
              >続きを読む</a
            >
          </div>

          <!-- Features -->
          <div class="mb-4">
            <strong>📊 特徴 (Features)</strong>
            <ul class="targets-grid">
              <li v-for="(val, key) in list_data.features" :key="key">
                <strong>{{ key }}</strong
                >: {{ val }}
              </li>
            </ul>
          </div>

          <!-- Targets -->
          <div>
            <strong>🎯 目標値 (Targets)</strong>
            <ul class="targets-grid">
              <li v-for="(val, key) in list_data.targets" :key="key">
                {{ key }}:
                <span :style="{ color: val > 0 ? 'red' : val < 0 ? 'blue' : 'inherit' }">{{
                  val
                }}</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <!-- 要約比較確認一覧 -->
    <div class="row" v-if="!loadingStore.isLoading && summarize_list.length">
      <div class="col table-scroll-wrapper">
        <table class="table table-scroll table-dark table-striped table-hover">
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
              <td class="summary" @click="dispCompSummarize(list_data.Link, list_data.Summarize, index)">
                {{ list_data.Summarize.slice(0,200) }}...
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
            <h5 class="modal-title" id="comparisonModalLabel">引数の比較 No.{{ dispIndex }}</h5>
            <button type="button" class="btn-close" @click="closeModal" aria-label="Close"></button>
          </div>
          <div class="modal-body">
            <div class="row">
              <div class="col-6">
                <p><strong>引数1:</strong></p>
                <p class="text-wrap diff-view" v-html="diff1"></p>
              </div>
              <div class="col-6">
                <p><strong>引数2:</strong></p>
                <p class="text-wrap diff-view" v-html="diff2"></p>
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
