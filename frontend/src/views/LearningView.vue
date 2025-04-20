<script setup>
import { inject, ref, nextTick, onMounted } from 'vue'
import { useLoadingStore } from '@/stores/loading'

// axiosをinject
const axios = inject('axios')
// ローディングを準備
const loadingStore = useLoadingStore()

const inago_list = ref([])

const selectedDate = ref('')
const is_work_data = ref(false)
const is_model_data = ref(false)
const tooltipRefs = ref([]) // すべてのTooltipの参照を格納

// 対象日の学習を行う
const callLearning = async (work_load) => {
  loadingStore.startLoading()

  nextTick(async () => {
    try {
      let params = {
        date: selectedDate.value,
        work_load: work_load,
      }
      let res = await axios.post('/disclosure/learning', params)

      if (res.status === 200) {
        // 現在の状態を再描画
        reLoading()
      }
    } catch (error) {
      alert(`call_error! ★tdnet detail=[${error}]`)
    } finally {
      loadingStore.stopLoading()
    }
  })
}

// Tdnet開示アップロード
const callTdnetUpload = async () => {

  loadingStore.startLoading()

  nextTick(async () => {
    try {
      let params = {
        date: selectedDate.value.replace(/-/g, ''),
      }
      let tdnet_res = await axios.post('/disclosure/upload', params)

      alert(tdnet_res.data.message)
    } catch (error) {
      alert(`call_error! ★tdnet detail=[${error}]`)
    } finally {
      loadingStore.stopLoading()
    }
  })
}

// 学習前作業データ削除
const callWorkDataClear = async () => {
  loadingStore.startLoading()

  try {
    let res = await axios.post('/disclosure/deleteworkdata')

    if (res.status == 200)
    {//削除正常
      reLoading()
    }
  }
  catch (err) {
    alert(`callWorkDataClear: ${err}`)
  }
  finally {
    loadingStore.stopLoading()
  }
}

// モデルクリア
const callModelDelete = async () => {
  loadingStore.startLoading()

  try {
    let res = await axios.post('/disclosure/deletemlpmodel')

    if (res.status === 200) {
      reLoading()
    }
  }
  catch (err)
  {
    alert(`callModelDelete:${err}`)
  }
  finally {
    loadingStore.stopLoading()
  }
}

// 現在の状態の読み込み
const reLoading = async () => {
  loadingStore.startLoading()

  try {
    let res = await axios.get('/disclosure/state')

    selectedDate.value = res.data.target_date
    is_work_data.value = res.data.is_work_data
    is_model_data.value = res.data.is_model_data
  }
  catch (err) {
    alert(`reLoading:${err}`)
  }
  finally
  {
    loadingStore.stopLoading()
  }
}

// mount
onMounted(async () => {
  reLoading()
})
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
        <label for="datePicker">モデル学習対象日:</label>
        <input type="date" id="datePicker" class="form-control" v-model="selectedDate" disabled />
      </div>
    </div>

    <!-- ボタン -->
    <div class="row mt-3">
      <div class="col d-flex justify-content-between">
        <button class="btn btn-success" @click="callTdnetUpload">Tdnet開示データ収集</button>
        <button class="btn btn-secondary" :disabled="is_work_data" @click="callLearning((work_load = false))">
          データ事前作成
        </button>
        <button class="btn btn-warning" :disabled="!is_work_data" @click="callLearning((work_load = true))">
          学習
        </button>
        <button class="btn btn-primary" :disabled="!is_work_data" @click="callWorkDataClear">作業データクリア</button>
        <button class="btn btn-danger" :disabled="!is_model_data" @click="callModelDelete">モデルクリア</button>
      </div>
    </div>

    <!-- 一覧 -->
    <div class="row" v-if="!loadingStore.isLoading && inago_list.length">
      <div class="col table-scroll-wrapper">
        <table class="table table-scroll">
          <!-- ヘッダー部分 -->
          <thead>
            <tr>
              <th>No.</th>
              <th class="time">時刻</th>
              <th class="code">証券コード</th>
              <th class="name">会社名</th>
              <th class="title">タイトル</th>
              <!-- <th class="link">要約</th> -->
            </tr>
          </thead>
          <tbody>
            <tr v-for="(list_data, index) in FilterList" :key="index">
              <td>{{ index + 1 }}</td>
              <td class="time">{{ list_data.Time }}</td>
              <td class="code">{{ list_data.Code }}</td>
              <td class="name">{{ list_data.Name }}</td>
              <td class="title" :ref="(el) => (tooltipRefs[index] = el)">
                {{ list_data.Title }}
              </td>
              <!-- <td class="link">{{ list_data.Link }}</td> -->
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>
