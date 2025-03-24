<script setup>
import { inject, ref, nextTick, onMounted } from 'vue'
import { useLoadingStore } from '@/stores/loading'

// axiosをinject
const axios = inject('axios')
// ローディングを準備
const loadingStore = useLoadingStore()

const inago_list = ref([])

const selectedDate = ref(new Date().toISOString().split('T')[0])
const tooltipRefs = ref([]) // すべてのTooltipの参照を格納

// Tdnet開示情報を収集
const callTdnetLearning = async () => {
  if (!selectedDate.value) {
    alert('日付いれいや')
    return
  }

  loadingStore.startLoading()

  nextTick(async () => {
    try {
      let params = {
        date: selectedDate.value.replace(/-/g, ''),
      }
      let tdnet_res = await axios.post('/disclosure/learning', params)

      alert(tdnet_res.data.message)
    } catch (error) {
      alert(`call_error! ★tdnet detail=[${error}]`)
    } finally {
      loadingStore.stopLoading()
    }
  })
}

// Tdnet開示アップロード
const callTdnetUpload = async () => {
  if (!selectedDate.value) {
    alert('日付いれいや')
    return
  }

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

const callPressReleaseLearning = async () => {
  if (!selectedDate.value) {
    alert('日付いれいや')
    return
  }

  loadingStore.startLoading()

  nextTick(async () => {
    try {
      let params = {
        date: selectedDate.value,
      }
      let pressrelease_res = await axios.post('/pressrelease/learning', params)

      alert(pressrelease_res.data.message)
    } catch (error) {
      alert(`call_error! ★pressrelease detail=[${error}]`)
    } finally {
      loadingStore.stopLoading()
    }
  })
}

onMounted(() => {})
</script>

<template>
  <div class="container mt-5">
    <!-- 日付入力 -->
    <div class="row mt-3">
      <div class="col">
        <label for="datePicker">日付:</label>
        <input type="date" id="datePicker" class="form-control" v-model="selectedDate" />
      </div>
    </div>

    <!-- ボタン -->
    <div class="row mt-3">
      <div class="col d-flex justify-content-between">
        <button class="btn btn-success" @click="callTdnetUpload">Tdnet開示データ収集</button>
        <button class="btn btn-secondary" @click="callTdnetLearning">Tdnet開示学習</button>
        <button class="btn btn-danger" @click="callPressReleaseLearning">
          プレスリリース開示学習
        </button>
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
