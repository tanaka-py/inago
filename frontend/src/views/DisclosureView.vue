<script setup>
import { inject, ref, nextTick, onMounted, watch, computed } from 'vue'
import { useLoadingStore } from '@/stores/loading'
import tippy from 'tippy.js'

// axiosをinject
const axios = inject('axios')
// ローディングを準備
const loadingStore = useLoadingStore()

const init_list = [
  {
    Time: '',
    Code: '',
    Name: '',
    Title: '',
    Link: '',
  },
]
const inago_list = ref([])

// 土日の場合は直近金曜日
const getAdjustedDate = () => {
  const today = new Date()
  let day = today.getDay() // 0:日曜, 1:月曜, ..., 5:金曜, 6:土曜

  if (day === 0) {
    // 日曜なら2日前の金曜に
    today.setDate(today.getDate() - 2)
  } else if (day === 6) {
    // 土曜なら1日前の金曜に
    today.setDate(today.getDate() - 1)
  }

  return today.toISOString().split('T')[0]
}

const selectedDate = ref(getAdjustedDate())
const tooltipRefs = ref([]) // すべてのTooltipの参照を格納
const isFilterList = ref(false)

// リストを絞り込む
const FilterList = computed(() => {
  let dipList = inago_list.value
  let excludes = ['短信']
  if (isFilterList.value) {
    dipList = inago_list.value.filter((item) => {
      return excludes.every((target) => !item.Title.includes(target))
    })
  }
  return dipList
})

// リスト絞り込みの切り替え
const changeList = () => {
  isFilterList.value = !isFilterList.value
}

// 開示とプレスリリース両方取得
const callList = async () => {
  loadingStore.startLoading()

  // いったんクリア
  inago_list.value = init_list
  nextTick(async () => {
    try {
      let tdnet_list = axios.get(`/disclosure/tdnetlist/${selectedDate.value.replace(/-/g, '')}`)
      //let pressrelease_list = axios.get(`/pressrelease/list/${selectedDate.value}`)

      // 同時に取得
      let responses = await Promise.all([tdnet_list])
      let [tdnet_res] = responses
      //let responses = await Promise.all([tdnet_list, pressrelease_list])
      //let [tdnet_res, pressrelease_res] = responses

      // 取得したものを結合して時間順にソート
      inago_list.value = tdnet_res.data.datalist
        .sort((a, b) => {
          let b_time = new Date(`${selectedDate.value.replace(/-/g, '/')} ${b.Time}:00`)
          let a_time = new Date(`${selectedDate.value.replace(/-/g, '/')} ${a.Time}:00`)
          return b_time - a_time
        })
    } catch (error) {
      alert(`call_error! ★tdnet detail=[${error}]`)
    } finally {
      loadingStore.stopLoading()
    }
  })
}

// Edinet開示情報を取得 ※使わない
// const callEdinetList = async () => {
//   try {
//     let getlist = await axios.get('/disclosure/edinetlist')
//     disclosure_list.value = getlist.data.datalist
//   } catch (error) {
//     alert(`call_error! ★ednet detail=[${error}]`)
//   }
// }

// disclosure_listが変更されたときにtooltipを再初期化する
watch(inago_list, (newList) => {
  nextTick(() => {
    // tooltipRefs 配列にアクセスして、各要素に対してツールチップをセット
    newList.forEach((_, index) => {
      const tooltipEl = tooltipRefs.value[index]
      if (tooltipEl) {
        tippy(tooltipEl, {
          content: inago_list.value[index].Link.replace(/\n/g, '<br>'), // Link をツールチップにセット
          maxWidth: 500, // ツールチップの最大幅
          maxHeight: 500, // ツールチップの最大高さ
          theme: 'light',
          allowHTML: true,
          animation: 'fade',
          arrow: true,
          placement: 'top', // 最初は上に表示
          interactive: true,
          flip: true, // 上に表示できない場合、下に表示する
          boundary: 'viewport', // ビューポート内に収めるように調整
          zIndex: 9999, // ツールチップのz-indexを高く設定
          appendTo: document.body, // ツールチップをbodyに追加
          onShow(instance) {
            tooltipEl.removeAttribute('title') // title属性を削除
            const tooltipContent = instance.popper.querySelector('.tippy-content')
            if (tooltipContent) {
              tooltipContent.style.overflowY = 'auto' // スクロールを有効にする
              tooltipContent.style.maxHeight = '500px' // 最大高さを設定
            }
          },
        })
      }
    })
  })
})

onMounted(() => {
  // nextTick(() => {
  //   callList()
  // })
})
</script>

<template>
  <div class="container mt-5">
    <!-- ボタン -->
    <div class="row mt-3">
      <div class="col d-flex justify-content-between">
        <!-- <button class="btn btn-primary" @click="callEdinetList">Edinet開示一覧取得</button> -->
        <div class="reload-button" @click="callList">
          <i class="fas fa-sync-alt"></i>
        </div>
        <button class="btn btn-danger" @click="changeList">リスト切り替え</button>
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
