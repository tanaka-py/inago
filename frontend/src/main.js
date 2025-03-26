import { createApp } from 'vue';
import { createPinia } from 'pinia'
import App from './App.vue';
import router from './router';
import axios from 'axios';
import 'tippy.js';
import 'tippy.js/dist/tippy.css'; // スタイルのインポート

import 'bootstrap/dist/css/bootstrap.min.css';
import './assets/main.css';

import * as bootstrap from 'bootstrap';
window.bootstrap = bootstrap;

const app = createApp(App);
app.use(createPinia()); // pinia登録 コンポーネント間管理

// provideでaxiosをコンポーネントに渡す
const axiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_URL
});
app.provide('axios', axiosInstance)

app.use(router);

app.mount('#app');
