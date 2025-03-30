import { createRouter, createWebHistory } from 'vue-router'
import DisclosureView from '../views/DisclosureView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      redirect: '/disclosure'
    },
    {
      path: '/disclosure',
      name: 'disclosure',
      component: DisclosureView,
    },
    {
      path: '/learning',
      name: 'learning',
      component: () => import('../views/LearningView.vue'),
    },
    {
      path: '/summarize',
      name: 'summarize',
      component: () => import('../views/SummariseView.vue'),
    },
  ],
})

export default router
