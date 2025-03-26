import { createRouter, createWebHistory } from 'vue-router'
import LearningView from '../views/LearningView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      redirect: '/learning'
    },
    // {
    //   path: '/about',
    //   name: 'about',
    //   // route level code-splitting
    //   // this generates a separate chunk (About.[hash].js) for this route
    //   // which is lazy-loaded when the route is visited.
    //   component: () => import('../views/AboutView.vue'),
    // },
    {
      path: '/learning',
      name: 'learning',
      // route level code-splitting
      // this generates a separate chunk (About.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: LearningView,
    },
    {
      path: '/disclosure',
      name: 'disclosure',
      // route level code-splitting
      // this generates a separate chunk (About.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import('../views/DisclosureView.vue'),
    },
    {
      path: '/summarize',
      name: 'summarize',
      component: () => import('../views/SummariseView.vue'),
    },
  ],
})

export default router
