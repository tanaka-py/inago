<script setup>
import { RouterLink, RouterView } from 'vue-router'
import LoadingSpinner from '@/components/LoadingSpinner.vue'
</script>

<template>
  <div class="container">
    <loading-spinner></loading-spinner>

    <!-- Header Section -->
    <header class="fixed-top bg-light py-2">
      <div class="d-flex align-items-center">
        <!-- Left aligned title -->
        <h1 class="display-4 custom-title">The イナゴ</h1>
      </div>

      <div class="wrapper">
        <!-- Navigation Bar -->
        <nav class="navbar navbar-expand-lg navbar-light">
          <div class="container-fluid">
            <div class="collapse navbar-collapse">
              <ul class="navbar-nav">
                <li class="nav-item">
                  <RouterLink to="/learning" class="nav-link" active-class="active-link"
                    >学習</RouterLink
                  >
                </li>
                <li class="nav-item">
                  <RouterLink to="/disclosure" class="nav-link" active-class="active-link"
                    >本日の開示</RouterLink
                  >
                </li>
              </ul>
            </div>
          </div>
        </nav>
      </div>
    </header>

    <!-- Main Content -->
    <div class="main-content">
      <RouterView />
    </div>

    <footer>©田中家</footer>
  </div>
</template>
