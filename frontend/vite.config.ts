import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react({
      // Явно указываем JSX runtime для React 19
      jsxRuntime: 'automatic',
      // Включаем Fast Refresh
      // fastRefresh: true,
      // Babel конфигурация для лучшей совместимости
      babel: {
        plugins: [],
        // Используем современный preset
        presets: []
      }
    })
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
    // Явно указываем расширения для резолвинга
    extensions: ['.mjs', '.js', '.ts', '.jsx', '.tsx', '.json']
  },
  server: {
    port: 3000,
    host: true,
    // Добавляем CORS для разработки
    cors: true,
    // HMR настройки
    hmr: {
      overlay: true
    }
  },
  // Детальное логирование для отладки
  logLevel: 'info',
  // Оптимизация для разработки
  optimizeDeps: {
    // Явно указываем зависимости для предварительной обработки
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      '@tanstack/react-query',
      'zustand',
      'axios',
      'clsx',
      'tailwind-merge',
      'lucide-react',
      'sonner'
    ],
    // Исключаем из оптимизации
    exclude: [],
    // Форсируем пересканирование при необходимости
    force: false
  },
  // Настройки сборки
  build: {
    // Указываем целевые браузеры
    target: 'es2022',
    // Минификация
    minify: 'esbuild',
    // Source maps для отладки
    sourcemap: true,
    // Размер чанков
    chunkSizeWarningLimit: 1000
  },
  // Настройки для ESBuild
  esbuild: {
    // JSX factory для React
    jsx: 'automatic',
    // Оптимизации
    treeShaking: true
  }
})