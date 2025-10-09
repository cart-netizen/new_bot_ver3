import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react({
      // Для @vitejs/plugin-react 5.0.4 достаточно пустой конфигурации
      // Плагин автоматически определит правильные настройки для React 19
    })
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    host: true,
  },
  // Детальное логирование для отладки
  logLevel: 'info',
  // Оптимизация для разработки
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      '@tanstack/react-query',
      'zustand',
      'axios',
    ],
  },
})