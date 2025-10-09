import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [
    react({
      // Убираем явную настройку jsxRuntime
      // Плагин сам определит правильную конфигурацию для React 19
      jsxImportSource: undefined,
      // Включаем Fast Refresh для React 19
      // fastRefresh: true,
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
  // Оптимизация сборки
  build: {
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'ui-vendor': ['lucide-react', 'sonner'],
        },
      },
    },
  },
  // Явное определение логирования
  logLevel: 'info',
})