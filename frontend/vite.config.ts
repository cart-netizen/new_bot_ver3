import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [
    react({
      // jsxRuntime: 'automatic' — включено по умолчанию в новых версиях, можно не указывать
      // fastRefresh: true — тоже по умолчанию
    })
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
    extensions: ['.mjs', '.js', '.ts', '.jsx', '.tsx', '.json']
  },
  server: {
    port: 3000,
    host: true,
    cors: true,
    hmr: {
      overlay: true
    }
  },
  logLevel: 'info',
  optimizeDeps: {
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
    ]
  },
  build: {
    target: 'es2022',
    minify: 'esbuild',
    sourcemap: true,
    chunkSizeWarningLimit: 1000
  },
  esbuild: {
    jsx: 'automatic', // Это нормально, но на самом деле esbuild в Vite не обрабатывает JSX напрямую — этим занимается plugin-react
    treeShaking: true
  }
});