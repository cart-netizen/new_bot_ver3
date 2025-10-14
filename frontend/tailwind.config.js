// /** @type {import('tailwindcss').Config} */
// export default {
//   content: [
//     "./index.html",
//     "./src/**/*.{js,ts,jsx,tsx}",
//   ],
//   theme: {
//     extend: {
//       colors: {
//         background: '#0a0e1a',
//         surface: '#1f2937',
//         primary: '#3b82f6',
//         success: '#10b981',
//         danger: '#ef4444',
//         warning: '#f59e0b',
//       },
//     },
//   },
//   plugins: [],
// }

// frontend/tailwind.config.js
/**
 * Конфигурация Tailwind CSS с кастомными анимациями.
 *
 * ОБНОВЛЕНО:
 * - Добавлены flash-анимации для изменения цен (green/red)
 * - Сохранены существующие настройки темы
 */

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Существующие цвета темы
        primary: '#3B82F6',
        secondary: '#8B5CF6',
        success: '#10B981',
        destructive: '#EF4444',
        warning: '#F59E0B',
        surface: '#1F2937',
        background: '#111827',
      },
      // ==================== НОВЫЕ АНИМАЦИИ ====================
      keyframes: {
        'flash-green': {
          '0%': { backgroundColor: 'rgba(16, 185, 129, 0.5)' },
          '100%': { backgroundColor: 'transparent' },
        },
        'flash-red': {
          '0%': { backgroundColor: 'rgba(239, 68, 68, 0.5)' },
          '100%': { backgroundColor: 'transparent' },
        },
        'pulse-slow': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.5' },
        },
      },
      animation: {
        'flash-green': 'flash-green 1s ease-out',
        'flash-red': 'flash-red 1s ease-out',
        'pulse-slow': 'pulse-slow 3s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}