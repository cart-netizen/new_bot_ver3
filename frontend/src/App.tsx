// frontend/src/App.tsx
/**
 * Главный компонент приложения с маршрутизацией.
 *
 * ОБНОВЛЕНО:
 * - Добавлены новые маршруты: /charts, /orders, /screener, /strategies
 * - Lazy loading для оптимизации производительности
 * - Сохранена существующая структура
 */

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'sonner';
import { lazy, Suspense } from 'react';

// Существующие компоненты
import { Layout } from './components/layout/Layout';
import { LoginPage } from './pages/LoginPage';
import { ProtectedRoute } from './components/auth/ProtectedRoute';

// Lazy loading страниц
const AccountPage = lazy(() => import('./pages/AccountPage').then(m => ({ default: m.AccountPage })));
const DashboardPage = lazy(() => import('./pages/DashboardPage').then(m => ({ default: m.DashboardPage })));
const MarketPage = lazy(() => import('./pages/MarketPage').then(m => ({ default: m.MarketPage })));
const TradingPage = lazy(() => import('./pages/TradingPage').then(m => ({ default: m.TradingPage })));

// ==================== НОВЫЕ СТРАНИЦЫ ====================
const ChartsPage = lazy(() => import('./pages/ChartsPage').then(m => ({ default: m.ChartsPage })));
const OrdersPage = lazy(() => import('./pages/OrdersPage').then(m => ({ default: m.OrdersPage })));
const ScreenerPage = lazy(() => import('./pages/ScreenerPage').then(m => ({ default: m.ScreenerPage })));
const StrategiesPage = lazy(() => import('./pages/StrategiesPage').then(m => ({ default: m.StrategiesPage })));

/**
 * Компонент загрузки для Suspense.
 */
function PageLoader() {
  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
    </div>
  );
}

/**
 * Главный компонент приложения.
 */
export function App() {
  return (
    <BrowserRouter>
      {/* Toast notifications */}
      <Toaster
        position="top-right"
        richColors
        closeButton
        theme="dark"
      />

      <Routes>
        {/* Публичный маршрут - Логин */}
        <Route path="/login" element={<LoginPage />} />

        {/* Защищенные маршруты */}
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <Layout />
            </ProtectedRoute>
          }
        >
          {/* Редирект с корня на dashboard */}
          <Route index element={<Navigate to="/dashboard" replace />} />

          {/* Существующие страницы */}
          <Route
            path="account"
            element={
              <Suspense fallback={<PageLoader />}>
                <AccountPage />
              </Suspense>
            }
          />
          <Route
            path="dashboard"
            element={
              <Suspense fallback={<PageLoader />}>
                <DashboardPage />
              </Suspense>
            }
          />
          <Route
            path="market"
            element={
              <Suspense fallback={<PageLoader />}>
                <MarketPage />
              </Suspense>
            }
          />
          <Route
            path="trading"
            element={
              <Suspense fallback={<PageLoader />}>
                <TradingPage />
              </Suspense>
            }
          />

          {/* ==================== НОВЫЕ МАРШРУТЫ ==================== */}
          <Route
            path="charts"
            element={
              <Suspense fallback={<PageLoader />}>
                <ChartsPage />
              </Suspense>
            }
          />
          <Route
            path="orders"
            element={
              <Suspense fallback={<PageLoader />}>
                <OrdersPage />
              </Suspense>
            }
          />
          <Route
            path="screener"
            element={
              <Suspense fallback={<PageLoader />}>
                <ScreenerPage />
              </Suspense>
            }
          />
          <Route
            path="strategies"
            element={
              <Suspense fallback={<PageLoader />}>
                <StrategiesPage />
              </Suspense>
            }
          />

          {/* 404 - Not Found */}
          <Route
            path="*"
            element={
              <div className="flex items-center justify-center min-h-screen">
                <div className="text-center">
                  <h1 className="text-4xl font-bold text-white mb-4">404</h1>
                  <p className="text-gray-400 mb-6">Страница не найдена</p>
                  <a
                    href="/dashboard"
                    className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/90 transition-colors"
                  >
                    Вернуться на главную
                  </a>
                </div>
              </div>
            }
          />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}