// frontend/src/App.tsx

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'sonner';
import { Layout } from './components/layout/Layout';
import { ProtectedRoute } from './components/auth/ProtectedRoute';
import { LoginPage } from './pages/LoginPage';
import { DashboardPage } from './pages/DashboardPage';
import { MarketPage } from './pages/MarketPage';
import { TradingPage } from './pages/TradingPage';
import { AccountPage } from './pages/AccountPage';
import { ChartsPage } from './pages/ChartsPage';
import { OrdersPage } from './pages/OrdersPage';
import { ScreenerPage } from './pages/ScreenerPage';
import { StrategiesPage } from './pages/StrategiesPage';
import MLManagementPage from "@/pages/MLManagement/frontend_example_MLManagement";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,
      refetchOnWindowFocus: false,
    },
  },
});

/**
 * Главный компонент приложения.
 * Управляет маршрутизацией и глобальными провайдерами.
 */
export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          {/* Страница входа */}
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
            {/* Перенаправление на Dashboard по умолчанию */}
            <Route index element={<Navigate to="/dashboard" replace />} />
            
            {/* Существующие страницы */}
            <Route path="account" element={<AccountPage />} />
            <Route path="dashboard" element={<DashboardPage />} />
            <Route path="market" element={<MarketPage />} />
            <Route path="trading" element={<TradingPage />} />

            {/* Новые страницы */}
            <Route path="charts" element={<ChartsPage />} />
            <Route path="orders" element={<OrdersPage />} />
            <Route path="screener" element={<ScreenerPage />} />
            <Route path="strategies" element={<StrategiesPage />} />
            <Route path="ML" element={<MLManagementPage />} />
          </Route>
        </Routes>
      </BrowserRouter>

      {/* Глобальные уведомления */}
      <Toaster position="top-right" />
    </QueryClientProvider>
  );
}