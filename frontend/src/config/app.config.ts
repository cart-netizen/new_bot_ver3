export const APP_CONFIG = {
  apiUrl: import.meta.env.VITE_API_URL || import.meta.env.PUBLIC_API_URL || 'http://localhost:8000',
  wsUrl: import.meta.env.VITE_WS_URL || import.meta.env.PUBLIC_WS_URL || 'ws://localhost:8000/ws',
} as const;

export const API_ENDPOINTS = {
  auth: {
    login: '/auth/login',
    verify: '/auth/verify',
  },
  bot: {
    start: '/bot/start',
    stop: '/bot/stop',
    status: '/bot/status',
    config: '/bot/config',
  },
  data: {
    pairs: '/data/pairs',
    orderbook: (symbol: string) => `/data/orderbook/${symbol}`,
    metrics: (symbol: string) => `/data/metrics/${symbol}`,
    allMetrics: '/data/metrics',
  },
  trading: {
    signals: '/trading/signals',
  },
} as const;