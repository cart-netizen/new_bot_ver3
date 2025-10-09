import { create } from 'zustand';
import { authApi } from '../api/auth.api.ts';
import type { LoginRequest } from '../types/api.types.ts';

interface AuthState {
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (data: LoginRequest) => Promise<void>;
  logout: () => void;
  setToken: (token: string) => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  token: localStorage.getItem('auth-token'),
  isAuthenticated: !!localStorage.getItem('auth-token'),
  isLoading: false,
  error: null,

  login: async (data) => {
    set({ isLoading: true, error: null });
    try {
      const response = await authApi.login(data);
      localStorage.setItem('auth-token', response.access_token);
      set({
        token: response.access_token,
        isAuthenticated: true,
        isLoading: false,
      });
    } catch (error: any) {
      set({
        error: error.response?.data?.detail || 'Ошибка входа',
        isLoading: false,
      });
      throw error;
    }
  },

  logout: () => {
    localStorage.removeItem('auth-token');
    set({
      token: null,
      isAuthenticated: false,
    });
  },

  setToken: (token) => {
    localStorage.setItem('auth-token', token);
    set({ token, isAuthenticated: true });
  },
}));