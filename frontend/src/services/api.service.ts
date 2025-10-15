// frontend/src/services/api.service.ts

import axios from 'axios';
import type { AxiosInstance, AxiosRequestConfig } from 'axios'; // ✅ type-only import

class ApiService {
  private api: AxiosInstance;

  constructor() {
    this.api = axios.create({
      baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
      timeout: 30000,
    });

    this.api.interceptors.request.use((config) => {
      const token = localStorage.getItem('auth_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });
  }

  async get(url: string, config?: AxiosRequestConfig) {
    const response = await this.api.get(url, config);
    return response.data;
  }

  // ✅ Убрали any, используем unknown
  async post(url: string, data?: unknown, config?: AxiosRequestConfig) {
    const response = await this.api.post(url, data, config);
    return response.data;
  }

  async put(url: string, data?: unknown, config?: AxiosRequestConfig) {
    const response = await this.api.put(url, data, config);
    return response.data;
  }

  async delete(url: string, config?: AxiosRequestConfig) {
    const response = await this.api.delete(url, config);
    return response.data;
  }
}

export const apiService = new ApiService();