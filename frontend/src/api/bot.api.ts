import apiClient from './client';
import { API_ENDPOINTS } from '../config/app.config';
import type { ConfigResponse } from '../types/api.types';

export const botApi = {
  start: async (): Promise<{ status: string; message: string }> => {
    const response = await apiClient.post(API_ENDPOINTS.bot.start);
    return response.data;
  },

  stop: async (): Promise<{ status: string; message: string }> => {
    const response = await apiClient.post(API_ENDPOINTS.bot.stop);
    return response.data;
  },

  getStatus: async (): Promise<any> => {
    const response = await apiClient.get(API_ENDPOINTS.bot.status);
    return response.data;
  },

  getConfig: async (): Promise<ConfigResponse> => {
    const response = await apiClient.get<ConfigResponse>(API_ENDPOINTS.bot.config);
    return response.data;
  },
};