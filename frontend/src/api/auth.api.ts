import apiClient from './client';
import { API_ENDPOINTS } from '../config/app.config';
import type { LoginRequest, LoginResponse } from '../types/api.types';

export const authApi = {
  login: async (data: LoginRequest): Promise<LoginResponse> => {
    const response = await apiClient.post<LoginResponse>(
      API_ENDPOINTS.auth.login,
      data
    );
    return response.data;
  },

  verify: async (): Promise<{ status: string }> => {
    const response = await apiClient.get(API_ENDPOINTS.auth.verify);
    return response.data;
  },
};