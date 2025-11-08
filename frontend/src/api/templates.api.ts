/**
 * API клиент для работы с шаблонами бэктестов
 */

import axios from 'axios';
import type { BacktestConfig } from './backtesting.api';

const apiClient = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json'
  }
});

/**
 * Partial configuration that can be stored in a template
 * Templates may not include all required fields (like name, dates, symbol)
 */
export type TemplateConfig = Partial<BacktestConfig>;

export interface Template {
  id: string;
  name: string;
  description?: string;
  config: TemplateConfig;
  tags: string[];
  is_public: boolean;
  usage_count: number;
  created_at: string;
  updated_at: string;
}

export interface CreateTemplateRequest {
  name: string;
  description?: string;
  config: TemplateConfig;
  tags?: string[];
  is_public?: boolean;
}

export interface UpdateTemplateRequest {
  name?: string;
  description?: string;
  config?: TemplateConfig;
  tags?: string[];
}

/**
 * Получить список шаблонов
 */
export async function listTemplates(tags?: string[]): Promise<{ templates: Template[]; total: number }> {
  const params = tags && tags.length > 0 ? { tags: tags.join(',') } : {};
  const response = await apiClient.get('/api/backtesting/templates', { params });
  return response.data;
}

/**
 * Получить шаблон по ID
 */
export async function getTemplate(templateId: string): Promise<Template> {
  const response = await apiClient.get(`/api/backtesting/templates/${templateId}`);
  return response.data;
}

/**
 * Создать новый шаблон
 */
export async function createTemplate(data: CreateTemplateRequest): Promise<{ success: boolean; template_id: string }> {
  const response = await apiClient.post('/api/backtesting/templates', data);
  return response.data;
}

/**
 * Обновить шаблон
 */
export async function updateTemplate(
  templateId: string,
  data: UpdateTemplateRequest
): Promise<{ success: boolean; template_id: string }> {
  const response = await apiClient.put(`/api/backtesting/templates/${templateId}`, data);
  return response.data;
}

/**
 * Удалить шаблон
 */
export async function deleteTemplate(templateId: string): Promise<{ success: boolean }> {
  const response = await apiClient.delete(`/api/backtesting/templates/${templateId}`);
  return response.data;
}
