-- Fix AuditAction enum - add missing values
-- Это исправляет ошибку: "неверное значение для перечисления auditaction: EMERGENCY_SHUTDOWN"

-- Добавляем недостающие значения в enum (IF NOT EXISTS безопасно для повторного запуска)
ALTER TYPE auditaction ADD VALUE IF NOT EXISTS 'system';
ALTER TYPE auditaction ADD VALUE IF NOT EXISTS 'emergency_shutdown';

-- Проверка: показать все значения enum
SELECT unnest(enum_range(NULL::auditaction)) AS audit_action_values;
