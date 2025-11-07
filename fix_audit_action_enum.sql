-- Fix AuditAction enum - add ALL missing values
-- Это исправляет ошибки: "неверное значение для перечисления auditaction"

-- Добавляем ВСЕ значения из Python модели (backend/database/models.py:54-65)
-- IF NOT EXISTS безопасно для повторного запуска

ALTER TYPE auditaction ADD VALUE IF NOT EXISTS 'order_place';
ALTER TYPE auditaction ADD VALUE IF NOT EXISTS 'order_cancel';
ALTER TYPE auditaction ADD VALUE IF NOT EXISTS 'order_modify';
ALTER TYPE auditaction ADD VALUE IF NOT EXISTS 'position_open';
ALTER TYPE auditaction ADD VALUE IF NOT EXISTS 'position_close';
ALTER TYPE auditaction ADD VALUE IF NOT EXISTS 'position_modify';
ALTER TYPE auditaction ADD VALUE IF NOT EXISTS 'balance_update';
ALTER TYPE auditaction ADD VALUE IF NOT EXISTS 'config_change';
ALTER TYPE auditaction ADD VALUE IF NOT EXISTS 'system';
ALTER TYPE auditaction ADD VALUE IF NOT EXISTS 'emergency_shutdown';

-- Проверка: показать все значения enum
SELECT unnest(enum_range(NULL::auditaction)) AS audit_action_values;
