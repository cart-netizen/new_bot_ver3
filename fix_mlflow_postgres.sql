-- Fix MLflow PostgreSQL Issue
-- Problem: alembic_version table contains '003_add_layering_patterns'
-- which is from main app migrations, not MLflow migrations

-- Option 1: Create separate schema for MLflow (RECOMMENDED)
-- This keeps main app and MLflow migrations isolated

CREATE SCHEMA IF NOT EXISTS mlflow;

-- Grant permissions to trading_bot user
GRANT ALL PRIVILEGES ON SCHEMA mlflow TO trading_bot;

-- Set search path to include mlflow schema
ALTER DATABASE trading_bot SET search_path TO public, mlflow;


-- Option 2: Drop MLflow tables and let it recreate (if schema separation not needed)
-- WARNING: This will delete all MLflow data!
-- Uncomment if you want to use this approach:

/*
-- Drop all MLflow tables
DROP TABLE IF EXISTS mlflow.experiments CASCADE;
DROP TABLE IF EXISTS mlflow.runs CASCADE;
DROP TABLE IF EXISTS mlflow.metrics CASCADE;
DROP TABLE IF EXISTS mlflow.params CASCADE;
DROP TABLE IF EXISTS mlflow.tags CASCADE;
DROP TABLE IF EXISTS mlflow.experiment_tags CASCADE;
DROP TABLE IF EXISTS mlflow.latest_metrics CASCADE;
DROP TABLE IF EXISTS mlflow.model_versions CASCADE;
DROP TABLE IF EXISTS mlflow.registered_models CASCADE;
DROP TABLE IF EXISTS mlflow.registered_model_tags CASCADE;
DROP TABLE IF EXISTS mlflow.model_version_tags CASCADE;
DROP TABLE IF EXISTS mlflow.datasets CASCADE;
DROP TABLE IF EXISTS mlflow.inputs CASCADE;
DROP TABLE IF EXISTS mlflow.input_tags CASCADE;
DROP TABLE IF EXISTS mlflow.trace_info CASCADE;
DROP TABLE IF EXISTS mlflow.trace_request_metadata CASCADE;
DROP TABLE IF EXISTS mlflow.trace_tags CASCADE;

-- If alembic_version is in public schema and contains MLflow version
-- DELETE FROM alembic_version WHERE version_num NOT IN (
--   SELECT version_num FROM alembic_version WHERE version_num LIKE '%layering%'
-- );
*/


-- Option 3: Just clear the problematic alembic_version entry
-- This is the simplest but may cause issues if there are multiple alembic migrations

/*
-- Remove the problematic migration version
DELETE FROM alembic_version WHERE version_num = '003_add_layering_patterns';

-- Or create a separate alembic_version table in mlflow schema
DROP TABLE IF EXISTS mlflow.alembic_version;
*/
