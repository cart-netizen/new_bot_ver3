-- Create separate database for MLflow tracking
-- This isolates MLflow migrations from main app migrations

-- Connect to postgres database first to create new database
-- Run: psql -U postgres -h localhost

-- Create MLflow database
CREATE DATABASE mlflow_tracking OWNER trading_bot;

-- Grant all privileges
GRANT ALL PRIVILEGES ON DATABASE mlflow_tracking TO trading_bot;

-- Now MLflow will use: postgresql://trading_bot:robocop@localhost:5432/mlflow_tracking
-- And main app will use: postgresql+asyncpg://trading_bot:robocop@localhost:5432/trading_bot

-- This way alembic_version tables are completely separate!
