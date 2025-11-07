#!/bin/bash
# Setup MLflow PostgreSQL Database
# This script creates a separate database for MLflow to avoid alembic_version conflicts

set -e

echo "================================="
echo "MLflow PostgreSQL Setup"
echo "================================="

# Configuration
DB_USER="trading_bot"
DB_PASSWORD="robocop"
DB_HOST="localhost"
DB_PORT="5432"
MLFLOW_DB="mlflow_tracking"
MAIN_DB="trading_bot"

echo ""
echo "This will:"
echo "1. Create separate database: $MLFLOW_DB"
echo "2. Keep main app database: $MAIN_DB"
echo "3. Update .env configuration"
echo ""

# Check if PostgreSQL is running
echo "Checking PostgreSQL connection..."
if ! psql -U postgres -h $DB_HOST -p $DB_PORT -c "SELECT 1" > /dev/null 2>&1; then
    echo "❌ ERROR: Cannot connect to PostgreSQL"
    echo "   Please ensure PostgreSQL is running on $DB_HOST:$DB_PORT"
    echo "   And you have postgres superuser access"
    exit 1
fi
echo "✓ PostgreSQL is running"

# Create MLflow database
echo ""
echo "Creating MLflow database..."
psql -U postgres -h $DB_HOST -p $DB_PORT <<EOF
-- Check if database exists
SELECT 'CREATE DATABASE $MLFLOW_DB OWNER $DB_USER'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$MLFLOW_DB')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE $MLFLOW_DB TO $DB_USER;

\c $MLFLOW_DB
GRANT ALL ON SCHEMA public TO $DB_USER;
ALTER SCHEMA public OWNER TO $DB_USER;
EOF

echo "✓ MLflow database created: $MLFLOW_DB"

# Update .env file
echo ""
echo "Updating .env configuration..."

# Backup .env
cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
echo "✓ Backed up .env"

# Update MLFLOW_TRACKING_URI in .env
if grep -q "^MLFLOW_TRACKING_URI=" .env; then
    # Update existing line
    sed -i.tmp "s|^MLFLOW_TRACKING_URI=.*|MLFLOW_TRACKING_URI=postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$MLFLOW_DB|" .env
    rm -f .env.tmp
    echo "✓ Updated MLFLOW_TRACKING_URI in .env"
else
    # Add new line
    echo "MLFLOW_TRACKING_URI=postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$MLFLOW_DB" >> .env
    echo "✓ Added MLFLOW_TRACKING_URI to .env"
fi

# Verify configuration
echo ""
echo "================================="
echo "Configuration Summary"
echo "================================="
echo "Main App Database:  postgresql+asyncpg://$DB_USER:***@$DB_HOST:$DB_PORT/$MAIN_DB"
echo "MLflow Tracking DB: postgresql://$DB_USER:***@$DB_HOST:$DB_PORT/$MLFLOW_DB"
echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Restart your application"
echo "2. MLflow will automatically create its tables in $MLFLOW_DB"
echo "3. No more alembic_version conflicts!"
echo ""
