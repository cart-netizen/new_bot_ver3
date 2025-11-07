# Setup MLflow PostgreSQL Database (Windows PowerShell)
# This script creates a separate database for MLflow to avoid alembic_version conflicts

$ErrorActionPreference = "Stop"

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "MLflow PostgreSQL Setup (Windows)" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Configuration
$DB_USER = "trading_bot"
$DB_PASSWORD = "robocop"
$DB_HOST = "localhost"
$DB_PORT = "5432"
$MLFLOW_DB = "mlflow_tracking"
$MAIN_DB = "trading_bot"

Write-Host ""
Write-Host "This will:" -ForegroundColor Yellow
Write-Host "1. Create separate database: $MLFLOW_DB"
Write-Host "2. Keep main app database: $MAIN_DB"
Write-Host "3. Update .env configuration"
Write-Host ""

# Check if psql is available
try {
    $null = Get-Command psql -ErrorAction Stop
} catch {
    Write-Host "❌ ERROR: psql command not found" -ForegroundColor Red
    Write-Host "   Please install PostgreSQL and add it to PATH" -ForegroundColor Red
    Write-Host "   Or run this from PostgreSQL bin directory" -ForegroundColor Red
    exit 1
}

# Check PostgreSQL connection
Write-Host "Checking PostgreSQL connection..." -ForegroundColor Yellow
$env:PGPASSWORD = "postgres_password"
try {
    psql -U postgres -h $DB_HOST -p $DB_PORT -c "SELECT 1" 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { throw }
    Write-Host "✓ PostgreSQL is running" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Cannot connect to PostgreSQL" -ForegroundColor Red
    Write-Host "   Please ensure PostgreSQL is running on ${DB_HOST}:${DB_PORT}" -ForegroundColor Red
    Write-Host "   And postgres superuser is accessible" -ForegroundColor Red
    exit 1
}

# Create MLflow database
Write-Host ""
Write-Host "Creating MLflow database..." -ForegroundColor Yellow

$sql = @"
-- Create database if not exists
SELECT 'CREATE DATABASE $MLFLOW_DB OWNER $DB_USER'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$MLFLOW_DB')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE $MLFLOW_DB TO $DB_USER;
"@

$sql | psql -U postgres -h $DB_HOST -p $DB_PORT 2>&1 | Out-Null

# Connect to new database and set permissions
$sql2 = @"
GRANT ALL ON SCHEMA public TO $DB_USER;
ALTER SCHEMA public OWNER TO $DB_USER;
"@

$sql2 | psql -U postgres -h $DB_HOST -p $DB_PORT -d $MLFLOW_DB 2>&1 | Out-Null

Write-Host "✓ MLflow database created: $MLFLOW_DB" -ForegroundColor Green

# Update .env file
Write-Host ""
Write-Host "Updating .env configuration..." -ForegroundColor Yellow

# Backup .env
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item .env ".env.backup.$timestamp"
Write-Host "✓ Backed up .env to .env.backup.$timestamp" -ForegroundColor Green

# Read .env content
$envContent = Get-Content .env -Raw

# Update or add MLFLOW_TRACKING_URI
$newUri = "MLFLOW_TRACKING_URI=postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${MLFLOW_DB}"

if ($envContent -match "(?m)^MLFLOW_TRACKING_URI=.*$") {
    # Replace existing line
    $envContent = $envContent -replace "(?m)^MLFLOW_TRACKING_URI=.*$", $newUri
    Write-Host "✓ Updated MLFLOW_TRACKING_URI in .env" -ForegroundColor Green
} else {
    # Add new line
    $envContent += "`n$newUri`n"
    Write-Host "✓ Added MLFLOW_TRACKING_URI to .env" -ForegroundColor Green
}

# Write back to file
$envContent | Set-Content .env -NoNewline

# Summary
Write-Host ""
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Configuration Summary" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Main App Database:  postgresql+asyncpg://${DB_USER}:***@${DB_HOST}:${DB_PORT}/${MAIN_DB}"
Write-Host "MLflow Tracking DB: postgresql://${DB_USER}:***@${DB_HOST}:${DB_PORT}/${MLFLOW_DB}"
Write-Host ""
Write-Host "✓ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Restart your application"
Write-Host "2. MLflow will automatically create its tables in $MLFLOW_DB"
Write-Host "3. No more alembic_version conflicts!"
Write-Host ""
