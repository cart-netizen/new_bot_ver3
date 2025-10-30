@echo off
echo ========================================
echo PostgreSQL Diagnostic and Fix Script
echo ========================================
echo.

REM Определяем версию PostgreSQL (пробуем 15, 16, 14)
set PG_VERSION=
for %%v in (16 15 14 13) do (
    sc query postgresql-x64-%%v >nul 2>&1
    if not errorlevel 1 (
        set PG_VERSION=%%v
        goto :found
    )
)

:found
if "%PG_VERSION%"=="" (
    echo [ERROR] PostgreSQL service not found!
    echo Please install PostgreSQL first.
    echo Download: https://www.enterprisedb.com/downloads/postgres-postgresql-downloads
    pause
    exit /b 1
)

echo [INFO] Found PostgreSQL version: %PG_VERSION%
echo.

REM Проверяем статус службы
echo ========================================
echo Step 1: Checking PostgreSQL service status
echo ========================================
sc query postgresql-x64-%PG_VERSION%
echo.

REM Проверяем порт 5432
echo ========================================
echo Step 2: Checking port 5432
echo ========================================
netstat -ano | findstr :5432
echo.

REM Спрашиваем, перезапустить ли службу
echo ========================================
echo Step 3: Restart PostgreSQL service?
echo ========================================
echo This will:
echo 1. Stop PostgreSQL service
echo 2. Wait 5 seconds
echo 3. Start PostgreSQL service
echo.
set /p RESTART="Do you want to restart PostgreSQL? (Y/N): "

if /i "%RESTART%"=="Y" (
    echo.
    echo [INFO] Stopping PostgreSQL service...
    net stop postgresql-x64-%PG_VERSION%

    echo [INFO] Waiting 5 seconds...
    timeout /t 5 /nobreak

    echo [INFO] Starting PostgreSQL service...
    net start postgresql-x64-%PG_VERSION%

    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to start PostgreSQL!
        echo.
        echo Please check the logs at:
        echo C:\Program Files\PostgreSQL\%PG_VERSION%\data\log\
        echo.
        echo Common issues:
        echo 1. Port 5432 is already in use
        echo 2. Data directory is corrupted
        echo 3. Permission issues
        echo.
        echo See FIX_POSTGRESQL_CRASH.md for detailed solutions.
        pause
        exit /b 1
    )

    echo.
    echo [SUCCESS] PostgreSQL service started successfully!
    echo.

    REM Ждем, пока PostgreSQL полностью запустится
    echo [INFO] Waiting for PostgreSQL to be ready...
    timeout /t 3 /nobreak

    REM Проверяем подключение
    echo.
    echo ========================================
    echo Step 4: Testing connection
    echo ========================================

    REM Проверяем порт снова
    netstat -ano | findstr :5432
    echo.

    echo [INFO] PostgreSQL should now be running on port 5432
    echo.
    echo Next steps:
    echo 1. Try connecting with pgAdmin
    echo 2. Run your application
    echo 3. If still not working, see FIX_POSTGRESQL_CRASH.md
    echo.
) else (
    echo.
    echo [INFO] Restart cancelled by user
    echo.
    echo To manually restart:
    echo 1. Open services.msc
    echo 2. Find 'postgresql-x64-%PG_VERSION%'
    echo 3. Right-click and select 'Restart'
    echo.
)

echo ========================================
echo Diagnostic Information
echo ========================================
echo PostgreSQL Version: %PG_VERSION%
echo Service Name: postgresql-x64-%PG_VERSION%
echo Data Directory: C:\Program Files\PostgreSQL\%PG_VERSION%\data\
echo Log Directory: C:\Program Files\PostgreSQL\%PG_VERSION%\data\log\
echo Configuration: C:\Program Files\PostgreSQL\%PG_VERSION%\data\postgresql.conf
echo.

pause
