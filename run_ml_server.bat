@echo off
REM ML Model Server - Windows Launcher
REM Запускает ML Model Server v2 на порту 8001

echo Starting ML Model Server v2...
echo Port: 8001
echo.

REM Активировать виртуальное окружение если есть
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Запустить сервер
python run_ml_server.py

pause
