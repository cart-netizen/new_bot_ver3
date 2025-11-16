@echo off
REM Скрипт для запуска ML Model Server на Windows

echo ================================================================================
echo 🚀 Запуск ML Model Server
echo ================================================================================
echo.
echo 📍 URL: http://localhost:8001
echo 📚 Документация: http://localhost:8001/docs
echo 🔍 Health check: http://localhost:8001/health
echo 🤖 Predict endpoint: POST http://localhost:8001/predict
echo.
echo Нажмите Ctrl+C для остановки
echo ================================================================================
echo.

REM Активируем виртуальное окружение если есть
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Запускаем ML сервер
python start_ml_server.py

pause
