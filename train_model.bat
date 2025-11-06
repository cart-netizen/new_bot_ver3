@echo off
REM Quick Model Training Script
REM Double-click to train model with default settings

echo ========================================
echo ML Model Training
echo ========================================
echo.

REM Activate virtual environment if exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo Virtual environment activated
) else (
    echo Warning: No virtual environment found
)

echo.
echo Starting training...
echo.

REM Run training script
python train_model.py

echo.
echo Press any key to exit...
pause >nul
