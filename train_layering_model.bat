@echo off
REM Training script for Layering ML Model (Windows)
REM Usage: Simply double-click this file or run: train_layering_model.bat

echo ================================================================================
echo LAYERING ML MODEL TRAINING
echo ================================================================================
echo.

REM Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found at .venv\
    echo Using system Python...
)

echo.
echo Starting training...
echo.

python train_layering_model.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo TRAINING COMPLETED SUCCESSFULLY
    echo ================================================================================
) else (
    echo.
    echo ================================================================================
    echo TRAINING FAILED - Check errors above
    echo ================================================================================
)

echo.
pause
