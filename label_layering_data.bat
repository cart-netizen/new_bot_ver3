@echo off
REM Label Collected Layering ML Data
REM Automatically labels unlabeled data based on detector confidence and metrics

echo.
echo ======================================================================
echo Layering ML Data Auto-Labeling
echo ======================================================================
echo.
echo This will automatically label unlabeled layering ML data
echo based on detector confidence, execution rate, and cancellation rate.
echo.
echo Press Ctrl+C to cancel, or
pause

python label_layering_data.py

echo.
pause
