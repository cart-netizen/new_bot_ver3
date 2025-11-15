@echo off
REM Start MLflow UI Server
REM Access at: http://localhost:5000

echo.
echo ======================================================================
echo Starting MLflow UI Server
echo ======================================================================
echo.
echo UI will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
echo ======================================================================
echo.

python start_mlflow_ui.py

pause
