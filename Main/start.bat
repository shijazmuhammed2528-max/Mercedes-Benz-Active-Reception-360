@echo off
title Mercedes-Benz Active Reception 360 Capture System
color 0F

echo ======================================================================
echo    MERCEDES-BENZ ACTIVE RECEPTION - 360 VEHICLE CAPTURE SYSTEM
echo ======================================================================
echo.

:: 1. Change to script directory
cd /d "%~dp0"

:: 2. Check Python availability
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not found in system PATH.
    echo Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

echo [OK] Python detected:
python --version
echo.

:: 3 & 4. Virtual environment check / activation
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating existing virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [INFO] Using system Python environment...
)

:: 5. Verify and install requirements if needed
echo [INFO] Checking required dependencies...
python -c "import flask, flask_cors, cv2" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing core web dependencies...
    pip install flask flask-cors
)

python -c "import torch, ultralytics" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Checking PyTorch and Ultralytics...
    echo Attempting to install requirements...
    pip install -r requirements.txt
)

echo.
echo ======================================================================
echo  Starting Active Reception Server on http://127.0.0.1:5000 ...
echo ======================================================================
echo.

:: Launch browser after short delay in background
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:5000"

:: Run the Flask application
python app.py

pause
