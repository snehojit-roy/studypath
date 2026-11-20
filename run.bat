@echo off
title StudyPath — Concept-Gap Tutor Launcher
echo ========================================================
echo        StudyPath — Concept-Gap Tutor Launcher
echo ========================================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b
)

:: 2. Create virtual environment if not already present
if not exist "venv" (
    echo [*] Creating isolated virtual environment (venv)...
    python -m venv venv
)

:: 3. Activate environment
call venv\Scripts\activate.bat

:: 4. Install / verify dependencies
echo [*] Checking dependencies...
pip install -r requirements.txt --quiet

:: 5. Launch application
echo.
echo ========================================================
echo  Launching StudyPath at http://localhost:8501
echo  Press Ctrl+C to stop the server
echo ========================================================
echo.
streamlit run app.py

pause
