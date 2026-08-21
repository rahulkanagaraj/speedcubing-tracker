@echo off
echo ===================================================
echo             Starting CubeMind AI Suite
echo ===================================================
echo.

:: Check if virtual environment folder exists
if not exist venv (
    echo [Setup] Virtual environment "venv" not found.
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [Error] Failed to create virtual environment. Please ensure Python is installed and in your PATH.
        pause
        exit /b 1
    )
    echo [Setup] Virtual environment created successfully.
    echo.
)

:: Install/update requirements
echo [Setup] Checking and installing missing dependencies...
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo [Error] Failed to install requirements. Please check your internet connection.
    pause
    exit /b 1
)
echo [Setup] Dependencies verified and ready.
echo.

echo Launching Unified Web Server & API...
start "CubeMind - Web Server" cmd /k ".\venv\Scripts\python.exe server.py"
echo.
echo ===================================================
echo  Server is active!
echo  Open http://localhost:5173/index.html in your browser.
echo ===================================================
echo.
pause
