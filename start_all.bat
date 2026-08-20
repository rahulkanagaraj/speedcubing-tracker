@echo off
echo ===================================================
echo             Starting CubeMind AI Suite
echo ===================================================
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
