@echo off
echo ========================================
echo Data Analyst Project - Phase 1 Setup
echo ========================================
echo.

echo Installing required packages...
pip install -r requirements.txt

echo.
echo Running main setup script...
python main.py

echo.
echo Setup complete! Press any key to exit.
pause >nul
