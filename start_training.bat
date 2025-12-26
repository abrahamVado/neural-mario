@echo off
echo 🍄 Starting Mario RL Training...
echo.

cd /d "%~dp0"
.\.venv\Scripts\python.exe train_mario.py

pause
