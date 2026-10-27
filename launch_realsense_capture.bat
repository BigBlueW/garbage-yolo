@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" "realsense_capture_gui.py"
if errorlevel 1 pause
endlocal
