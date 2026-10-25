@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" order_label_tool.py
) else (
  python order_label_tool.py
)
if errorlevel 1 pause
