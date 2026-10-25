@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" "image_test_gui.py"
) else (
  python "image_test_gui.py"
)
