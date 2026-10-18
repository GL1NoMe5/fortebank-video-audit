@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
"%~dp0.venv\Scripts\python.exe" -u build_timeline_memory.py
if errorlevel 1 goto failed
"%~dp0.venv\Scripts\python.exe" render_timeline_report.py
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\start_local_working.ps1
goto end
:failed
echo ERROR: complete reference profiles are preserved in runtime/timeline_memory.
:end
pause
