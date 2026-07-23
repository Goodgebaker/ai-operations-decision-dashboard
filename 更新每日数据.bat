@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "AI_MONITOR_PYTHON=.venv\Scripts\python.exe"
) else (
  set "AI_MONITOR_PYTHON=python"
)

"%AI_MONITOR_PYTHON%" scripts\update_daily_data.py
if errorlevel 1 (
  echo.
  echo 更新失败。请查看上方错误；已通过校验的原始文件可能已安全移入 archive 目录。
  pause
  exit /b 1
)

echo.
echo 更新完成，可以重新打开或刷新看板。
pause
