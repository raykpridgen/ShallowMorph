@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Launch morph_wrap.run_sweep in a detached cmd process and log everything.
REM Usage examples:
REM   run_sweep_background.bat --sweep A --execute --continue-on-failure
REM   run_sweep_background.bat --sweep B --dry-run

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
set "LOG_DIR=%REPO_ROOT%\out"
set "LOG_FILE=%LOG_DIR%\run_sweep_background.log"
set "VENV_PY=%REPO_ROOT%\.venv\Scripts\python.exe"
set "PY_CMD=python"
if exist "%VENV_PY%" set "PY_CMD=%VENV_PY%"
if /I "%RUN_SWEEP_BG_WORKER%"=="1" goto worker

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo Starting background sweep...
echo Log file: "%LOG_FILE%"

start "MORPH Sweep" /min cmd /v:on /c "set RUN_SWEEP_BG_WORKER=1&& call ""%~f0"" %*"

echo Started. You can lock the computer; process will continue in background.
echo To monitor:
echo   powershell -NoProfile -Command "Get-Content -Path '%LOG_FILE%' -Wait"

endlocal
exit /b 0

:worker
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
cd /d "%REPO_ROOT%"
set "PYTHONPATH=%REPO_ROOT%\code;%PYTHONPATH%"

(
echo.
echo ===== START %DATE% %TIME% =====
echo CWD: %CD%
echo PY_CMD: %PY_CMD%
"%PY_CMD%" --version
echo Command: "%PY_CMD%" -m morph_wrap.run_sweep %*
"%PY_CMD%" -m morph_wrap.run_sweep %*
set "RUN_EXIT=!ERRORLEVEL!"
echo ===== END %DATE% %TIME% exit=!RUN_EXIT! =====
) >> "%LOG_FILE%" 2>&1

exit /b %RUN_EXIT%
