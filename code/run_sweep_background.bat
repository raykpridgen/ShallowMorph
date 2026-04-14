@echo off
setlocal EnableExtensions

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

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo Starting background sweep...
echo Log file: "%LOG_FILE%"

start "MORPH Sweep" /min cmd /c ^
"cd /d ""%REPO_ROOT%"" && ^
set ""PYTHONPATH=%REPO_ROOT%\code;%PYTHONPATH%"" && ^
echo.>> ""%LOG_FILE%"" && ^
echo ===== START %DATE% %TIME% =====>> ""%LOG_FILE%"" && ^
echo CWD: %REPO_ROOT%>> ""%LOG_FILE%"" && ^
echo PY_CMD: %PY_CMD%>> ""%LOG_FILE%"" && ^
""%PY_CMD%"" --version >> ""%LOG_FILE%"" 2>&1 && ^
echo Command: ""%PY_CMD%"" -m morph_wrap.run_sweep %* >> ""%LOG_FILE%"" && ^
""%PY_CMD%"" -m morph_wrap.run_sweep %* >> ""%LOG_FILE%"" 2>&1 && ^
echo ===== END %DATE% %TIME% exit=0 =====>> ""%LOG_FILE%"" || ^
echo ===== END %DATE% %TIME% exit=1 =====>> ""%LOG_FILE%"""

echo Started. You can lock the computer; process will continue in background.
echo To monitor:
echo   powershell -NoProfile -Command "Get-Content -Path '%LOG_FILE%' -Wait"

endlocal
