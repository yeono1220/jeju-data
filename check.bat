@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
title 데이터올레 설치 점검
if not exist ".venv\Scripts\python.exe" goto :noinstall
set "PYTHONUTF8=1"
".venv\Scripts\python.exe" "tools\preflight.py"
echo.
echo 자세한 결과는 data\점검결과.txt 에 저장했습니다.
pause
exit /b 0

:noinstall
echo 먼저 설치.bat 을 실행하세요.
pause
exit /b 1
