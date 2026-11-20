@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
title 데이터올레
if not exist ".venv\Scripts\python.exe" goto :noinstall
set "PYTHONUTF8=1"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
echo ==================================================
echo   데이터올레를 시작합니다.
echo   잠시 후 브라우저에 http://127.0.0.1:8501 이 열립니다.
echo   화면이 비어 있거나 연결할 수 없다고 나오면 10초쯤 뒤 새로 고침(F5) 하세요.
echo   이 검은 창을 닫으면 프로그램이 종료됩니다.
echo ==================================================
start "" /min cmd /c "timeout /t 10 /nobreak >nul & start http://127.0.0.1:8501"
".venv\Scripts\python.exe" -m streamlit run "src\dataolle\app\main.py" --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
echo.
echo 프로그램이 종료되었습니다.
pause
exit /b 0

:noinstall
echo 먼저 설치.bat 을 실행하세요.
pause
exit /b 1
