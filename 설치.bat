@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
title 데이터올레 설치
echo ==================================================
echo   데이터올레 설치 - 처음 한 번만 실행합니다
echo ==================================================
set "PY="
py -3.13 -c "import sys" >nul 2>nul && set "PY=py -3.13"
if defined PY goto :havepy
python -c "import sys; sys.exit(0 if (3, 12) <= sys.version_info[:2] <= (3, 14) else 1)" >nul 2>nul && set "PY=python"
if defined PY goto :havepy
echo.
echo [안내] Python 3.13이 설치되어 있지 않습니다(3.12~3.14만 지원, 3.15 이상은 패키지가 아직 없습니다).
echo   1. https://www.python.org/downloads/windows/ 에서
echo      Python 3.13.17 - Windows installer 64-bit 를 내려받아 실행합니다.
echo   2. 첫 화면 아래의 "Add python.exe to PATH" 에 체크하고 "Install Now" 를 누릅니다.
echo   3. 설치가 끝나면 이 창을 닫고 설치.bat 을 다시 실행합니다.
echo.
pause
exit /b 1

:havepy
echo [1/4] Python 확인
%PY% --version
if exist ".venv\Scripts\python.exe" goto :venvok
echo [2/4] 프로그램 전용 환경(.venv)을 만드는 중...
%PY% -m venv .venv
if errorlevel 1 goto :fail
goto :pkgs
:venvok
echo [2/4] 프로그램 전용 환경(.venv)이 이미 있습니다.

:pkgs
echo [3/4] 패키지를 설치하는 중... 처음에는 5~10분 걸립니다(약 120MB 내려받기).
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --no-input --only-binary=:all: -r requirements-lock.txt
if errorlevel 1 goto :fail
echo [4/4] 설정 폴더를 준비하는 중...
if not exist "data" mkdir "data"
if not exist "data\.env" copy /y ".env.example" "data\.env" >nul
set "PYTHONUTF8=1"
".venv\Scripts\python.exe" "tools\preflight.py"
if errorlevel 1 goto :blocked
echo.
echo 설치가 끝났습니다. 이제 실행.bat 을 더블클릭하세요.
echo.
pause
exit /b 0

:blocked
echo.
echo [안내] 패키지는 내려받았지만 점검에서 조치가 필요한 항목이 나왔습니다.
echo   - [차단] 이 보이면: 보안정책이 서명 없는 확장 모듈(.pyd, .dll) 실행을 막은 것입니다.
echo     data\점검결과.txt 의 "허용 등록이 필요한 폴더와 파일"을 운영자에게 전달해 허용 등록을 요청하세요.
echo   - 허용 등록 뒤 점검.bat 을 실행해 "정상"이 나오면 실행.bat 을 더블클릭합니다.
echo   - 자세한 방법은 docs\설치_매뉴얼.pdf 3-1절을 보세요.
echo.
pause
exit /b 1

:fail
echo.
echo [오류] 설치 중 문제가 생겼습니다. 위의 메시지를 확인하세요.
echo   - 인터넷에서 pypi.org, files.pythonhosted.org 에 접속할 수 있어야 합니다.
echo   - 기관 프록시가 있으면 이 창에서 set HTTPS_PROXY=http://주소:포트 를 입력한 뒤 다시 실행하세요.
echo   - 자세한 방법은 docs\설치_매뉴얼.pdf 를 보세요.
echo.
pause
exit /b 1
