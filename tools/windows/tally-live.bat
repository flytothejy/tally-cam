@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
rem ================================================================
rem  tally-cam 라이브 계수 — 더블클릭으로 실행
rem  처음 한 번은 설치(10~20분), 다음부터는 바로 시작해요.
rem  아래 값만 바꿔 쓰세요.
rem ================================================================
set "CAMERA=L010426"
set "TITLE=서울 역삼역 앞 테헤란로"
rem GATES: auto = 처음 2분 30초 동안 차 흐름을 보고 기준선을 자동으로 정함 / 또는 기준선 JSON 경로
set "GATES=auto"
rem OUT: preview = 이 PC 화면에만 띄움(q로 끝) / live.mp4 = 파일로 저장 / rtmp = 유튜브(YT_STREAM_KEY 필요, UTIC 약관 확인 후)
set "OUT=preview"
rem ================================================================

where py >nul 2>nul
if errorlevel 1 (
  echo [!] 파이썬이 없어요. https://www.python.org/downloads/ 에서 3.11 또는 3.12를 설치하세요.
  echo     설치 첫 화면에서 "Add python.exe to PATH"를 꼭 체크하세요.
  pause & exit /b 1
)

echo [1/4] 최신 코드 받는 중...
powershell -NoProfile -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri https://github.com/flytothejy/tally-cam/archive/refs/heads/main.zip -OutFile app.zip; Expand-Archive -Path app.zip -DestinationPath . -Force; Remove-Item app.zip"
if not exist "tally-cam-main\tools\live.py" ( echo [!] 코드를 받지 못했어요. 인터넷 연결을 확인하세요. & pause & exit /b 1 )

if not exist "tally-cam-main\.env" (
  if exist "..\tally-cam\.env" ( copy /y "..\tally-cam\.env" "tally-cam-main\.env" >nul ) else (
    echo [!] UTIC 키 파일^(.env^)이 없어요. Project\tally-cam\.env 를 이 폴더의 tally-cam-main 안에 넣어 주세요.
    pause & exit /b 1 )
)

set "MODEL=nano"
set "TORCH_IDX=https://download.pytorch.org/whl/cpu"
nvidia-smi >nul 2>nul
if not errorlevel 1 ( set "MODEL=small" & set "TORCH_IDX=https://download.pytorch.org/whl/cu124" )

if not exist "venv\Scripts\python.exe" (
  echo [2/4] 파이썬 환경 만드는 중...
  py -3 -m venv venv || ( echo [!] 가상환경을 만들지 못했어요. & pause & exit /b 1 )
)
if not exist "venv\installed.txt" (
  echo [3/4] 패키지 설치 중^(처음 한 번, 10~20분^)...
  "venv\Scripts\python.exe" -m pip install -q -U pip
  "venv\Scripts\python.exe" -m pip install -q torch torchvision --index-url %TORCH_IDX% || ( echo [!] torch 설치 실패 & pause & exit /b 1 )
  "venv\Scripts\python.exe" -m pip install -q rfdetr ultralytics lap opencv-python pillow || ( echo [!] 패키지 설치 실패 & pause & exit /b 1 )
  echo ok> "venv\installed.txt"
)

if /i not "%OUT%"=="preview" (
  where ffmpeg >nul 2>nul
  if errorlevel 1 ( echo [!] 파일 저장·유튜브 송출에는 ffmpeg가 필요해요: 관리자 PowerShell에서  winget install Gyan.FFmpeg & pause & exit /b 1 )
)

echo [4/4] 시작^(검출 모델: %MODEL%^) — 창에서 q를 누르면 끝나요.
"venv\Scripts\python.exe" "tally-cam-main\tools\live.py" --utic %CAMERA% --gates "%GATES%" --title "%TITLE%" --model %MODEL% --out "%OUT%"
pause
