@echo off
setlocal
cd /d %~dp0

if not exist .venv\Scripts\python.exe (
  echo [DSS] Creating venv...
  python -m venv .venv || exit /b 1
)

echo [DSS] Installing requirements...
.venv\Scripts\python.exe -m pip install -r requirements.txt || exit /b 1

echo [DSS] Training model (FAST + progress)...
.venv\Scripts\python.exe train.py --cars .\cars.csv --out .\models\car_advisor_rf.pkl --fast
