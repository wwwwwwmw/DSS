@echo off
setlocal
cd /d %~dp0

if not exist .venv\Scripts\python.exe (
  echo [DSS] Creating venv...
  python -m venv .venv || exit /b 1
)

echo [DSS] Installing requirements...
.venv\Scripts\python.exe -m pip install -r requirements.txt || exit /b 1

echo [DSS] Starting web at http://127.0.0.1:5002 ...
.venv\Scripts\python.exe app.py
