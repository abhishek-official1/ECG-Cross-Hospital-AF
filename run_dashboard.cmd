@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -c "import streamlit, torch, wfdb" >nul 2>&1
if errorlevel 1 (
    echo Installing dashboard and inference dependencies in the project environment...
    ".venv\Scripts\python.exe" -m pip install -r requirements-inference.txt
    if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py
