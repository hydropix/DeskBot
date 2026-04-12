@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)
echo.
echo   Press R in the viewer to generate a NEW random terrain
echo.
python -m deskbot --random %*
