@echo off
echo ========================================
echo FaceSort with Native Explorer Support
echo ========================================
echo.
echo This version enables native Windows Explorer integration
echo for folder selection.
echo.

REM Check if virtual environment exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Please run:
    echo python -m venv venv
    echo pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if requirements are installed
python -c "import streamlit, insightface, sklearn, cv2, PIL" 2>nul
if %errorlevel% neq 0 (
    echo Installing missing dependencies...
    pip install -r requirements.txt
)

REM Check for tkinter (needed for local folder picker on Windows)
python -c "import tkinter" 2>nul
if %errorlevel% neq 0 (
    echo ========================================
    echo WARNING: tkinter not found!
    echo ========================================
    echo.
    echo tkinter is required for native Windows Explorer integration.
    echo.
    echo Solutions:
    echo 1. Install tkinter: pip install tk
    echo 2. Or reinstall Python from python.org (includes tkinter)
    echo.
    echo You can still use browser-based folder selection.
    echo ========================================
    echo.
)

echo ========================================
echo Starting FaceSort with Explorer Support
echo ========================================
echo.
echo Available at: http://localhost:8501
echo.
echo NEW FEATURES:
echo - Native Windows Explorer folder picker
echo - System file dialogs for folder selection
echo - Improved folder navigation
echo - Clean, focused interface
echo - Permanent results storage in user home
echo - Automatic temp file cleanup
echo.
echo HOW TO USE:
echo 1. Go to "🗂️ Проводник" tab
echo 2. Click "🔍 Выбрать папку через Windows проводник"
echo 3. Choose your folder with photos
echo 4. Click "🚀 Начать обработку" or "➕ Добавить в очередь"
echo 5. Go to "🚀 Обработка" tab - processing starts automatically!
echo.
echo ========================================

REM Start Streamlit with headless=false for local system integration
streamlit run app.py --server.headless false

pause
