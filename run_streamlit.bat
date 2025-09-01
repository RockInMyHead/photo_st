@echo off
echo Starting FaceSort Streamlit Web Interface...
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
    echo Warning: tkinter not available. Local folder picker will not work.
    echo You can still use the browser-based folder selection.
    echo.
)

REM Start Streamlit app
echo Starting Streamlit server...
echo.
echo Available at: http://localhost:8501
echo.
echo Features:
echo - Native Windows Explorer integration
echo - Browser-based folder selection (fallback)
echo - File explorer with image thumbnails
echo - Processing queue system
echo - Real-time logs with detailed information
echo.
echo HOW TO USE:
echo 1. Go to "🗂️ Проводник" tab
echo 2. Click "🔍 Выбрать папку через Windows проводник"
echo 3. Choose your folder with photos
echo 4. Click "🚀 Начать обработку" or "➕ Добавить в очередь"
echo 5. Go to "🚀 Обработка" tab - processing starts automatically!
echo.
echo Main Interface:
echo - Simplified file explorer with folder picker
echo - Image thumbnails and file icons
echo - Queue system for batch processing
echo - Native Windows Explorer integration
echo - Clean, user-friendly interface
echo.
echo Results Storage:
echo - Results saved to ~/FaceSort_Results/ with timestamps
echo - Temporary files in ~/FaceSort_Temp/ (auto-cleaned)
echo - Easy access to processed folders
echo.
streamlit run app.py

pause
