#!/bin/bash

echo "========================================"
echo "FaceSort with Native Explorer Support"
echo "========================================"
echo
echo "This version enables native system explorer integration"
echo "for folder selection."
echo

# Check if virtual environment exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    echo "Activating virtual environment (Windows)..."
    source venv/Scripts/activate
else
    echo "Virtual environment not found. Please run:"
    echo "python -m venv venv"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Check if requirements are installed
python -c "import streamlit, insightface, sklearn, cv2, PIL" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing missing dependencies..."
    pip install -r requirements.txt
fi

# Check for system dialog tools
echo "Checking system dialog tools..."
echo

# Check for Linux dialog tools
if command -v zenity &> /dev/null; then
    echo "✓ zenity found (Linux native folder picker ready)"
else
    echo "! zenity not found (install: sudo apt-get install zenity)"
    echo "  You can still use browser-based folder selection."
fi

# Check for macOS dialog tools
if command -v osascript &> /dev/null; then
    echo "✓ osascript found (macOS native folder picker ready)"
else
    echo "! osascript not found (macOS dialog may not work)"
    echo "  You can still use browser-based folder selection."
fi

echo
echo "========================================"
echo "Starting FaceSort with Explorer Support"
echo "========================================"
echo
echo "Available at: http://localhost:8501"
echo
echo "NEW FEATURES:"
echo "- Native system file explorer integration"
echo "- System dialogs for folder selection"
echo "- Improved folder navigation"
echo "- Clean, focused interface"
echo "- Permanent results storage in user home"
echo "- Automatic temp file cleanup"
echo
echo "HOW TO USE:"
echo "1. Go to '🗂️ Проводник' tab"
echo "2. Click '🔍 Выбрать папку через Windows проводник'"
echo "3. Choose your folder with photos"
echo "4. Click '🚀 Начать обработку' or '➕ Добавить в очередь'"
echo "5. Go to '🚀 Обработка' tab - processing starts automatically!"
echo
echo "========================================"

# Start Streamlit with headless=false for local system integration
streamlit run app.py --server.headless false
