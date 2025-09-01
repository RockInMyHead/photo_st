#!/bin/bash

echo "Starting FaceSort Streamlit Web Interface..."
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

# Check for Linux dialog tools
if command -v zenity &> /dev/null; then
    echo "✓ zenity found (Linux folder picker available)"
else
    echo "! zenity not found (install with: sudo apt-get install zenity)"
fi

# Check for macOS dialog tools
if command -v osascript &> /dev/null; then
    echo "✓ osascript found (macOS folder picker available)"
else
    echo "! osascript not found (macOS dialog may not work)"
fi

echo
echo "Starting Streamlit server..."
echo "App will be available at: http://localhost:8501"
echo
echo "Features:"
echo "- Native system file explorer integration"
echo "- Browser-based folder selection (fallback)"
echo "- File explorer with image thumbnails"
echo "- Processing queue system"
echo "- Real-time logs with detailed information"
echo
echo "HOW TO USE:"
echo "1. Go to '🗂️ Проводник' tab"
echo "2. Click '🔍 Выбрать папку через Windows проводник'"
echo "3. Choose your folder with photos"
echo "4. Click '🚀 Начать обработку' or '➕ Добавить в очередь'"
echo "5. Go to '🚀 Обработка' tab - processing starts automatically!"
echo ""
echo "Main Interface:"
echo "- Simplified file explorer with folder picker"
echo "- Image thumbnails and file icons"
echo "- Queue system for batch processing"
echo "- Native system folder picker integration"
echo "- Clean, user-friendly interface"
echo ""
echo "Results Storage:"
echo "- Results saved to ~/FaceSort_Results/ with timestamps"
echo "- Temporary files in ~/FaceSort_Temp/ (auto-cleaned)"
echo "- Easy access to processed folders"
echo
echo "System Requirements:"
if command -v zenity &> /dev/null; then
    echo "✓ zenity available (Linux native folder picker ready)"
else
    echo "! zenity not available (install: sudo apt-get install zenity)"
fi

if command -v osascript &> /dev/null; then
    echo "✓ osascript available (macOS native folder picker ready)"
else
    echo "! osascript not available (macOS dialog may not work)"
fi
echo
streamlit run app.py
